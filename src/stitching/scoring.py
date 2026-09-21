"""
src.stitching.scoring - 跨镜连接评分模块

对候选边进行多维度评分:

跨镜连接得分 = w1*外观相似度 + w2*属性一致性 + w3*车牌一致性
              + w4*时间可达性 + w5*空间可达性 + w6*方向一致性
              - p1*路径分叉惩罚 - p2*观测缺失惩罚

车辆侧权重: 车牌(0.35) > 时间(0.40) > 拓扑(0.15) > ReID(0.00) = 属性(0.00)
行人侧权重: 时间(0.35) > ReID(0.30) > 属性(0.25) > 背包(0.10)

车辆侧 ReID 与属性的软评分权重为 **0**，这不是遗漏（P-A 网格搜索实测）：
跨镜外观连续相似度的可分性很弱（d-prime 仅 0.78）、属性跨镜一致率仅 28–37%，
两者按权重加进加权和只会给所有候选加上同一份噪声、压缩区分度。
它们改由 `candidate_edge.py` 的硬门控承担（规则 7 属性门控、规则 8 外观门控，
阈值 `stitching.min_appearance_score`，默认 0.95 → 精度 0.97）。
**同一个弱信号，放进加权和是噪声，放进阈值是过滤器。**

⚠️ **本数据集上的实际生效权重（实测 `explain_dimensions()`）**
五维配置权重之和是 0.90，余下 0.10 由方向加分项补齐（总分上限 1.00）。
但 `plate` 这一维在本数据集上**恒为无证据** —— 实测 68,349 条检测中
带 `plate_number` 的为 **0 条**，因此它每次都被整维剔除。结果是：

    配置权重: plate 0.35 / temporal 0.40 / topology 0.15 / reid 0.0 / attribute 0.0
    生效权重: temporal 0.6545 / topology 0.2455          （plate 被剔除）

即"六维评分函数"在当前数据上**实际退化为「时间可达性 + 路网拓扑 + 方向」三项**。
这不是缺陷，而是缺失维度重分摊正常工作；但读这段代码时必须知道它，
否则会以为车牌一致性真的在参与打分。

**权重唯一来源是 `configs/default.yaml` 的 `stitching.weights`**（T6）：
不传 `vehicle_weights` / `pedestrian_weights` 时本模块自动从配置读取；只在
配置不可用时才退回本文件里的 `DEFAULT_*_WEIGHTS` 兜底值（兜底值已与配置
对齐，两者不再各说各话）。

权重缺失维度的处理（T6）
------------------------
某个分项"没有证据"时（如 ReID 与 CLIP 向量都没有 → 外观分恒为中性 0.5，
或双方都没有车牌 → 车牌分恒为中性 0.5），把它按原权重乘进加权和只会给
**所有候选**加上同一个常数偏移，白白压缩区分度。因此默认启用
`reweight_missing_dimensions`（可由 `stitching.reweight_missing_dimensions`
关闭）：把无证据维度的权重**按比例分摊**到有证据的维度上，权重总和与原来
一致（不会放大任何分项），被剔除的维度在 `reasoning` 与原子的
`explain_dimensions()` 里显式列出，可测试、可观测。

每个评分分项都有独立的计算方法，方便后续消融实验。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.config import get_config
from src.common.data_models import CrossCameraEdge, Tracklet
from src.common.logger import get_logger
from src.common.utils import cosine_similarity, time_diff_seconds
from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology

logger = get_logger("stitching.scoring")

# ============================================================
# 常量定义
# ============================================================

# 城区合理速度范围 (km/h)
MIN_CITY_SPEED_KMH = 20.0
MAX_CITY_SPEED_KMH = 80.0

# 分项"无证据"时的中性分（不奖励也不惩罚）
NEUTRAL_SCORE = 0.5

# 方向一致性是额外加分项，不计入 stitching.weights 的主权重归一化
DIRECTION_BONUS_WEIGHT = 0.10

# 配置缺失时的兜底权重——**必须与 configs/default.yaml 的 stitching.weights 一致**，
# 否则又会出现"配置一套、代码一套"的两套权重问题（T6 修的正是这个）
DEFAULT_VEHICLE_WEIGHTS: Dict[str, float] = {
    "plate": 0.35,
    "temporal": 0.40,
    "topology": 0.15,
    # reid / attribute 软评分权重置零：P-A 网格搜索实测（eval_chain_idf1.py）
    # 它们的连续相似度值跨镜不可分（d-prime 0.78 / 属性一致率 28-37%），
    # 加权进软评分只会加噪声。真正的过滤由规则 8（外观硬门控）与规则 7（属性硬门控）承担。
    "reid": 0.0,
    "attribute": 0.0,
}
DEFAULT_PEDESTRIAN_WEIGHTS: Dict[str, float] = {
    "temporal": 0.35,
    "reid": 0.30,
    "attribute": 0.25,
    "bag": 0.10,
}
DEFAULT_PENALTIES: Dict[str, float] = {
    "path_divergence": 0.10,
    "observation_missing": 0.05,
}


def load_stitching_weights(
    config: Any = None,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    从 `configs/default.yaml` 的 `stitching` 段读取权重与惩罚项

    Args:
        config: Config 实例；缺省用全局单例 `get_config()`

    Returns:
        (vehicle_weights, pedestrian_weights, penalties)；配置读取失败或缺少
        对应段落时该项为 None（由调用方退回 DEFAULT_* 兜底值）
    """
    try:
        cfg = config if config is not None else get_config()
        weights_cfg = cfg.get("stitching.weights", {}) or {}
        penalties_cfg = cfg.get("stitching.penalties", {}) or {}
    except Exception as e:  # 配置文件缺失/损坏不应让评分器初始化失败
        logger.warning("读取 stitching 权重配置失败，使用内置兜底权重: %s", e)
        return None, None, None

    vehicle = weights_cfg.get("vehicle") or None
    pedestrian = weights_cfg.get("pedestrian") or None
    penalties = penalties_cfg or None
    if vehicle is None or pedestrian is None:
        logger.warning(
            "configs/default.yaml 的 stitching.weights 缺少 vehicle/pedestrian，"
            "缺失项将使用内置兜底权重"
        )
    return vehicle, pedestrian, penalties

# 方向映射: 方向字符串 → 角度 (正北为0°, 顺时针)
DIRECTION_TO_ANGLE: Dict[str, float] = {
    "eastbound": 90.0,        # 由西向东
    "westbound": 270.0,       # 由东向西
    "northbound": 0.0,        # 由南向北
    "southbound": 180.0,      # 由北向南
    "east": 90.0,
    "west": 270.0,
    "north": 0.0,
    "south": 180.0,
    "northeast": 45.0,
    "northwest": 315.0,
    "southeast": 135.0,
    "southwest": 225.0,
    "east-west": 90.0,
    "north-south": 0.0,
    "northeast-southwest": 45.0,
    "northwest-southeast": 315.0,
}

# 车辆属性比较的键及其权重
VEHICLE_ATTRIBUTE_KEYS = ["color", "vehicle_type"]
VEHICLE_ATTRIBUTE_WEIGHTS = {"color": 0.5, "vehicle_type": 0.5}

# 行人属性比较的键及其权重
PEDESTRIAN_ATTRIBUTE_KEYS = ["gender", "clothing_color", "bag", "bag_color"]
PEDESTRIAN_ATTRIBUTE_WEIGHTS = {
    "gender": 0.25,
    "clothing_color": 0.30,
    "bag": 0.20,
    "bag_color": 0.25,
}


class CrossCameraScorer:
    """
    跨镜连接评分器

    对候选连接边进行多维度综合评分，支持车辆和行人两种目标类型的差异化权重。

    评分公式:
        score = Σ(wi * score_i) - Σ(pj * penalty_j)

    使用方式:
        scorer = CrossCameraScorer(camera_manager, road_topology)
        edge = scorer.score(source_tracklet, target_tracklet)
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        road_topology: RoadTopology,
        vehicle_weights: Optional[Dict[str, float]] = None,
        pedestrian_weights: Optional[Dict[str, float]] = None,
        penalties: Optional[Dict[str, float]] = None,
        min_speed_kmh: float = MIN_CITY_SPEED_KMH,
        max_speed_kmh: float = MAX_CITY_SPEED_KMH,
        camera_time_offsets: Optional[Dict[str, float]] = None,
        config: Any = None,
    ) -> None:
        """
        初始化评分器

        权重优先级（T6）:
            显式传参 > `configs/default.yaml` 的 `stitching.weights` > 内置兜底值
        三者数值口径一致，不再出现"配置一套、代码一套"。

        Args:
            camera_manager: 摄像头管理器
            road_topology: 道路拓扑
            vehicle_weights: 车辆评分权重字典（None 则读配置）
            pedestrian_weights: 行人评分权重字典（None 则读配置）
            penalties: 惩罚项权重字典（None 则读配置）
            min_speed_kmh: 城区最低速度 (km/h)
            max_speed_kmh: 城区最高速度 (km/h)
            camera_time_offsets: 摄像头 → 全局时间偏移(秒)；传入后时间/空间可达性
                会先把本机时间换算成全局时间再算跨镜间隔（T5 对齐）。None 表示
                不做换算，等价于旧行为（本机时间直接相减）
            config: Config 实例；缺省用全局单例（仅在需要读权重时使用）
        """
        self.camera_manager = camera_manager
        self.road_topology = road_topology
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh

        # 摄像头全局时间偏移（T5）：空字典/None → 不做换算
        self.camera_time_offsets: Dict[str, float] = dict(camera_time_offsets or {})

        # 权重：显式传参 > 配置 > 兜底（兜底值与配置对齐）
        cfg_vehicle, cfg_pedestrian, cfg_penalties = load_stitching_weights(config)
        self.vehicle_weights = vehicle_weights or cfg_vehicle or dict(DEFAULT_VEHICLE_WEIGHTS)
        self.pedestrian_weights = (
            pedestrian_weights or cfg_pedestrian or dict(DEFAULT_PEDESTRIAN_WEIGHTS)
        )
        self.penalties = penalties or cfg_penalties or dict(DEFAULT_PENALTIES)

        # "生效权重是否等于配置里的值"——显式传入但值与配置相同的也算（构建器
        # 就是把配置值透传进来的），避免这个标志位对线上路径失真
        self.weights_from_config = bool(
            cfg_vehicle is not None
            and (vehicle_weights is None or dict(vehicle_weights) == dict(cfg_vehicle))
        )

        # 无证据维度是否剔除并重新分摊权重（T6）
        self.reweight_missing_dimensions = True
        try:
            cfg = config if config is not None else get_config()
            self.reweight_missing_dimensions = bool(
                cfg.get("stitching.reweight_missing_dimensions", True)
            )
        except Exception as e:
            logger.warning("读取 reweight_missing_dimensions 失败，保持默认 True: %s", e)

        logger.info(
            "跨镜连接评分器初始化 | 车辆权重=%s(来源=%s) | 行人权重=%s | 惩罚=%s | "
            "缺失维度重分摊=%s | 时间偏移=%d 个摄像头",
            self.vehicle_weights, "配置" if self.weights_from_config else "传参/兜底",
            self.pedestrian_weights, self.penalties,
            self.reweight_missing_dimensions, len(self.camera_time_offsets),
        )

    # ================================================================
    # 核心评分入口
    # ================================================================

    def score(
        self,
        source: Tracklet,
        target: Tracklet,
    ) -> CrossCameraEdge:
        """
        计算两个 Tracklet 之间的跨镜连接综合评分

        流程:
            1. 根据目标类别选择权重配置
            2. 计算各维度分项分数
            3. 剔除"无证据"维度并把其权重按比例分摊给有证据的维度（T6）
            4. 加权求和得到基础分
            5. 计算并减去惩罚项
            6. 截断到 [0, 1] 范围
            7. 生成可解释推理说明（含有效权重与剔除明细）

        Args:
            source: 源 Tracklet (时间较早)
            target: 目标 Tracklet (时间较晚)

        Returns:
            带完整评分信息的跨镜候选连接边
        """
        target_type = source.target_type.lower()

        # ---------- 选择权重 ----------
        if target_type == "vehicle":
            weights = self.vehicle_weights
        elif target_type == "pedestrian":
            weights = self.pedestrian_weights
        else:
            # 非机动车等默认使用行人权重
            weights = self.pedestrian_weights

        # ---------- 维度可用性 + 有效权重（T6）----------
        availability = self._dimension_availability(source, target)
        effective_weights, dropped = self._effective_weights(weights, availability)

        # ---------- 计算各维度分数 ----------
        appearance_score = self._score_appearance(source, target)
        attribute_score = self._score_attribute(source, target)
        plate_score = self._score_plate(source, target)
        temporal_score = self._score_temporal(source, target)
        spatial_score = self._score_spatial(source, target)
        direction_score = self._score_direction(source, target)

        # ---------- 计算惩罚 ----------
        path_div_penalty, obs_missing_penalty = self._compute_penalty_components(
            source, target
        )
        total_penalty = (
            self.penalties["path_divergence"] * path_div_penalty
            + self.penalties["observation_missing"] * obs_missing_penalty
        )

        # ---------- 加权求和 ----------
        # 根据权重配置映射到各分项
        weighted_score = self._weighted_sum(
            weights=effective_weights,
            appearance_score=appearance_score,
            attribute_score=attribute_score,
            plate_score=plate_score,
            temporal_score=temporal_score,
            spatial_score=spatial_score,
            direction_score=direction_score,
        )

        # ---------- 最终得分 (减去惩罚, 截断到 [0, 1]) ----------
        final_score = max(0.0, min(1.0, weighted_score - total_penalty))

        # ---------- 判断有效性 ----------
        is_valid = final_score > 0.0 and plate_score > 0.0

        # ---------- 生成推理说明 ----------
        reasoning = self._build_reasoning(
            source=source,
            target=target,
            appearance_score=appearance_score,
            attribute_score=attribute_score,
            plate_score=plate_score,
            temporal_score=temporal_score,
            spatial_score=spatial_score,
            direction_score=direction_score,
            total_penalty=total_penalty,
            final_score=final_score,
            weights=weights,
            effective_weights=effective_weights,
            dropped_dimensions=dropped,
            availability=availability,
        )

        logger.debug(
            "评分完成 | %s → %s | score=%.4f | appearance=%.3f attr=%.3f "
            "plate=%.3f temporal=%.3f spatial=%.3f direction=%.3f penalty=%.3f "
            "| 有效权重=%s | 剔除=%s",
            source.tracklet_id, target.tracklet_id, final_score,
            appearance_score, attribute_score, plate_score,
            temporal_score, spatial_score, direction_score, total_penalty,
            effective_weights, dropped,
        )

        return CrossCameraEdge(
            source_tracklet_id=source.tracklet_id,
            target_tracklet_id=target.tracklet_id,
            score=final_score,
            appearance_score=appearance_score,
            attribute_score=attribute_score,
            plate_score=plate_score,
            temporal_score=temporal_score,
            spatial_score=spatial_score,
            direction_score=direction_score,
            penalty=total_penalty,
            is_valid=is_valid,
            reasoning=reasoning,
        )

    # 兼容原始接口别名
    score_edge = score

    # ================================================================
    # 维度可用性与权重分摊 (T6: 让"无证据维度"的退化路径显式、可测、可观测)
    # ================================================================

    def _dimension_availability(
        self, source: Tracklet, target: Tracklet
    ) -> Dict[str, bool]:
        """
        判断每个评分维度这一对 Tracklet 是否**有真实证据**

        没有证据的维度分项恒为 NEUTRAL_SCORE(0.5)，把它算进加权和只会给所有
        候选加上同一个常数偏移。判定口径:

            - reid:      双方都有 ReID 或都有 CLIP 向量（与 `_score_appearance` 一致）
            - plate:     双方都有车牌（否则 `_score_plate` 恒为 0.5）
            - attribute: 至少一个属性键双方都有值（否则 `_score_attribute` 恒为 0.5）
            - temporal:  恒为 True（一定算得出时间差）
            - topology:  恒为 True（拓扑距离/可达性一定给出分数）

        Returns:
            {"reid": bool, "plate": bool, "attribute": bool, "temporal": bool,
             "topology": bool}
        """
        target_type = source.target_type.lower()
        attr_keys = (
            VEHICLE_ATTRIBUTE_KEYS if target_type == "vehicle" else PEDESTRIAN_ATTRIBUTE_KEYS
        )
        src_attrs = source.attributes or {}
        tgt_attrs = target.attributes or {}
        attribute_available = any(
            src_attrs.get(key) is not None and tgt_attrs.get(key) is not None
            for key in attr_keys
        )

        return {
            "reid": self._appearance_source(source, target) != "none",
            "plate": bool(source.has_plate and target.has_plate),
            "attribute": attribute_available,
            "temporal": True,
            "topology": True,
        }

    def _effective_weights(
        self,
        weights: Dict[str, float],
        availability: Dict[str, bool],
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        剔除无证据维度的权重，并按比例分摊到有证据的维度

        分摊方式：保持**权重总和不变**，只按原有比例重新分配。例如车辆的
        plate(0.35)+reid(0.15) 都无证据时，剩下的 temporal(0.25)、
        topology(0.15)、attribute(0.10) 会被放大 2 倍变为 0.5 / 0.3 / 0.2，
        总和仍是 1.0——因此**不会**放大任何单个分项的绝对贡献，只是让有限的
        区分度全部来自真正有信息的维度。

        关闭 `self.reweight_missing_dimensions`（配置
        `stitching.reweight_missing_dimensions=false`）时原样返回，退化为旧行为：
        无证据维度以中性 0.5 乘原权重参与求和。

        Args:
            weights: 原始权重（来自配置或构造参数）
            availability: `_dimension_availability` 的结果

        Returns:
            (有效权重, 被剔除的权重键列表)
        """
        if not self.reweight_missing_dimensions:
            return dict(weights), []

        kept: Dict[str, float] = {}
        dropped: List[str] = []
        for key, value in weights.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            # 行人背包权重落在属性分上，可用性跟随 attribute
            availability_key = "attribute" if key == "bag" else key
            if not availability.get(availability_key, True):
                dropped.append(key)
                continue
            kept[key] = weight

        total_kept = sum(kept.values())
        total_original = sum(
            float(v) for v in weights.values()
            if isinstance(v, (int, float)) and float(v) > 0
        )
        if total_kept <= 0 or total_original <= 0:
            # 所有维度都无证据 → 保持原权重（各项均为中性分，总分也就中性）
            return dict(weights), dropped

        scale = total_original / total_kept
        return {key: weight * scale for key, weight in kept.items()}, dropped

    def explain_dimensions(self, source: Tracklet, target: Tracklet) -> Dict[str, Any]:
        """
        输出一对 Tracklet 的评分维度可解释信息（供测试与线上排障）

        Returns:
            {"weights", "effective_weights", "dropped_dimensions",
             "availability", "appearance_source", "reweighting_enabled",
             "time_aligned"}
        """
        target_type = source.target_type.lower()
        weights = self.vehicle_weights if target_type == "vehicle" else self.pedestrian_weights
        availability = self._dimension_availability(source, target)
        effective_weights, dropped = self._effective_weights(weights, availability)
        return {
            "target_type": target_type,
            "weights": dict(weights),
            "effective_weights": effective_weights,
            "dropped_dimensions": dropped,
            "availability": availability,
            "appearance_source": self._appearance_source(source, target),
            "reweighting_enabled": self.reweight_missing_dimensions,
            "time_aligned": bool(
                self.camera_time_offsets.get(source.camera_id) is not None
                and self.camera_time_offsets.get(target.camera_id) is not None
            ) if self.camera_time_offsets else False,
        }

    # ================================================================
    # 分项评分方法 (每个方法独立可调用, 方便消融实验)
    # ================================================================

    def _appearance_source(self, source: Tracklet, target: Tracklet) -> str:
        """
        外观分的证据来源（显式化退化路径）

        Returns:
            "reid"（双方都有 ReID 向量）/ "clip"（回退到 CLIP 向量）/
            "none"（都没有 → 外观分只能给中性值，此时该维度不参与加权）
        """
        if source.avg_reid_vector is not None and target.avg_reid_vector is not None:
            return "reid"
        if source.avg_clip_vector is not None and target.avg_clip_vector is not None:
            return "clip"
        return "none"

    def _score_appearance(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算外观相似度

        优先使用 ReID 向量的余弦相似度；
        若无 ReID 向量则回退到 CLIP 向量。

        余弦相似度原始范围 [-1, 1]，线性映射到 [0, 1]:
            score = (cos_sim + 1) / 2

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            外观相似度 [0, 1]；无任何外观特征时返回 NEUTRAL_SCORE (0.5)。

        注意: 返回 0.5 只表示"这一维没有信息"，**不代表**两目标外观相似——是否
        把这一维算进总分由 `_dimension_availability` / `_effective_weights`
        决定（默认会把无证据维度的权重剔除并分摊给其它维度）。
        """
        source_kind = self._appearance_source(source, target)

        if source_kind == "none":
            return NEUTRAL_SCORE

        if source_kind == "reid":
            cos_sim = cosine_similarity(source.avg_reid_vector, target.avg_reid_vector)
        else:
            cos_sim = cosine_similarity(source.avg_clip_vector, target.avg_clip_vector)
        return float((cos_sim + 1.0) / 2.0)

    def _score_attribute(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算属性一致性分数

        对每个属性:
            - 匹配: 1.0 分
            - 不匹配: 0.0 分
            - 一方缺失: 0.5 分 (中性, 不惩罚也不奖励)
            - 双方缺失: 0.5 分

        最终按属性权重加权平均。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            属性一致性分数 [0, 1]
        """
        target_type = source.target_type.lower()

        if target_type == "vehicle":
            attr_keys = VEHICLE_ATTRIBUTE_KEYS
            attr_weights = VEHICLE_ATTRIBUTE_WEIGHTS
        else:
            attr_keys = PEDESTRIAN_ATTRIBUTE_KEYS
            attr_weights = PEDESTRIAN_ATTRIBUTE_WEIGHTS

        src_attrs = source.attributes or {}
        tgt_attrs = target.attributes or {}

        weighted_sum = 0.0
        total_weight = 0.0

        for key in attr_keys:
            w = attr_weights.get(key, 1.0)
            total_weight += w

            src_val = src_attrs.get(key)
            tgt_val = tgt_attrs.get(key)

            if src_val is None or tgt_val is None:
                # 一方或双方缺失 → 中性分
                weighted_sum += NEUTRAL_SCORE * w
            elif str(src_val).lower() == str(tgt_val).lower():
                # 匹配
                weighted_sum += 1.0 * w
            else:
                # 不匹配
                weighted_sum += 0.0 * w

        if total_weight == 0:
            return NEUTRAL_SCORE

        return weighted_sum / total_weight

    def _score_plate(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算车牌一致性分数

        规则:
            - 两个都有车牌且相同 → 1.0 (强证据)
            - 两个都有车牌但不同 → 0.0 (直接否决)
            - 一个有一个没有 → 0.5 (中性)
            - 两个都没有 → 0.5 (不惩罚也不奖励)

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            车牌一致性分数 [0, 1]
        """
        src_has = source.has_plate
        tgt_has = target.has_plate

        if src_has and tgt_has:
            # 双方都有车牌
            src_plate = source.plate_number.strip().upper()
            tgt_plate = target.plate_number.strip().upper()
            if src_plate == tgt_plate:
                return 1.0
            else:
                return 0.0  # 车牌冲突 → 直接否决

        # 其他情况 (一方无车牌或双方都无) → 无证据的中性分
        return NEUTRAL_SCORE

    def _aligned_travel_time(self, source: Tracklet, target: Tracklet) -> float:
        """
        跨镜旅行时间（秒）: target 首现 − source 末现

        传入 `camera_time_offsets` 时先把双方的本机时间换算成全局时间再相减
        （T5：各摄像头的 timestamp 都从 0 起跳，直接相减会得到错位的负值）。
        未传偏移表时就是本机时间相减，与旧行为完全一致。
        """
        gap = time_diff_seconds(source.end_time, target.start_time)
        if not self.camera_time_offsets:
            return gap
        return gap + (
            self.camera_time_offsets.get(target.camera_id, 0.0)
            - self.camera_time_offsets.get(source.camera_id, 0.0)
        )

    def _score_temporal(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算时间可达性分数

        流程:
            1. 计算实际旅行时间 = target.start_time - source.end_time
               （传入摄像头偏移表时按全局时间计算，见 `_aligned_travel_time`）
            2. 获取摄像头间距离, 计算合理旅行时间范围:
               - 最短时间 = distance / max_speed
               - 最长时间 = distance / min_speed
            3. 在合理范围内: 1.0
            4. 偏离合理范围: 高斯衰减 exp(-0.5 * ((actual - expected) / sigma)^2)

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            时间可达性分数 [0, 1]
        """
        # 实际旅行时间 (秒)
        actual_travel_time = self._aligned_travel_time(source, target)
        if actual_travel_time < 0:
            # 时间倒序 → 不可能
            return 0.0

        # 尝试从拓扑获取距离
        distance = self._get_camera_distance(source.camera_id, target.camera_id)

        if distance is None or distance <= 0:
            # 无法获取距离 → 仅基于时间间隔给一个宽松分数
            # 优化: 对中等时间间隔使用更平缓的衰减
            # 期望值 120 秒, sigma 400 秒 (更平缓)
            expected = 120.0
            sigma = 400.0
            return float(math.exp(-0.5 * ((actual_travel_time - expected) / sigma) ** 2))

        # 计算合理时间范围
        min_time = distance / (self.max_speed_kmh * 1000.0 / 3600.0)  # 最快
        max_time = distance / (self.min_speed_kmh * 1000.0 / 3600.0)  # 最慢

        if min_time <= actual_travel_time <= max_time:
            # 在合理范围内
            return 1.0

        # 偏离合理范围 → 高斯衰减
        if actual_travel_time < min_time:
            # 太快了, 偏离下限
            deviation = min_time - actual_travel_time
            expected = min_time
            sigma = max(min_time * 0.3, 10.0)  # sigma 为最短时间的 30%, 至少 10 秒
        else:
            # 太慢了, 偏离上限
            deviation = actual_travel_time - max_time
            expected = max_time
            sigma = max(max_time * 0.5, 30.0)  # sigma 为最长时间的 50%, 至少 30 秒

        score = math.exp(-0.5 * (deviation / sigma) ** 2)
        return float(max(0.0, min(1.0, score)))

    def _score_spatial(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算空间可达性分数

        基于摄像头间的最短路径距离和旅行时间的一致性:
            - 拓扑不可达 → 0.0
            - 可达 → 基于距离与旅行时间匹配度评分

        匹配度计算:
            估算速度 = distance / actual_travel_time
            如果估算速度在 [min_speed, max_speed] 范围内 → 高分
            偏离越大分数越低 (高斯衰减)

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            空间可达性分数 [0, 1]
        """
        # 检查拓扑可达性
        try:
            reachable = self.road_topology.is_reachable(
                source.camera_id, target.camera_id
            )
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            # 如果拓扑未实现, 给一个宽松默认值
            reachable = True

        if not reachable:
            return 0.0

        # 获取距离
        distance = self._get_camera_distance(source.camera_id, target.camera_id)
        if distance is None or distance <= 0:
            # 无距离信息 → 中等分数 (不惩罚)
            return 0.5

        # 实际旅行时间（传入偏移表时按全局时间计算，见 `_aligned_travel_time`）
        actual_travel_time = self._aligned_travel_time(source, target)
        if actual_travel_time <= 0:
            return 0.0

        # 估算实际速度 (km/h)
        estimated_speed = (distance / 1000.0) / (actual_travel_time / 3600.0)

        # 判断估算速度是否在合理范围内
        if self.min_speed_kmh <= estimated_speed <= self.max_speed_kmh:
            # 速度合理 → 高分
            # 越接近中间值分数越高
            mid_speed = (self.min_speed_kmh + self.max_speed_kmh) / 2.0
            deviation = abs(estimated_speed - mid_speed)
            sigma = (self.max_speed_kmh - self.min_speed_kmh) / 4.0
            return float(0.7 + 0.3 * math.exp(-0.5 * (deviation / sigma) ** 2))
        else:
            # 速度超出合理范围 → 衰减
            if estimated_speed < self.min_speed_kmh:
                deviation = self.min_speed_kmh - estimated_speed
            else:
                deviation = estimated_speed - self.max_speed_kmh
            sigma = 15.0  # km/h
            return float(max(0.0, 0.7 * math.exp(-0.5 * (deviation / sigma) ** 2)))

    def _score_direction(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算方向一致性分数

        比较 Tracklet 的运动方向与道路期望方向:
            - 一致 (角度差 < 45°): 1.0
            - 垂直 (45° <= 角度差 < 135°): 0.5
            - 相反 (角度差 >= 135°): 0.0

        对于行人, 方向约束较弱, 最低 0.3 分。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            方向一致性分数 [0, 1]
        """
        src_direction = source.direction.lower().strip() if source.direction else ""
        tgt_direction = target.direction.lower().strip() if target.direction else ""

        if not src_direction or not tgt_direction:
            # 方向信息缺失 → 中性
            return NEUTRAL_SCORE

        # 获取源 tracklet 运动方向角度
        src_angle = DIRECTION_TO_ANGLE.get(src_direction)
        # 获取目标 tracklet 运动方向角度
        tgt_angle = DIRECTION_TO_ANGLE.get(tgt_direction)

        if src_angle is None or tgt_angle is None:
            # 无法解析方向 → 中性
            return NEUTRAL_SCORE

        # 计算角度差 [0, 180]
        angle_diff = abs(src_angle - tgt_angle) % 360
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        # 分段评分
        if angle_diff < 45.0:
            # 一致
            score = 1.0
        elif angle_diff < 135.0:
            # 垂直/斜交
            score = 0.5
        else:
            # 相反
            score = 0.0

        # 行人方向约束较弱
        if source.target_type.lower() == "pedestrian":
            score = max(score, 0.3)

        return score

    # ================================================================
    # 惩罚计算
    # ================================================================

    def _compute_penalty_components(
        self, source: Tracklet, target: Tracklet
    ) -> Tuple[float, float]:
        """
        分别计算路径分叉惩罚和观测缺失惩罚

        路径分叉惩罚 (path_divergence_penalty):
            从源摄像头到目标摄像头如果有多条可能路径, 不确定性增大。
            penalty = 1 - 1/num_paths
            只有 1 条路径时惩罚为 0, 路径越多惩罚越大。

        观测缺失惩罚 (observation_missing_penalty):
            如果两个摄像头之间应该经过其他中间摄像头但没有观测到目标。
            penalty = 1 - exp(-intermediate_count * 0.5)
            中间摄像头越多, 惩罚越大。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            (path_divergence_penalty, observation_missing_penalty) 各自范围 [0, 1]
        """
        src_cam = source.camera_id
        tgt_cam = target.camera_id

        # ---- 路径分叉惩罚 ----
        num_paths = self._count_possible_paths(src_cam, tgt_cam)
        if num_paths <= 1:
            path_div_penalty = 0.0
        else:
            # 路径越多, 不确定性越大
            path_div_penalty = 1.0 - 1.0 / num_paths

        # ---- 观测缺失惩罚 ----
        intermediate_count = self._count_intermediate_cameras(src_cam, tgt_cam)
        if intermediate_count <= 0:
            obs_missing_penalty = 0.0
        else:
            # 中间摄像头越多, 缺失惩罚越大, 但上限为 1
            obs_missing_penalty = 1.0 - math.exp(-intermediate_count * 0.5)

        return path_div_penalty, obs_missing_penalty

    def _compute_penalty(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算总惩罚值 (加权后)

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            总惩罚值
        """
        path_div, obs_missing = self._compute_penalty_components(source, target)
        return (
            self.penalties["path_divergence"] * path_div
            + self.penalties["observation_missing"] * obs_missing
        )

    # ================================================================
    # 加权求和
    # ================================================================

    def _weighted_sum(
        self,
        weights: Dict[str, float],
        appearance_score: float,
        attribute_score: float,
        plate_score: float,
        temporal_score: float,
        spatial_score: float,
        direction_score: float,
    ) -> float:
        """
        根据目标类型对应的权重配置, 计算加权总分

        权重映射规则:
            - "reid" → appearance_score
            - "attribute" → attribute_score
            - "plate" → plate_score (仅车辆)
            - "temporal" → temporal_score
            - "topology" → spatial_score
            - "bag" → attribute_score (行人背包归入属性)
            - "direction" → direction_score (始终参与, 作为额外加分)

        方向一致性作为额外加分项（DIRECTION_BONUS_WEIGHT，与 config 无关），
        不计入主权重归一化。本方法**不做归一化**：调用方传入的应当是已经处理过
        缺失维度、总和与配置一致的 `effective_weights`（见 `_effective_weights`）；
        直接传原始权重则保持旧的"中性分 × 原权重"行为。

        Args:
            weights: 权重字典（通常来自 `_effective_weights`）
            各分项分数

        Returns:
            加权总分 (未减惩罚)
        """
        score = 0.0

        # 外观相似度 (ReID)
        w_reid = weights.get("reid", 0.0)
        score += w_reid * appearance_score

        # 属性一致性
        w_attr = weights.get("attribute", 0.0)
        score += w_attr * attribute_score

        # 车牌一致性 (仅车辆有此项)
        w_plate = weights.get("plate", 0.0)
        score += w_plate * plate_score

        # 时间可达性
        w_temporal = weights.get("temporal", 0.0)
        score += w_temporal * temporal_score

        # 空间可达性 (拓扑)
        w_topology = weights.get("topology", 0.0)
        score += w_topology * spatial_score

        # 行人背包 (归入属性分)
        w_bag = weights.get("bag", 0.0)
        if w_bag > 0:
            score += w_bag * attribute_score

        # 方向一致性作为额外加分项 (权重提升以增强同方向车辆匹配)
        score += DIRECTION_BONUS_WEIGHT * direction_score

        return score

    # ================================================================
    # 辅助方法
    # ================================================================

    def _get_camera_distance(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> Optional[float]:
        """
        获取两个摄像头之间的距离 (米)

        优先使用 RoadTopology 的 shortest_path 距离,
        回退到 CameraManager 的直线距离。

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            距离 (米), 无法获取时返回 None
        """
        # 尝试从 RoadTopology 获取
        try:
            dist = self.road_topology.get_segment_distance(
                src_camera_id, tgt_camera_id
            )
            if dist is not None and dist > 0:
                return dist
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        # 回退: 从 CameraManager 获取直线距离
        try:
            src_cam = self.camera_manager.get_camera(src_camera_id)
            tgt_cam = self.camera_manager.get_camera(tgt_camera_id)
            if src_cam is not None and tgt_cam is not None:
                dist = src_cam.distance_to(tgt_cam)
                # 同 P1：数据集只有场景近似中心，同一场景摄像头坐标相同 → haversine=0，
                # 0 应视为「未知距离」而非「真的 0 米」，返回 None 走中性分。
                if dist is not None and dist > 0:
                    return dist
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        return None

    # 路径计数的两个上限（2026-09-21 加，见 `_count_possible_paths` 的说明）
    #
    # 这两个常量原先只存在于 `src/trajectory/builder.py` 的
    # `BoundedCrossCameraScorer` 子类里 —— 也就是说**基类一直是坏的**，
    # 只有走 builder 的那条路绕开了它。任何直接用 `CrossCameraScorer` 的代码
    # （如 `scripts/train_edge_scorer.py` 的构边）仍会撞上组合爆炸。
    # 现已下沉到基类，子类保留为兼容别名。
    MAX_PATH_HOPS = 6          # 超过 6 跳的绕行不计入"可能路径数"
    PATH_EXPANSION_BUDGET = 20000   # 兜底上限，防异常拓扑再次爆炸

    def _count_possible_paths(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> int:
        """
        计算两个摄像头之间可能的路径数量（有界枚举，最多 5 条）

        ## 为什么必须「有界」

        这个函数的语义是"数一数有几条路可走"，用来算路径分叉惩罚
        （`penalty = 1 - 1/num_paths`）。但简单路径的数量在图上是**指数级**的，
        而原来的写法只限制了**计数**（`paths_found < max_paths`），
        没有限制**搜索**：

            while stack and paths_found < max_paths:   # ← 只挡计数
                for neighbor in adj.get(current, []):  # ← 搜索无界
                    stack.append(...)

        当两点之间**根本不可达**（或只有很长的绕行）时，`paths_found` 永远是 0，
        循环条件恒真，DFS 会把整棵指数级的简单路径树走完。

        **实测**（46 摄像头 / 91 条邻接边）：单对摄像头 3,486,156 次递归调用、
        耗时 1.84 秒（c029→c030）；builder 侧的记录更早测到 c001→c002 达 23.2 秒。
        而候选边生成时**每一对都要调它一次** ⇒ 直接超时。

        ## 影响范围（别把这条说过头）

        **生产路径此前是有保护的**：`TrajectoryBuilder._scene_edge_pool` 构造
        `CandidateEdgeGenerator` 时显式传了 `scorer=self._get_scorer()`
        （返回 `BoundedCrossCameraScorer`），所以线上走的是子类的有界实现。

        真正暴露的是**直接用基类的代码** —— `CandidateEdgeGenerator` 在不传
        `scorer` 时会自己 `CrossCameraScorer(...)`（`candidate_edge.py:114`），
        以及 `scripts/train_edge_scorer.py` 这类直接实例化基类的脚本。
        本次把上限下沉到基类，**不改变任何线上行为**，只是把地雷拆掉。

        **修复**：加两个上限 —— `MAX_PATH_HOPS`（超过 6 跳的路径不计入）与
        `PATH_EXPANSION_BUDGET`（兜底预算）。修复后单对耗时降到亚毫秒。

        ## ⚠️ 这**不是**纯粹的等价重构 —— 返回值会变（已实测，别覆盖这条）

        我最初按"等价重构"来写，实测后发现**不等价**。在本拓扑（46 摄像头 /
        1035 对）上逐对比对：

        | 结果 | 对数 |
        |---|---|
        | 旧实现能终止、可参与比对 | 709 |
        | 其中**返回值不同** | **278（39.2%）** |
        | 旧实现组合爆炸、根本给不出值 | 326 |

        差异集中在 `(旧 5, 新 1)` 共 229 对 —— 即旧版数出了 ≥5 条**超过 6 跳**的
        绕行，新版不再计入。对 `path_divergence_penalty = 1 - 1/num_paths` 的影响：
        最大 **0.80**、平均 0.27，乘惩罚权重 0.1 后对总分影响最大 **0.08**。

        **为什么接受这个语义变化**：系统自己的可达性假设就是 ≤6 跳
        （`is_reachable` 与 builder 侧 `MAX_HOPS`），把 9 跳绕行算作"同一辆车
        可能走的路线"本来就可疑 —— 新版数的是**合理路径**，不是图论意义上的
        全部简单路径。且**线上路径早已在用这个语义**（builder 一直在用
        `BoundedCrossCameraScorer`），所以本次改动让基类与线上口径**趋于一致**，
        而不是引入新的偏离。

        **为什么不能做成精确版**：曾尝试"反向可达性剪枝 + 压入即计数"来保住
        精确语义，实测能把可比对的对数从 709 提到 814 且**零差异**，但 221 对
        仍然爆炸（最坏 c011→c040 需 195 万次扩展）。简单路径计数是 #P-hard，
        指数下界绕不过去。因此**有界近似是唯一可行的选择**，代价就是把上面
        这张差异表如实写在这里。

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            可能路径数量 (上限 5；不可达或不在邻接表中返回 1)
        """
        max_paths = 5
        try:
            adj = self.road_topology._adjacency
        except AttributeError:
            return 1

        if src_camera_id not in adj or tgt_camera_id not in adj:
            return 1

        found = 0
        budget = self.PATH_EXPANSION_BUDGET
        # (当前节点, 已访问集合, 已走跳数)
        stack: List[Tuple[str, set, int]] = [(src_camera_id, {src_camera_id}, 0)]

        while stack and found < max_paths and budget > 0:
            current, visited, depth = stack.pop()
            budget -= 1
            if depth >= self.MAX_PATH_HOPS:
                continue
            for neighbor in adj.get(current, []):
                if neighbor in visited:
                    continue
                if neighbor == tgt_camera_id:
                    found += 1
                    if found >= max_paths:
                        break
                else:
                    stack.append((neighbor, visited | {neighbor}, depth + 1))

        return max(found, 1)

    def _count_intermediate_cameras(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> int:
        """
        计算两个摄像头之间最短路径上的中间摄像头数量

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            中间摄像头数量
        """
        try:
            path = self.road_topology.shortest_path(src_camera_id, tgt_camera_id)
            if path is not None and len(path) > 2:
                return len(path) - 2  # 去掉起点和终点
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass
        return 0

    def _build_reasoning(
        self,
        source: Tracklet,
        target: Tracklet,
        appearance_score: float,
        attribute_score: float,
        plate_score: float,
        temporal_score: float,
        spatial_score: float,
        direction_score: float,
        total_penalty: float,
        final_score: float,
        weights: Dict[str, float],
        effective_weights: Optional[Dict[str, float]] = None,
        dropped_dimensions: Optional[List[str]] = None,
        availability: Optional[Dict[str, bool]] = None,
    ) -> str:
        """
        生成可解释推理说明字符串

        Args:
            各分项分数和配置
            effective_weights: 剔除无证据维度后的实际权重（T6 可观测性）
            dropped_dimensions: 被剔除的权重键
            availability: 各维度是否有真实证据

        Returns:
            推理说明字符串
        """
        parts = [
            f"跨镜连接评分 [{source.tracklet_id} → {target.tracklet_id}]:",
            f"  外观相似度={appearance_score:.3f}",
            f"  属性一致性={attribute_score:.3f}",
            f"  车牌一致性={plate_score:.3f}",
            f"  时间可达性={temporal_score:.3f}",
            f"  空间可达性={spatial_score:.3f}",
            f"  方向一致性={direction_score:.3f}",
            f"  总惩罚={total_penalty:.3f}",
            f"  最终得分={final_score:.3f}",
        ]

        # 权重来源与缺失维度的处理（显式化，便于排障与消融）
        if effective_weights is not None:
            parts.append(f"  配置权重={weights}")
            parts.append(
                "  有效权重="
                + " ".join(f"{k}={v:.3f}" for k, v in sorted(effective_weights.items()))
                + (
                    f"（无证据维度已剔除并按比例分摊: {', '.join(sorted(dropped_dimensions))}）"
                    if dropped_dimensions else "（全部维度均有证据）"
                )
            )
        if availability is not None:
            parts.append(
                "  维度证据="
                + " ".join(
                    f"{k}={'有' if v else '无'}" for k, v in sorted(availability.items())
                )
            )

        # 关键证据摘要
        evidence = []
        if plate_score == 1.0:
            evidence.append("车牌完全匹配")
        elif plate_score == 0.0:
            evidence.append("车牌冲突(否决)")
        elif availability is not None and not availability.get("plate", True):
            evidence.append("双方均无车牌(车牌维度无证据, 不计入总分)")

        if availability is not None and not availability.get("reid", True):
            evidence.append("无 ReID/CLIP 外观特征(外观维度无证据, 不计入总分)")
        elif appearance_score >= 0.8:
            evidence.append("外观高度相似")
        elif appearance_score < 0.4:
            evidence.append("外观差异较大")

        if temporal_score >= 0.9:
            evidence.append("时间窗口完全吻合")
        elif temporal_score < 0.3:
            evidence.append("时间窗口匹配度低")

        if evidence:
            parts.append("  关键证据: " + "; ".join(evidence))

        return "\n".join(parts)
