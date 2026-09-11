"""
src.stitching.scoring - 跨镜连接评分模块

对候选边进行多维度评分:

跨镜连接得分 = w1*外观相似度 + w2*属性一致性 + w3*车牌一致性
              + w4*时间可达性 + w5*空间可达性 + w6*方向一致性
              - p1*路径分叉惩罚 - p2*观测缺失惩罚

车辆侧权重: 车牌(0.35) > 时间(0.25) > 拓扑(0.15) > ReID(0.15) > 属性(0.10)
行人侧权重: 时间(0.35) > ReID(0.30) > 属性(0.25) > 背包(0.10)

每个评分分项都有独立的计算方法，方便后续消融实验。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

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
    ) -> None:
        """
        初始化评分器

        Args:
            camera_manager: 摄像头管理器
            road_topology: 道路拓扑
            vehicle_weights: 车辆评分权重字典
            pedestrian_weights: 行人评分权重字典
            penalties: 惩罚项权重字典
            min_speed_kmh: 城区最低速度 (km/h)
            max_speed_kmh: 城区最高速度 (km/h)
        """
        self.camera_manager = camera_manager
        self.road_topology = road_topology
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh

        # 车辆权重: 车牌 > 时间 > ReID > 拓扑 > 属性
        # 优化: 增加reid和temporal权重，降低topology权重，提升密集场景区分度
        self.vehicle_weights = vehicle_weights or {
            "plate": 0.30,
            "temporal": 0.25,
            "topology": 0.10,   # 对应空间可达性 (降低权重减少误关联)
            "reid": 0.25,
            "attribute": 0.10,
        }

        # 行人权重: 时间 > ReID > 属性 > 背包
        # 优化: 增加reid权重，提升外观区分度
        self.pedestrian_weights = pedestrian_weights or {
            "temporal": 0.30,
            "reid": 0.35,
            "attribute": 0.25,
            "bag": 0.10,
        }

        # 惩罚项配置
        self.penalties = penalties or {
            "path_divergence": 0.1,      # 路径分叉惩罚系数
            "observation_missing": 0.05,  # 观测缺失惩罚系数
        }

        logger.info(
            "跨镜连接评分器初始化 | 车辆权重=%s | 行人权重=%s | 惩罚=%s",
            self.vehicle_weights, self.pedestrian_weights, self.penalties,
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
            3. 加权求和得到基础分
            4. 计算并减去惩罚项
            5. 截断到 [0, 1] 范围
            6. 生成可解释推理说明

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
            weights=weights,
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
        )

        logger.debug(
            "评分完成 | %s → %s | score=%.4f | appearance=%.3f attr=%.3f "
            "plate=%.3f temporal=%.3f spatial=%.3f direction=%.3f penalty=%.3f",
            source.tracklet_id, target.tracklet_id, final_score,
            appearance_score, attribute_score, plate_score,
            temporal_score, spatial_score, direction_score, total_penalty,
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
    # 分项评分方法 (每个方法独立可调用, 方便消融实验)
    # ================================================================

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
            外观相似度 [0, 1]，无特征时返回 0.5 (中性)
        """
        # 优先使用 ReID 向量
        if source.avg_reid_vector is not None and target.avg_reid_vector is not None:
            cos_sim = cosine_similarity(source.avg_reid_vector, target.avg_reid_vector)
            return float((cos_sim + 1.0) / 2.0)

        # 回退到 CLIP 向量
        if source.avg_clip_vector is not None and target.avg_clip_vector is not None:
            cos_sim = cosine_similarity(source.avg_clip_vector, target.avg_clip_vector)
            return float((cos_sim + 1.0) / 2.0)

        # 无任何外观特征 → 返回中性值
        return 0.5

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
                weighted_sum += 0.5 * w
            elif str(src_val).lower() == str(tgt_val).lower():
                # 匹配
                weighted_sum += 1.0 * w
            else:
                # 不匹配
                weighted_sum += 0.0 * w

        if total_weight == 0:
            return 0.5

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

        # 其他情况 (一方无车牌或双方都无)
        return 0.5

    def _score_temporal(self, source: Tracklet, target: Tracklet) -> float:
        """
        计算时间可达性分数

        流程:
            1. 计算实际旅行时间 = target.start_time - source.end_time
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
        actual_travel_time = time_diff_seconds(source.end_time, target.start_time)
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
            # 如果拓扑未实现, 给一个宽松默认值
            reachable = True

        if not reachable:
            return 0.0

        # 获取距离
        distance = self._get_camera_distance(source.camera_id, target.camera_id)
        if distance is None or distance <= 0:
            # 无距离信息 → 中等分数 (不惩罚)
            return 0.5

        # 实际旅行时间
        actual_travel_time = time_diff_seconds(source.end_time, target.start_time)
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
            return 0.5

        # 获取源 tracklet 运动方向角度
        src_angle = DIRECTION_TO_ANGLE.get(src_direction)
        # 获取目标 tracklet 运动方向角度
        tgt_angle = DIRECTION_TO_ANGLE.get(tgt_direction)

        if src_angle is None or tgt_angle is None:
            # 无法解析方向 → 中性
            return 0.5

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

        方向一致性作为额外加分项 (权重 0.05), 不计入主权重归一化。

        Args:
            weights: 权重字典
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
        direction_weight = 0.10
        score += direction_weight * direction_score

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
            pass

        # 回退: 从 CameraManager 获取直线距离
        try:
            src_cam = self.camera_manager.get_camera(src_camera_id)
            tgt_cam = self.camera_manager.get_camera(tgt_camera_id)
            if src_cam is not None and tgt_cam is not None:
                return src_cam.distance_to(tgt_cam)
        except (NotImplementedError, AttributeError):
            pass

        return None

    def _count_possible_paths(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> int:
        """
        计算两个摄像头之间可能的路径数量

        使用 BFS 枚举所有简单路径 (最多搜索 5 条)。

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            可能路径数量 (上限 5)
        """
        max_paths = 5
        try:
            adj = self.road_topology._adjacency
        except AttributeError:
            return 1

        if src_camera_id not in adj or tgt_camera_id not in adj:
            return 1

        # BFS 枚举简单路径
        paths_found = 0
        stack = [(src_camera_id, {src_camera_id})]
        while stack and paths_found < max_paths:
            current, visited = stack.pop()
            if current == tgt_camera_id:
                paths_found += 1
                continue
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_visited))

        return max(paths_found, 1)

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
    ) -> str:
        """
        生成可解释推理说明字符串

        Args:
            各分项分数和配置

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

        # 关键证据摘要
        evidence = []
        if plate_score == 1.0:
            evidence.append("车牌完全匹配")
        elif plate_score == 0.0:
            evidence.append("车牌冲突(否决)")

        if appearance_score >= 0.8:
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
