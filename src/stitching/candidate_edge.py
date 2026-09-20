"""
src.stitching.candidate_edge - 候选边生成模块

基于交通约束过滤生成跨镜候选连接边。
候选边表示两个 Tracklet 可能属于同一目标。

候选边生成规则(交通约束缩小候选范围):
1. 摄像头拓扑可达
2. 时间顺序合理
3. 旅行时间在合理范围内
4. 道路方向一致
5. 目标类别一致
6. 车牌不冲突
7. 颜色、车型、服饰等属性不冲突
8. ReID 外观相似度达到阈值

实现策略: 先用交通约束 (规则1-5) 做粗筛, 大幅减少候选对数量;
         再计算外观相似度 (规则6-8) 做精筛。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.common.data_models import CrossCameraEdge, Tracklet
from src.common.logger import get_logger
from src.common.utils import cosine_similarity, time_diff_seconds
from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology
from src.stitching.scoring import CrossCameraScorer

logger = get_logger("stitching.candidate_edge")

# ============================================================
# 常量
# ============================================================

# 城区合理速度范围 (km/h)
MIN_CITY_SPEED_KMH = 20.0
MAX_CITY_SPEED_KMH = 80.0

# 属性里的「未知」占位值 = 无信息，属性一致性检查时等同缺失，不与具体值判为冲突
_MISSING_ATTR = frozenset({"", "none", "null", "未知", "unknown", "other", "其他", "n/a"})

# 方向角度映射 (正北为0°, 顺时针)
DIRECTION_TO_ANGLE: Dict[str, float] = {
    "eastbound": 90.0,
    "westbound": 270.0,
    "northbound": 0.0,
    "southbound": 180.0,
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
}


class CandidateEdgeGenerator:
    """
    跨镜候选边生成器

    对 Tracklet 对进行交通约束过滤，生成有效的候选连接边。
    采用"粗筛 + 精筛"两阶段策略:
        - 粗筛阶段: 交通约束 (拓扑/时间/距离/方向/类别) 快速排除不可能配对
        - 精筛阶段: 外观特征 (车牌/属性/ReID) 精细过滤

    使用方式:
        generator = CandidateEdgeGenerator(camera_manager, road_topology)
        edges = generator.generate_candidates(all_tracklets, road_topology)
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        road_topology: RoadTopology,
        max_time_gap_seconds: float = 600.0,
        min_appearance_score: float = 0.6,
        spatial_search_radius_km: float = 10.0,
        min_speed_kmh: float = MIN_CITY_SPEED_KMH,
        max_speed_kmh: float = MAX_CITY_SPEED_KMH,
        scorer: Optional[CrossCameraScorer] = None,
    ) -> None:
        """
        初始化候选边生成器

        Args:
            camera_manager: 摄像头管理器
            road_topology: 道路拓扑
            max_time_gap_seconds: 最大时间间隔(秒)
            min_appearance_score: 最低外观相似度阈值
            spatial_search_radius_km: 空间搜索半径(公里)
            min_speed_kmh: 城区最低速度 (km/h)
            max_speed_kmh: 城区最高速度 (km/h)
            scorer: 跨镜评分器 (可选, 不传则内部创建)
        """
        self.camera_manager = camera_manager
        self.road_topology = road_topology
        self.max_time_gap_seconds = max_time_gap_seconds
        self.min_appearance_score = min_appearance_score
        self.spatial_search_radius_km = spatial_search_radius_km
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh

        # 评分器 (用于精筛阶段计算 ReID 相似度等)
        self.scorer = scorer or CrossCameraScorer(
            camera_manager=camera_manager,
            road_topology=road_topology,
            min_speed_kmh=min_speed_kmh,
            max_speed_kmh=max_speed_kmh,
        )

        logger.info(
            "候选边生成器初始化 | max_time_gap=%.1fs | min_appearance=%.2f | "
            "speed_range=[%.1f, %.1f] km/h",
            max_time_gap_seconds, min_appearance_score,
            min_speed_kmh, max_speed_kmh,
        )

    # ================================================================
    # 主入口: 批量生成候选边
    # ================================================================

    def generate_candidates(
        self,
        tracklets: List[Tracklet],
        road_topology: Optional[RoadTopology] = None,
    ) -> List[CrossCameraEdge]:
        """
        为所有 Tracklet 对生成候选连接边

        流程:
            1. 按摄像头分组 Tracklet
            2. 对不同摄像头的 Tracklet 对进行粗筛 (规则1-5)
            3. 对通过粗筛的候选对进行精筛 (规则6-8)
            4. 使用 CrossCameraScorer 计算完整评分

        Args:
            tracklets: 所有 Tracklet 列表
            road_topology: 道路拓扑 (可选, 默认使用初始化时的)

        Returns:
            通过所有规则的候选边列表
        """
        topology = road_topology or self.road_topology
        valid_edges: List[CrossCameraEdge] = []

        if len(tracklets) < 2:
            logger.info("Tracklet 数量不足, 无需生成候选边")
            return valid_edges

        # 按摄像头分组, 方便后续按对处理
        cam_groups: Dict[str, List[Tracklet]] = {}
        for t in tracklets:
            cam_groups.setdefault(t.camera_id, []).append(t)

        camera_ids = list(cam_groups.keys())
        total_pairs = 0
        coarse_passed = 0

        # 遍历不同摄像头的所有 Tracklet 对
        for i in range(len(camera_ids)):
            for j in range(len(camera_ids)):
                if i == j:
                    continue  # 跳过同一摄像头内的 Tracklet 对

                src_cam_id = camera_ids[i]
                tgt_cam_id = camera_ids[j]

                for src_t in cam_groups[src_cam_id]:
                    for tgt_t in cam_groups[tgt_cam_id]:
                        total_pairs += 1

                        # ---- 粗筛阶段 (规则1-5) ----
                        coarse_result, fail_reason = self._coarse_filter(
                            src_t, tgt_t, topology
                        )
                        if not coarse_result:
                            continue

                        coarse_passed += 1

                        # ---- 精筛阶段 (规则6-8) + 完整评分 ----
                        edge = self._fine_filter_and_score(src_t, tgt_t, topology)
                        if edge is not None and edge.is_valid:
                            valid_edges.append(edge)

        logger.info(
            "候选边生成完成 | 总对数=%d | 粗筛通过=%d | 有效候选边=%d",
            total_pairs, coarse_passed, len(valid_edges),
        )
        return valid_edges

    # ================================================================
    # 单源 Tracklet 生成候选边 (兼容原有接口)
    # ================================================================

    def generate(
        self,
        source_tracklet: Tracklet,
        candidate_tracklets: List[Tracklet],
    ) -> List[CrossCameraEdge]:
        """
        为源 Tracklet 生成候选连接边

        Args:
            source_tracklet: 源 Tracklet
            candidate_tracklets: 候选 Tracklet 列表

        Returns:
            有效的候选连接边列表
        """
        valid_edges: List[CrossCameraEdge] = []

        for candidate in candidate_tracklets:
            # 跳过同一摄像头
            if source_tracklet.camera_id == candidate.camera_id:
                continue

            # 粗筛
            coarse_result, fail_reason = self._coarse_filter(
                source_tracklet, candidate, self.road_topology
            )
            if not coarse_result:
                continue

            # 精筛 + 评分
            edge = self._fine_filter_and_score(
                source_tracklet, candidate, self.road_topology
            )
            if edge is not None and edge.is_valid:
                valid_edges.append(edge)

        return valid_edges

    # ================================================================
    # 粗筛阶段: 交通约束快速过滤 (规则1-5)
    # ================================================================

    def _coarse_filter(
        self,
        source: Tracklet,
        target: Tracklet,
        topology: RoadTopology,
    ) -> Tuple[bool, str]:
        """
        粗筛阶段: 应用交通约束规则 1-5 快速过滤不可能的配对

        规则:
            1. 摄像头拓扑可达
            2. 时间顺序合理 (source.end_time < target.start_time)
            3. 旅行时间在合理范围内
            4. 道路方向一致
            5. 目标类别一致

        Args:
            source: 源 Tracklet (时间较早)
            target: 目标 Tracklet (时间较晚)
            topology: 道路拓扑

        Returns:
            (是否通过, 失败原因描述)
        """
        # ---- 规则5: 目标类别一致 ----
        if source.target_type.lower() != target.target_type.lower():
            return False, f"目标类别不一致: {source.target_type} vs {target.target_type}"

        # ---- 规则2: 时间顺序合理 ----
        if not self._check_temporal_order(source, target):
            return False, "时间顺序不合理: 源不在目标之前"



        # ---- 规则1: 摄像头拓扑可达 ----
        if not self._check_topology_reachable(source.camera_id, target.camera_id, topology):
            return False, f"拓扑不可达: {source.camera_id} → {target.camera_id}"

        # ---- 规则3: 旅行时间在合理范围内 ----
        if not self._check_travel_time_feasibility(source, target, topology):
            return False, "旅行时间不在合理范围内"

        # ---- 规则4: 道路方向一致 ----
        if not self._check_direction_compatibility(source, target):
            return False, "道路方向矛盾"

        return True, ""

    # ================================================================
    # 精筛阶段: 外观特征过滤 + 完整评分 (规则6-8)
    # ================================================================

    def _fine_filter_and_score(
        self,
        source: Tracklet,
        target: Tracklet,
        topology: RoadTopology,
    ) -> Optional[CrossCameraEdge]:
        """
        精筛阶段: 应用外观特征规则 6-8 并使用评分器计算完整评分

        规则:
            6. 车牌不冲突
            7. 属性不冲突
            8. ReID 外观相似度达阈值

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet
            topology: 道路拓扑

        Returns:
            通过所有规则的 CrossCameraEdge, 不通过返回 None
        """
        # ---- 规则6: 车牌不冲突 ----
        if not self._check_plate_consistency(source, target):
            logger.debug(
                "车牌冲突: %s(%s) vs %s(%s)",
                source.tracklet_id, source.plate_number,
                target.tracklet_id, target.plate_number,
            )
            return None

        # ---- 规则7: 属性不冲突 ----
        if not self._check_attribute_consistency(source, target):
            logger.debug(
                "属性冲突: %s vs %s",
                source.tracklet_id, target.tracklet_id,
            )
            return None

        # ---- 规则8: ReID 外观相似度达阈值 ----
        # 优化: 对长时间间隔提高阈值，对低质量目标增加过滤
        time_gap = time_diff_seconds(source.end_time, target.start_time)
        effective_threshold = self.min_appearance_score

        # 长时间间隔(>100帧)提高阈值
        if time_gap > 100:
            effective_threshold = min(0.85, self.min_appearance_score + 0.1)

        # 低质量目标提高阈值
        src_quality = getattr(source, 'avg_quality', None)
        tgt_quality = getattr(target, 'avg_quality', None)
        if src_quality is not None and src_quality < 0.4:
            effective_threshold += 0.05
        if tgt_quality is not None and tgt_quality < 0.4:
            effective_threshold += 0.05

        appearance_sim = self._compute_appearance_similarity(source, target)
        if appearance_sim < effective_threshold:
            logger.debug(
                "外观相似度不足: %s vs %s, sim=%.3f < %.3f",
                source.tracklet_id, target.tracklet_id,
                appearance_sim, effective_threshold,
            )
            return None

        # ---- 所有规则通过, 使用评分器计算完整评分 ----
        edge = self.scorer.score(source, target)
        return edge

    # ================================================================
    # 各规则检查方法
    # ================================================================

    def _check_temporal_order(self, source: Tracklet, target: Tracklet) -> bool:
        """
        检查时间顺序: 源 Tracklet 的离开时间早于目标 Tracklet 的进入时间

        用全局对齐时间判定（T5 偏移）。本数据集相邻摄像头有重叠视野，
        本地时间下真值相邻对的旅行时间常为负（重叠），用本地时间会误杀真边。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            是否满足时间顺序
        """
        return source.end_time < target.start_time

    def _check_topology_reachable(
        self,
        src_camera_id: str,
        tgt_camera_id: str,
        topology: RoadTopology,
    ) -> bool:
        """
        检查摄像头拓扑可达性 (规则1)

        优先使用 RoadTopology.is_reachable(),
        回退到 CameraManager.is_topologically_reachable()。

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID
            topology: 道路拓扑

        Returns:
            是否拓扑可达
        """
        # 同一摄像头直接可达
        if src_camera_id == tgt_camera_id:
            return True

        # 尝试 RoadTopology
        try:
            return topology.is_reachable(src_camera_id, tgt_camera_id)
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        # 回退到 CameraManager 的 BFS
        try:
            return self.camera_manager.is_topologically_reachable(
                src_camera_id, tgt_camera_id
            )
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            # 如果都未实现, 默认认为可达 (宽松策略)
            return True

    def _check_travel_time_feasibility(
        self,
        source: Tracklet,
        target: Tracklet,
        topology: RoadTopology,
    ) -> bool:
        """
        检查旅行时间是否在合理范围内 (规则3)

        根据路段距离和城区速度范围 [20, 80] km/h 判断:
            - 最短时间 = distance / max_speed
            - 最长时间 = distance / min_speed
            - 实际旅行时间 = target.start_time - source.end_time
            - 额外允许 20% 的容差

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet
            topology: 道路拓扑

        Returns:
            旅行时间是否合理
        """
        actual_time = time_diff_seconds(source.end_time, target.start_time)
        if actual_time <= 0:
            return False

        # 获取距离
        distance = self._get_camera_distance(source.camera_id, target.camera_id, topology)

        if distance is None or distance <= 0:
            # 无距离信息, 仅检查时间间隔是否在阈值内
            return actual_time <= self.max_time_gap_seconds

        # 计算合理时间范围
        min_time = distance / (self.max_speed_kmh * 1000.0 / 3600.0)
        max_time = distance / (self.min_speed_kmh * 1000.0 / 3600.0)

        # 允许 20% 容差
        tolerance = 0.2
        return min_time * (1 - tolerance) <= actual_time <= max_time * (1 + tolerance)

    def _check_direction_compatibility(
        self,
        source: Tracklet,
        target: Tracklet,
    ) -> bool:
        """
        检查道路方向一致性 (规则4)

        判断两个 Tracklet 的运动方向是否矛盾:
            - 方向一致 (角度差 < 90°): 通过
            - 方向垂直 (90° <= 角度差 < 150°): 通过 (路口转弯可能)
            - 方向相反 (角度差 >= 150°): 拒绝

        对于行人, 方向约束放宽 (不做方向过滤)。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            方向是否兼容
        """
        # 行人方向约束较弱, 不做方向过滤
        if source.target_type.lower() == "pedestrian":
            return True

        src_dir = source.direction.lower().strip() if source.direction else ""
        tgt_dir = target.direction.lower().strip() if target.direction else ""

        # 方向信息缺失时不过滤
        if not src_dir or not tgt_dir:
            return True

        src_angle = DIRECTION_TO_ANGLE.get(src_dir)
        tgt_angle = DIRECTION_TO_ANGLE.get(tgt_dir)

        if src_angle is None or tgt_angle is None:
            return True

        # 计算角度差 [0, 180]
        angle_diff = abs(src_angle - tgt_angle) % 360
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        # 方向完全相反 (>= 150°) 则拒绝
        return angle_diff < 150.0

    def _check_plate_consistency(
        self,
        source: Tracklet,
        target: Tracklet,
    ) -> bool:
        """
        检查车牌一致性 (规则6)

        如果两个 Tracklet 都有车牌, 车牌必须相同。
        一方或双方无车牌时不冲突。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            车牌是否一致 (不冲突)
        """
        if not source.has_plate or not target.has_plate:
            return True  # 至少一方无车牌, 不冲突

        src_plate = source.plate_number.strip().upper()
        tgt_plate = target.plate_number.strip().upper()
        return src_plate == tgt_plate

    def _check_attribute_consistency(
        self,
        source: Tracklet,
        target: Tracklet,
    ) -> bool:
        """
        检查属性一致性 (规则7)

        比较颜色、车型 (车辆) 或性别、服饰 (行人) 等关键属性。
        如果双方都有某属性且值不同, 则判定为冲突。
        一方缺失或双方一致则不冲突。

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            属性是否一致 (不冲突)
        """
        target_type = source.target_type.lower()
        src_attrs = source.attributes or {}
        tgt_attrs = target.attributes or {}

        if target_type == "vehicle":
            # 车辆: 检查颜色和车型
            conflict_keys = ["color", "vehicle_type"]
        else:
            # 行人: 检查性别和上衣颜色
            conflict_keys = ["gender", "clothing_color"]

        for key in conflict_keys:
            src_val = src_attrs.get(key)
            tgt_val = tgt_attrs.get(key)

            # 「未知」等占位值 = 无信息，等同缺失，不应与任何具体值判为冲突。
            # 修复前把 '未知' 当具体值：'未知' != '轿车' 会被误判为冲突、误删真实边
            # （docstring 语义是「一方缺失则不冲突」，这里补上对未知值的识别）。
            if src_val is not None and str(src_val).strip().lower() in _MISSING_ATTR:
                src_val = None
            if tgt_val is not None and str(tgt_val).strip().lower() in _MISSING_ATTR:
                tgt_val = None

            # 双方都有该属性且值不同 → 冲突
            if src_val is not None and tgt_val is not None:
                if str(src_val).lower() != str(tgt_val).lower():
                    return False

        return True

    def _compute_appearance_similarity(
        self,
        source: Tracklet,
        target: Tracklet,
    ) -> float:
        """
        计算外观相似度 (规则8 使用)

        优先使用 ReID 向量的余弦相似度;
        无 ReID 时回退到 CLIP 向量。

        余弦相似度从 [-1, 1] 映射到 [0, 1]:
            score = (cos_sim + 1) / 2

        Args:
            source: 源 Tracklet
            target: 目标 Tracklet

        Returns:
            外观相似度 [0, 1], 无特征时返回 0.5
        """
        # 优先 ReID
        if source.avg_reid_vector is not None and target.avg_reid_vector is not None:
            cos_sim = cosine_similarity(source.avg_reid_vector, target.avg_reid_vector)
            return float((cos_sim + 1.0) / 2.0)

        # 回退 CLIP
        if source.avg_clip_vector is not None and target.avg_clip_vector is not None:
            cos_sim = cosine_similarity(source.avg_clip_vector, target.avg_clip_vector)
            return float((cos_sim + 1.0) / 2.0)

        # 无特征
        return 0.5

    # ================================================================
    # 辅助方法
    # ================================================================

    def _get_camera_distance(
        self,
        src_camera_id: str,
        tgt_camera_id: str,
        topology: RoadTopology,
    ) -> Optional[float]:
        """
        获取两个摄像头之间的距离 (米)

        优先使用 RoadTopology, 回退到 CameraManager 的 Haversine 距离。

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID
            topology: 道路拓扑

        Returns:
            距离 (米), 无法获取时返回 None
        """
        # 尝试 RoadTopology
        try:
            dist = topology.get_segment_distance(src_camera_id, tgt_camera_id)
            if dist is not None and dist > 0:
                return dist
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        # 回退: CameraManager 直线距离
        try:
            src_cam = self.camera_manager.get_camera(src_camera_id)
            tgt_cam = self.camera_manager.get_camera(tgt_camera_id)
            if src_cam is not None and tgt_cam is not None:
                from src.common.utils import haversine_distance
                dist = haversine_distance(
                    src_cam.latitude, src_cam.longitude,
                    tgt_cam.latitude, tgt_cam.longitude,
                )
                # 数据集没有逐摄像头 GPS，只有场景近似中心：同一场景的摄像头坐标相同，
                # haversine=0。0 应视为「未知距离」而非「真的 0 米」，返回 None 让调用方
                # 走中性分，避免用 0 距离算出速度为 0 的伪结论。
                if dist is not None and dist > 0:
                    return dist
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        return None
