"""
src.backtrack.chain_expander - 上下游扩展模块

在初始观测链基础上进行上下游扩展:
- 向更远的摄像头扩展
- 补充遗漏的中间观测节点
- 生成候选路径

数据流向: ObservationChain → ChainExpander → 扩展后的完整轨迹
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.common.data_models import (
    CandidatePath,
    CrossCameraEdge,
    InferenceSegment,
    ObservationChain,
    ObservationNode,
    ObservationSegment,
    Tracklet,
)
from src.common.logger import get_logger
from src.common.utils import generate_id
from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology

logger = get_logger("backtrack.chain_expander")


class ChainExpander:
    """
    观测链扩展器

    在初始观测链基础上进行扩展和补充。

    使用方式:
        expander = ChainExpander(camera_manager, road_topology)
        expanded = expander.expand(observation_chain)
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        road_topology: RoadTopology,
        tracklets: Optional[Dict[str, Tracklet]] = None,
        max_upstream_depth: int = 10,
        max_downstream_depth: int = 10,
        min_confidence_threshold: float = 0.5,
    ) -> None:
        """
        初始化链扩展器

        Args:
            camera_manager: 摄像头管理器
            road_topology: 道路拓扑
            tracklets: Tracklet 字典 {tracklet_id: Tracklet}
            max_upstream_depth: 最大上游扩展深度
            max_downstream_depth: 最大下游扩展深度
            min_confidence_threshold: 最低置信度阈值
        """
        self.camera_manager = camera_manager
        self.road_topology = road_topology
        self._tracklets = tracklets or {}
        self.max_upstream_depth = max_upstream_depth
        self.max_downstream_depth = max_downstream_depth
        self.min_confidence_threshold = min_confidence_threshold

        logger.info(
            f"观测链扩展器初始化: upstream_depth={max_upstream_depth}, "
            f"downstream_depth={max_downstream_depth}, "
            f"min_confidence={min_confidence_threshold}"
        )

    def expand(
        self,
        chain: ObservationChain,
    ) -> Tuple[
        List[ObservationNode],
        List[ObservationSegment],
        List[InferenceSegment],
        List[CandidatePath],
    ]:
        """
        扩展观测链，生成完整的轨迹输出组件

        Args:
            chain: 输入观测链

        Returns:
            (观测节点列表, 观测段列表, 推断段列表, 候选路径列表)
        """
        logger.info(f"开始扩展观测链: {chain.chain_id}")

        # 1. 从链中提取观测节点
        observation_nodes = self._build_observation_nodes(chain)

        # 2. 生成观测段(每个摄像头内的真实轨迹)
        observation_segments = self._build_observation_segments(chain)

        # 3. 生成推断段(摄像头之间的推断连接)
        inference_segments = self._build_inference_segments(chain)

        # 4. 生成候选路径(摄像头之间的可能道路)
        candidate_paths = self._build_candidate_paths(chain)

        logger.info(
            f"扩展完成: {len(observation_nodes)} 个观测节点, "
            f"{len(observation_segments)} 个观测段, "
            f"{len(inference_segments)} 个推断段, "
            f"{len(candidate_paths)} 条候选路径"
        )

        return observation_nodes, observation_segments, inference_segments, candidate_paths

    def _build_observation_nodes(
        self,
        chain: ObservationChain,
    ) -> List[ObservationNode]:
        """
        从观测链构建观测节点列表

        按时间排序，每个 Tracklet 对应一个观测节点。

        Args:
            chain: 观测链

        Returns:
            观测节点列表
        """
        nodes = []

        for tracklet_id in chain.ordered_tracklets:
            tracklet = self._tracklets.get(tracklet_id)
            if tracklet is None:
                logger.warning(f"未找到 Tracklet: {tracklet_id}")
                continue

            # 获取关键帧路径
            keyframe_path = tracklet.keyframe_paths[0] if tracklet.keyframe_paths else ""

            # 计算观测置信度(基于 Tracklet 的质量和连接边)
            confidence = self._compute_node_confidence(tracklet_id, chain)

            node = ObservationNode(
                camera_id=tracklet.camera_id,
                tracklet_id=tracklet_id,
                timestamp=tracklet.start_time,
                keyframe_path=keyframe_path,
                confidence=confidence,
            )
            nodes.append(node)

        # 按时间排序
        nodes.sort(key=lambda n: n.timestamp)

        return nodes

    def _build_observation_segments(
        self,
        chain: ObservationChain,
    ) -> List[ObservationSegment]:
        """
        从观测链构建观测段列表

        每个 Tracklet 对应一个观测段，描述摄像头视野内的真实运动。

        Args:
            chain: 观测链

        Returns:
            观测段列表
        """
        segments = []

        for tracklet_id in chain.ordered_tracklets:
            tracklet = self._tracklets.get(tracklet_id)
            if tracklet is None:
                continue

            # 生成进入和离开描述
            entry_desc = self._describe_entry(tracklet)
            exit_desc = self._describe_exit(tracklet)

            segment = ObservationSegment(
                tracklet_id=tracklet_id,
                camera_id=tracklet.camera_id,
                start_time=tracklet.start_time,
                end_time=tracklet.end_time,
                entry_description=entry_desc,
                exit_description=exit_desc,
                direction=tracklet.direction,
            )
            segments.append(segment)

        # 按时间排序
        segments.sort(key=lambda s: s.start_time)

        return segments

    def _build_inference_segments(
        self,
        chain: ObservationChain,
    ) -> List[InferenceSegment]:
        """
        从观测链构建推断段列表

        相邻 Tracklet 之间的跨摄像头连接作为推断段。

        Args:
            chain: 观测链

        Returns:
            推断段列表
        """
        segments = []

        for edge in chain.edges:
            if not edge.is_valid:
                continue

            source_tracklet = self._tracklets.get(edge.source_tracklet_id)
            target_tracklet = self._tracklets.get(edge.target_tracklet_id)

            if source_tracklet is None or target_tracklet is None:
                continue

            # 计算实际时间差
            actual_time = (target_tracklet.start_time - source_tracklet.end_time).total_seconds()

            # 估算理论行驶时间
            estimated_time = self._estimate_travel_time(
                source_tracklet.camera_id,
                target_tracklet.camera_id,
            )

            # 生成路径描述
            route_desc = f"{source_tracklet.camera_id} → {target_tracklet.camera_id}"

            segment = InferenceSegment(
                source_camera_id=source_tracklet.camera_id,
                target_camera_id=target_tracklet.camera_id,
                source_tracklet_id=edge.source_tracklet_id,
                target_tracklet_id=edge.target_tracklet_id,
                confidence=edge.score,
                estimated_travel_time=estimated_time,
                actual_travel_time=actual_time,
                route_description=route_desc,
            )
            segments.append(segment)

        return segments

    def _build_candidate_paths(
        self,
        chain: ObservationChain,
    ) -> List[CandidatePath]:
        """
        从观测链构建候选路径列表

        对于每对相邻摄像头，生成可能的候选路径。

        Args:
            chain: 观测链

        Returns:
            候选路径列表
        """
        paths = []

        # 对于推断段，生成候选路径
        for edge in chain.edges:
            if not edge.is_valid:
                continue

            source_tracklet = self._tracklets.get(edge.source_tracklet_id)
            target_tracklet = self._tracklets.get(edge.target_tracklet_id)

            if source_tracklet is None or target_tracklet is None:
                continue

            # 尝试获取路段信息
            source_cam = self.camera_manager.get_camera(source_tracklet.camera_id)
            target_cam = self.camera_manager.get_camera(target_tracklet.camera_id)

            if source_cam and target_cam:
                # 生成主路径
                path_id = generate_id("PATH")
                distance = self._estimate_distance(
                    source_tracklet.camera_id,
                    target_tracklet.camera_id,
                )
                estimated_time = self._estimate_travel_time(
                    source_tracklet.camera_id,
                    target_tracklet.camera_id,
                )

                main_path = CandidatePath(
                    path_id=path_id,
                    road_segments=[source_cam.covered_road_segment, target_cam.covered_road_segment],
                    confidence=edge.score,
                    distance_meters=distance,
                    estimated_time=estimated_time,
                )
                paths.append(main_path)

                # 如果有替代路径，添加备选路径(降低置信度)
                adjacent = self.camera_manager.get_adjacent_cameras(source_tracklet.camera_id)
                for alt_cam_id in adjacent:
                    if alt_cam_id == target_tracklet.camera_id:
                        continue
                    # 检查是否可达目标
                    if self.camera_manager.is_topologically_reachable(alt_cam_id, target_tracklet.camera_id, max_hops=3):
                        alt_path_id = generate_id("PATH")
                        alt_path = CandidatePath(
                            path_id=alt_path_id,
                            road_segments=[source_cam.covered_road_segment],
                            confidence=edge.score * 0.5,  # 降低置信度
                            distance_meters=distance * 1.5,
                            estimated_time=estimated_time * 1.3,
                        )
                        paths.append(alt_path)
                        break  # 只添加一条备选路径

        # 按置信度排序
        paths.sort(key=lambda p: p.confidence, reverse=True)

        return paths

    def _compute_node_confidence(
        self,
        tracklet_id: str,
        chain: ObservationChain,
    ) -> float:
        """
        计算观测节点置信度

        基于 Tracklet 质量和连接边置信度。

        Args:
            tracklet_id: Tracklet ID
            chain: 观测链

        Returns:
            节点置信度 [0, 1]
        """
        # 基础置信度来自相关边的分数
        related_scores = []
        for edge in chain.edges:
            if edge.source_tracklet_id == tracklet_id or edge.target_tracklet_id == tracklet_id:
                related_scores.append(edge.score)

        if related_scores:
            return sum(related_scores) / len(related_scores)

        # 如果没有相关边，使用默认值
        return 0.5

    def _describe_entry(self, tracklet: Tracklet) -> str:
        """
        生成进入画面描述

        Args:
            tracklet: Tracklet 对象

        Returns:
            进入描述字符串
        """
        if tracklet.entry_point:
            x, y = tracklet.entry_point
            if x < 0.3:
                return "从画面左侧进入"
            elif x > 0.7:
                return "从画面右侧进入"
            elif y < 0.3:
                return "从画面上方进入"
            else:
                return "从画面下方进入"
        return "进入画面"

    def _describe_exit(self, tracklet: Tracklet) -> str:
        """
        生成离开画面描述

        Args:
            tracklet: Tracklet 对象

        Returns:
            离开描述字符串
        """
        if tracklet.exit_point:
            x, y = tracklet.exit_point
            if x > 0.7:
                return "从画面右侧离开"
            elif x < 0.3:
                return "从画面左侧离开"
            elif y > 0.7:
                return "从画面下方离开"
            else:
                return "从画面上方离开"
        return "离开画面"

    def _estimate_travel_time(
        self,
        source_camera_id: str,
        target_camera_id: str,
    ) -> float:
        """
        估算两个摄像头之间的行驶时间

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID

        Returns:
            估算时间(秒)
        """
        source_cam = self.camera_manager.get_camera(source_camera_id)
        target_cam = self.camera_manager.get_camera(target_camera_id)

        if source_cam is None or target_cam is None:
            return 60.0  # 默认值

        # 计算直线距离
        from src.common.utils import haversine_distance
        distance = haversine_distance(
            source_cam.latitude, source_cam.longitude,
            target_cam.latitude, target_cam.longitude,
        )

        # 假设平均速度 40 km/h
        avg_speed = 40.0 * 1000 / 3600  # m/s
        return distance / avg_speed if avg_speed > 0 else 60.0

    def _estimate_distance(
        self,
        source_camera_id: str,
        target_camera_id: str,
    ) -> float:
        """
        估算两个摄像头之间的距离

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID

        Returns:
            估算距离(米)
        """
        source_cam = self.camera_manager.get_camera(source_camera_id)
        target_cam = self.camera_manager.get_camera(target_camera_id)

        if source_cam is None or target_cam is None:
            return 1000.0  # 默认值

        from src.common.utils import haversine_distance
        return haversine_distance(
            source_cam.latitude, source_cam.longitude,
            target_cam.latitude, target_cam.longitude,
        )
