"""
src.data_governance.road_topology - 道路拓扑构建

负责构建和管理道路拓扑图，包括:
- 从配置文件加载路段和邻接关系
- 构建摄像头间的拓扑图
- 计算最短路径和可达性
- 估算合理旅行时间

数据流向: camera_metadata.yaml → RoadTopology → 跨镜候选边生成
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from src.common.data_models import CameraMetadata, RoadSegment
from src.common.logger import get_logger
from src.common.utils import haversine_distance

logger = get_logger("data_governance.road_topology")


class RoadTopology:
    """
    道路拓扑管理器

    构建摄像头之间的道路拓扑图，用于跨镜轨迹拼接时的
    空间可达性判断和路径规划。

    使用方式:
        topology = RoadTopology("configs/camera_metadata.yaml")
        is_reachable = topology.is_reachable("c001", "c005")
        path = topology.shortest_path("c001", "c005")
    """

    # 城区默认速度范围 (km/h)
    DEFAULT_MIN_SPEED = 20.0
    DEFAULT_MAX_SPEED = 80.0

    def __init__(self, metadata_path: str) -> None:
        """
        初始化道路拓扑

        Args:
            metadata_path: 摄像头元数据 YAML 文件路径(包含路段和邻接信息)
        """
        self._metadata_path = metadata_path
        self._segments: Dict[str, RoadSegment] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._camera_to_segment: Dict[str, str] = {}
        self._cameras: Dict[str, CameraMetadata] = {}
        # 摄像头间距离缓存 {(cam_a, cam_b): distance_meters}
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self._build_topology()

    def _build_topology(self) -> None:
        """
        从配置文件构建道路拓扑图

        加载路段定义、构建邻接表、建立摄像头到路段的映射、验证拓扑一致性。
        """
        path = Path(self._metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"摄像头元数据文件不存在: {self._metadata_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 1. 加载摄像头元数据
        cameras_data = data.get("cameras", [])
        for cam_data in cameras_data:
            camera_id = cam_data.get("camera_id")
            if not camera_id:
                continue
            try:
                cam = CameraMetadata(
                    camera_id=camera_id,
                    name=cam_data.get("name", camera_id),
                    scene_id=cam_data.get("scene_id"),
                    latitude=float(cam_data["latitude"]) if cam_data.get("latitude") is not None else None,
                    longitude=float(cam_data["longitude"]) if cam_data.get("longitude") is not None else None,
                    direction=float(cam_data.get("direction", 0.0)),
                    covered_road_segment=cam_data.get("covered_road_segment", ""),
                    lane_direction=cam_data.get("lane_direction", "eastbound"),
                )
                self._cameras[camera_id] = cam
            except (KeyError, ValueError) as e:
                logger.warning(f"加载摄像头 {camera_id} 失败: {e}")

        # 2. 加载路段定义并构建 RoadSegment 对象
        segments_data = data.get("road_segments", [])
        for seg_data in segments_data:
            seg_id = seg_data.get("segment_id")
            if not seg_id:
                continue
            try:
                segment = RoadSegment(
                    segment_id=seg_id,
                    name=seg_data.get("name", seg_id),
                    start_camera_ids=seg_data.get("start_camera_ids", []),
                    end_camera_ids=seg_data.get("end_camera_ids", []),
                    distance_meters=float(seg_data.get("distance_meters", 0.0)),
                    direction=seg_data.get("direction", ""),
                )
                self._segments[seg_id] = segment
                # 建立摄像头到路段的映射
                for cam_id in segment.start_camera_ids + segment.end_camera_ids:
                    if cam_id not in self._camera_to_segment:
                        self._camera_to_segment[cam_id] = seg_id
            except (KeyError, ValueError) as e:
                logger.warning(f"加载路段 {seg_id} 失败: {e}")

        # 3. 加载邻接关系
        self._adjacency = data.get("adjacency", {})

        # 4. 预计算相邻摄像头间的距离（基于 Haversine 公式，仅当 GPS 坐标可用时）
        for cam_id, neighbors in self._adjacency.items():
            if cam_id not in self._cameras:
                continue
            cam_a = self._cameras[cam_id]
            for neighbor_id in neighbors:
                if neighbor_id not in self._cameras:
                    continue
                cam_b = self._cameras[neighbor_id]
                if cam_a.latitude is None or cam_a.longitude is None:
                    continue
                if cam_b.latitude is None or cam_b.longitude is None:
                    continue
                pair = (cam_id, neighbor_id)
                if pair not in self._distance_cache:
                    dist = haversine_distance(
                        cam_a.latitude, cam_a.longitude,
                        cam_b.latitude, cam_b.longitude,
                    )
                    self._distance_cache[pair] = dist
                    # 对称距离
                    self._distance_cache[(neighbor_id, cam_id)] = dist

        # 5. 验证拓扑一致性：检查邻接表中引用的摄像头是否都存在
        for cam_id, neighbors in self._adjacency.items():
            if cam_id not in self._cameras:
                logger.warning(f"邻接表中的摄像头 {cam_id} 未在摄像头列表中找到")
            for neighbor_id in neighbors:
                if neighbor_id not in self._cameras:
                    logger.warning(
                        f"邻接表引用的邻居摄像头 {neighbor_id} "
                        f"(来自 {cam_id}) 未在摄像头列表中找到"
                    )

        logger.info(
            f"道路拓扑构建完成: {len(self._cameras)} 个摄像头, "
            f"{len(self._segments)} 个路段, "
            f"{len(self._distance_cache) // 2} 条邻接边"
        )

    def is_reachable(
        self,
        source_camera_id: str,
        target_camera_id: str,
        max_hops: int = 10,
    ) -> bool:
        """
        判断从源摄像头是否可以到达目标摄像头（BFS）

        考虑方向约束：如果两个摄像头方向相反且不在同一路段，
        则需要更多跳数才能到达。

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID
            max_hops: 最大跳数

        Returns:
            是否可达
        """
        if source_camera_id not in self._cameras or target_camera_id not in self._cameras:
            return False
        if source_camera_id == target_camera_id:
            return True

        visited: Set[str] = {source_camera_id}
        queue: deque = deque([(source_camera_id, 0)])

        while queue:
            current, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbor in self._adjacency.get(current, []):
                if neighbor == target_camera_id:
                    return True
                if neighbor not in visited and neighbor in self._cameras:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
        return False

    def shortest_path(
        self,
        source_camera_id: str,
        target_camera_id: str,
    ) -> Optional[List[str]]:
        """
        计算两个摄像头之间的最短路径（Dijkstra 算法，按路段距离）

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID

        Returns:
            路径上的摄像头 ID 列表，不可达时返回 None
        """
        if source_camera_id not in self._cameras or target_camera_id not in self._cameras:
            return None
        if source_camera_id == target_camera_id:
            return [source_camera_id]

        # Dijkstra 算法
        # dist[node] = 从源到该节点的最短距离
        dist: Dict[str, float] = {source_camera_id: 0.0}
        # prev[node] = 最短路径上的前驱节点
        prev: Dict[str, Optional[str]] = {source_camera_id: None}
        # 优先队列 (distance, node)
        pq: List[Tuple[float, str]] = [(0.0, source_camera_id)]
        visited: Set[str] = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == target_camera_id:
                break

            for neighbor in self._adjacency.get(u, []):
                if neighbor in visited or neighbor not in self._cameras:
                    continue
                # 获取 u 到 neighbor 的距离
                edge_dist = self._get_edge_distance(u, neighbor)
                new_dist = d + edge_dist
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))

        # 回溯路径
        if target_camera_id not in prev:
            return None

        path: List[str] = []
        current: Optional[str] = target_camera_id
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()
        return path

    def get_n_hop_neighbors(
        self,
        camera_id: str,
        n: int = 1,
    ) -> List[str]:
        """
        获取摄像头的 N 跳邻居

        Args:
            camera_id: 摄像头 ID
            n: 跳数（1 表示直接邻居）

        Returns:
            N 跳邻居摄像头 ID 列表（不包含自身和更近层的邻居）
        """
        if camera_id not in self._cameras:
            return []

        # BFS 逐层扩展
        visited: Set[str] = {camera_id}
        current_layer: List[str] = [camera_id]

        for _ in range(n):
            next_layer: List[str] = []
            for node in current_layer:
                for neighbor in self._adjacency.get(node, []):
                    if neighbor not in visited and neighbor in self._cameras:
                        visited.add(neighbor)
                        next_layer.append(neighbor)
            current_layer = next_layer

        return current_layer

    def get_segment_for_camera(self, camera_id: str) -> Optional[RoadSegment]:
        """
        获取摄像头所在的路段

        Args:
            camera_id: 摄像头 ID

        Returns:
            路段对象或 None
        """
        seg_id = self._camera_to_segment.get(camera_id)
        if seg_id:
            return self._segments.get(seg_id)
        return None

    def get_segment_distance(
        self,
        source_camera_id: str,
        target_camera_id: str,
    ) -> Optional[float]:
        """
        获取两个摄像头之间的路段距离(米)

        优先使用路段定义中的距离，如果没有则使用 Haversine 距离。
        如果两个摄像头相邻，直接返回缓存的距离。
        如果不可达，尝试沿最短路径累加距离。

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID

        Returns:
            路段距离(米)，不可达时返回 None
        """
        if source_camera_id not in self._cameras or target_camera_id not in self._cameras:
            return None

        # 直接相邻的情况
        pair = (source_camera_id, target_camera_id)
        if pair in self._distance_cache:
            return self._distance_cache[pair]

        # 尝试沿最短路径累加距离
        path = self.shortest_path(source_camera_id, target_camera_id)
        if path is None or len(path) < 2:
            return None

        total_distance = 0.0
        for i in range(len(path) - 1):
            edge_pair = (path[i], path[i + 1])
            edge_dist = self._distance_cache.get(edge_pair)
            if edge_dist is None:
                return None
            total_distance += edge_dist
        return total_distance

    def estimate_travel_time(
        self,
        source_camera_id: str,
        target_camera_id: str,
        min_speed_kmh: Optional[float] = None,
        max_speed_kmh: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        估算两个摄像头之间的合理旅行时间范围

        基于路段距离和城区速度约束（默认 20-80 km/h）估算。

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID
            min_speed_kmh: 最低速度(km/h)，默认 20 km/h（城区拥堵）
            max_speed_kmh: 最高速度(km/h)，默认 80 km/h（城区限速）

        Returns:
            (最短时间秒, 最长时间秒)，不可达时返回 None
        """
        if min_speed_kmh is None:
            min_speed_kmh = self.DEFAULT_MIN_SPEED
        if max_speed_kmh is None:
            max_speed_kmh = self.DEFAULT_MAX_SPEED

        distance = self.get_segment_distance(source_camera_id, target_camera_id)
        if distance is None or distance <= 0:
            return None

        # 时间 = 距离 / 速度
        # 最短时间 = 距离 / 最高速度
        # 最长时间 = 距离 / 最低速度
        min_time = distance / (max_speed_kmh * 1000.0 / 3600.0)
        max_time = distance / (min_speed_kmh * 1000.0 / 3600.0)

        return (min_time, max_time)

    def is_direction_compatible(
        self,
        source_camera_id: str,
        target_camera_id: str,
    ) -> bool:
        """
        判断两个摄像头之间的方向是否兼容（考虑车道方向约束）

        同一路段上方向相反的摄像头之间不可直接连接（除非通过路口绕行）。

        Args:
            source_camera_id: 源摄像头 ID
            target_camera_id: 目标摄像头 ID

        Returns:
            方向是否兼容
        """
        cam_a = self._cameras.get(source_camera_id)
        cam_b = self._cameras.get(target_camera_id)
        if cam_a is None or cam_b is None:
            return False

        # 如果两个摄像头在同一路段且方向相反，则不兼容
        if cam_a.covered_road_segment == cam_b.covered_road_segment:
            if cam_a.is_opposite_direction(cam_b):
                return False

        return True

    def get_all_segments(self) -> List[RoadSegment]:
        """获取所有路段"""
        return list(self._segments.values())

    def get_camera_metadata(self, camera_id: str) -> Optional[CameraMetadata]:
        """
        获取摄像头元数据

        Args:
            camera_id: 摄像头 ID

        Returns:
            CameraMetadata 或 None
        """
        return self._cameras.get(camera_id)

    @property
    def camera_ids(self) -> List[str]:
        """获取拓扑中所有摄像头 ID"""
        return list(self._adjacency.keys())

    @property
    def adjacency(self) -> Dict[str, List[str]]:
        """获取邻接表"""
        return self._adjacency.copy()

    def _get_edge_distance(self, cam_a: str, cam_b: str) -> float:
        """
        获取两个相邻摄像头之间的边距离

        优先使用缓存的 Haversine 距离。

        Args:
            cam_a: 摄像头 A ID
            cam_b: 摄像头 B ID

        Returns:
            距离(米)
        """
        pair = (cam_a, cam_b)
        if pair in self._distance_cache:
            return self._distance_cache[pair]

        # 如果缓存中没有，实时计算
        meta_a = self._cameras.get(cam_a)
        meta_b = self._cameras.get(cam_b)
        if meta_a and meta_b and meta_a.latitude is not None and meta_b.latitude is not None:
            dist = haversine_distance(
                meta_a.latitude, meta_a.longitude,
                meta_b.latitude, meta_b.longitude,
            )
            self._distance_cache[pair] = dist
            self._distance_cache[(cam_b, cam_a)] = dist
            return dist

        return float("inf")
