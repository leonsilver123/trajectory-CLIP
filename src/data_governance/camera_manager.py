"""
src.data_governance.camera_manager - 摄像头元数据管理

负责加载和管理摄像头元数据，包括:
- 从 YAML 配置文件加载摄像头信息
- 提供摄像头查询接口(按 ID、按场景)
- 计算摄像头间距离
- 管理摄像头覆盖的路段信息
- 支持动态注册新摄像头
- 支持 AICity22 场景-摄像头映射查询

数据流向: camera_metadata.yaml → CameraManager → 跨镜拼接/轨迹输出
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.common.data_models import CameraMetadata
from src.common.logger import get_logger
from src.common.utils import haversine_distance

logger = get_logger("data_governance.camera_manager")


class CameraManager:
    """
    摄像头元数据管理器

    管理所有摄像头的元数据信息，提供查询和空间计算接口。
    是视频接入与数据治理层的核心类。

    使用方式:
        manager = CameraManager("configs/camera_metadata.yaml")
        cam = manager.get_camera("c001")
        neighbors = manager.get_nearby_cameras(31.3621, 120.6182, radius_km=5.0)
        scene_cams = manager.get_cameras_by_scene("S01")
    """

    def __init__(self, metadata_path: str) -> None:
        """
        初始化摄像头管理器

        Args:
            metadata_path: 摄像头元数据 YAML 文件路径
        """
        self._metadata_path = metadata_path
        self._cameras: Dict[str, CameraMetadata] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._road_segments: Dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        从 YAML 文件加载摄像头元数据

        加载摄像头列表、路段定义和邻接关系，并验证数据完整性。
        """
        path = Path(self._metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"摄像头元数据文件不存在: {self._metadata_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 加载摄像头列表
        cameras_data = data.get("cameras", [])
        seen_ids = set()
        for cam_data in cameras_data:
            camera_id = cam_data.get("camera_id")
            if not camera_id:
                logger.warning("跳过缺少 camera_id 的摄像头配置项")
                continue
            if camera_id in seen_ids:
                logger.warning(f"重复的摄像头 ID: {camera_id}，跳过")
                continue
            seen_ids.add(camera_id)

            try:
                cam = CameraMetadata(
                    camera_id=camera_id,
                    name=cam_data.get("name", camera_id),
                    scene_id=cam_data.get("scene_id") or cam_data.get("scene"),
                    latitude=float(cam_data["latitude"]) if cam_data.get("latitude") is not None else None,
                    longitude=float(cam_data["longitude"]) if cam_data.get("longitude") is not None else None,
                    direction=float(cam_data.get("direction", 0.0)),
                    covered_road_segment=cam_data.get("covered_road_segment", ""),
                    lane_direction=cam_data.get("lane_direction", "eastbound"),
                )
                self._cameras[camera_id] = cam
            except (KeyError, ValueError) as e:
                logger.warning(f"加载摄像头 {camera_id} 失败: {e}")

        # 加载路段定义
        segments_data = data.get("road_segments", [])
        for seg_data in segments_data:
            seg_id = seg_data.get("segment_id")
            if seg_id:
                self._road_segments[seg_id] = seg_data

        # 加载邻接关系
        self._adjacency = data.get("adjacency", {})

        logger.info(
            f"摄像头元数据加载完成: {len(self._cameras)} 个摄像头, "
            f"{len(self._road_segments)} 个路段"
        )

    def get_camera(self, camera_id: str) -> Optional[CameraMetadata]:
        """
        根据 ID 获取摄像头元数据

        Args:
            camera_id: 摄像头 ID

        Returns:
            CameraMetadata 或 None
        """
        return self._cameras.get(camera_id)

    def get_all_cameras(self) -> List[CameraMetadata]:
        """
        获取所有摄像头

        Returns:
            所有摄像头元数据列表
        """
        return list(self._cameras.values())

    def get_nearby_cameras(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> List[CameraMetadata]:
        """
        获取指定坐标附近的摄像头

        使用 Haversine 公式计算球面距离进行过滤。

        Args:
            latitude: 纬度
            longitude: 经度
            radius_km: 搜索半径(公里)

        Returns:
            指定范围内的摄像头列表，按距离升序排列
        """
        radius_m = radius_km * 1000.0
        results = []
        for cam in self._cameras.values():
            if cam.latitude is None or cam.longitude is None:
                continue
            dist = haversine_distance(latitude, longitude, cam.latitude, cam.longitude)
            if dist <= radius_m:
                results.append((dist, cam))
        results.sort(key=lambda x: x[0])
        return [cam for _, cam in results]

    def get_cameras_by_region(self, lane_direction: str) -> List[CameraMetadata]:
        """
        按车道方向筛选摄像头

        Args:
            lane_direction: 车道方向 (eastbound/westbound/northbound/southbound)

        Returns:
            匹配的摄像头列表
        """
        return [
            cam for cam in self._cameras.values()
            if cam.lane_direction == lane_direction
        ]

    def get_cameras_by_scene(self, scene_id: str) -> List[CameraMetadata]:
        """
        按场景 ID 筛选摄像头（AICity22 数据集）

        Args:
            scene_id: 场景 ID（如 'S01', 'S02'）

        Returns:
            该场景下的摄像头列表
        """
        return [
            cam for cam in self._cameras.values()
            if cam.scene_id == scene_id
        ]

    def get_cameras_in_latlon_range(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> List[CameraMetadata]:
        """
        按经纬度范围查询摄像头

        Args:
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度

        Returns:
            范围内的摄像头列表
        """
        return [
            cam for cam in self._cameras.values()
            if cam.latitude is not None and cam.longitude is not None
            and min_lat <= cam.latitude <= max_lat
            and min_lon <= cam.longitude <= max_lon
        ]

    def get_adjacent_cameras(self, camera_id: str) -> List[str]:
        """
        获取拓扑相邻的摄像头 ID 列表

        Args:
            camera_id: 摄像头 ID

        Returns:
            相邻摄像头 ID 列表
        """
        return self._adjacency.get(camera_id, [])

    def are_adjacent(self, camera_id_a: str, camera_id_b: str) -> bool:
        """
        判断两个摄像头是否拓扑相邻

        Args:
            camera_id_a: 摄像头 A ID
            camera_id_b: 摄像头 B ID

        Returns:
            是否相邻
        """
        neighbors_a = self._adjacency.get(camera_id_a, [])
        neighbors_b = self._adjacency.get(camera_id_b, [])
        return camera_id_b in neighbors_a or camera_id_a in neighbors_b

    def is_topologically_reachable(
        self,
        camera_id_a: str,
        camera_id_b: str,
        max_hops: int = 10,
    ) -> bool:
        """
        判断两个摄像头之间是否拓扑可达（BFS）

        Args:
            camera_id_a: 起始摄像头 ID
            camera_id_b: 目标摄像头 ID
            max_hops: 最大跳数

        Returns:
            是否可达
        """
        if camera_id_a not in self._cameras or camera_id_b not in self._cameras:
            return False
        if camera_id_a == camera_id_b:
            return True

        visited = {camera_id_a}
        queue = [(camera_id_a, 0)]
        while queue:
            current, hops = queue.pop(0)
            if hops >= max_hops:
                continue
            for neighbor in self._adjacency.get(current, []):
                if neighbor == camera_id_b:
                    return True
                if neighbor not in visited and neighbor in self._cameras:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
        return False

    def register_camera(
        self,
        camera_id: str,
        name: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        direction: float = 0.0,
        covered_road_segment: str = "",
        lane_direction: str = "eastbound",
        adjacent_cameras: Optional[List[str]] = None,
        scene_id: Optional[str] = None,
    ) -> CameraMetadata:
        """
        动态注册新摄像头

        Args:
            camera_id: 摄像头 ID（如 'c001'）
            name: 摄像头名称
            latitude: 纬度（AICity22 数据集可为空）
            longitude: 经度（AICity22 数据集可为空）
            direction: 朝向角度
            covered_road_segment: 覆盖路段 ID
            lane_direction: 车道方向
            adjacent_cameras: 相邻摄像头 ID 列表
            scene_id: 所属场景 ID（如 'S01'）

        Returns:
            注册的 CameraMetadata 对象

        Raises:
            ValueError: 摄像头 ID 已存在
        """
        if camera_id in self._cameras:
            raise ValueError(f"摄像头 ID 已存在: {camera_id}")

        cam = CameraMetadata(
            camera_id=camera_id,
            name=name,
            scene_id=scene_id,
            latitude=latitude,
            longitude=longitude,
            direction=direction,
            covered_road_segment=covered_road_segment,
            lane_direction=lane_direction,
        )
        self._cameras[camera_id] = cam

        # 更新邻接关系
        if adjacent_cameras:
            self._adjacency[camera_id] = adjacent_cameras
            for adj_id in adjacent_cameras:
                if adj_id in self._adjacency:
                    if camera_id not in self._adjacency[adj_id]:
                        self._adjacency[adj_id].append(camera_id)
                else:
                    self._adjacency[adj_id] = [camera_id]
        else:
            self._adjacency[camera_id] = []

        logger.info(f"注册新摄像头: {camera_id} ({name})")
        return cam

    def get_camera_count(self) -> int:
        """获取摄像头总数"""
        return len(self._cameras)

    @property
    def camera_ids(self) -> List[str]:
        """获取所有摄像头 ID"""
        return list(self._cameras.keys())

    @property
    def road_segments(self) -> Dict[str, dict]:
        """获取所有路段定义"""
        return self._road_segments.copy()
