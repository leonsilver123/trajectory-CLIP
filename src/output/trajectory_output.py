"""
src.output.trajectory_output - 轨迹输出生成模块

将 TrajectoryResult 格式化为各种输出格式:
- JSON 格式(API 返回用)
- 文本摘要格式(人类可读)
- 地图标注格式(前端可视化用，包含经纬度坐标)
- 时间轴数据格式

数据流向: TrajectoryResult → TrajectoryOutputFormatter → 多种输出格式
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.data_models import (
    CandidatePath,
    InferenceSegment,
    ObservationNode,
    ObservationSegment,
    TrajectoryResult,
)
from src.common.logger import get_logger
from src.data_governance.camera_manager import CameraManager

logger = get_logger("output.trajectory_output")


class TrajectoryOutputFormatter:
    """
    轨迹输出格式化器

    将 TrajectoryResult 转换为多种输出格式。

    使用方式:
        formatter = TrajectoryOutputFormatter(camera_manager)
        json_output = formatter.to_json(result)
        text_summary = formatter.to_text_summary(result)
        map_data = formatter.to_map_data(result)
    """

    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
    ) -> None:
        """
        初始化轨迹输出格式化器

        Args:
            camera_manager: 摄像头管理器(用于获取经纬度)
        """
        self.camera_manager = camera_manager
        logger.info("轨迹输出格式化器初始化")

    def to_json(self, result: TrajectoryResult) -> dict:
        """
        将轨迹结果转换为 JSON 格式

        用于 API 返回。

        Args:
            result: 轨迹结果

        Returns:
            JSON 字典
        """
        return {
            "query_id": result.query_id,
            "target_instance": self._instance_to_dict(result.target_instance),
            "observation_nodes": [self._node_to_dict(n) for n in result.observation_nodes],
            "observation_segments": [self._obs_segment_to_dict(s) for s in result.observation_segments],
            "inference_segments": [self._inf_segment_to_dict(s) for s in result.inference_segments],
            "candidate_paths": [self._path_to_dict(p) for p in result.candidate_paths],
            "evidence": result.evidence,
            "overall_confidence": result.overall_confidence,
            "camera_sequence": result.camera_sequence,
            "time_sequence": [t.isoformat() for t in result.time_sequence],
        }

    def to_text_summary(self, result: TrajectoryResult) -> str:
        """
        将轨迹结果转换为文本摘要

        人类可读的格式。

        Args:
            result: 轨迹结果

        Returns:
            文本摘要字符串
        """
        lines = []

        # 标题
        lines.append("=" * 50)
        lines.append("轨迹回溯结果")
        lines.append("=" * 50)

        # 目标信息
        instance = result.target_instance
        lines.append(f"\n【目标信息】")
        lines.append(f"  实例ID: {instance.instance_id}")
        lines.append(f"  目标类别: {instance.target_type}")
        lines.append(f"  来源摄像头: {instance.camera_id}")
        lines.append(f"  检测时间: {instance.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if instance.plate_number:
            lines.append(f"  车牌号: {instance.plate_number}")

        # 摄像头经过序列
        lines.append(f"\n【摄像头经过序列】")
        lines.append(f"  共经过 {len(result.observation_nodes)} 个摄像头")
        for i, node in enumerate(result.observation_nodes, 1):
            time_str = node.timestamp.strftime('%H:%M:%S')
            lines.append(f"  {i}. {node.camera_id} @ {time_str} (置信度: {node.confidence:.2f})")

        # 观测段
        if result.observation_segments:
            lines.append(f"\n【观测段】(摄像头视野内真实轨迹)")
            for seg in result.observation_segments:
                start_time = seg.start_time.strftime('%H:%M:%S')
                end_time = seg.end_time.strftime('%H:%M:%S')
                lines.append(f"  {seg.camera_id}: {start_time} → {end_time}")
                lines.append(f"    {seg.entry_description} → {seg.exit_description}")
                lines.append(f"    方向: {seg.direction}")

        # 推断段
        if result.inference_segments:
            lines.append(f"\n【推断段】(摄像头之间推断连接)")
            for seg in result.inference_segments:
                lines.append(f"  {seg.source_camera_id} → {seg.target_camera_id}")
                lines.append(f"    置信度: {seg.confidence:.2f}")
                lines.append(f"    估算行驶时间: {seg.estimated_travel_time:.0f}秒")
                lines.append(f"    实际时间差: {seg.actual_travel_time:.0f}秒")

        # 候选路径
        if result.candidate_paths:
            lines.append(f"\n【候选路径】")
            for i, path in enumerate(result.candidate_paths, 1):
                lines.append(f"  路径 {i}: 置信度 {path.confidence:.2f}")
                lines.append(f"    距离: {path.distance_meters:.0f}米")
                lines.append(f"    预估时间: {path.estimated_time:.0f}秒")

        # 整体置信度
        lines.append(f"\n【整体轨迹置信度】: {result.overall_confidence:.2f}")
        lines.append("=" * 50)

        return "\n".join(lines)

    def to_map_data(self, result: TrajectoryResult) -> dict:
        """
        将轨迹结果转换为地图标注格式

        包含 GeoJSON 格式的路径数据，供前端地图使用。

        Args:
            result: 轨迹结果

        Returns:
            地图数据字典(GeoJSON 兼容)
        """
        # 构建观测点 FeatureCollection
        observation_features = []
        coordinates = []

        for node in result.observation_nodes:
            # 获取摄像头经纬度
            lat, lon = self._get_camera_latlon(node.camera_id)

            # 添加点要素
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],  # GeoJSON 格式: [经度, 纬度]
                },
                "properties": {
                    "camera_id": node.camera_id,
                    "tracklet_id": node.tracklet_id,
                    "timestamp": node.timestamp.isoformat(),
                    "confidence": node.confidence,
                    "keyframe_path": node.keyframe_path,
                    "marker_type": "observation",
                },
            }
            observation_features.append(feature)
            coordinates.append([lon, lat])

        # 构建路径 LineString
        path_features = []
        if len(coordinates) >= 2:
            # 主路径(实线)
            main_path_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "path_type": "observed",
                    "style": {"color": "#0066CC", "weight": 4, "opacity": 0.8},
                    "description": "观测路径(实线)",
                },
            }
            path_features.append(main_path_feature)

        # 推断段路径(虚线)
        for seg in result.inference_segments:
            source_lat, source_lon = self._get_camera_latlon(seg.source_camera_id)
            target_lat, target_lon = self._get_camera_latlon(seg.target_camera_id)

            inference_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[source_lon, source_lat], [target_lon, target_lat]],
                },
                "properties": {
                    "path_type": "inferred",
                    "confidence": seg.confidence,
                    "style": {
                        "color": "#FF6600",
                        "weight": 2,
                        "opacity": seg.confidence,
                        "dashArray": "5, 10",
                    },
                    "description": f"推断连接 (置信度: {seg.confidence:.2f})",
                },
            }
            path_features.append(inference_feature)

        # 候选路径(不同颜色)
        for i, path in enumerate(result.candidate_paths):
            if path.road_segments:
                candidate_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates,  # 简化表示
                    },
                    "properties": {
                        "path_type": "candidate",
                        "path_id": path.path_id,
                        "confidence": path.confidence,
                        "distance_meters": path.distance_meters,
                        "style": {
                            "color": self._confidence_to_color(path.confidence),
                            "weight": 2,
                            "opacity": 0.5,
                        },
                        "description": f"候选路径 {i+1} (置信度: {path.confidence:.2f})",
                    },
                }
                path_features.append(candidate_feature)

        # 构建完整 GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": observation_features + path_features,
        }

        # 构建地图数据
        map_data = {
            "center": self._compute_center(coordinates),
            "zoom": 14,
            "geojson": geojson,
            "bounds": self._compute_bounds(coordinates),
        }

        return map_data

    def to_timeline(self, result: TrajectoryResult) -> List[dict]:
        """
        将轨迹结果转换为时间轴数据

        每个节点包含时间、摄像头、关键帧路径。

        Args:
            result: 轨迹结果

        Returns:
            时间轴数据列表
        """
        timeline = []

        # 按时间排序的观测节点
        sorted_nodes = sorted(result.observation_nodes, key=lambda n: n.timestamp)

        for i, node in enumerate(sorted_nodes):
            # 查找对应的观测段
            obs_segment = None
            for seg in result.observation_segments:
                if seg.tracklet_id == node.tracklet_id:
                    obs_segment = seg
                    break

            # 查找对应的推断段(进入该节点的连接)
            incoming_inference = None
            for seg in result.inference_segments:
                if seg.target_tracklet_id == node.tracklet_id:
                    incoming_inference = seg
                    break

            # 查找对应的推断段(离开该节点的连接)
            outgoing_inference = None
            for seg in result.inference_segments:
                if seg.source_tracklet_id == node.tracklet_id:
                    outgoing_inference = seg
                    break

            timeline_entry = {
                "index": i + 1,
                "timestamp": node.timestamp.isoformat(),
                "time_display": node.timestamp.strftime('%H:%M:%S'),
                "camera_id": node.camera_id,
                "tracklet_id": node.tracklet_id,
                "keyframe_path": node.keyframe_path,
                "confidence": node.confidence,
                "observation": {
                    "start_time": obs_segment.start_time.isoformat() if obs_segment else None,
                    "end_time": obs_segment.end_time.isoformat() if obs_segment else None,
                    "entry_description": obs_segment.entry_description if obs_segment else None,
                    "exit_description": obs_segment.exit_description if obs_segment else None,
                    "direction": obs_segment.direction if obs_segment else None,
                } if obs_segment else None,
                "incoming_connection": {
                    "source_camera": incoming_inference.source_camera_id,
                    "confidence": incoming_inference.confidence,
                    "travel_time": incoming_inference.actual_travel_time,
                } if incoming_inference else None,
                "outgoing_connection": {
                    "target_camera": outgoing_inference.target_camera_id,
                    "confidence": outgoing_inference.confidence,
                    "travel_time": outgoing_inference.actual_travel_time,
                } if outgoing_inference else None,
            }
            timeline.append(timeline_entry)

        return timeline

    def _instance_to_dict(self, instance) -> dict:
        """将目标实例转换为字典"""
        return {
            "instance_id": instance.instance_id,
            "camera_id": instance.camera_id,
            "timestamp": instance.timestamp.isoformat(),
            "target_type": instance.target_type,
            "plate_number": instance.plate_number,
            "attributes": instance.attributes,
            "quality_score": instance.quality_score,
            "keyframe_path": instance.keyframe_path,
        }

    def _node_to_dict(self, node: ObservationNode) -> dict:
        """将观测节点转换为字典"""
        lat, lon = self._get_camera_latlon(node.camera_id)
        return {
            "camera_id": node.camera_id,
            "tracklet_id": node.tracklet_id,
            "timestamp": node.timestamp.isoformat(),
            "keyframe_path": node.keyframe_path,
            "confidence": node.confidence,
            "latitude": lat,
            "longitude": lon,
        }

    def _obs_segment_to_dict(self, segment: ObservationSegment) -> dict:
        """将观测段转换为字典"""
        return {
            "tracklet_id": segment.tracklet_id,
            "camera_id": segment.camera_id,
            "start_time": segment.start_time.isoformat(),
            "end_time": segment.end_time.isoformat(),
            "entry_description": segment.entry_description,
            "exit_description": segment.exit_description,
            "direction": segment.direction,
        }

    def _inf_segment_to_dict(self, segment: InferenceSegment) -> dict:
        """将推断段转换为字典"""
        return {
            "source_camera_id": segment.source_camera_id,
            "target_camera_id": segment.target_camera_id,
            "source_tracklet_id": segment.source_tracklet_id,
            "target_tracklet_id": segment.target_tracklet_id,
            "confidence": segment.confidence,
            "estimated_travel_time": segment.estimated_travel_time,
            "actual_travel_time": segment.actual_travel_time,
            "route_description": segment.route_description,
        }

    def _path_to_dict(self, path: CandidatePath) -> dict:
        """将候选路径转换为字典"""
        return {
            "path_id": path.path_id,
            "road_segments": path.road_segments,
            "confidence": path.confidence,
            "distance_meters": path.distance_meters,
            "estimated_time": path.estimated_time,
        }

    def _get_camera_latlon(self, camera_id: str) -> tuple:
        """
        获取摄像头经纬度

        Args:
            camera_id: 摄像头 ID

        Returns:
            (纬度, 经度) 元组
        """
        if self.camera_manager:
            cam = self.camera_manager.get_camera(camera_id)
            if cam:
                return cam.latitude, cam.longitude

        # 默认值(苏州相城区)
        return 31.3621, 120.6182

    def _compute_center(self, coordinates: List[List[float]]) -> List[float]:
        """
        计算坐标中心点

        Args:
            coordinates: [[lon, lat], ...] 坐标列表

        Returns:
            [lon, lat] 中心点
        """
        if not coordinates:
            return [120.6182, 31.3621]  # 默认值

        avg_lon = sum(c[0] for c in coordinates) / len(coordinates)
        avg_lat = sum(c[1] for c in coordinates) / len(coordinates)
        return [avg_lon, avg_lat]

    def _compute_bounds(self, coordinates: List[List[float]]) -> dict:
        """
        计算坐标边界

        Args:
            coordinates: [[lon, lat], ...] 坐标列表

        Returns:
            边界字典
        """
        if not coordinates:
            return {
                "south": 31.3,
                "west": 120.5,
                "north": 31.4,
                "east": 120.7,
            }

        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]

        return {
            "south": min(lats) - 0.01,
            "west": min(lons) - 0.01,
            "north": max(lats) + 0.01,
            "east": max(lons) + 0.01,
        }

    def _confidence_to_color(self, confidence: float) -> str:
        """
        根据置信度生成颜色

        Args:
            confidence: 置信度 [0, 1]

        Returns:
            颜色字符串
        """
        if confidence >= 0.8:
            return "#00AA00"  # 绿色
        elif confidence >= 0.6:
            return "#FFAA00"  # 橙色
        else:
            return "#FF0000"  # 红色
