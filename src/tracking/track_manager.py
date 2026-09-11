"""
src.tracking.track_manager - 轨迹管理器

管理所有摄像头的 Tracklet 生命周期，提供:
- Tracklet 注册与查询
- 按摄像头/时间/目标类别检索 Tracklet
- Tracklet 持久化存储（JSON 文件）
- Tracklet 的增删改查

数据流向: TrackletGenerator → TrackManager → 跨镜拼接/检索模块
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.data_models import Tracklet
from src.common.logger import get_logger

logger = get_logger("tracking.track_manager")


class TrackManager:
    """
    轨迹管理器

    集中管理所有摄像头的 Tracklet，提供查询和索引接口。
    支持按摄像头、时间范围、目标类型查询，以及 JSON 持久化。

    使用方式:
        manager = TrackManager()
        manager.register_tracklet(tracklet)
        tracklets = manager.get_tracklets_by_camera("c001")
        manager.save_to_json("output/tracklets.json")
    """

    def __init__(self) -> None:
        """初始化轨迹管理器"""
        # Tracklet 存储 {tracklet_id: Tracklet}
        self._tracklets: Dict[str, Tracklet] = {}
        # 摄像头索引 {camera_id: [tracklet_id, ...]}
        self._camera_index: Dict[str, List[str]] = {}
        # 车牌索引 {plate_number: [tracklet_id, ...]}
        self._plate_index: Dict[str, List[str]] = {}
        # 目标类型索引 {target_type: [tracklet_id, ...]}
        self._type_index: Dict[str, List[str]] = {}

        logger.info("轨迹管理器初始化")

    def register_tracklet(self, tracklet: Tracklet) -> None:
        """
        注册一个 Tracklet

        Args:
            tracklet: Tracklet 对象

        Raises:
            ValueError: Tracklet ID 已存在
        """
        tid = tracklet.tracklet_id
        if tid in self._tracklets:
            logger.warning(f"Tracklet ID 已存在，将覆盖: {tid}")

        self._tracklets[tid] = tracklet

        # 更新摄像头索引
        cam_id = tracklet.camera_id
        if cam_id not in self._camera_index:
            self._camera_index[cam_id] = []
        if tid not in self._camera_index[cam_id]:
            self._camera_index[cam_id].append(tid)

        # 更新车牌索引
        if tracklet.has_plate:
            plate = tracklet.plate_number
            if plate not in self._plate_index:
                self._plate_index[plate] = []
            if tid not in self._plate_index[plate]:
                self._plate_index[plate].append(tid)

        # 更新目标类型索引
        t_type = tracklet.target_type
        if t_type not in self._type_index:
            self._type_index[t_type] = []
        if tid not in self._type_index[t_type]:
            self._type_index[t_type].append(tid)

        logger.debug(f"注册 Tracklet: {tid} (camera={cam_id}, type={t_type})")

    def get_tracklet(self, tracklet_id: str) -> Optional[Tracklet]:
        """
        根据 ID 获取 Tracklet

        Args:
            tracklet_id: Tracklet ID

        Returns:
            Tracklet 或 None
        """
        return self._tracklets.get(tracklet_id)

    def get_tracklets_by_camera(
        self,
        camera_id: str,
        target_type: Optional[str] = None,
    ) -> List[Tracklet]:
        """
        获取指定摄像头的 Tracklet 列表

        Args:
            camera_id: 摄像头 ID
            target_type: 目标类别过滤（可选）

        Returns:
            Tracklet 列表，按开始时间升序排列
        """
        tracklet_ids = self._camera_index.get(camera_id, [])
        results = [self._tracklets[tid] for tid in tracklet_ids if tid in self._tracklets]

        # 目标类别过滤
        if target_type is not None:
            results = [t for t in results if t.target_type == target_type]

        # 按开始时间排序
        results.sort(key=lambda t: t.start_time)
        return results

    def get_tracklets_by_plate(
        self,
        plate_number: str,
    ) -> List[Tracklet]:
        """
        根据车牌号查询 Tracklet

        Args:
            plate_number: 车牌号

        Returns:
            匹配的 Tracklet 列表
        """
        tracklet_ids = self._plate_index.get(plate_number, [])
        return [self._tracklets[tid] for tid in tracklet_ids if tid in self._tracklets]

    def get_tracklets_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        camera_id: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> List[Tracklet]:
        """
        按时间范围查询 Tracklet

        返回在指定时间范围内有重叠的 Tracklet。

        Args:
            start_time: 查询起始时间
            end_time: 查询结束时间
            camera_id: 摄像头 ID 过滤（可选）
            target_type: 目标类别过滤（可选）

        Returns:
            匹配的 Tracklet 列表
        """
        results = []
        for tracklet in self._tracklets.values():
            # 时间范围重叠判断
            if tracklet.end_time < start_time or tracklet.start_time > end_time:
                continue
            # 摄像头过滤
            if camera_id is not None and tracklet.camera_id != camera_id:
                continue
            # 目标类别过滤
            if target_type is not None and tracklet.target_type != target_type:
                continue
            results.append(tracklet)

        results.sort(key=lambda t: t.start_time)
        return results

    def get_tracklets_by_type(
        self,
        target_type: str,
    ) -> List[Tracklet]:
        """
        按目标类型查询 Tracklet

        Args:
            target_type: 目标类别 ("vehicle" / "pedestrian" / "non_motor_vehicle")

        Returns:
            匹配的 Tracklet 列表
        """
        tracklet_ids = self._type_index.get(target_type, [])
        return [self._tracklets[tid] for tid in tracklet_ids if tid in self._tracklets]

    def remove_tracklet(self, tracklet_id: str) -> bool:
        """
        删除一个 Tracklet

        Args:
            tracklet_id: Tracklet ID

        Returns:
            是否删除成功
        """
        tracklet = self._tracklets.pop(tracklet_id, None)
        if tracklet is None:
            return False

        # 清理索引
        cam_id = tracklet.camera_id
        if cam_id in self._camera_index:
            self._camera_index[cam_id] = [
                tid for tid in self._camera_index[cam_id] if tid != tracklet_id
            ]

        if tracklet.has_plate:
            plate = tracklet.plate_number
            if plate in self._plate_index:
                self._plate_index[plate] = [
                    tid for tid in self._plate_index[plate] if tid != tracklet_id
                ]

        t_type = tracklet.target_type
        if t_type in self._type_index:
            self._type_index[t_type] = [
                tid for tid in self._type_index[t_type] if tid != tracklet_id
            ]

        logger.debug(f"删除 Tracklet: {tracklet_id}")
        return True

    def get_all_tracklets(self) -> List[Tracklet]:
        """获取所有 Tracklet"""
        return list(self._tracklets.values())

    @property
    def tracklet_count(self) -> int:
        """获取 Tracklet 总数"""
        return len(self._tracklets)

    @property
    def camera_ids(self) -> List[str]:
        """获取有 Tracklet 的摄像头 ID 列表"""
        return list(self._camera_index.keys())

    def save_to_json(self, filepath: str) -> None:
        """
        将 Tracklet 数据持久化为 JSON 文件

        Args:
            filepath: JSON 文件路径
        """
        try:
            data = {
                "tracklet_count": len(self._tracklets),
                "tracklets": [],
            }

            for tracklet in self._tracklets.values():
                trk_data = self._tracklet_to_dict(tracklet)
                data["tracklets"].append(trk_data)

            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Tracklet 数据已保存到: {filepath} ({len(self._tracklets)} 条)")

        except Exception as e:
            logger.error(f"Tracklet 数据保存失败: {e}")
            raise

    def load_from_json(self, filepath: str) -> int:
        """
        从 JSON 文件加载 Tracklet 数据

        Args:
            filepath: JSON 文件路径

        Returns:
            加载的 Tracklet 数量
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            count = 0
            for trk_data in data.get("tracklets", []):
                tracklet = self._dict_to_tracklet(trk_data)
                if tracklet is not None:
                    self.register_tracklet(tracklet)
                    count += 1

            logger.info(f"从 {filepath} 加载了 {count} 条 Tracklet")
            return count

        except Exception as e:
            logger.error(f"Tracklet 数据加载失败: {e}")
            return 0

    def _tracklet_to_dict(self, tracklet: Tracklet) -> Dict[str, Any]:
        """
        将 Tracklet 转换为可 JSON 序列化的字典

        Args:
            tracklet: Tracklet 对象

        Returns:
            字典
        """
        return {
            "tracklet_id": tracklet.tracklet_id,
            "camera_id": tracklet.camera_id,
            "scene_id": tracklet.scene_id,
            "target_type": tracklet.target_type,
            "start_time": tracklet.start_time.isoformat(),
            "end_time": tracklet.end_time.isoformat(),
            "direction": tracklet.direction,
            "plate_number": tracklet.plate_number,
            "attributes": tracklet.attributes,
            "avg_reid_vector": (
                tracklet.avg_reid_vector.tolist()
                if tracklet.avg_reid_vector is not None else None
            ),
            "avg_clip_vector": (
                tracklet.avg_clip_vector.tolist()
                if tracklet.avg_clip_vector is not None else None
            ),
            "keyframe_paths": tracklet.keyframe_paths,
            "entry_point": list(tracklet.entry_point) if tracklet.entry_point else None,
            "exit_point": list(tracklet.exit_point) if tracklet.exit_point else None,
            "instance_count": tracklet.instance_count,
            "duration_seconds": tracklet.duration_seconds,
        }

    def _dict_to_tracklet(self, data: Dict[str, Any]) -> Optional[Tracklet]:
        """
        从字典恢复 Tracklet 对象

        Args:
            data: 字典数据

        Returns:
            Tracklet 对象，解析失败返回 None
        """
        try:
            avg_reid = None
            if data.get("avg_reid_vector") is not None:
                avg_reid = np.array(data["avg_reid_vector"], dtype=np.float32)

            avg_clip = None
            if data.get("avg_clip_vector") is not None:
                avg_clip = np.array(data["avg_clip_vector"], dtype=np.float32)

            entry_point = tuple(data["entry_point"]) if data.get("entry_point") else None
            exit_point = tuple(data["exit_point"]) if data.get("exit_point") else None

            tracklet = Tracklet(
                tracklet_id=data["tracklet_id"],
                camera_id=data["camera_id"],
                scene_id=data.get("scene_id"),
                target_type=data["target_type"],
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]),
                instances=[],  # 实例列表不序列化
                direction=data.get("direction", ""),
                plate_number=data.get("plate_number"),
                attributes=data.get("attributes", {}),
                avg_reid_vector=avg_reid,
                avg_clip_vector=avg_clip,
                keyframe_paths=data.get("keyframe_paths", []),
                entry_point=entry_point,
                exit_point=exit_point,
            )
            return tracklet

        except Exception as e:
            logger.warning(f"Tracklet 数据解析失败: {e}")
            return None
