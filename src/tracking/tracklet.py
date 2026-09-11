"""
src.tracking.tracklet - Tracklet 生成与管理

将单摄跟踪结果聚合为 Tracklet (单摄轨迹片段)。
每个 Tracklet 代表一个目标在单个摄像头视野内的完整运动过程。

输入: 单摄跟踪结果序列 (track_id, 类别, bbox, timestamp)
输出: Tracklet 对象列表
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.data_models import BoundingBox, TargetInstance, Tracklet
from src.common.logger import get_logger
from src.common.utils import generate_id

logger = get_logger("tracking.tracklet")


class TrackletGenerator:
    """
    Tracklet 生成器

    将单摄跟踪器的输出聚合为 Tracklet 对象。
    跟踪 ID 不再出现时，将对应的实例序列聚合为一个 Tracklet。

    使用方式:
        generator = TrackletGenerator(camera_id="c001")
        # 逐帧喂入跟踪结果
        generator.add_frame(track_results, timestamp, frame_id)
        # 获取完成的 Tracklet
        tracklets = generator.get_completed_tracklets()
    """

    def __init__(
        self,
        camera_id: str,
        min_instances: int = 3,
        keyframe_interval: int = 5,
        scene_id: Optional[str] = None,
    ) -> None:
        """
        初始化 Tracklet 生成器

        Args:
            camera_id: 所属摄像头 ID（如 'c001'）
            min_instances: 最少实例数（少于此数不生成 Tracklet）
            keyframe_interval: 关键帧间隔（每隔 N 帧保存一张）
            scene_id: 所属场景 ID（如 'S01'，AICity22 数据集）
        """
        self.camera_id = camera_id
        self.scene_id = scene_id
        self.min_instances = min_instances
        self.keyframe_interval = keyframe_interval

        # 活跃的跟踪轨迹 {track_id: {"instances": [TargetInstance, ...], "last_frame": int}}
        self._active_tracks: Dict[int, Dict] = {}
        # 已完成的 Tracklet
        self._completed_tracklets: List[Tracklet] = []
        # 上一次处理的帧 ID (用于检测轨迹终止)
        self._last_frame_id: int = -1

        logger.info(f"Tracklet 生成器初始化: camera={camera_id}")

    def add_frame(
        self,
        track_results: List[Tuple[int, str, BoundingBox]],
        timestamp: datetime,
        frame_id: int,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        添加一帧的跟踪结果

        检测已终止的轨迹（当前帧没有出现的 track_id），
        将其聚合为 Tracklet 并输出。

        Args:
            track_results: 跟踪结果 [(track_id, 类别, bbox), ...]
            timestamp: 当前帧时间戳
            frame_id: 帧编号
            frame: 原始帧（用于保存关键帧，可选）
        """
        # 当前帧中出现的 track_id 集合
        current_track_ids = {tr[0] for tr in track_results}

        # 1. 检测已终止的轨迹：在活跃轨迹中但不在当前帧中
        terminated_ids = []
        for tid in list(self._active_tracks.keys()):
            if tid not in current_track_ids:
                terminated_ids.append(tid)

        # 2. 将终止的轨迹聚合为 Tracklet
        for tid in terminated_ids:
            track_data = self._active_tracks.pop(tid)
            instances = track_data["instances"]
            tracklet = self._build_tracklet(tid, instances)
            if tracklet is not None:
                self._completed_tracklets.append(tracklet)

        # 3. 更新活跃轨迹
        for track_id, target_type, bbox in track_results:
            # 创建 TargetInstance
            instance = TargetInstance(
                instance_id="",
                camera_id=self.camera_id,
                scene_id=self.scene_id,
                timestamp=timestamp,
                frame_id=frame_id,
                target_type=target_type,
                bbox=bbox,
                attributes={},
                plate_number=None,
                plate_confidence=0.0,
                quality_score=0.0,
                reid_vector=None,
                clip_vector=None,
                keyframe_path=None,
            )

            if track_id not in self._active_tracks:
                # 新轨迹
                self._active_tracks[track_id] = {
                    "instances": [instance],
                    "last_frame": frame_id,
                    "target_type": target_type,
                }
            else:
                # 更新已有轨迹
                self._active_tracks[track_id]["instances"].append(instance)
                self._active_tracks[track_id]["last_frame"] = frame_id

            # 保存关键帧路径（如果有原始帧）
            if frame is not None:
                instances_count = len(self._active_tracks[track_id]["instances"])
                if instances_count % self.keyframe_interval == 0:
                    # 记录关键帧索引（后续可由外部模块保存实际图片）
                    instance.keyframe_path = f"{self.camera_id}_trk{track_id}_f{frame_id}"

        self._last_frame_id = frame_id

    def _build_tracklet(
        self,
        track_id: int,
        instances: List[TargetInstance],
    ) -> Optional[Tracklet]:
        """
        将一组 TargetInstance 聚合为 Tracklet

        Args:
            track_id: 跟踪 ID
            instances: 目标实例序列

        Returns:
            Tracklet 对象，实例数不足时返回 None
        """
        # 1. 检查实例数是否满足最低要求
        if len(instances) < self.min_instances:
            return None

        # 按时间排序
        instances.sort(key=lambda inst: (inst.timestamp, inst.frame_id))

        # 2. 聚合属性（取众数）
        aggregated_attrs = self._aggregate_attributes(instances)

        # 3. 计算平均 ReID/CLIP 向量
        avg_reid = self._average_vectors([
            inst.reid_vector for inst in instances if inst.reid_vector is not None
        ])
        avg_clip = self._average_vectors([
            inst.clip_vector for inst in instances if inst.clip_vector is not None
        ])

        # 4. 确定进入/离开位置
        first_bbox = instances[0].bbox
        last_bbox = instances[-1].bbox
        entry_point = first_bbox.center
        exit_point = last_bbox.center

        # 5. 确定运动方向
        direction = self._compute_direction(entry_point, exit_point)

        # 6. 聚合车牌号（取出现最多的）
        plate_numbers = [
            inst.plate_number for inst in instances
            if inst.plate_number is not None and inst.plate_number.strip()
        ]
        plate_number = None
        if plate_numbers:
            plate_counter = Counter(plate_numbers)
            plate_number = plate_counter.most_common(1)[0][0]

        # 7. 提取关键帧路径
        keyframe_paths = [
            inst.keyframe_path for inst in instances
            if inst.keyframe_path is not None
        ]

        # 8. 生成 Tracklet
        tracklet_id = generate_id(f"TRK_{self.camera_id}")

        tracklet = Tracklet(
            tracklet_id=tracklet_id,
            camera_id=self.camera_id,
            scene_id=self.scene_id,
            target_type=instances[0].target_type,
            start_time=instances[0].timestamp,
            end_time=instances[-1].timestamp,
            instances=instances,
            direction=direction,
            plate_number=plate_number,
            attributes=aggregated_attrs,
            avg_reid_vector=avg_reid,
            avg_clip_vector=avg_clip,
            keyframe_paths=keyframe_paths,
            entry_point=entry_point,
            exit_point=exit_point,
        )

        logger.debug(
            f"Tracklet 生成: id={tracklet_id}, camera={self.camera_id}, "
            f"type={tracklet.target_type}, instances={len(instances)}, "
            f"duration={tracklet.duration_seconds:.1f}s"
        )

        return tracklet

    def _aggregate_attributes(self, instances: List[TargetInstance]) -> Dict:
        """
        聚合多个实例的属性（取众数）

        Args:
            instances: 目标实例列表

        Returns:
            聚合后的属性字典
        """
        if not instances:
            return {}

        # 收集所有属性键
        all_keys = set()
        for inst in instances:
            all_keys.update(inst.attributes.keys())

        aggregated = {}
        for key in all_keys:
            values = [
                inst.attributes[key] for inst in instances
                if key in inst.attributes and inst.attributes[key] is not None
            ]
            if values:
                # 对数值型取平均，对字符串取众数
                if isinstance(values[0], (int, float)):
                    aggregated[key] = sum(values) / len(values)
                else:
                    counter = Counter(values)
                    aggregated[key] = counter.most_common(1)[0][0]

        return aggregated

    def _average_vectors(
        self,
        vectors: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        计算向量列表的平均值并 L2 归一化

        Args:
            vectors: 向量列表

        Returns:
            归一化后的平均向量，列表为空时返回 None
        """
        if not vectors:
            return None

        try:
            stacked = np.stack(vectors)
            mean_vec = stacked.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            return mean_vec.astype(np.float32)
        except Exception as e:
            logger.warning(f"向量平均计算失败: {e}")
            return None

    def _compute_direction(
        self,
        entry_point: Tuple[float, float],
        exit_point: Tuple[float, float],
    ) -> str:
        """
        基于轨迹起止位置计算运动方向

        使用角度描述运动方向（如"由西向东"、"由左向右"等）。

        Args:
            entry_point: 进入画面位置 (x, y)
            exit_point: 离开画面位置 (x, y)

        Returns:
            方向描述字符串
        """
        dx = exit_point[0] - entry_point[0]
        dy = exit_point[1] - entry_point[1]

        # 计算角度（以正东为 0°，逆时针为正）
        import math
        angle = math.degrees(math.atan2(-dy, dx))  # y 轴向下取反
        if angle < 0:
            angle += 360

        # 将角度映射到方向描述
        if 315 <= angle or angle < 45:
            return "由西向东"
        elif 45 <= angle < 135:
            return "由南向北"
        elif 135 <= angle < 225:
            return "由东向西"
        elif 225 <= angle < 315:
            return "由北向南"
        else:
            return "未知方向"

    def get_completed_tracklets(self) -> List[Tracklet]:
        """
        获取所有已完成的 Tracklet

        Returns:
            已完成的 Tracklet 列表
        """
        result = self._completed_tracklets.copy()
        self._completed_tracklets.clear()
        return result

    def flush(self) -> List[Tracklet]:
        """
        强制输出所有活跃轨迹为 Tracklet（视频结束时调用）

        Returns:
            强制输出的 Tracklet 列表
        """
        flushed = []
        for track_id, track_data in self._active_tracks.items():
            instances = track_data["instances"]
            tracklet = self._build_tracklet(track_id, instances)
            if tracklet is not None:
                flushed.append(tracklet)

        self._active_tracks.clear()
        logger.info(f"强制输出 {len(flushed)} 个 Tracklet (camera={self.camera_id})")
        return flushed
