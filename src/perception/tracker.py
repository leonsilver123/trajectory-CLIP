"""
src.perception.tracker - 单摄多目标跟踪模块

实现简化的 ByteTrack 风格跟踪，不依赖外部跟踪库。
核心逻辑：IoU 匹配 + 卡尔曼滤波预测。

输入: 逐帧检测结果
输出: 带 track_id 的跟踪结果序列
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.data_models import BoundingBox
from src.common.logger import get_logger

logger = get_logger("perception.tracker")


class _KalmanFilter:
    """
    卡尔曼滤波器

    状态向量: [x, y, w, h, vx, vy, vw, vh]
    其中 (x, y) 是检测框中心, (w, h) 是宽高,
    (vx, vy, vw, vh) 是对应速度。
    """

    def __init__(self) -> None:
        """初始化卡尔曼滤波器"""
        # 状态向量 8 维
        self.state = np.zeros(8, dtype=np.float64)
        # 状态转移矩阵 (匀速模型)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # x += vx
        self.F[1, 5] = 1.0  # y += vy
        self.F[2, 6] = 1.0  # w += vw
        self.F[3, 7] = 1.0  # h += vh
        # 观测矩阵 (只观测 [x, y, w, h])
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        # 状态协方差矩阵
        self.P = np.eye(8, dtype=np.float64) * 10.0
        # 过程噪声协方差
        self.Q = np.eye(8, dtype=np.float64) * 1.0
        self.Q[4:, 4:] *= 5.0  # 速度分量噪声更大
        # 观测噪声协方差
        self.R = np.eye(4, dtype=np.float64) * 5.0

    def predict(self) -> np.ndarray:
        """
        预测步骤

        Returns:
            预测的状态向量
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新步骤

        Args:
            measurement: 观测值 [x, y, w, h]

        Returns:
            更新后的状态向量
        """
        # 卡尔曼增益
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # 更新状态和协方差
        self.state = self.state + K @ y
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.state

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        从状态向量获取检测框 [x1, y1, x2, y2]

        Returns:
            (x1, y1, x2, y2)
        """
        cx, cy, w, h = self.state[0], self.state[1], self.state[2], self.state[3]
        w = max(w, 1.0)
        h = max(h, 1.0)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)


def _compute_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    计算两组检测框之间的 IoU 矩阵

    Args:
        boxes_a: (N, 4) 格式 [x1, y1, x2, y2]
        boxes_b: (M, 4) 格式 [x1, y1, x2, y2]

    Returns:
        IoU 矩阵 (N, M)
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    # 计算交集面积
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # 计算各自面积
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return intersection / union


class _Track:
    """
    单条跟踪轨迹

    封装卡尔曼滤波器和轨迹状态管理。
    """

    _next_id: int = 0

    def __init__(self, bbox: BoundingBox, target_type: str, frame_id: int) -> None:
        """
        初始化跟踪轨迹

        Args:
            bbox: 初始检测框
            target_type: 目标类别
            frame_id: 创建时的帧编号
        """
        self.track_id = _Track._next_id
        _Track._next_id += 1

        self.target_type = target_type
        self.kf = _KalmanFilter()
        self.age = 0            # 轨迹存在总帧数
        self.hits = 0           # 成功匹配帧数
        self.time_since_update = 0  # 距上次更新的帧数

        # 用初始检测框初始化卡尔曼滤波器状态
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w = bbox.width
        h = bbox.height
        self.kf.state[:4] = [cx, cy, w, h]
        self.hits = 1
        self.last_bbox = bbox

    def predict(self) -> np.ndarray:
        """
        预测步骤（仅更新卡尔曼滤波器状态，不修改跟踪元数据）

        Returns:
            预测的状态向量前 4 维 [cx, cy, w, h]
        """
        return self.kf.predict()[:4]

    def update(self, bbox: BoundingBox, frame_id: int) -> None:
        """
        用新的检测结果更新轨迹

        Args:
            bbox: 新的检测框
            frame_id: 帧编号
        """
        self.time_since_update = 0
        self.hits += 1
        self.last_bbox = bbox

        # 构造观测值 [cx, cy, w, h]
        measurement = np.array([
            (bbox.x1 + bbox.x2) / 2,
            (bbox.y1 + bbox.y2) / 2,
            bbox.width,
            bbox.height,
        ], dtype=np.float64)
        self.kf.update(measurement)

    def get_predicted_bbox(self) -> BoundingBox:
        """
        获取预测的检测框（用于输出）

        Returns:
            预测的 BoundingBox
        """
        x1, y1, x2, y2 = self.kf.get_bbox()
        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=self.last_bbox.confidence,
        )

    @property
    def is_confirmed(self) -> bool:
        """轨迹是否已确认（命中次数足够）"""
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        """轨迹是否丢失"""
        return self.time_since_update > 0


class SingleCameraTracker:
    """
    单摄多目标跟踪器

    实现简化的 ByteTrack 风格跟踪：IoU 匹配 + 卡尔曼滤波预测。
    为每个目标分配稳定的 track_id。

    使用方式:
        tracker = SingleCameraTracker(max_age=30, iou_threshold=0.3)
        tracks = tracker.update(detections, frame_id)
        # tracks: List[Tuple[int, str, BoundingBox]]  (track_id, 类别, 检测框)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        初始化多目标跟踪器

        Args:
            max_age: 最大丢失帧数（超过则终止轨迹）
            min_hits: 最小命中帧数（少于此数则不输出）
            iou_threshold: IOU 匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # 活跃的跟踪轨迹列表
        self._tracks: List[_Track] = []
        self._frame_id = 0

        logger.info(
            f"单摄多目标跟踪器初始化: max_age={max_age}, "
            f"min_hits={min_hits}, iou_threshold={iou_threshold}"
        )

    def update(
        self,
        detections: List[Tuple[str, BoundingBox]],
        frame_id: int,
    ) -> List[Tuple[int, str, BoundingBox]]:
        """
        更新跟踪状态

        Args:
            detections: 当前帧检测结果 [(类别, 检测框), ...]
            frame_id: 当前帧编号

        Returns:
            跟踪结果 [(track_id, 类别, 检测框), ...]
        """
        self._frame_id = frame_id

        # 1. 对所有活跃轨迹进行卡尔曼预测
        predicted_states = []
        for track in self._tracks:
            pred = track.predict()
            predicted_states.append(pred)
            track.age += 1
            track.time_since_update += 1

        # 2. IoU 匹配（仅当有活跃轨迹和有检测框时）
        matched_trk: List[int] = []
        matched_det: List[int] = []
        unmatched_det: List[int] = list(range(len(detections)))

        if len(detections) > 0 and len(self._tracks) > 0:
            # 检测框坐标
            det_boxes = np.array([
                [d[1].x1, d[1].y1, d[1].x2, d[1].y2] for d in detections
            ], dtype=np.float64)

            # 使用预测状态构造轨迹预测框 [x1, y1, x2, y2]
            trk_boxes = np.array([
                self._pred_to_xyxy(pred) for pred in predicted_states
            ], dtype=np.float64)

            iou_matrix = _compute_iou_matrix(trk_boxes, det_boxes)

            # 贪心匹配
            matched_trk, matched_det, _, unmatched_det = \
                self._greedy_match(iou_matrix)

            # 更新匹配的轨迹
            for trk_idx, det_idx in zip(matched_trk, matched_det):
                self._tracks[trk_idx].update(detections[det_idx][1], frame_id)

        # 3. 为未匹配的检测创建新轨迹
        for det_idx in unmatched_det:
            target_type, bbox = detections[det_idx]
            new_track = _Track(bbox, target_type, frame_id)
            self._tracks.append(new_track)

        # 7. 移除丢失时间过长的轨迹
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
        ]

        # 8. 输出已确认的轨迹
        results: List[Tuple[int, str, BoundingBox]] = []
        for track in self._tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                pred_bbox = track.get_predicted_bbox()
                results.append((track.track_id, track.target_type, pred_bbox))

        return results

    @staticmethod
    def _pred_to_xyxy(pred: np.ndarray) -> Tuple[float, float, float, float]:
        """将预测状态 [cx, cy, w, h] 转为 [x1, y1, x2, y2]"""
        cx, cy, w, h = pred[0], pred[1], max(pred[2], 1.0), max(pred[3], 1.0)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _greedy_match(
        self,
        iou_matrix: np.ndarray,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        贪心匹配算法

        按 IoU 从高到低贪心匹配轨迹和检测框。

        Args:
            iou_matrix: IoU 矩阵 (num_tracks, num_dets)

        Returns:
            (matched_trk, matched_det, unmatched_trk, unmatched_det)
        """
        n_trk, n_det = iou_matrix.shape
        matched_trk: List[int] = []
        matched_det: List[int] = []
        unmatched_trk: List[int] = list(range(n_trk))
        unmatched_det: List[int] = list(range(n_det))

        # 贪心匹配：每次选 IoU 最大的配对
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            trk_idx, det_idx = int(max_idx[0]), int(max_idx[1])

            matched_trk.append(trk_idx)
            matched_det.append(det_idx)
            unmatched_trk.remove(trk_idx)
            unmatched_det.remove(det_idx)

            # 将该轨迹和检测框从矩阵中移除（设为 -1）
            iou_matrix[trk_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        return matched_trk, matched_det, unmatched_trk, unmatched_det

    def reset(self) -> None:
        """重置跟踪器状态（切换视频/摄像头时调用）"""
        self._tracks.clear()
        self._frame_id = 0
        _Track._next_id = 0
        logger.info("跟踪器状态已重置")
