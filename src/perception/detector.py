"""
src.perception.detector - 目标检测模块

基于 YOLO (ultralytics) 实现交通场景目标检测，支持:
- 机动车、行人、非机动车检测
- 批量帧检测
- 检测结果后处理(NMS、置信度过滤)

输入: 视频帧 (numpy array)
输出: BoundingBox 列表 + 类别标签
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.data_models import BoundingBox
from src.common.logger import get_logger

logger = get_logger("perception.detector")


class VehicleDetector:
    """
    目标检测器

    使用 YOLO 模型检测视频帧中的交通目标（机动车、行人、非机动车）。
    虽然类名为 VehicleDetector，实际检测所有配置的目标类型。

    使用方式:
        detector = VehicleDetector(model_name="yolov8x", device="cuda")
        detections = detector.detect(frame)
        # detections: List[Tuple[str, BoundingBox]]  (类别, 检测框)
    """

    # YOLO COCO 类别 ID 到系统类别的映射
    CLASS_MAPPING: Dict[int, str] = {
        0: "pedestrian",          # person
        1: "non_motor_vehicle",   # bicycle
        2: "vehicle",             # car
        3: "non_motor_vehicle",   # motorcycle
        5: "vehicle",             # bus
        7: "vehicle",             # truck
    }

    def __init__(
        self,
        model_name: str = "yolov8x",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cuda",
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        """
        初始化目标检测器

        Args:
            model_name: YOLO 模型名称
            confidence_threshold: 置信度阈值
            nms_threshold: NMS 阈值
            device: 推理设备 ("cuda" 或 "cpu")
            input_size: 输入尺寸 (宽, 高)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.input_size = input_size
        self._model = None

        logger.info(f"目标检测器初始化: model={model_name}, device={device}")

    def _lazy_init(self) -> None:
        """
        延迟初始化 YOLO 模型

        首次调用检测时加载模型，避免不必要的内存占用。
        """
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"{self.model_name}.pt")
            # 将模型移到指定设备
            if self.device == "cuda":
                self._model.to("cuda")
            logger.info(f"YOLO 模型加载完成: {self.model_name}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Tuple[str, BoundingBox]]:
        """
        检测单帧图像中的目标

        Args:
            frame: 输入图像 (H, W, 3) BGR 格式

        Returns:
            检测结果列表，每个元素为 (目标类别, 检测框)
        """
        self._lazy_init()

        results = []
        try:
            # ultralytics 推理
            yolo_results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                imgsz=list(self.input_size),
                device=self.device,
                verbose=False,
            )

            for result in yolo_results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    # 获取 COCO 类别 ID
                    cls_id = int(boxes.cls[i].item())
                    # 映射到系统类别
                    target_type = self._map_class(cls_id)
                    if target_type is None:
                        continue

                    # 获取检测框坐标和置信度
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].item())

                    bbox = BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                    )
                    results.append((target_type, bbox))

        except Exception as e:
            logger.error(f"目标检测推理失败: {e}")

        return results

    def detect_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[Tuple[str, BoundingBox]]]:
        """
        批量检测多帧图像

        Args:
            frames: 图像列表

        Returns:
            每帧的检测结果列表
        """
        self._lazy_init()

        all_results: List[List[Tuple[str, BoundingBox]]] = []
        try:
            # ultralytics 支持批量推理
            yolo_results = self._model.predict(
                source=frames,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                imgsz=list(self.input_size),
                device=self.device,
                verbose=False,
            )

            for result in yolo_results:
                frame_results = []
                boxes = result.boxes
                if boxes is None:
                    all_results.append(frame_results)
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    target_type = self._map_class(cls_id)
                    if target_type is None:
                        continue

                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].item())

                    bbox = BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                    )
                    frame_results.append((target_type, bbox))

                all_results.append(frame_results)

        except Exception as e:
            logger.error(f"批量检测推理失败: {e}")
            # 回退为逐帧检测
            all_results = [self.detect(frame) for frame in frames]

        return all_results

    def _map_class(self, class_id: int) -> Optional[str]:
        """
        将 YOLO/COCO 类别 ID 映射为系统目标类别

        Args:
            class_id: COCO 类别 ID

        Returns:
            系统类别名称，不在映射中的类别返回 None
        """
        return self.CLASS_MAPPING.get(class_id)
