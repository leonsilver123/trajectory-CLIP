"""
tests.test_detector - 目标检测器测试

约定：本文件**不加载 YOLO 权重**。检测的解析逻辑（类别映射、检测框转换、
异常吞噬、批量的逐帧回退）用一个桩模型注入 `detector._model` 来测，
因此任何机器上都能跑，不受模型文件与显存影响。
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.perception.detector import VehicleDetector


# ============================================================
# 桩：模拟 ultralytics 的 Results / Boxes 接口
# 只实现 detector.py 实际用到的三个访问路径：
#   boxes.cls[i].item()        boxes.conf[i].item()        boxes.xyxy[i].cpu().numpy()
# ============================================================


class _Scalar:
    """模拟 tensor 标量（支持 .item()）"""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _Vec:
    """模拟 tensor 向量（支持 .cpu().numpy()）"""

    def __init__(self, values: List[float]) -> None:
        self._arr = np.asarray(values, dtype=float)

    def cpu(self) -> "_Vec":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    """模拟 ultralytics 的 Boxes"""

    def __init__(self, cls_ids, xyxy_list, confs) -> None:
        self.cls = [_Scalar(c) for c in cls_ids]
        self.xyxy = [_Vec(x) for x in xyxy_list]
        self.conf = [_Scalar(c) for c in confs]

    def __len__(self) -> int:
        return len(self.cls)


class _FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class _FakeModel:
    """桩模型：predict() 返回预置结果，或按 source 类型决定是否抛异常"""

    def __init__(self, results=None, raise_on=None) -> None:
        self._results = results or []
        self._raise_on = raise_on   # "list" / "frame" / None
        self.calls: List[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs.get("source")
        if self._raise_on == "list" and isinstance(source, list):
            raise RuntimeError("模拟批量推理失败")
        if self._raise_on == "frame" and isinstance(source, np.ndarray):
            raise RuntimeError("模拟单帧推理失败")
        if isinstance(source, list):
            return self._results
        return self._results[:1]


@pytest.fixture
def detector() -> VehicleDetector:
    """构造检测器（不触发模型加载）"""
    return VehicleDetector(model_name="yolov8x", device="cpu")


# ============================================================
# 初始化与懒加载契约
# ============================================================


class TestInit:
    def test_init_does_not_load_model(self, detector):
        """构造时不得加载模型 —— 懒加载是本模块的性能契约"""
        assert detector._model is None

    def test_init_stores_params(self):
        """构造参数应原样保存"""
        det = VehicleDetector(
            model_name="yolov8n",
            confidence_threshold=0.7,
            nms_threshold=0.3,
            device="cpu",
            input_size=(320, 320),
        )
        assert det.model_name == "yolov8n"
        assert det.confidence_threshold == 0.7
        assert det.nms_threshold == 0.3
        assert det.input_size == (320, 320)


# ============================================================
# 类别映射
# ============================================================


class TestClassMapping:
    def test_mapping_covers_traffic_classes(self):
        """COCO 的交通相关类别都应映射到系统三类之一"""
        assert VehicleDetector.CLASS_MAPPING[2] == "vehicle"      # car
        assert VehicleDetector.CLASS_MAPPING[5] == "vehicle"      # bus
        assert VehicleDetector.CLASS_MAPPING[7] == "vehicle"      # truck
        assert VehicleDetector.CLASS_MAPPING[0] == "pedestrian"   # person
        assert VehicleDetector.CLASS_MAPPING[1] == "non_motor_vehicle"   # bicycle
        assert VehicleDetector.CLASS_MAPPING[3] == "non_motor_vehicle"   # motorcycle

    def test_mapping_values_are_known_system_types(self):
        """映射的目标值只能是系统定义的三类"""
        assert set(VehicleDetector.CLASS_MAPPING.values()) == {
            "vehicle", "pedestrian", "non_motor_vehicle"
        }

    @pytest.mark.parametrize("class_id", [2, 5, 7])
    def test_map_class_vehicle(self, detector, class_id):
        assert detector._map_class(class_id) == "vehicle"

    @pytest.mark.parametrize("class_id", [11, 13, 99, -1])
    def test_map_class_unmapped_returns_none(self, detector, class_id):
        """非交通类别（如 stop sign=11）应返回 None，由调用方跳过"""
        assert detector._map_class(class_id) is None


# ============================================================
# 单帧检测的解析逻辑
# ============================================================


class TestDetectParsing:
    def test_detect_maps_classes_and_converts_boxes(self, detector):
        """应把 COCO 类别翻译成系统类别，并把框转成 float 的 BoundingBox"""
        boxes = _FakeBoxes(
            cls_ids=[2, 0],
            xyxy_list=[[10.0, 20.0, 110.0, 220.0], [1.0, 2.0, 51.0, 102.0]],
            confs=[0.9, 0.75],
        )
        detector._model = _FakeModel([_FakeResult(boxes)])

        results = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(results) == 2
        assert results[0][0] == "vehicle"
        assert results[1][0] == "pedestrian"

        bbox = results[0][1]
        assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10.0, 20.0, 110.0, 220.0)
        assert bbox.confidence == pytest.approx(0.9)
        assert results[1][1].confidence == pytest.approx(0.75)

    def test_detect_skips_unmapped_classes(self, detector):
        """不在映射里的类别必须被跳过，不得混进结果"""
        boxes = _FakeBoxes(
            cls_ids=[2, 11, 0],          # car / stop sign / person
            xyxy_list=[[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
            confs=[0.9, 0.9, 0.9],
        )
        detector._model = _FakeModel([_FakeResult(boxes)])

        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(results) == 2
        assert [r[0] for r in results] == ["vehicle", "pedestrian"]

    def test_detect_returns_empty_when_boxes_is_none(self, detector):
        """boxes 为 None（无检出）时应返回空列表而非报错"""
        detector._model = _FakeModel([_FakeResult(None)])

        assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []

    def test_detect_passes_thresholds_to_model(self, detector):
        """置信度阈值 / NMS 阈值 / 输入尺寸必须真的传给模型"""
        detector._model = _FakeModel([_FakeResult(None)])
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        kwargs = detector._model.calls[0]
        assert kwargs["conf"] == detector.confidence_threshold
        assert kwargs["iou"] == detector.nms_threshold
        assert kwargs["imgsz"] == list(detector.input_size)

    def test_detect_swallows_inference_error(self, detector):
        """推理抛异常时应记日志并返回空列表，不得把异常抛给调用方"""
        detector._model = _FakeModel(raise_on="frame")

        assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []


# ============================================================
# 批量检测与回退
# ============================================================


class TestDetectBatch:
    def test_detect_batch_returns_one_list_per_frame(self, detector):
        """批量检测应为每一帧返回一个结果列表，顺序与输入一致"""
        boxes_a = _FakeBoxes([2], [[0, 0, 10, 10]], [0.9])
        boxes_b = _FakeBoxes([], [], [])
        detector._model = _FakeModel([_FakeResult(boxes_a), _FakeResult(boxes_b)])

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        out = detector.detect_batch(frames)

        assert len(out) == 2
        assert len(out[0]) == 1 and out[0][0][0] == "vehicle"
        assert out[1] == []

    def test_detect_batch_falls_back_to_per_frame(self, detector):
        """批量推理失败时应回退为逐帧检测，而不是整体返回空"""
        boxes = _FakeBoxes([2], [[0, 0, 10, 10]], [0.9])
        detector._model = _FakeModel([_FakeResult(boxes)], raise_on="list")

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        out = detector.detect_batch(frames)

        assert len(out) == 3
        assert all(len(r) == 1 and r[0][0] == "vehicle" for r in out)

    def test_detect_batch_empty_input(self, detector):
        """空输入应返回空列表且不抛异常

        注意：当前实现仍会把空列表传给 `model.predict`（`_lazy_init` 之后无条件调用），
        本测试只锁定「不炸、返回空」这一行为，不主张「不调用模型」。
        """
        detector._model = _FakeModel([])

        assert detector.detect_batch([]) == []
