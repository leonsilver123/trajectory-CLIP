"""
tests/test_data_models.py - 数据模型测试

测试目标: 验证 src/common/data_models.py 中所有核心数据结构的正确性
覆盖:
  - BoundingBox: 创建、width/height/area/center 属性计算
  - BoundingBox.iou(): 测试 IoU 计算（未实现则标记 xfail）
  - CameraMetadata: 创建、可选字段、distance_to() 方法
  - TargetInstance: 创建、has_plate/has_reid/has_clip 属性
  - Tracklet: 创建、duration_seconds/instance_count 属性
  - CrossCameraEdge: 创建、评分字段
  - TrajectoryResult: 创建、camera_sequence/time_sequence 属性
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.common.data_models import (
    BoundingBox,
    CameraMetadata,
    TargetInstance,
    Tracklet,
    CrossCameraEdge,
    TrajectoryResult,
    ObservationNode,
    ObservationSegment,
    InferenceSegment,
    CandidatePath,
    ParsedQuery,
    RetrievalCandidate,
    ObservationChain,
    TargetType,
    RoadSegment,
)


# ============================================================
# BoundingBox 测试
# ============================================================

class TestBoundingBox:
    """BoundingBox 数据模型测试"""

    def test_create_basic(self):
        """测试基本创建"""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120, confidence=0.9)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 120
        assert bbox.confidence == 0.9

    def test_width(self):
        """测试宽度计算"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        assert bbox.width == 100.0

    def test_height(self):
        """测试高度计算"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        assert bbox.height == 50.0

    def test_area(self):
        """测试面积计算"""
        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=80, confidence=0.8)
        assert bbox.width == 50
        assert bbox.height == 60
        assert bbox.area == 3000.0

    def test_center(self):
        """测试中心点计算"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.8)
        cx, cy = bbox.center
        assert cx == 50.0
        assert cy == 50.0

    def test_center_nonzero_origin(self):
        """测试非零原点的中心点"""
        bbox = BoundingBox(x1=200, y1=300, x2=400, y2=500, confidence=0.5)
        cx, cy = bbox.center
        assert cx == 300.0
        assert cy == 400.0

    def test_zero_area_bbox(self):
        """测试零面积 bbox（退化情况）"""
        bbox = BoundingBox(x1=50, y1=50, x2=50, y2=50, confidence=0.5)
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0

    @pytest.mark.xfail(reason="BoundingBox.iou() 尚未实现，抛出 NotImplementedError")
    def test_iou_overlap(self):
        """测试 IoU 计算 - 有重叠"""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150, confidence=0.8)
        iou = bbox1.iou(bbox2)
        # 交集: 50*50=2500, 并集: 10000+10000-2500=17500
        expected = 2500.0 / 17500.0
        assert abs(iou - expected) < 1e-6

    @pytest.mark.xfail(reason="BoundingBox.iou() 尚未实现，抛出 NotImplementedError")
    def test_iou_no_overlap(self):
        """测试 IoU 计算 - 无重叠"""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        bbox2 = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
        iou = bbox1.iou(bbox2)
        assert iou == 0.0

    @pytest.mark.xfail(reason="BoundingBox.iou() 尚未实现，抛出 NotImplementedError")
    def test_iou_identical(self):
        """测试 IoU 计算 - 完全相同"""
        bbox1 = BoundingBox(x1=10, y1=10, x2=90, y2=90, confidence=0.9)
        bbox2 = BoundingBox(x1=10, y1=10, x2=90, y2=90, confidence=0.8)
        iou = bbox1.iou(bbox2)
        assert abs(iou - 1.0) < 1e-6

    def test_iou_computed_successfully(self):
        """验证 iou() 已实现并可正常计算"""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150, confidence=0.8)
        iou = bbox1.iou(bbox2)
        # 不再抛出 NotImplementedError，返回有效 IoU 值
        assert 0.0 <= iou <= 1.0

    def test_confidence_range(self):
        """测试置信度值可以是边界值"""
        bbox_low = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.0)
        bbox_high = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=1.0)
        assert bbox_low.confidence == 0.0
        assert bbox_high.confidence == 1.0


# ============================================================
# CameraMetadata 测试
# ============================================================

class TestCameraMetadata:
    """CameraMetadata 数据模型测试"""

    def test_create_basic(self):
        """测试基本创建"""
        cam = CameraMetadata(
            camera_id="c001",
            name="Scene01-Camera01",
            latitude=42.526,
            longitude=-90.7236,
        )
        assert cam.camera_id == "c001"
        assert cam.name == "Scene01-Camera01"
        assert cam.latitude == 42.526
        assert cam.longitude == -90.7236

    def test_optional_fields_default(self):
        """测试可选字段默认值"""
        cam = CameraMetadata(camera_id="c002", name="TestCam")
        assert cam.scene_id is None
        assert cam.latitude is None
        assert cam.longitude is None
        assert cam.direction == 0.0
        assert cam.covered_road_segment == ""
        assert cam.lane_direction == "eastbound"

    def test_with_scene_id(self):
        """测试带场景 ID"""
        cam = CameraMetadata(
            camera_id="c001", name="Cam1", scene_id="S01"
        )
        assert cam.scene_id == "S01"

    def test_distance_to(self):
        """测试 distance_to 方法（Haversine 距离）"""
        cam1 = CameraMetadata(
            camera_id="c001", name="Cam1",
            latitude=42.526, longitude=-90.7236,
        )
        cam2 = CameraMetadata(
            camera_id="c002", name="Cam2",
            latitude=42.5258, longitude=-90.7232,
        )
        dist = cam1.distance_to(cam2)
        # 距离应该很小（约几十米）
        assert 0 < dist < 1000

    def test_distance_to_missing_gps_self(self):
        """测试缺少 GPS 坐标时抛出异常（自身）"""
        cam1 = CameraMetadata(camera_id="c001", name="Cam1")
        cam2 = CameraMetadata(
            camera_id="c002", name="Cam2",
            latitude=42.526, longitude=-90.7236,
        )
        with pytest.raises(ValueError, match="缺少 GPS 坐标"):
            cam1.distance_to(cam2)

    def test_distance_to_missing_gps_other(self):
        """测试缺少 GPS 坐标时抛出异常（对方）"""
        cam1 = CameraMetadata(
            camera_id="c001", name="Cam1",
            latitude=42.526, longitude=-90.7236,
        )
        cam2 = CameraMetadata(camera_id="c002", name="Cam2")
        with pytest.raises(ValueError, match="缺少 GPS 坐标"):
            cam1.distance_to(cam2)

    def test_is_opposite_direction(self):
        """测试方向相反判断"""
        cam_north = CameraMetadata(
            camera_id="c001", name="Cam1", lane_direction="northbound"
        )
        cam_south = CameraMetadata(
            camera_id="c002", name="Cam2", lane_direction="southbound"
        )
        cam_east = CameraMetadata(
            camera_id="c003", name="Cam3", lane_direction="eastbound"
        )
        assert cam_north.is_opposite_direction(cam_south) is True
        assert cam_north.is_opposite_direction(cam_east) is False


# ============================================================
# TargetInstance 测试
# ============================================================

class TestTargetInstance:
    """TargetInstance 数据模型测试"""

    def _make_instance(self, **kwargs):
        """辅助创建 TargetInstance"""
        defaults = dict(
            instance_id="INST_001",
            camera_id="c001",
            timestamp=datetime.now(),
            frame_id=100,
            target_type="vehicle",
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9),
            attributes={"color": "白色"},
            plate_number=None,
            plate_confidence=0.0,
            quality_score=0.8,
            reid_vector=None,
            clip_vector=None,
            keyframe_path=None,
        )
        defaults.update(kwargs)
        return TargetInstance(**defaults)

    def test_create_basic(self):
        """测试基本创建"""
        inst = self._make_instance()
        assert inst.instance_id == "INST_001"
        assert inst.camera_id == "c001"
        assert inst.target_type == "vehicle"

    def test_auto_instance_id(self):
        """测试自动生成 instance_id"""
        inst = self._make_instance(instance_id="")
        assert inst.instance_id.startswith("INST_")
        assert len(inst.instance_id) > 5

    def test_has_plate_true(self):
        """测试有车牌"""
        inst = self._make_instance(plate_number="苏E12345")
        assert inst.has_plate is True

    def test_has_plate_false_none(self):
        """测试无车牌 (None)"""
        inst = self._make_instance(plate_number=None)
        assert inst.has_plate is False

    def test_has_plate_false_empty(self):
        """测试无车牌 (空字符串)"""
        inst = self._make_instance(plate_number="  ")
        assert inst.has_plate is False

    def test_has_reid_true(self):
        """测试有 ReID 特征"""
        inst = self._make_instance(reid_vector=np.random.randn(512).astype(np.float32))
        assert inst.has_reid is True

    def test_has_reid_false(self):
        """测试无 ReID 特征"""
        inst = self._make_instance(reid_vector=None)
        assert inst.has_reid is False

    def test_has_clip_true(self):
        """测试有 CLIP 特征"""
        inst = self._make_instance(clip_vector=np.random.randn(768).astype(np.float32))
        assert inst.has_clip is True

    def test_has_clip_false(self):
        """测试无 CLIP 特征"""
        inst = self._make_instance(clip_vector=None)
        assert inst.has_clip is False


# ============================================================
# Tracklet 测试
# ============================================================

class TestTracklet:
    """Tracklet 数据模型测试"""

    def _make_tracklet(self, duration_sec=10.0, num_instances=5):
        """辅助创建 Tracklet"""
        start = datetime(2024, 1, 1, 8, 30, 0)
        end = start + timedelta(seconds=duration_sec)
        instances = []
        for i in range(num_instances):
            inst = TargetInstance(
                instance_id=f"INST_{i:03d}",
                camera_id="c001",
                timestamp=start + timedelta(seconds=i * duration_sec / max(num_instances - 1, 1)),
                frame_id=i * 10,
                target_type="vehicle",
                bbox=BoundingBox(x1=i * 10, y1=50, x2=i * 10 + 80, y2=150, confidence=0.85),
                attributes={},
                plate_number=None,
                plate_confidence=0.0,
                quality_score=0.8,
                reid_vector=None,
                clip_vector=None,
                keyframe_path=None,
            )
            instances.append(inst)
        return Tracklet(
            tracklet_id="TRK_001",
            camera_id="c001",
            target_type="vehicle",
            start_time=start,
            end_time=end,
            instances=instances,
            direction="由西向东",
            plate_number=None,
            attributes={},
            avg_reid_vector=None,
            avg_clip_vector=None,
            keyframe_paths=[],
        )

    def test_create_basic(self):
        """测试基本创建"""
        trk = self._make_tracklet()
        assert trk.tracklet_id == "TRK_001"
        assert trk.camera_id == "c001"

    def test_duration_seconds(self):
        """测试持续时间计算"""
        trk = self._make_tracklet(duration_sec=30.0)
        assert trk.duration_seconds == 30.0

    def test_instance_count(self):
        """测试实例数量"""
        trk = self._make_tracklet(num_instances=7)
        assert trk.instance_count == 7

    def test_has_plate_true(self):
        """测试有车牌"""
        trk = self._make_tracklet()
        trk.plate_number = "苏E12345"
        assert trk.has_plate is True

    def test_has_plate_false(self):
        """测试无车牌"""
        trk = self._make_tracklet()
        assert trk.has_plate is False


# ============================================================
# CrossCameraEdge 测试
# ============================================================

class TestCrossCameraEdge:
    """CrossCameraEdge 数据模型测试"""

    def test_create_basic(self):
        """测试基本创建"""
        edge = CrossCameraEdge(
            source_tracklet_id="TRK_001",
            target_tracklet_id="TRK_002",
            score=0.85,
            appearance_score=0.9,
            attribute_score=0.8,
            plate_score=1.0,
            temporal_score=0.7,
            spatial_score=0.8,
            direction_score=0.9,
            penalty=0.05,
            is_valid=True,
            reasoning="外观高度相似，时间可达",
        )
        assert edge.score == 0.85
        assert edge.is_valid is True
        assert edge.source_tracklet_id == "TRK_001"
        assert edge.target_tracklet_id == "TRK_002"

    def test_score_range(self):
        """测试评分字段范围"""
        edge = CrossCameraEdge(
            source_tracklet_id="TRK_A",
            target_tracklet_id="TRK_B",
            score=0.0,
            appearance_score=0.0,
            attribute_score=0.0,
            plate_score=0.0,
            temporal_score=0.0,
            spatial_score=0.0,
            direction_score=0.0,
            penalty=0.0,
            is_valid=False,
            reasoning="低分连接",
        )
        assert edge.score == 0.0
        assert edge.is_valid is False


# ============================================================
# TrajectoryResult 测试
# ============================================================

class TestTrajectoryResult:
    """TrajectoryResult 数据模型测试"""

    def _make_trajectory_result(self):
        """辅助创建 TrajectoryResult"""
        target = TargetInstance(
            instance_id="INST_ANCHOR",
            camera_id="c001",
            timestamp=datetime(2024, 1, 1, 8, 30, 0),
            frame_id=100,
            target_type="vehicle",
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9),
            attributes={},
            plate_number=None,
            plate_confidence=0.0,
            quality_score=0.8,
            reid_vector=None,
            clip_vector=None,
            keyframe_path=None,
        )
        nodes = [
            ObservationNode(
                camera_id=f"c{i:03d}",
                tracklet_id=f"TRK_{i:03d}",
                timestamp=datetime(2024, 1, 1, 8, 30 + i * 5, 0),
                keyframe_path=f"/tmp/keyframe_c{i:03d}.jpg",
                confidence=0.9,
            )
            for i in range(1, 4)
        ]
        return TrajectoryResult(
            query_id="Q_001",
            target_instance=target,
            observation_nodes=nodes,
            observation_segments=[],
            inference_segments=[],
            candidate_paths=[],
            evidence={"plate_consistency": 0.0},
            overall_confidence=0.85,
        )

    def test_create_basic(self):
        """测试基本创建"""
        traj = self._make_trajectory_result()
        assert traj.query_id == "Q_001"
        assert traj.overall_confidence == 0.85

    def test_camera_sequence(self):
        """测试摄像头序列"""
        traj = self._make_trajectory_result()
        seq = traj.camera_sequence
        assert seq == ["c001", "c002", "c003"]

    def test_time_sequence(self):
        """测试时间序列"""
        traj = self._make_trajectory_result()
        times = traj.time_sequence
        assert len(times) == 3
        assert times[0] < times[1] < times[2]


# ============================================================
# 枚举类型测试
# ============================================================

class TestEnums:
    """枚举类型测试"""

    def test_target_type_values(self):
        """测试目标类别枚举值"""
        assert TargetType.VEHICLE.value == "vehicle"
        assert TargetType.PEDESTRIAN.value == "pedestrian"
        assert TargetType.NON_MOTOR_VEHICLE.value == "non_motor_vehicle"

    def test_target_type_is_str(self):
        """测试枚举是字符串类型"""
        assert isinstance(TargetType.VEHICLE, str)


# ============================================================
# ParsedQuery 测试
# ============================================================

class TestParsedQuery:
    """ParsedQuery 数据模型测试"""

    def test_create_basic(self):
        """测试基本创建"""
        pq = ParsedQuery(
            raw_text="黑色轿车",
            target_type="vehicle",
            attributes={"color": "黑色", "vehicle_type": "轿车"},
            plate_number=None,
            clip_text_embedding=None,
        )
        assert pq.raw_text == "黑色轿车"
        assert pq.target_type == "vehicle"
        assert pq.plate_number is None

    def test_with_plate(self):
        """测试带车牌号"""
        pq = ParsedQuery(
            raw_text="苏E12345",
            target_type="vehicle",
            attributes={},
            plate_number="苏E12345",
            clip_text_embedding=None,
        )
        assert pq.plate_number == "苏E12345"


# ============================================================
# RoadSegment 测试
# ============================================================

class TestRoadSegment:
    """RoadSegment 数据模型测试"""

    def test_create_basic(self):
        """测试基本创建"""
        seg = RoadSegment(
            segment_id="SEG_001",
            name="Test Segment",
            start_camera_ids=["c001"],
            end_camera_ids=["c002"],
            distance_meters=100.0,
            direction="east-west",
        )
        assert seg.segment_id == "SEG_001"
        assert seg.distance_meters == 100.0
