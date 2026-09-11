"""
tests/test_tracking.py - 跟踪功能测试

测试目标: 验证单摄跟踪与 Tracklet 生成功能（不需要模型权重）
覆盖:
  - TrackletGenerator 帧输入和 tracklet 生成
  - TrackManager 注册/查询/序列化
  - TrackletGenerator 方向计算
  - TrackletGenerator 属性聚合
"""

import pytest
import json
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.common.data_models import BoundingBox, TargetInstance, Tracklet
from src.tracking.tracklet import TrackletGenerator
from src.tracking.track_manager import TrackManager


# ============================================================
# Fixtures
# ============================================================

def _make_bbox(x1=0, y1=0, x2=100, y2=100, conf=0.9):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


@pytest.fixture
def tracklet_gen():
    return TrackletGenerator(camera_id="c001", min_instances=3, scene_id="S01")


@pytest.fixture
def track_manager():
    return TrackManager()


def _make_tracklet(tid="TRK_001", camera_id="c001", target_type="vehicle",
                    start_offset=0, duration=10, plate=None):
    """辅助创建 Tracklet"""
    start = datetime(2024, 1, 1, 8, 30, 0) + timedelta(seconds=start_offset)
    end = start + timedelta(seconds=duration)
    return Tracklet(
        tracklet_id=tid,
        camera_id=camera_id,
        target_type=target_type,
        start_time=start,
        end_time=end,
        instances=[],
        direction="由西向东",
        plate_number=plate,
        attributes={"color": "白色"},
        avg_reid_vector=None,
        avg_clip_vector=None,
        keyframe_paths=[],
    )


# ============================================================
# TrackletGenerator 测试
# ============================================================

class TestTrackletGenerator:
    """TrackletGenerator 测试"""

    def test_init(self, tracklet_gen):
        """测试初始化"""
        assert tracklet_gen.camera_id == "c001"
        assert tracklet_gen.scene_id == "S01"
        assert tracklet_gen.min_instances == 3

    def test_add_single_frame(self, tracklet_gen):
        """测试添加单帧"""
        ts = datetime(2024, 1, 1, 8, 30, 0)
        tracklet_gen.add_frame(
            track_results=[(1, "vehicle", _make_bbox())],
            timestamp=ts,
            frame_id=0,
        )
        # 单帧不应产生完成的 tracklet
        completed = tracklet_gen.get_completed_tracklets()
        assert len(completed) == 0

    def test_tracklet_generated_on_track_termination(self, tracklet_gen):
        """测试跟踪终止时生成 tracklet"""
        base_time = datetime(2024, 1, 1, 8, 30, 0)
        # 添加 5 帧，track_id=1 持续出现
        for i in range(5):
            tracklet_gen.add_frame(
                track_results=[(1, "vehicle", _make_bbox(x1=i * 10))],
                timestamp=base_time + timedelta(seconds=i),
                frame_id=i,
            )
        # track_id=1 仍在活跃，不应有完成的 tracklet
        completed = tracklet_gen.get_completed_tracklets()
        assert len(completed) == 0

        # 下一帧 track_id=1 不出现 → 终止
        tracklet_gen.add_frame(
            track_results=[],
            timestamp=base_time + timedelta(seconds=5),
            frame_id=5,
        )
        completed = tracklet_gen.get_completed_tracklets()
        assert len(completed) == 1
        assert completed[0].camera_id == "c001"
        assert completed[0].target_type == "vehicle"
        assert completed[0].instance_count == 5

    def test_tracklet_min_instances_filter(self):
        """测试最少实例数过滤"""
        gen = TrackletGenerator(camera_id="c001", min_instances=5)
        base_time = datetime(2024, 1, 1, 8, 30, 0)
        # 只添加 3 帧
        for i in range(3):
            gen.add_frame(
                track_results=[(1, "vehicle", _make_bbox())],
                timestamp=base_time + timedelta(seconds=i),
                frame_id=i,
            )
        # 终止
        gen.add_frame(
            track_results=[],
            timestamp=base_time + timedelta(seconds=3),
            frame_id=3,
        )
        completed = gen.get_completed_tracklets()
        # 实例数不足 min_instances=5，不应生成 tracklet
        assert len(completed) == 0

    def test_multiple_tracks(self, tracklet_gen):
        """测试多个同时跟踪目标"""
        base_time = datetime(2024, 1, 1, 8, 30, 0)
        for i in range(5):
            tracklet_gen.add_frame(
                track_results=[
                    (1, "vehicle", _make_bbox(x1=i * 10)),
                    (2, "pedestrian", _make_bbox(x1=200 + i * 5, y1=300)),
                ],
                timestamp=base_time + timedelta(seconds=i),
                frame_id=i,
            )
        # 终止两个 track
        tracklet_gen.add_frame(
            track_results=[],
            timestamp=base_time + timedelta(seconds=5),
            frame_id=5,
        )
        completed = tracklet_gen.get_completed_tracklets()
        assert len(completed) == 2
        types = {c.target_type for c in completed}
        assert "vehicle" in types
        assert "pedestrian" in types

    def test_flush(self, tracklet_gen):
        """测试强制输出"""
        base_time = datetime(2024, 1, 1, 8, 30, 0)
        for i in range(5):
            tracklet_gen.add_frame(
                track_results=[(1, "vehicle", _make_bbox())],
                timestamp=base_time + timedelta(seconds=i),
                frame_id=i,
            )
        flushed = tracklet_gen.flush()
        assert len(flushed) == 1
        # flush 后活跃轨迹应为空
        assert len(tracklet_gen._active_tracks) == 0

    def test_direction_computation(self, tracklet_gen):
        """测试方向计算"""
        # 由西向东（x 增加）
        direction = tracklet_gen._compute_direction((0, 50), (100, 50))
        assert "东" in direction

        # 由北向南（y 增加，图像坐标系 y 向下）
        direction = tracklet_gen._compute_direction((50, 0), (50, 100))
        assert "南" in direction

    def test_tracklet_duration(self, tracklet_gen):
        """测试 tracklet 持续时间"""
        base_time = datetime(2024, 1, 1, 8, 30, 0)
        for i in range(10):
            tracklet_gen.add_frame(
                track_results=[(1, "vehicle", _make_bbox())],
                timestamp=base_time + timedelta(seconds=i * 2),
                frame_id=i * 10,
            )
        tracklet_gen.add_frame(
            track_results=[],
            timestamp=base_time + timedelta(seconds=20),
            frame_id=100,
        )
        completed = tracklet_gen.get_completed_tracklets()
        assert len(completed) == 1
        assert completed[0].duration_seconds == 18.0  # 0 到 18 秒


# ============================================================
# TrackManager 测试
# ============================================================

class TestTrackManager:
    """TrackManager 轨迹管理器测试"""

    def test_init(self, track_manager):
        """测试初始化"""
        assert track_manager.tracklet_count == 0

    def test_register_tracklet(self, track_manager):
        """测试注册 Tracklet"""
        trk = _make_tracklet()
        track_manager.register_tracklet(trk)
        assert track_manager.tracklet_count == 1

    def test_get_tracklet(self, track_manager):
        """测试获取 Tracklet"""
        trk = _make_tracklet(tid="TRK_GET")
        track_manager.register_tracklet(trk)
        result = track_manager.get_tracklet("TRK_GET")
        assert result is not None
        assert result.tracklet_id == "TRK_GET"

    def test_get_nonexistent_tracklet(self, track_manager):
        """测试获取不存在的 Tracklet"""
        result = track_manager.get_tracklet("NONEXISTENT")
        assert result is None

    def test_get_tracklets_by_camera(self, track_manager):
        """测试按摄像头查询"""
        track_manager.register_tracklet(_make_tracklet("TRK_A", camera_id="c001"))
        track_manager.register_tracklet(_make_tracklet("TRK_B", camera_id="c001"))
        track_manager.register_tracklet(_make_tracklet("TRK_C", camera_id="c002"))

        c001_tracklets = track_manager.get_tracklets_by_camera("c001")
        assert len(c001_tracklets) == 2

        c002_tracklets = track_manager.get_tracklets_by_camera("c002")
        assert len(c002_tracklets) == 1

    def test_get_tracklets_by_type(self, track_manager):
        """测试按目标类型查询"""
        track_manager.register_tracklet(_make_tracklet("TRK_V1", target_type="vehicle"))
        track_manager.register_tracklet(_make_tracklet("TRK_V2", target_type="vehicle"))
        track_manager.register_tracklet(_make_tracklet("TRK_P1", target_type="pedestrian"))

        vehicles = track_manager.get_tracklets_by_type("vehicle")
        assert len(vehicles) == 2

        pedestrians = track_manager.get_tracklets_by_type("pedestrian")
        assert len(pedestrians) == 1

    def test_get_tracklets_by_plate(self, track_manager):
        """测试按车牌查询"""
        track_manager.register_tracklet(_make_tracklet("TRK_PL1", plate="苏E12345"))
        track_manager.register_tracklet(_make_tracklet("TRK_PL2", plate="苏E67890"))

        results = track_manager.get_tracklets_by_plate("苏E12345")
        assert len(results) == 1
        assert results[0].plate_number == "苏E12345"

    def test_remove_tracklet(self, track_manager):
        """测试删除 Tracklet"""
        track_manager.register_tracklet(_make_tracklet("TRK_DEL"))
        assert track_manager.tracklet_count == 1

        success = track_manager.remove_tracklet("TRK_DEL")
        assert success is True
        assert track_manager.tracklet_count == 0

    def test_remove_nonexistent(self, track_manager):
        """测试删除不存在的 Tracklet"""
        success = track_manager.remove_tracklet("NONEXISTENT")
        assert success is False

    def test_camera_ids(self, track_manager):
        """测试获取有 Tracklet 的摄像头 ID"""
        track_manager.register_tracklet(_make_tracklet("TRK_1", camera_id="c001"))
        track_manager.register_tracklet(_make_tracklet("TRK_2", camera_id="c002"))
        ids = track_manager.camera_ids
        assert "c001" in ids
        assert "c002" in ids

    def test_save_and_load_json(self, track_manager):
        """测试 JSON 序列化与反序列化"""
        track_manager.register_tracklet(_make_tracklet("TRK_S1", plate="苏E11111"))
        track_manager.register_tracklet(_make_tracklet("TRK_S2"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name

        try:
            track_manager.save_to_json(filepath)

            # 验证文件存在且可解析
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["tracklet_count"] == 2
            assert len(data["tracklets"]) == 2

            # 加载到新 manager
            new_manager = TrackManager()
            loaded = new_manager.load_from_json(filepath)
            assert loaded == 2
            assert new_manager.tracklet_count == 2
            assert new_manager.get_tracklet("TRK_S1") is not None
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_tracklets_by_time_range(self, track_manager):
        """测试按时间范围查询"""
        track_manager.register_tracklet(_make_tracklet("TRK_T1", start_offset=0, duration=10))
        track_manager.register_tracklet(_make_tracklet("TRK_T2", start_offset=30, duration=10))

        base = datetime(2024, 1, 1, 8, 30, 0)
        # 查询 0-15 秒范围
        results = track_manager.get_tracklets_by_time_range(
            start_time=base,
            end_time=base + timedelta(seconds=15),
        )
        assert len(results) == 1
        assert results[0].tracklet_id == "TRK_T1"
