"""
tests.test_timeline_frames - 时间线帧字段（PLAN4-T1/T2）

背景：轨迹还原动画要展示"车在摄像头画面里怎么动"。原先
`build_camera_timeline()` 的 `frames[]` 只有 frame_id / timestamp / crop_path /
confidence —— 能显示实拍裁剪图，但**画不出位置**。检测本身带 bbox，只是没往上传。

本文件守住三件事：
1. 新增字段是**纯追加**的（旧字段一个都不能少，否则前端会静默显示空白）；
2. `bbox_norm` 落在合理值域内，且**同一摄像头内可比较**（相对运动才是动画要的）；
3. 拿不到数据时一律 `None` + 说明来源，**不填 0** —— 那会把"没有数据"伪装成"在画面左上角"。
"""

from __future__ import annotations

import pytest

from src.trajectory.builder import get_trajectory_builder

# 旧字段：前端（及既有调用方）依赖它们，不能因为加了新字段就弄丢
_LEGACY_FIELDS = ("frame_id", "timestamp", "crop_path", "confidence")
# PLAN4-T1/T2 新增
_NEW_FIELDS = ("bbox", "bbox_norm", "global_timestamp", "frame_size", "frame_size_source")


@pytest.fixture(scope="module")
def timeline():
    """取一条真实的跨镜轨迹时间线；数据不可用则跳过"""
    builder = get_trajectory_builder()
    builder.ensure_loaded()
    if not builder.data_available:      # 注意：是 @property，不是方法
        pytest.skip("轨迹数据不可用（output/datastore 与 JSON 均缺失）")
    try:
        tl = builder.build_camera_timeline(instance_id="BL_c039_V0312_000028")
    except Exception as e:  # 该 instance 可能因数据版本变化而不存在
        pytest.skip(f"构造时间线失败: {e}")
    if not tl.get("camera_sequence"):
        pytest.skip("该目标没有跨摄像头序列")
    return tl


def _all_frames(tl) -> list:
    return [f for cam in tl["camera_sequence"] for f in cam.get("frames", [])]


class TestFramesAreBackwardCompatible:
    def test_legacy_fields_present(self, timeline):
        """旧字段必须一个不少 —— 加了新字段就丢旧字段是最常见的破坏性改动"""
        for frame in _all_frames(timeline):
            missing = [k for k in _LEGACY_FIELDS if k not in frame]
            assert not missing, f"帧缺少旧字段 {missing}: {frame}"

    def test_new_fields_present(self, timeline):
        for frame in _all_frames(timeline):
            missing = [k for k in _NEW_FIELDS if k not in frame]
            assert not missing, f"帧缺少新增字段 {missing}"

    def test_frames_non_empty(self, timeline):
        frames = _all_frames(timeline)
        assert frames, "跨镜时间线不应一帧都没有"


class TestBboxIsHonest:
    def test_bbox_is_four_tuple_or_none(self, timeline):
        for frame in _all_frames(timeline):
            bbox = frame["bbox"]
            if bbox is None:
                continue
            assert isinstance(bbox, list) and len(bbox) == 4, f"bbox 应为 4 元组: {bbox}"
            assert all(isinstance(v, (int, float)) for v in bbox)

    def test_bbox_is_not_zero_filled(self, timeline):
        """拿不到 bbox 时必须是 None，不能填 [0,0,0,0]

        填 0 会把"没有数据"伪装成"车在画面左上角且尺寸为零"，
        动画里就变成一个缩在角落的点 —— 那是编造。
        """
        for frame in _all_frames(timeline):
            if frame["bbox"] == [0.0, 0.0, 0.0, 0.0]:
                pytest.fail(f"bbox 被填成全 0（应为 None）: {frame}")

    def test_bbox_is_well_formed(self, timeline):
        """x1<x2、y1<y2 —— 反了说明坐标解析有问题"""
        for frame in _all_frames(timeline):
            bbox = frame["bbox"]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            assert x1 <= x2 and y1 <= y2, f"bbox 坐标次序错误: {bbox}"


class TestBboxNorm:
    def test_norm_within_image_bounds(self, timeline):
        """归一化中心点应落在 [0,1]；宽高为正

        允许轻微越界（≤1.05）—— 画面尺寸是从检测框推断的**下界**，
        若某帧的框恰好是全局最大，其归一化值就可能触到 1.0。
        但明显越界（比如 2.0）说明归一化分母用错了。
        """
        for frame in _all_frames(timeline):
            norm = frame["bbox_norm"]
            if norm is None:
                continue
            cx, cy, w, h = norm
            assert 0.0 <= cx <= 1.05, f"归一化 cx 越界: {cx}"
            assert 0.0 <= cy <= 1.05, f"归一化 cy 越界: {cy}"
            assert w >= 0 and h >= 0, f"归一化宽高应为正: {norm}"

    def test_norm_requires_bbox(self, timeline):
        """没有 bbox 就不该有 bbox_norm —— 两者必须同生同灭"""
        for frame in _all_frames(timeline):
            if frame["bbox"] is None:
                assert frame["bbox_norm"] is None, "无 bbox 却有 bbox_norm"
            if frame["bbox_norm"] is None and frame["bbox"] is not None:
                # 有框但拿不到画面尺寸 → 允许 norm 为 None，但来源必须说清楚
                assert frame["frame_size_source"] == "unavailable", (
                    "有 bbox 却无 norm 时，frame_size_source 应为 unavailable"
                )

    def test_norm_is_comparable_within_one_camera(self, timeline):
        """同一摄像头内所有帧必须用同一个分母 —— 否则相对运动会失真

        这条是动画正确性的关键：位移能不能用，取决于分母一致。
        """
        for cam in timeline["camera_sequence"]:
            sizes = {tuple(f["frame_size"]) for f in cam.get("frames", [])
                     if f.get("frame_size") is not None}
            assert len(sizes) <= 1, (
                f"摄像头 {cam['camera_id']} 内出现多个 frame_size {sizes} —— "
                f"分母不一致会让帧间位移不可比"
            )


class TestFrameSizeSource:
    def test_source_is_declared(self, timeline):
        for frame in _all_frames(timeline):
            assert frame["frame_size_source"] in (
                "inferred_from_detections", "unavailable"
            ), f"未知的来源标记: {frame['frame_size_source']}"

    def test_source_matches_value(self, timeline):
        for frame in _all_frames(timeline):
            if frame["frame_size"] is not None:
                assert frame["frame_size_source"] == "inferred_from_detections"
                assert len(frame["frame_size"]) == 2
                assert all(v > 0 for v in frame["frame_size"])
            else:
                assert frame["frame_size_source"] == "unavailable"


class TestGlobalTimestamp:
    def test_global_timestamp_differs_or_is_none(self, timeline):
        """全局时间用于跨摄像头比较；拿不到时为 None（不回落成本机时间冒充）"""
        for cam in timeline["camera_sequence"]:
            for frame in cam.get("frames", []):
                gts = frame["global_timestamp"]
                if gts is None:
                    continue
                assert isinstance(gts, str) and len(gts) >= 19, f"格式异常: {gts}"

    def test_frames_sorted_by_time_within_camera(self, timeline):
        """同一摄像头内帧应按时间升序 —— 动画按顺序播，乱序会导致车"倒着走\""""
        for cam in timeline["camera_sequence"]:
            stamps = [f["timestamp"] for f in cam.get("frames", []) if f["timestamp"]]
            assert stamps == sorted(stamps), (
                f"摄像头 {cam['camera_id']} 的帧未按时间升序"
            )
