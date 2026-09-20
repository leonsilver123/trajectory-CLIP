"""
tests.test_enum_vocabulary - 枚举与真实数据的词表一致性（PLAN3-C4）

背景：`src/common/data_models.py` 定义了 `TargetType` / `LaneDirection` / `RoadDirection`
三个枚举，但**生产代码零引用** —— 线上一律用字符串字面量（如
`builder.py` 的 `target_type == "vehicle"`）。枚举因此沦为摆设。

这里不删枚举，而是给它们一个真实职责：**当作摄像头元数据的词表校验器**。
枚举从此是"合法取值"的唯一定义处，数据写错了会在这里红，而不是在运行时
表现为某个摄像头静默匹配不上。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.common.data_models import LaneDirection, RoadDirection, TargetType

_METADATA = Path(__file__).resolve().parent.parent / "configs" / "cityflow_camera_metadata.yaml"


@pytest.fixture(scope="module")
def metadata() -> dict:
    if not _METADATA.exists():
        pytest.skip(f"缺少摄像头元数据 {_METADATA}")
    with open(_METADATA, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _values(enum_cls) -> set:
    return {m.value for m in enum_cls}


# ============================================================
# 枚举自身的完整性
# ============================================================


class TestEnumVocabulary:
    def test_target_type_covers_system_classes(self):
        """系统三类目标必须齐全 —— 少了哪类，检测结果就无处归类"""
        assert _values(TargetType) == {"vehicle", "pedestrian", "non_motor_vehicle"}

    def test_lane_direction_covers_four_compass_directions(self):
        assert _values(LaneDirection) == {"eastbound", "westbound", "northbound", "southbound"}

    def test_enums_are_str_subclass(self):
        """都是 str 枚举，才能和 YAML/JSON 里的字符串直接比较"""
        for enum_cls in (TargetType, LaneDirection, RoadDirection):
            assert issubclass(enum_cls, str)
            assert isinstance(enum_cls(list(enum_cls)[0]), str)


# ============================================================
# 与真实元数据的一致性（这才是给枚举的真实职责）
# ============================================================


class TestMetadataMatchesEnums:
    def test_camera_lane_direction_is_in_enum(self, metadata):
        """每个摄像头的 lane_direction 必须是 LaneDirection 的合法取值"""
        cameras = metadata.get("cameras") or []
        if not cameras:
            pytest.skip("元数据里没有 cameras 段")

        allowed = _values(LaneDirection)
        seen = set()
        for cam in cameras:
            value = cam.get("lane_direction")
            if value is None:
                continue
            seen.add(value)
            assert value in allowed, (
                f"摄像头 {cam.get('camera_id')} 的 lane_direction={value!r} "
                f"不在 LaneDirection 词表 {sorted(allowed)} 中"
            )
        assert seen, "没有任何摄像头带 lane_direction，校验形同虚设"

    def test_camera_direction_is_numeric_degrees(self, metadata):
        """摄像头的 direction 是**角度值**（全部 46 个都是 float），不是词表字段

        这条测试同时防一个易犯的混淆：YAML 里 `lane_direction: eastbound` 含有
        `direction: eastbound` 这个子串，粗略 grep 会误以为"摄像头的 direction 是字符串"。
        真正的字符串方向只出现在 road_segments 段（见下一个测试）。
        """
        cameras = metadata.get("cameras") or []
        if not cameras:
            pytest.skip("元数据里没有 cameras 段")

        bad = [
            (c.get("camera_id"), c.get("direction"))
            for c in cameras
            if not isinstance(c.get("direction"), (int, float))
        ]
        assert not bad, f"摄像头 direction 应为角度数值，以下不是: {bad[:5]}"

    def test_road_segment_direction_is_in_road_direction_enum(self, metadata):
        """路段的 direction 必须是 RoadDirection 的合法取值"""
        segments = metadata.get("road_segments") or []
        if not segments:
            pytest.skip("元数据里没有 road_segments 段")

        allowed = _values(RoadDirection)
        checked = 0
        for seg in segments:
            value = seg.get("direction")
            if value is None:
                continue
            checked += 1
            assert value in allowed, (
                f"路段 {seg.get('segment_id')} 的 direction={value!r} "
                f"不在 RoadDirection 词表 {sorted(allowed)} 中"
            )
        assert checked > 0, "没有任何路段带 direction，校验形同虚设"

    def test_road_segments_reference_existing_cameras(self, metadata):
        """路段两端引用的摄像头必须真实存在（防止拼写错误导致拓扑断裂）"""
        cameras = metadata.get("cameras") or []
        segments = metadata.get("road_segments") or []
        if not cameras or not segments:
            pytest.skip("缺少 cameras 或 road_segments 段")

        known = {c["camera_id"] for c in cameras if c.get("camera_id")}
        missing = []
        for seg in segments:
            refs = list(seg.get("start_camera_ids") or []) + list(seg.get("end_camera_ids") or [])
            for ref in refs:
                if ref not in known:
                    missing.append((seg.get("segment_id"), ref))

        assert not missing, f"路段引用了不存在的摄像头: {missing[:5]}"
