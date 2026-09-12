"""
tests.test_time_alignment - T5 跨摄时间对齐测试

背景（已实测取证）:
    `output/cityflow_results.json` 里每辆车的 `timestamp` 是**各摄像头自己的
    本地时间**（每路都从 2020-01-01 00:00:00 附近起跳），不同摄像头的时间戳
    直接比较是错位的。数据集 `cam_timestamp/{scene}.txt` 给出每个摄像头的
    全局起始偏移（秒），正确全局时间 = 本地时间 + 该摄像头偏移。

覆盖点:
1. 偏移表按 (场景, 摄像头) 精确加载，且与数据集文件逐值一致
2. 同一摄像头在不同场景的偏移不同时，按场景命中正确的那一个
3. 全局时间 = 本机时间 + 偏移
4. **真实数据案例**：某车跨镜的 actual_travel_time 在「对齐前是负值（故留
   None）→ 对齐后为正」——这是本任务的核心结论，用真实 detection 数据证明
5. 每个检测自身的 `timestamp` 字段不被改写，全局时间只作为新增字段给出
6. `time_alignment.enabled=false` 时完全退回本机时间（对照组）

本文件不依赖任何模型权重，只读 output/cityflow_results.json 与
cityflow/AICity22_Track1_MTMC_Tracking/cam_timestamp/*.txt。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.common.config import Config
from src.trajectory.builder import TrajectoryBuilder, get_trajectory_builder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAM_TIMESTAMP_DIR = PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking" / "cam_timestamp"

# 真实存在于 output/cityflow_results.json 的目标（V0272 跨 11 个摄像头，全在 S04）
ANCHOR_V0272 = "CF3_c034_V0272_000739"

# 真实案例：V0272 的 c034 → c035 相邻摄像头对
# 对齐前（各摄像头本机时间直接相减）≈ -16.601s → 判定为错位、actual_travel_time 只能留 None
# 对齐后（本机时间 + cam_timestamp 偏移）≈ +8.749s → 得到有意义的跨镜行程时间
CASE_SRC_CAMERA = "c034"
CASE_TGT_CAMERA = "c035"
CASE_SCENE = "S04"


def _read_dataset_offsets(scene: str) -> dict:
    """
    直接从数据集文件读偏移（测试的自校验基线，不经过被测代码）

    这样测试断言的是"实现是否与数据集一致"，而不是"实现是否与实现一致"。
    """
    offsets = {}
    with open(CAM_TIMESTAMP_DIR / f"{scene}.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                offsets[parts[0]] = float(parts[1])
    return offsets


@pytest.fixture(scope="module")
def builder() -> TrajectoryBuilder:
    """整个模块共用一个构建器（全量数据加载较慢，只做一次）"""
    return get_trajectory_builder()


@pytest.fixture(scope="module")
def result_v0272(builder) -> dict:
    """V0272 的强身份回溯结果（模块内共享，避免重复计算）"""
    return builder.build(ANCHOR_V0272)


@pytest.fixture(scope="module")
def disabled_builder() -> TrajectoryBuilder:
    """
    关闭时间对齐的构建器（对照组）

    独立 Config 实例，不污染全局单例；只读 6 个小文本文件 + 一次数据加载。
    """
    config = Config()
    config.set("time_alignment.enabled", False)
    return TrajectoryBuilder(config=config)


# ============================================================
# 1. 偏移表加载
# ============================================================

class TestCameraOffsetLoading:

    def test_offsets_match_dataset_file(self, builder):
        """S04 每个摄像头的偏移必须与数据集文件逐值一致"""
        assert builder.ensure_loaded() is True
        expected = _read_dataset_offsets(CASE_SCENE)
        assert expected, "数据集 cam_timestamp/S04.txt 应当非空"
        for camera_id, offset in expected.items():
            assert builder.camera_time_offset(camera_id, CASE_SCENE) == pytest.approx(offset), \
                f"{camera_id} 偏移与数据集不一致"

    def test_s01_baseline_offsets(self, builder):
        """S01 的典型值（c001=0、c002=1.640、c005=2.235）"""
        assert builder.camera_time_offset("c001", "S01") == pytest.approx(0.0)
        assert builder.camera_time_offset("c002", "S01") == pytest.approx(1.640)
        assert builder.camera_time_offset("c005", "S01") == pytest.approx(2.235)

    def test_s04_large_offsets(self, builder):
        """S04 的偏移可达上百秒（c018=29.955、c035=165.568）"""
        assert builder.camera_time_offset("c018", "S04") == pytest.approx(29.955)
        assert builder.camera_time_offset("c035", "S04") == pytest.approx(165.568)

    def test_same_camera_different_scene(self, builder):
        """
        c010 同时在 S03(8.715) 与 S05(0.000) 出现

        这说明偏移**必须**按 (场景, 摄像头) 查表，只按摄像头查会串场景。
        """
        assert builder.camera_time_offset("c010", "S03") == pytest.approx(8.715)
        assert builder.camera_time_offset("c010", "S05") == pytest.approx(0.0)

    def test_unknown_camera_returns_none(self, builder):
        """查不到的摄像头必须返回 None（表示"未知"），不能拿 0.0 冒充已对齐"""
        assert builder.camera_time_offset("c999", "S04") is None


# ============================================================
# 2. 本机时间 → 全局时间
# ============================================================

class TestGlobalTimeConversion:

    def test_global_time_is_local_plus_offset(self, builder):
        local = datetime(2020, 1, 1, 0, 0, 10)
        offset = _read_dataset_offsets(CASE_SCENE)[CASE_TGT_CAMERA]
        assert builder.to_global_time(CASE_TGT_CAMERA, local, CASE_SCENE) == \
            local + timedelta(seconds=offset)

    def test_unknown_camera_keeps_local_time(self, builder):
        """偏移未知时原样返回本机时间，且 is_time_aligned 如实回答 False"""
        local = datetime(2020, 1, 1, 0, 0, 10)
        assert builder.to_global_time("c999", local, "S04") == local
        assert builder.is_time_aligned("c999", CASE_TGT_CAMERA, "S04", CASE_SCENE) is False
        assert builder.is_time_aligned(CASE_SRC_CAMERA, CASE_TGT_CAMERA, CASE_SCENE, CASE_SCENE) is True

    def test_none_time_stays_none(self, builder):
        assert builder.to_global_time(CASE_TGT_CAMERA, None, CASE_SCENE) is None


# ============================================================
# 3. 真实数据案例：对齐前为负 → 对齐后为正
# ============================================================

class TestRealCaseTravelTime:
    """V0272 的 c034 → c035：T5 的核心证据"""

    def _find_segment(self, result_v0272):
        segments = result_v0272["inference_segments"]
        assert segments, "强身份路径应当产出推断段"
        for seg in segments:
            if (seg["source_camera_id"], seg["target_camera_id"]) == \
                    (CASE_SRC_CAMERA, CASE_TGT_CAMERA):
                return seg
        pytest.fail(f"未找到 {CASE_SRC_CAMERA} → {CASE_TGT_CAMERA} 的跨镜推断段")

    @pytest.fixture
    def segment(self, result_v0272):
        return self._find_segment(result_v0272)

    def test_local_travel_time_is_negative(self, segment):
        """对齐前：各摄像头本机时间直接相减得到负值（错位的直接证据）"""
        assert segment["local_travel_time"] < 0, segment["local_travel_time"]

    def test_actual_travel_time_positive_after_alignment(self, segment):
        """对齐后：跨镜行程时间为有意义的正值，不再是 None"""
        assert segment["actual_travel_time"] is not None
        assert segment["actual_travel_time"] > 0

    def test_actual_equals_local_plus_offset_delta(self, segment):
        """
        actual = local + (目标摄像头偏移 − 源摄像头偏移)

        这是"全局时间 = 本机时间 + 偏移"的直接推论，用数据集文件里的偏移
        独立算一遍，避免只验证实现自洽。
        """
        offsets = _read_dataset_offsets(CASE_SCENE)
        expected = segment["local_travel_time"] + (
            offsets[CASE_TGT_CAMERA] - offsets[CASE_SRC_CAMERA]
        )
        assert segment["actual_travel_time"] == pytest.approx(expected, abs=0.01)

    def test_alignment_annotations_present(self, segment):
        """对齐过程必须可解释：标注对齐与否 + 两侧偏移值"""
        offsets = _read_dataset_offsets(CASE_SCENE)
        assert segment["time_aligned"] is True
        assert segment["source_time_offset_seconds"] == pytest.approx(offsets[CASE_SRC_CAMERA])
        assert segment["target_time_offset_seconds"] == pytest.approx(offsets[CASE_TGT_CAMERA])

    def test_timeline_total_duration_uses_global_time(self, builder):
        """跨镜总时长必须用全局时间，不能是本机时间的差值"""
        timeline = builder.build_camera_timeline(instance_id=ANCHOR_V0272)
        assert timeline["global_first_appearance"]
        assert timeline["global_last_appearance"]
        first = datetime.strptime(timeline["global_first_appearance"], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(timeline["global_last_appearance"], "%Y-%m-%d %H:%M:%S")
        expected = int((last - first).total_seconds())
        assert timeline["total_duration_seconds"] == expected
        # 本机时间跨度与全局时间跨度差别很大 —— 证明两者不是同一个数
        local_first = datetime.strptime(timeline["first_appearance"], "%Y-%m-%d %H:%M:%S")
        local_last = datetime.strptime(timeline["last_appearance"], "%Y-%m-%d %H:%M:%S")
        assert expected != int((local_last - local_first).total_seconds())


# ============================================================
# 4. 展示字段不被改写
# ============================================================

class TestDisplayFieldsUnchanged:
    """timestamp 仍是摄像头本机时间；全局时间只作为新增字段"""

    def test_timestamp_remains_local(self, result_v0272):
        """本机时间就是各摄像头从 0 起跳的时间，不应被叠加偏移"""
        for node in result_v0272["observation_nodes"]:
            assert node["timestamp"].startswith("2020-01-01 00:00"), node["timestamp"]

    def test_global_timestamp_is_new_field(self, result_v0272):
        """
        global_timestamp = timestamp + time_offset_seconds（新增字段，不改原字段）

        两个展示字段都精确到秒，因此与浮点偏移的对照允许 1 秒内的格式截断误差。
        """
        for node in result_v0272["observation_nodes"]:
            assert node["global_timestamp"] is not None
            local = datetime.strptime(node["timestamp"], "%Y-%m-%d %H:%M:%S")
            glob = datetime.strptime(node["global_timestamp"], "%Y-%m-%d %H:%M:%S")
            assert (glob - local).total_seconds() == pytest.approx(
                node["time_offset_seconds"], abs=1.0
            )
            assert node["global_timestamp"] != node["timestamp"]

    def test_original_contract_fields_intact(self, result_v0272):
        """原有字段一个都不少（前端已依赖）"""
        for key in ("camera_id", "camera_name", "tracklet_id", "timestamp",
                    "latitude", "longitude", "keyframe_path", "confidence",
                    "basis", "basis_text"):
            assert key in result_v0272["observation_nodes"][0], key

    def test_camera_sequence_is_global_order(self, result_v0272):
        result = result_v0272
        """摄像头先后必须按全局首现时间排序"""
        times = []
        for node in result["observation_nodes"]:
            times.append(datetime.strptime(node["global_timestamp"], "%Y-%m-%d %H:%M:%S"))
        assert times == sorted(times), [n["camera_id"] for n in result["observation_nodes"]]


# ============================================================
# 5. 对照组：关闭对齐后退回本机时间
# ============================================================

class TestAlignmentDisabled:

    def test_offset_lookup_disabled(self, disabled_builder):
        """关闭后偏移一律视为未知 → 保持本机时间"""
        assert disabled_builder.camera_time_offset("c002", "S01") is None
        local = datetime(2020, 1, 1, 0, 0, 10)
        assert disabled_builder.to_global_time("c002", local, "S01") == local

    def test_real_case_falls_back_to_none(self, disabled_builder):
        """
        同一辆车、同一对摄像头，关闭对齐后拿不到正的行程时间

        对齐前要么该摄像头对不再相邻（本机时间排序不同），要么本机时间差为负
        只能留 None —— 两种情况都证明"对齐后为正"是对齐带来的，不是本来就有的。
        """
        result = disabled_builder.build(ANCHOR_V0272)
        by_pair = {
            (seg["source_camera_id"], seg["target_camera_id"]): seg
            for seg in result["inference_segments"]
        }
        pair = by_pair.get((CASE_SRC_CAMERA, CASE_TGT_CAMERA)) or \
            by_pair.get((CASE_TGT_CAMERA, CASE_SRC_CAMERA))
        assert pair is None or pair["actual_travel_time"] is None
        # 关闭后所有段都标注为"未对齐"
        assert all(seg["time_aligned"] is False for seg in result["inference_segments"])
        # 对照组里不存在正的 c034→c035 行程时间
        assert not (pair is not None and (pair["actual_travel_time"] or 0) > 0)
