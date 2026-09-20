"""
tests.test_scoring_evidence_coverage - 评分维度在本数据集上的真实证据覆盖（PLAN3-D6）

## 背景

车辆侧评分配置了五个权重：`plate 0.35 / temporal 0.40 / topology 0.15 /
reid 0.0 / attribute 0.0`。看起来"车牌一致性"是最重要的一维。

**但本数据集里一条车牌都没有。** 实测 68,349 条检测中带 `plate_number`
的为 **0 条**。于是 `plate` 每次评分都因"无证据"被整维剔除
（`reweight_missing_dimensions` 的正常行为），实际生效的只有：

    temporal 0.6545 / topology 0.2455

即"六维评分函数"在本数据集上**实际退化为「时间可达性 + 路网拓扑 + 方向」**。

本文件把这个事实钉住：
- 数据侧：车牌覆盖率为 0（若将来有了车牌，这里会红，提示重新评估权重）；
- 算法侧：生效权重确实剔除了 plate，且数值与预期一致；
- 一个此前没有文档的设计：**五项权重之和是 0.90**，余下 0.10 由方向加分项
  （`DIRECTION_BONUS_WEIGHT`）补齐，因此总分上限恰为 1.00。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology
from src.stitching.scoring import (
    DEFAULT_VEHICLE_WEIGHTS,
    DIRECTION_BONUS_WEIGHT,
    CrossCameraScorer,
)
from src.common.data_models import Tracklet

_ROOT = Path(__file__).resolve().parent.parent
_META = _ROOT / "configs" / "cityflow_camera_metadata.yaml"


@pytest.fixture(scope="module")
def scorer():
    if not _META.exists():
        pytest.skip(f"缺少摄像头元数据 {_META}")
    return CrossCameraScorer(CameraManager(str(_META)), RoadTopology(str(_META)))


def _vehicle_tracklet(camera_id: str, plate=None, reid: bool = False) -> Tracklet:
    """构造一个"本数据集真实情形"的车辆 tracklet（有颜色/车型属性）"""
    return Tracklet(
        tracklet_id=f"T_{camera_id}",
        camera_id=camera_id,
        target_type="vehicle",
        start_time="2020-01-01 00:00:00",
        end_time="2020-01-01 00:00:10",
        instances=[],
        direction="northbound",
        plate_number=plate,
        attributes={"color": "白色", "vehicle_type": "轿车"},
        avg_reid_vector=(np.ones(2048, dtype=np.float32) if reid else None),
        avg_clip_vector=None,
        keyframe_paths=[],
    )


class TestWeightArithmetic:
    """权重之和 0.90 + 方向加分 0.10 = 总分上限 1.00"""

    def test_vehicle_weights_sum_to_point_nine(self):
        total = sum(DEFAULT_VEHICLE_WEIGHTS.values())
        assert total == pytest.approx(0.90), (
            f"车辆侧五维权重之和为 {total}，预期 0.90 —— "
            f"余下部分应由方向加分项补足，见下一个断言"
        )

    def test_direction_bonus_fills_the_remainder(self):
        total = sum(DEFAULT_VEHICLE_WEIGHTS.values()) + DIRECTION_BONUS_WEIGHT
        assert total == pytest.approx(1.00), (
            f"权重和 + 方向加分 = {total}，应恰为 1.00（否则总分上限不是 1）"
        )


class TestEffectiveWeightsOnRealData:
    """无车牌时 plate 被整维剔除，生效权重与预期一致"""

    def test_plate_is_dropped_without_evidence(self, scorer):
        info = scorer.explain_dimensions(
            _vehicle_tracklet("c001"), _vehicle_tracklet("c002")
        )
        assert info["availability"]["plate"] is False
        assert "plate" in info["dropped_dimensions"]
        assert "plate" not in info["effective_weights"]

    def test_effective_weights_are_temporal_and_topology(self, scorer):
        info = scorer.explain_dimensions(
            _vehicle_tracklet("c001"), _vehicle_tracklet("c002")
        )
        eff = info["effective_weights"]
        assert set(eff) == {"temporal", "topology"}, (
            f"生效维度应为 temporal + topology，实际 {sorted(eff)}"
        )
        # 放大系数 = 配置总权重 / 保留维度权重 = 0.90 / 0.55
        scale = 0.90 / 0.55
        assert eff["temporal"] == pytest.approx(0.40 * scale, rel=1e-6)
        assert eff["topology"] == pytest.approx(0.15 * scale, rel=1e-6)

    def test_effective_weights_preserve_original_total(self, scorer):
        """重分摊保持**配置里的总权重**不变（0.90），即被剔除维度的份额真的转给了其余维度

        注意不是"保留维度之和不变"（那是 0.55）—— 若只保持 0.55，
        被剔除的份额就凭空消失了，等于整体缩分。
        """
        info = scorer.explain_dimensions(
            _vehicle_tracklet("c001"), _vehicle_tracklet("c002")
        )
        eff_total = sum(info["effective_weights"].values())
        original_total = sum(DEFAULT_VEHICLE_WEIGHTS.values())
        assert eff_total == pytest.approx(original_total, rel=1e-9)

    def test_plate_becomes_available_when_present(self, scorer):
        """一旦双方都有车牌，该维就应重新参与打分（证明剔除是"无证据"而非"被禁用"）"""
        info = scorer.explain_dimensions(
            _vehicle_tracklet("c001", plate="苏E12345"),
            _vehicle_tracklet("c002", plate="苏E12345"),
        )
        assert info["availability"]["plate"] is True
        assert "plate" in info["effective_weights"]


class TestDatasetPlateCoverage:
    """数据侧事实：本数据集车牌覆盖率为 0"""

    def test_no_detection_has_a_plate_number(self):
        """若这条测试红了，说明数据里出现了车牌 —— 请重新评估车辆侧权重分配"""
        from src.storage.datastore import load_results

        data = load_results()
        if not data:
            pytest.skip("datastore 与 JSON 均不可用")

        dets = data.get("detections", [])
        if not dets:
            pytest.skip("detections 为空")

        with_plate = sum(1 for d in dets if d.get("plate_number"))
        assert with_plate == 0, (
            f"数据集中有 {with_plate}/{len(dets)} 条检测带车牌。"
            f"车牌维度从此有证据了 —— 请重新评估 stitching.weights.vehicle 的分配，"
            f"并同步更新 configs/default.yaml 与 src/stitching/scoring.py 的注释"
        )

    def test_color_and_type_have_full_coverage(self):
        """颜色与车型属性是满覆盖的（按 detect 计数）—— 这是属性硬门控能工作的前提"""
        from src.storage.datastore import load_results

        data = load_results()
        if not data:
            pytest.skip("datastore 与 JSON 均不可用")

        dets = data.get("detections", [])
        if not dets:
            pytest.skip("detections 为空")

        with_color = sum(1 for d in dets if (d.get("attributes") or {}).get("color"))
        assert with_color == len(dets), (
            f"带 color 的检测 {with_color}/{len(dets)}，不是满覆盖"
        )
