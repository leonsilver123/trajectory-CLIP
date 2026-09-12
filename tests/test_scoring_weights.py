"""
tests.test_scoring_weights - T6 评分权重修复测试

背景（已实测取证）:
    `configs/default.yaml` 的 `stitching.weights.vehicle` 是
    plate:0.35 / temporal:0.25 / topology:0.15 / reid:0.15 / attribute:0.10，
    但 `src/stitching/scoring.py` 里另有一套硬编码默认权重
    （plate:0.30 / topology:0.10 / reid:0.25）——"配置驱动"被破坏。
    本数据集里 **没有任何车牌真值、也没有任何 ReID 向量**，于是
    plate(0.35) 与 reid(0.15) 两维恒为中性分 0.5，等于 50% 的权重只贡献
    一个常数偏移，剩下一半的区分度还要被摊薄。

覆盖点:
1. 权重唯一来源是配置；硬编码兜底值与配置逐键一致（不再两套）
2. 显式传参优先级最高
3. ReID 缺失时：不崩、退化路径显式（维度被剔除）、分数可解释
4. ReID 软评分权重为 0（P-A 实测软 reid 是噪声）：appearance_score 仍如实记录，
   但不乘进总分——外观过滤由规则 8 硬门控承担
5. 缺失维度重分摊：权重总和不变（不放大任何单个分项）
6. 可关闭重分摊 → 退回旧行为
7. 跨摄时间对齐传入评分器后，时间可达性维度按全局时间计算（T5 与 T6 打通）

本文件用轻量 stub 提供 CameraManager / RoadTopology，不加载任何模型权重，
也不做秒级的路径枚举（真实 RoadTopology 的路径计数单对耗时 4–23 秒）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.common.config import Config
from src.common.data_models import BoundingBox, TargetInstance, Tracklet
from src.stitching.scoring import (
    DEFAULT_PEDESTRIAN_WEIGHTS,
    DEFAULT_VEHICLE_WEIGHTS,
    NEUTRAL_SCORE,
    CrossCameraScorer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# ============================================================
# 轻量 stub：只提供评分需要的几个方法，避免真实拓扑的秒级路径枚举
# ============================================================

class _StubTopology:
    """可达 + 固定距离的拓扑（不提供 _adjacency → 路径计数退化为 1）"""

    def __init__(self, distance_meters: float = 1000.0) -> None:
        self.distance_meters = distance_meters

    def is_reachable(self, src_camera_id: str, tgt_camera_id: str) -> bool:
        return True

    def get_segment_distance(self, src_camera_id: str, tgt_camera_id: str):
        return self.distance_meters

    def shortest_path(self, src_camera_id: str, tgt_camera_id: str):
        return [src_camera_id, tgt_camera_id]


class _StubCameraManager:
    """无摄像头元数据（评分回退到拓扑距离）"""

    def get_camera(self, camera_id: str):
        return None


def _make_tracklet(
    tracklet_id: str,
    camera_id: str,
    start_second: int = 0,
    duration: int = 5,
    target_type: str = "vehicle",
    attributes: dict | None = None,
    reid_vector=None,
    clip_vector=None,
    plate_number=None,
    direction: str = "eastbound",
) -> Tracklet:
    """构造一个最小可用的 Tracklet（单一检测即可，属性/向量直接给聚合值）"""
    start = datetime(2020, 1, 1, 0, 0, start_second)
    end = start + timedelta(seconds=duration)
    instance = TargetInstance(
        instance_id=f"INST_{tracklet_id}",
        camera_id=camera_id,
        timestamp=start,
        frame_id=0,
        target_type=target_type,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        attributes=attributes or {},
        plate_number=plate_number,
        plate_confidence=0.9 if plate_number else 0.0,
        quality_score=0.9,
        reid_vector=reid_vector,
        clip_vector=clip_vector,
        keyframe_path=None,
    )
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        target_type=target_type,
        start_time=start,
        end_time=end,
        instances=[instance],
        direction=direction,
        plate_number=plate_number,
        attributes=attributes or {},
        avg_reid_vector=reid_vector,
        avg_clip_vector=clip_vector,
        keyframe_paths=[],
    )


@pytest.fixture
def scorer() -> CrossCameraScorer:
    """默认评分器（权重应当来自 configs/default.yaml）"""
    return CrossCameraScorer(
        camera_manager=_StubCameraManager(),
        road_topology=_StubTopology(),
    )


# ============================================================
# 1. 权重来源
# ============================================================

class TestWeightSource:

    @staticmethod
    def _config_weights() -> dict:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["stitching"]["weights"]

    def test_scorer_reads_weights_from_config(self, scorer):
        """不传权重时必须读 configs/default.yaml，而不是用另一套硬编码值"""
        expected = self._config_weights()
        assert scorer.vehicle_weights == expected["vehicle"]
        assert scorer.pedestrian_weights == expected["pedestrian"]
        assert scorer.weights_from_config is True

    def test_config_weights_are_not_the_old_hardcoded_ones(self, scorer):
        """
        回归保护：旧硬编码值（plate 0.30 / topology 0.10 / reid 0.25）必须消失

        这条断言直接卡住"配置一套、代码一套"的复发。
        """
        old_vehicle = {"plate": 0.30, "temporal": 0.25, "topology": 0.10,
                       "reid": 0.25, "attribute": 0.10}
        assert scorer.vehicle_weights != old_vehicle
        assert scorer.vehicle_weights["plate"] == pytest.approx(0.35)
        assert scorer.vehicle_weights["topology"] == pytest.approx(0.15)
        # P-A（网格搜索实测）：软 reid / attribute 权重归零，不再是 0.15 / 0.10。
        # 依据：它们的连续相似度跨镜不可分（reid d-prime 0.78、属性一致率 28-37%），
        # 加权进软评分只会加噪声；真正的过滤由规则 8 / 规则 7 硬门控承担。
        assert scorer.vehicle_weights["reid"] == pytest.approx(0.0)
        assert scorer.vehicle_weights["attribute"] == pytest.approx(0.0)
        assert scorer.vehicle_weights["temporal"] == pytest.approx(0.40)

    def test_hardcoded_fallback_matches_config(self):
        """兜底常量必须与配置逐键一致（它是"配置读不到"时的替身）"""
        expected = self._config_weights()
        assert DEFAULT_VEHICLE_WEIGHTS == expected["vehicle"]
        assert DEFAULT_PEDESTRIAN_WEIGHTS == expected["pedestrian"]

    def test_explicit_weights_win(self):
        """显式传参优先级最高"""
        custom = {"plate": 0.5, "temporal": 0.5}
        scorer = CrossCameraScorer(
            camera_manager=_StubCameraManager(),
            road_topology=_StubTopology(),
            vehicle_weights=custom,
        )
        assert scorer.vehicle_weights == custom
        assert scorer.weights_from_config is False

    def test_penalties_from_config(self, scorer):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            expected = yaml.safe_load(f)["stitching"]["penalties"]
        assert scorer.penalties == expected


# ============================================================
# 2. ReID 缺失时的退化路径（显式、可解释、不崩）
# ============================================================

class TestMissingReidDegradation:

    @pytest.fixture
    def pair_without_reid(self):
        src = _make_tracklet("TRK_A", "c001", start_second=0,
                             attributes={"color": "black", "vehicle_type": "sedan"})
        tgt = _make_tracklet("TRK_B", "c002", start_second=30,
                             attributes={"color": "black", "vehicle_type": "sedan"})
        return src, tgt

    def test_score_does_not_crash(self, scorer, pair_without_reid):
        """两个 avg_reid_vector=None 的 Tracklet 必须能正常打分"""
        src, tgt = pair_without_reid
        assert src.avg_reid_vector is None and tgt.avg_reid_vector is None
        edge = scorer.score(src, tgt)
        assert 0.0 <= edge.score <= 1.0
        assert edge.appearance_score == pytest.approx(NEUTRAL_SCORE)

    def test_appearance_source_is_none(self, scorer, pair_without_reid):
        src, tgt = pair_without_reid
        assert scorer._appearance_source(src, tgt) == "none"

    def test_dimension_availability_explicit(self, scorer, pair_without_reid):
        """reid / plate 两维明确判为"无证据"，temporal / topology 恒为有证据"""
        src, tgt = pair_without_reid
        availability = scorer._dimension_availability(src, tgt)
        assert availability["reid"] is False
        assert availability["plate"] is False
        assert availability["attribute"] is True   # 双方都有 color/vehicle_type
        assert availability["temporal"] is True
        assert availability["topology"] is True

    def test_effective_weights_drop_missing_and_keep_total(self, scorer, pair_without_reid):
        """
        剔除 reid(0.15) 与 plate(0.35) 后，权重按比例分摊给其余维度

        车辆原权重总和 1.0，剩余 temporal 0.25 / topology 0.15 / attribute 0.10
        总和 0.5 → 放大 2 倍，总和仍是 1.0：不放大任何单个分项的绝对贡献。
        """
        src, tgt = pair_without_reid
        weights = scorer.vehicle_weights
        availability = scorer._dimension_availability(src, tgt)
        effective, dropped = scorer._effective_weights(weights, availability)

        # reid/attribute 软权重为 0（P-A），_effective_weights 对 weight<=0 直接跳过，
        # 因此只有 plate 是"被剔除"的缺失维度；有效维度只剩 temporal/topology。
        assert sorted(dropped) == ["plate"]
        assert set(effective) == {"temporal", "topology"}
        # total_original = plate 0.35 + temporal 0.40 + topology 0.15 = 0.90，
        # total_kept = 0.55 → scale = 0.90/0.55
        assert effective["temporal"] == pytest.approx(0.40 * 0.90 / 0.55)
        assert effective["topology"] == pytest.approx(0.15 * 0.90 / 0.55)
        # 权重总和不变 —— 没有"把权重乘到一个不存在的分数上"这种事
        assert sum(effective.values()) == pytest.approx(sum(weights.values()))

    def test_reasoning_explains_dropped_dimensions(self, scorer, pair_without_reid):
        """退化路径必须写进 reasoning，排障时看得见"""
        src, tgt = pair_without_reid
        edge = scorer.score(src, tgt)
        assert "有效权重" in edge.reasoning
        assert "plate" in edge.reasoning and "reid" in edge.reasoning
        assert "无 ReID/CLIP 外观特征" in edge.reasoning
        assert "双方均无车牌" in edge.reasoning

    def test_explain_dimensions_api(self, scorer, pair_without_reid):
        """explain_dimensions 是给测试与线上排障用的可观测入口"""
        src, tgt = pair_without_reid
        info = scorer.explain_dimensions(src, tgt)
        assert info["appearance_source"] == "none"
        # reid 软权重为 0（P-A），weight<=0 被跳过而非"剔除"，故 dropped 只剩 plate
        assert sorted(info["dropped_dimensions"]) == ["plate"]
        assert info["availability"]["reid"] is False
        assert info["reweighting_enabled"] is True
        assert info["time_aligned"] is False   # 未传偏移表

    def test_reweight_can_be_disabled(self, pair_without_reid):
        """关闭重分摊 → 退回旧行为（中性分 × 原权重直接求和）"""
        config = Config()
        config.set("stitching.reweight_missing_dimensions", False)
        scorer = CrossCameraScorer(
            camera_manager=_StubCameraManager(),
            road_topology=_StubTopology(),
            config=config,
        )
        src, tgt = pair_without_reid
        assert scorer.reweight_missing_dimensions is False
        availability = scorer._dimension_availability(src, tgt)
        effective, dropped = scorer._effective_weights(scorer.vehicle_weights, availability)
        assert dropped == []
        assert effective == scorer.vehicle_weights

    def test_clip_vector_counts_as_appearance_evidence(self, scorer):
        """有 CLIP 向量时外观维仍有证据（退化只发生在两者都缺时）"""
        src = _make_tracklet("TRK_C", "c001", start_second=0,
                             clip_vector=np.array([1.0, 0.0], dtype=np.float32))
        tgt = _make_tracklet("TRK_D", "c002", start_second=30,
                             clip_vector=np.array([1.0, 0.0], dtype=np.float32))
        assert scorer._appearance_source(src, tgt) == "clip"
        assert scorer._dimension_availability(src, tgt)["reid"] is True
        edge = scorer.score(src, tgt)
        assert edge.appearance_score == pytest.approx(1.0)


# ============================================================
# 3. ReID 存在时确实贡献到总分
# ============================================================

class TestReidContributes:

    @staticmethod
    def _pair(similarity_second_vector) -> tuple:
        src = _make_tracklet("TRK_R1", "c001", start_second=0,
                             reid_vector=np.array([1.0, 0.0], dtype=np.float32))
        tgt = _make_tracklet("TRK_R2", "c002", start_second=30,
                             reid_vector=similarity_second_vector)
        return src, tgt

    def test_reid_tracked_but_zero_soft_weight(self, scorer):
        """有 ReID 向量时外观来源仍判为 reid，但软评分权重为 0（P-A：软 reid 是噪声）"""
        src, tgt = self._pair(np.array([1.0, 0.0], dtype=np.float32))
        info = scorer.explain_dimensions(src, tgt)
        assert info["appearance_source"] == "reid"
        # reid 软权重=0 → 既不进 effective_weights（weight<=0 被跳过），也不进 dropped
        assert "reid" not in info["effective_weights"]
        assert "reid" not in info["dropped_dimensions"]

    def test_reid_similarity_does_not_change_soft_score(self, scorer):
        """软 reid 权重=0 → 只改 ReID 相似度不改变总分（外观过滤交给规则 8 硬门控）"""
        src_high, tgt_high = self._pair(np.array([1.0, 0.0], dtype=np.float32))    # cos=1.0
        src_low, tgt_low = self._pair(np.array([-1.0, 0.0], dtype=np.float32))    # cos=-1.0

        edge_high = scorer.score(src_high, tgt_high)
        edge_low = scorer.score(src_low, tgt_low)

        # appearance_score 仍如实记录（1.0 vs 0.0），只是不乘进总分
        assert edge_high.appearance_score == pytest.approx(1.0)
        assert edge_low.appearance_score == pytest.approx(0.0)
        # 两对除外观外完全相同 → 软评分权重为 0 时总分应相同
        assert edge_high.score == pytest.approx(edge_low.score)

    def test_reid_presence_does_not_change_effective_weights(self, scorer):
        """软 reid 权重=0 → 有无 ReID 时有效权重相同（reid 都被跳过）"""
        with_reid_src, with_reid_tgt = self._pair(np.array([1.0, 0.0], dtype=np.float32))
        without_src = _make_tracklet("TRK_N1", "c001", start_second=0)
        without_tgt = _make_tracklet("TRK_N2", "c002", start_second=30)

        with_reid = scorer.explain_dimensions(with_reid_src, with_reid_tgt)["effective_weights"]
        without_reid = scorer.explain_dimensions(without_src, without_tgt)["effective_weights"]
        assert with_reid == without_reid
        assert "reid" not in with_reid

    def test_pedestrian_weights_used_for_pedestrian(self, scorer):
        """行人走另一套权重（temporal 0.35 / reid 0.30 / attribute 0.25 / bag 0.10）"""
        src = _make_tracklet("TRK_P1", "c001", start_second=0, target_type="pedestrian",
                             attributes={"gender": "male", "clothing_color": "black"})
        tgt = _make_tracklet("TRK_P2", "c002", start_second=30, target_type="pedestrian",
                             attributes={"gender": "male", "clothing_color": "black"})
        info = scorer.explain_dimensions(src, tgt)
        assert info["target_type"] == "pedestrian"
        assert info["weights"] == scorer.pedestrian_weights


# ============================================================
# 4. 时间对齐传入评分器（T5 × T6）
# ============================================================

class TestTemporalAlignmentInScorer:

    # c034 偏移 140.218s / c035 偏移 165.568s（S04，来自数据集 cam_timestamp/S04.txt）
    OFFSETS = {"c034": 140.218, "c035": 165.568}

    @staticmethod
    def _pair() -> tuple:
        """源 c034 末现于 00:00:21.5，目标 c035 首现于 00:00:04.599（本机时间）"""
        src = _make_tracklet("TRK_T1", "c034", start_second=10, duration=11)   # end = 00:00:21
        tgt = _make_tracklet("TRK_T2", "c035", start_second=0, duration=1)    # start = 00:00:00
        return src, tgt

    def test_local_travel_time_is_negative_without_offsets(self):
        scorer = CrossCameraScorer(
            camera_manager=_StubCameraManager(), road_topology=_StubTopology()
        )
        src, tgt = self._pair()
        assert scorer._aligned_travel_time(src, tgt) < 0
        # 时间倒序 → 不可能，时间可达性给 0
        assert scorer._score_temporal(src, tgt) == 0.0

    def test_aligned_travel_time_positive_with_offsets(self):
        scorer = CrossCameraScorer(
            camera_manager=_StubCameraManager(),
            road_topology=_StubTopology(),
            camera_time_offsets=self.OFFSETS,
        )
        src, tgt = self._pair()
        expected = (tgt.start_time - src.end_time).total_seconds() + (
            self.OFFSETS["c035"] - self.OFFSETS["c034"]
        )
        assert scorer._aligned_travel_time(src, tgt) == pytest.approx(expected)
        # 对齐后不再是"时间倒序"，时间可达性恢复为正常分数
        assert scorer._score_temporal(src, tgt) > 0.0

    def test_alignment_flagged_in_explain_dimensions(self):
        scorer = CrossCameraScorer(
            camera_manager=_StubCameraManager(),
            road_topology=_StubTopology(),
            camera_time_offsets=self.OFFSETS,
        )
        src, tgt = self._pair()
        assert scorer.explain_dimensions(src, tgt)["time_aligned"] is True


# ============================================================
# 5. 线上构建器用的评分器也吃配置
# ============================================================

class TestBuilderScorerUsesConfig:

    def test_builder_scorer_weights_and_offsets(self):
        """src.trajectory.builder 造出来的评分器：权重来自配置 + 带时间偏移表"""
        from src.trajectory.builder import get_trajectory_builder

        builder = get_trajectory_builder()
        assert builder.ensure_loaded() is True
        scorer = builder._get_scorer()
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            expected = yaml.safe_load(f)["stitching"]["weights"]
        assert scorer.vehicle_weights == expected["vehicle"]
        assert scorer.pedestrian_weights == expected["pedestrian"]
        # T5：偏移表已注入，c034/c035 的跨镜时间才会被正确换算
        assert scorer.camera_time_offsets.get("c034") == pytest.approx(140.218)
        assert scorer.camera_time_offsets.get("c035") == pytest.approx(165.568)
        assert len(scorer.camera_time_offsets) == 46
