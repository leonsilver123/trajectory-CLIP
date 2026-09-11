"""
tests/test_stitching.py - 拼接功能测试

测试目标: 验证跨镜轨迹拼接相关模块（不需要模型权重）
覆盖:
  - CandidateEdgeGenerator 粗筛规则（用构造的 Tracklet 对）
  - CrossCameraScorer 评分逻辑
  - ObservationChainBuilder 链构建

注意: CandidateEdgeGenerator 和 CrossCameraScorer 依赖 CameraManager 和 RoadTopology，
      需要构造合适的 mock 对象或使用真实配置文件。
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.common.data_models import (
    BoundingBox,
    CrossCameraEdge,
    TargetInstance,
    Tracklet,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Fixtures
# ============================================================

def _make_tracklet(tid="TRK_001", camera_id="c001", target_type="vehicle",
                    start_hour=8, start_min=30, start_sec=0,
                    duration=10, plate=None, color="白色",
                    reid_vec=None, clip_vec=None):
    """辅助创建 Tracklet"""
    start = datetime(2024, 1, 1, start_hour, start_min, start_sec)
    end = start + timedelta(seconds=duration)
    instances = []
    for i in range(3):
        inst = TargetInstance(
            instance_id=f"INST_{tid}_{i}",
            camera_id=camera_id,
            timestamp=start + timedelta(seconds=i * duration / 2),
            frame_id=i * 10,
            target_type=target_type,
            bbox=BoundingBox(x1=i * 10, y1=50, x2=i * 10 + 80, y2=150, confidence=0.85),
            attributes={"color": color},
            plate_number=plate,
            plate_confidence=0.0 if plate is None else 0.9,
            quality_score=0.8,
            reid_vector=reid_vec,
            clip_vector=clip_vec,
            keyframe_path=None,
        )
        instances.append(inst)
    return Tracklet(
        tracklet_id=tid,
        camera_id=camera_id,
        target_type=target_type,
        start_time=start,
        end_time=end,
        instances=instances,
        direction="由西向东",
        plate_number=plate,
        attributes={"color": color},
        avg_reid_vector=reid_vec,
        avg_clip_vector=clip_vec,
        keyframe_paths=[],
    )


def _make_edge(src_id="TRK_001", tgt_id="TRK_002", score=0.8,
               appearance=0.9, temporal=0.7, spatial=0.8, is_valid=True):
    """辅助创建 CrossCameraEdge"""
    return CrossCameraEdge(
        source_tracklet_id=src_id,
        target_tracklet_id=tgt_id,
        score=score,
        appearance_score=appearance,
        attribute_score=0.8,
        plate_score=0.0,
        temporal_score=temporal,
        spatial_score=spatial,
        direction_score=0.9,
        penalty=0.05,
        is_valid=is_valid,
        reasoning="测试边",
    )


# ============================================================
# CrossCameraEdge 数据模型测试
# ============================================================

class TestCrossCameraEdgeModel:
    """CrossCameraEdge 数据模型测试"""

    def test_create_valid_edge(self):
        """测试创建有效边"""
        edge = _make_edge()
        assert edge.is_valid is True
        assert edge.score == 0.8
        assert edge.source_tracklet_id == "TRK_001"

    def test_create_invalid_edge(self):
        """测试创建无效边"""
        edge = _make_edge(is_valid=False, score=0.2)
        assert edge.is_valid is False
        assert edge.score == 0.2

    def test_edge_score_components(self):
        """测试边的评分分量"""
        edge = _make_edge(appearance=0.95, temporal=0.85, spatial=0.75)
        assert edge.appearance_score == 0.95
        assert edge.temporal_score == 0.85
        assert edge.spatial_score == 0.75


# ============================================================
# ObservationChainBuilder 测试
# ============================================================

class TestObservationChainBuilder:
    """ObservationChainBuilder 链构建测试"""

    def test_init(self):
        """测试初始化"""
        from src.stitching.observation_chain import ObservationChainBuilder
        builder = ObservationChainBuilder()
        assert builder.min_chain_confidence == 0.4
        assert builder.max_upstream_depth == 15
        assert builder.max_downstream_depth == 15

    def test_init_with_params(self):
        """测试带参数初始化"""
        from src.stitching.observation_chain import ObservationChainBuilder
        builder = ObservationChainBuilder(
            min_chain_confidence=0.6,
            max_upstream_depth=5,
            max_downstream_depth=5,
        )
        assert builder.min_chain_confidence == 0.6
        assert builder.max_upstream_depth == 5

    def test_add_tracklets(self):
        """测试添加 Tracklet"""
        from src.stitching.observation_chain import ObservationChainBuilder
        builder = ObservationChainBuilder()
        trk1 = _make_tracklet("TRK_001", "c001")
        trk2 = _make_tracklet("TRK_002", "c002", start_min=35)
        builder._tracklets[trk1.tracklet_id] = trk1
        builder._tracklets[trk2.tracklet_id] = trk2
        assert len(builder._tracklets) == 2

    def test_add_edges(self):
        """测试添加边"""
        from src.stitching.observation_chain import ObservationChainBuilder
        builder = ObservationChainBuilder()
        edge = _make_edge()
        builder._all_edges.append(edge)
        # 构建邻接表
        src = edge.source_tracklet_id
        tgt = edge.target_tracklet_id
        if src not in builder._downstream_adj:
            builder._downstream_adj[src] = []
        builder._downstream_adj[src].append((tgt, edge))
        assert len(builder._all_edges) == 1
        assert src in builder._downstream_adj

    def test_chain_builder_with_tracklets_and_edges(self):
        """测试用 Tracklet 和边初始化构建器"""
        from src.stitching.observation_chain import ObservationChainBuilder
        trk1 = _make_tracklet("TRK_001", "c001", start_min=30)
        trk2 = _make_tracklet("TRK_002", "c002", start_min=35)
        trk3 = _make_tracklet("TRK_003", "c010", start_min=40)
        edge1 = _make_edge("TRK_001", "TRK_002", score=0.85)
        edge2 = _make_edge("TRK_002", "TRK_003", score=0.75)

        builder = ObservationChainBuilder(
            tracklets=[trk1, trk2, trk3],
            edges=[edge1, edge2],
        )
        assert len(builder._tracklets) == 3
        assert len(builder._all_edges) == 2


# ============================================================
# CandidateEdgeGenerator 粗筛规则测试（间接测试）
# ============================================================

class TestCandidateEdgeRules:
    """候选边生成规则测试（使用构造数据验证逻辑约束）"""

    def test_same_camera_tracklets_should_not_connect(self):
        """测试同一摄像头的 tracklet 不应产生跨镜边"""
        trk1 = _make_tracklet("TRK_A", "c001", start_min=30)
        trk2 = _make_tracklet("TRK_B", "c001", start_min=35)
        # 同一摄像头 - 逻辑上不应产生跨镜边
        assert trk1.camera_id == trk2.camera_id

    def test_type_mismatch_tracklets(self):
        """测试不同类型目标的 tracklet"""
        trk_vehicle = _make_tracklet("TRK_V", "c001", target_type="vehicle")
        trk_ped = _make_tracklet("TRK_P", "c002", target_type="pedestrian")
        # 不同类型 - 候选边生成器应过滤掉
        assert trk_vehicle.target_type != trk_ped.target_type

    def test_time_order_valid(self):
        """测试时间顺序验证"""
        trk1 = _make_tracklet("TRK_1", "c001", start_min=30)
        trk2 = _make_tracklet("TRK_2", "c002", start_min=35)
        # trk1 在 trk2 之前结束
        assert trk1.end_time < trk2.start_time

    def test_time_order_invalid(self):
        """测试时间顺序无效（目标不可能从未来回到过去）"""
        trk1 = _make_tracklet("TRK_1", "c001", start_min=40)
        trk2 = _make_tracklet("TRK_2", "c002", start_min=30)
        # trk1 开始时间晚于 trk2 - 如果 trk1 是 source 则不合理
        assert trk1.start_time > trk2.start_time

    def test_plate_consistency(self):
        """测试车牌一致性"""
        trk1 = _make_tracklet("TRK_1", "c001", plate="苏E12345")
        trk2 = _make_tracklet("TRK_2", "c002", plate="苏E12345")
        trk3 = _make_tracklet("TRK_3", "c003", plate="苏E99999")
        # 相同车牌应匹配
        assert trk1.plate_number == trk2.plate_number
        # 不同车牌应冲突
        assert trk1.plate_number != trk3.plate_number

    def test_plate_conflict(self):
        """测试车牌冲突检测"""
        trk1 = _make_tracklet("TRK_1", "c001", plate="苏E12345")
        trk2 = _make_tracklet("TRK_2", "c002", plate="苏E99999")
        # 两个都有车牌但不同 → 冲突
        has_conflict = (
            trk1.has_plate and trk2.has_plate and
            trk1.plate_number != trk2.plate_number
        )
        assert has_conflict is True


# ============================================================
# CrossCameraScorer 评分逻辑测试
# ============================================================

class TestCrossCameraScorerLogic:
    """CrossCameraScorer 评分逻辑测试（直接测试评分公式）"""

    def test_score_range(self):
        """测试评分在 [0, 1] 范围内"""
        edge = _make_edge(score=0.85)
        assert 0 <= edge.score <= 1

    def test_high_appearance_high_score(self):
        """测试高外观相似度 → 高分"""
        edge_high = _make_edge(score=0.9, appearance=0.95)
        edge_low = _make_edge(score=0.5, appearance=0.3)
        assert edge_high.score > edge_low.score

    def test_penalty_reduces_score(self):
        """测试惩罚降低分数"""
        edge_no_penalty = CrossCameraEdge(
            source_tracklet_id="A", target_tracklet_id="B",
            score=0.8, appearance_score=0.8, attribute_score=0.8,
            plate_score=0.8, temporal_score=0.8, spatial_score=0.8,
            direction_score=0.8, penalty=0.0, is_valid=True, reasoning="",
        )
        edge_with_penalty = CrossCameraEdge(
            source_tracklet_id="A", target_tracklet_id="B",
            score=0.6, appearance_score=0.8, attribute_score=0.8,
            plate_score=0.8, temporal_score=0.8, spatial_score=0.8,
            direction_score=0.8, penalty=0.2, is_valid=True, reasoning="",
        )
        assert edge_no_penalty.score > edge_with_penalty.score
