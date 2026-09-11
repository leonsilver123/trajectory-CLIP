"""
tests.test_trajectory_builder - src.trajectory.builder 与回溯接口的回归测试

覆盖点:
1. 强身份路径（真实 vehicle_id）跨镜聚合，且每段标注「强身份匹配」
2. 弱身份路径（src.stitching 六维评分拼接），且每段标注「概率推断」
3. 同一目标重复回溯结果完全一致（主路径无随机数）
4. ID 格式不被改写（target_id / vehicle_id / track_id）
5. HTTP 接口 /trace、/trajectory 仍保持原有响应字段
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.common.ids import extract_vehicle_id, parse_target_id
from src.trajectory.builder import (
    BASIS_STRONG,
    BASIS_WEAK,
    TrajectoryNotFoundError,
    get_trajectory_builder,
)

# 真实存在于 output/cityflow_results.json 中的目标（跨 5 个摄像头）
ANCHOR = "CF3_c001_V0034_000001"
ANCHOR_VEHICLE = "V0034"


@pytest.fixture(scope="module")
def builder():
    """整个模块共用一个构建器（数据加载较慢，只做一次）"""
    return get_trajectory_builder()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ============================================================
# 数据与锚点定位
# ============================================================

class TestAnchorResolve:
    """锚点定位与 ID 解析"""

    def test_data_available(self, builder):
        assert builder.data_available is True

    def test_find_target_id_direct_hit(self, builder):
        assert builder.find_target_id(ANCHOR) == ANCHOR

    def test_find_target_id_from_track_id(self, builder):
        """track_id 形式也能解析回真实 target_id，且 vehicle_id 一致"""
        found = builder.find_target_id(f"CF3_TRACK_c001_{ANCHOR_VEHICLE}")
        assert found is not None
        assert extract_vehicle_id(found) == ANCHOR_VEHICLE

    def test_find_target_id_unknown_returns_none(self, builder):
        assert builder.find_target_id("NOT_A_REAL_TARGET") is None

    def test_ids_not_rewritten(self, builder):
        """ID 格式必须与数据集一致，不得由构建器改写"""
        target_id = parse_target_id(ANCHOR)
        assert target_id["camera_id"] == "c001"
        assert target_id["vehicle_id"] == ANCHOR_VEHICLE

        for node in builder.build(ANCHOR)["observation_nodes"]:
            assert re.fullmatch(r"CF3_TRACK_c\d{3}_V\d{4}", node["tracklet_id"]), node["tracklet_id"]

    def test_missing_target_raises(self, builder):
        with pytest.raises(TrajectoryNotFoundError):
            builder.build("NOT_A_REAL_TARGET")


# ============================================================
# 强身份路径
# ============================================================

class TestStrongPath:
    """强身份直接匹配"""

    def test_cross_camera(self, builder):
        result = builder.build(ANCHOR)
        assert result["identity"]["mode"] == "strong"
        assert result["identity"]["basis"] == BASIS_STRONG
        assert result["identity"]["vehicle_id"] == ANCHOR_VEHICLE
        # V0034 真实覆盖 5 个摄像头
        assert len(result["camera_sequence"]) > 1

    def test_every_segment_labelled(self, builder):
        result = builder.build(ANCHOR)
        for key in ("observation_nodes", "observation_segments"):
            for item in result[key]:
                assert item["basis"] == BASIS_STRONG
                assert "强身份匹配" in item["basis_text"]
        for seg in result["inference_segments"]:
            assert seg["basis"] == BASIS_STRONG
            assert "强身份匹配" in seg["basis_text"]

    def test_confidence_filled_from_real_scores(self, builder):
        """overall_confidence 与证据分项必须是真实算出来的，不是占位 0.0"""
        result = builder.build(ANCHOR)
        assert 0.0 < result["overall_confidence"] <= 1.0
        evidence = result["evidence"]
        assert evidence["source"] == "cityflow_results"
        assert evidence["link_count"] > 0
        # 外观分来自检测自带的 clip_image_vector 或中性回退，必须落在 [0, 1]
        assert 0.0 <= evidence["appearance_similarity"] <= 1.0

    def test_inference_segment_fields(self, builder):
        result = builder.build(ANCHOR)
        for seg in result["inference_segments"]:
            assert 0.0 <= seg["confidence"] <= 1.0
            # 行程时间要么是真实推算值，要么为 None——不允许造数
            assert seg["estimated_travel_time"] is None or seg["estimated_travel_time"] >= 0
            assert seg["actual_travel_time"] is None or seg["actual_travel_time"] >= 0

    def test_deterministic(self, builder):
        """同一锚点重复构建，结果逐字节一致（无随机数）"""
        first = json.dumps(builder.build(ANCHOR), sort_keys=True, ensure_ascii=False)
        second = json.dumps(builder.build(ANCHOR), sort_keys=True, ensure_ascii=False)
        assert first == second


# ============================================================
# 弱身份路径（src.stitching 六维评分拼接）
# ============================================================

class TestWeakPath:
    """弱身份概率推断"""

    def test_stitching_scores_used(self, builder):
        result = builder.build(ANCHOR, mode="stitch")
        assert result["identity"]["mode"] == "weak"
        assert result["identity"]["basis"] == BASIS_WEAK
        assert result["identity"]["vehicle_id"] is None
        # 证据来自 src.stitching，且候选边/链置信度都是真实计算值
        assert result["evidence"]["source"] == "src.stitching"
        assert result["evidence"]["candidate_edges"] > 0
        assert 0.0 < result["evidence"]["chain_confidence"] <= 1.0

    def test_every_segment_labelled(self, builder):
        result = builder.build(ANCHOR, mode="stitch")
        inferred = [n for n in result["observation_nodes"] if n["basis"] == BASIS_WEAK]
        assert inferred, "弱身份路径至少要有一个概率推断节点"
        for node in inferred:
            assert "概率推断" in node["basis_text"]
        for seg in result["inference_segments"]:
            assert seg["basis"] == BASIS_WEAK
            assert "概率推断" in seg["basis_text"]

    def test_score_detail_present(self, builder):
        """推断段必须带六维评分明细，且各维落在 [0, 1]"""
        result = builder.build(ANCHOR, mode="stitch")
        assert result["inference_segments"]
        for seg in result["inference_segments"]:
            detail = seg["score_detail"]
            for key in ("appearance", "attribute", "plate", "temporal", "spatial", "direction", "penalty"):
                assert 0.0 <= detail[key] <= 1.0, (key, detail[key])

    def test_deterministic(self, builder):
        first = json.dumps(builder.build(ANCHOR, mode="stitch"), sort_keys=True, ensure_ascii=False)
        second = json.dumps(builder.build(ANCHOR, mode="stitch"), sort_keys=True, ensure_ascii=False)
        assert first == second


# ============================================================
# HTTP 接口
# ============================================================

class TestBacktrackApi:
    """回溯接口的字段兼容性与确定性"""

    def test_trace_ok_and_repeatable(self, client):
        payload = {"instance_id": ANCHOR}
        first = client.post("/api/v1/backtrack/trace", json=payload)
        second = client.post("/api/v1/backtrack/trace", json=payload)
        assert first.status_code == 200
        assert first.content == second.content

    def test_trace_required_fields(self, client):
        """原有响应字段一个都不能少"""
        data = client.post("/api/v1/backtrack/trace", json={"instance_id": ANCHOR}).json()
        for key in (
            "query_id", "camera_sequence", "observation_nodes",
            "observation_segments", "inference_segments",
            "candidate_paths", "overall_confidence",
        ):
            assert key in data
        node = data["observation_nodes"][0]
        for key in ("camera_id", "camera_name", "tracklet_id", "timestamp",
                    "latitude", "longitude", "keyframe_path", "confidence"):
            assert key in node

    def test_trace_weak_mode(self, client):
        data = client.post(
            "/api/v1/backtrack/trace", json={"instance_id": ANCHOR, "mode": "stitch"}
        ).json()
        assert data["identity"]["mode"] == "weak"
        assert data["evidence"]["source"] == "src.stitching"

    def test_trace_missing_target_404(self, client):
        resp = client.post("/api/v1/backtrack/trace", json={"instance_id": "NOT_A_REAL_TARGET"})
        assert resp.status_code == 404

    def test_trajectory_ok(self, client):
        resp = client.post("/api/v1/backtrack/trajectory", json={"instance_id": ANCHOR})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        trajectory = body["trajectory"]
        for key in ("track_id", "vehicle_id", "first_appearance", "last_appearance",
                    "total_duration_seconds", "total_cameras", "total_detections",
                    "camera_sequence", "attributes"):
            assert key in trajectory
        camera = trajectory["camera_sequence"][0]
        for key in ("camera_id", "camera_name", "arrival_time", "departure_time",
                    "duration_seconds", "detection_count", "direction", "frames"):
            assert key in camera
        assert trajectory["vehicle_id"] == ANCHOR_VEHICLE
        assert trajectory["total_cameras"] > 1

    def test_trajectory_repeatable(self, client):
        payload = {"instance_id": ANCHOR}
        assert client.post("/api/v1/backtrack/trajectory", json=payload).content == \
            client.post("/api/v1/backtrack/trajectory", json=payload).content

    def test_candidate_path_numeric_fields(self, client):
        """
        前端按 {:.0%} / {:.0f} 直接格式化候选路径的数值字段，不能给 null

        强身份与弱身份两条路径都要满足这个契约。
        """
        for mode in ("auto", "stitch"):
            data = client.post(
                "/api/v1/backtrack/trace", json={"instance_id": ANCHOR, "mode": mode}
            ).json()
            assert data["candidate_paths"], mode
            for path in data["candidate_paths"]:
                for key in ("confidence", "distance_meters", "estimated_time"):
                    assert isinstance(path[key], (int, float)), (mode, key, path[key])

    def test_result_endpoint(self, client):
        """回溯结果可通过 /result/{query_id} 取回"""
        query_id = client.post(
            "/api/v1/backtrack/trace", json={"instance_id": ANCHOR}
        ).json()["query_id"]
        resp = client.get(f"/api/v1/backtrack/result/{query_id}")
        assert resp.status_code == 200
        assert resp.json()["query_id"] == query_id
