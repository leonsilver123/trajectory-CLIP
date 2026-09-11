"""
tests/test_api.py - API 接口测试

测试目标: 验证 FastAPI 各端点的正确性
覆盖:
  - GET /health 健康检查
  - GET /api/v1/dashboard/stats 系统统计
  - GET /api/v1/dashboard/cameras 摄像头列表
  - GET /api/v1/dashboard/health 服务健康
  - POST /api/v1/search/query 文本搜索
  - POST /api/v1/search/plate 车牌搜索
  - POST /api/v1/backtrack/trace 轨迹回溯
  - POST /api/v1/confirm/target 目标确认

注意: 使用 FastAPI TestClient，不需要运行真实服务器
"""

import pytest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from src.common.config import reset_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_config():
    """每个测试前重置全局配置"""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def client():
    """创建 FastAPI TestClient"""
    # 延迟导入以确保配置重置后重新创建 app
    from api.main import create_app
    app = create_app()
    return TestClient(app)


# output/cityflow_results.json 中真实存在的目标（用于正向用例）
REAL_TARGET_ID = "CF3_c001_V0034_000001"
REAL_TRACK_ID = "CF3_TRACK_c001_V0034"


@pytest.fixture
def searched_query_id(client):
    """先跑一次真实检索，拿到后端会话中真实存在的 query_id"""
    resp = client.post("/api/v1/search/query", json={
        "query_text": "黑色轿车",
        "top_k": 3,
    })
    assert resp.status_code == 200
    return resp.json()["query_id"]


# ============================================================
# 健康检查
# ============================================================

class TestHealthCheck:
    """健康检查接口测试"""

    def test_health_endpoint(self, client):
        """测试 GET /health 返回 200"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_version(self, client):
        """测试健康检查返回版本号"""
        resp = client.get("/health")
        data = resp.json()
        assert data["version"] == "1.0.0"


# ============================================================
# Dashboard API
# ============================================================

class TestDashboardAPI:
    """仪表盘 API 测试"""

    def test_stats_endpoint(self, client):
        """测试 GET /api/v1/dashboard/stats 返回 200"""
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "camera_count" in data
        assert "instance_count" in data
        assert "tracklet_count" in data

    def test_stats_has_confidence_distribution(self, client):
        """测试统计数据包含置信度分布"""
        resp = client.get("/api/v1/dashboard/stats")
        data = resp.json()
        assert "confidence_distribution" in data
        assert isinstance(data["confidence_distribution"], list)

    def test_cameras_endpoint(self, client):
        """测试 GET /api/v1/dashboard/cameras 返回 200"""
        resp = client.get("/api/v1/dashboard/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert "cameras" in data
        assert isinstance(data["cameras"], list)

    def test_cameras_have_required_fields(self, client):
        """测试摄像头数据包含必要字段"""
        resp = client.get("/api/v1/dashboard/cameras")
        data = resp.json()
        if data["cameras"]:
            cam = data["cameras"][0]
            assert "camera_id" in cam
            assert "name" in cam

    def test_dashboard_health(self, client):
        """测试 GET /api/v1/dashboard/health"""
        resp = client.get("/api/v1/dashboard/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "version" in data
        assert "device" in data
        assert "results_loaded" in data

    def test_dashboard_health_results_loaded(self, client):
        """测试结果文件加载状态"""
        resp = client.get("/api/v1/dashboard/health")
        data = resp.json()
        results_path = PROJECT_ROOT / "output" / "cityflow_results.json"
        assert data["results_loaded"] == results_path.exists()


# ============================================================
# Search API
# ============================================================

class TestSearchAPI:
    """检索 API 测试"""

    def test_search_query_basic(self, client):
        """测试基本文本搜索"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "车辆",
            "top_k": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "query_id" in data
        assert "candidates" in data
        assert "total_count" in data

    def test_search_query_returns_candidates(self, client):
        """测试搜索返回候选结果"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "车辆",
            "top_k": 5,
        })
        data = resp.json()
        assert data["total_count"] <= 5
        if data["candidates"]:
            c = data["candidates"][0]
            assert "instance_id" in c
            assert "camera_id" in c
            assert "combined_score" in c
            assert "rank" in c

    def test_search_query_with_target_type(self, client):
        """测试带目标类型过滤的搜索"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "",
            "top_k": 20,
            "target_type": "vehicle",
        })
        data = resp.json()
        for c in data["candidates"]:
            assert c["target_type"] == "vehicle"

    def test_search_query_empty_text(self, client):
        """测试空文本搜索（返回所有结果）"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] > 0

    def test_search_plate_basic(self, client):
        """测试车牌搜索"""
        resp = client.post("/api/v1/search/plate", json={
            "plate_number": "NONEXIST_PLATE_XYZ",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "query_id" in data
        # 不存在的车牌应该返回空
        assert data["total_count"] == 0

    def test_search_query_candidates_sorted(self, client):
        """测试搜索结果按分数排序"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "车辆",
            "top_k": 20,
        })
        data = resp.json()
        scores = [c["combined_score"] for c in data["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_search_query_rank_sequential(self, client):
        """测试结果排名连续"""
        resp = client.post("/api/v1/search/query", json={
            "query_text": "",
            "top_k": 10,
        })
        data = resp.json()
        if data["candidates"]:
            ranks = [c["rank"] for c in data["candidates"]]
            assert ranks == list(range(1, len(ranks) + 1))


# ============================================================
# Backtrack API
# ============================================================

class TestBacktrackAPI:
    """回溯 API 测试"""

    def test_backtrack_trace(self, client):
        """测试轨迹回溯接口（真实 target_id）"""
        resp = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
            "max_upstream": 5,
            "max_downstream": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "query_id" in data
        assert "camera_sequence" in data
        assert "observation_nodes" in data
        assert "observation_segments" in data
        assert "inference_segments" in data
        assert "overall_confidence" in data

    def test_backtrack_trace_has_cameras(self, client):
        """测试回溯结果包含真实摄像头序列（非空壳）"""
        resp = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["camera_sequence"]) > 0
        # 观测节点来自真实检测，必须带出真实时间戳与摄像头
        assert len(data["observation_nodes"]) > 0
        assert data["observation_nodes"][0]["camera_id"] == data["camera_sequence"][0]
        assert data["observation_nodes"][0]["timestamp"]

    def test_backtrack_trace_accepts_track_id(self, client):
        """track_id 形式（CF3_TRACK_c001_V0034）同样能定位真实目标"""
        resp = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TRACK_ID,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["camera_sequence"]) > 0
        assert "c001" in data["camera_sequence"]

    def test_backtrack_trace_confidence_range(self, client):
        """测试回溯置信度为 [0,1] 的数值（非 None、非造假随机数）"""
        resp = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
        })
        data = resp.json()
        overall = data["overall_confidence"]
        assert isinstance(overall, float)
        assert 0 <= overall <= 1

    def test_backtrack_trace_deterministic(self, client):
        """同一 instance_id 连续回溯两次，结果必须完全一致（无随机性）"""
        first = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
        })
        second = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
        })
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()

    def test_backtrack_trace_via_session_query_id(self, client, searched_query_id):
        """检索 → 确认 → 回溯 靠 query_id 串联（请求体不需要 instance_id）"""
        confirm = client.post("/api/v1/confirm/target", json={
            "query_id": searched_query_id,
            "instance_id": REAL_TARGET_ID,
        })
        assert confirm.status_code == 200

        resp = client.post("/api/v1/backtrack/trace", json={
            "query_id": searched_query_id,
        })
        assert resp.status_code == 200
        assert len(resp.json()["camera_sequence"]) > 0

    def test_backtrack_trace_nonexistent_instance(self, client):
        """目标不存在时返回 404，不得回落到随机目标或演示数据"""
        resp = client.post("/api/v1/backtrack/trace", json={
            "instance_id": "CF3_c999_V9999_000001",
        })
        assert resp.status_code == 404

    def test_backtrack_trace_without_any_anchor(self, client):
        """既没有 instance_id 也没有会话确认目标时返回 404"""
        resp = client.post("/api/v1/backtrack/trace", json={})
        assert resp.status_code == 404

    def test_backtrack_get_result(self, client):
        """测试获取回溯结果"""
        # 先执行回溯
        resp1 = client.post("/api/v1/backtrack/trace", json={
            "instance_id": REAL_TARGET_ID,
        })
        assert resp1.status_code == 200
        query_id = resp1.json()["query_id"]

        # 再查询结果
        resp2 = client.get(f"/api/v1/backtrack/result/{query_id}")
        assert resp2.status_code == 200

    def test_backtrack_get_nonexistent_result(self, client):
        """测试获取不存在的回溯结果返回 404"""
        resp = client.get("/api/v1/backtrack/result/NONEXISTENT_ID")
        assert resp.status_code == 404


# ============================================================
# Confirm API
# ============================================================

class TestConfirmAPI:
    """确认 API 测试"""

    def test_confirm_target(self, client, searched_query_id):
        """测试目标确认接口（用检索返回的真实 query_id）"""
        resp = client.post("/api/v1/confirm/target", json={
            "query_id": searched_query_id,
            "instance_id": REAL_TARGET_ID,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "confirmed"
        assert data["query_id"] == searched_query_id
        assert data["instance_id"] == REAL_TARGET_ID
        assert "message" in data

    def test_confirm_target_message_content(self, client, searched_query_id):
        """测试确认消息内容"""
        resp = client.post("/api/v1/confirm/target", json={
            "query_id": searched_query_id,
            "instance_id": "INST_XYZ",
        })
        data = resp.json()
        assert "INST_XYZ" in data["message"]

    def test_confirm_nonexistent_query_id(self, client):
        """未经过检索的 query_id 必须返回 404，不能静默确认"""
        resp = client.post("/api/v1/confirm/target", json={
            "query_id": "Q_NOT_EXIST",
            "instance_id": REAL_TARGET_ID,
        })
        assert resp.status_code == 404
