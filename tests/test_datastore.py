"""
tests.test_datastore - 数据契约落盘与统一读取层的回归测试

覆盖点:
1. datastore 还原结果与 cityflow_results.json 逐字段等价（若已构建）
2. datastore 缺失时自动回退 JSON 直读，行为与改造前一致（叠加式，不是替换）
3. 散落的 json.load(cityflow_results.json) 已收敛到 src.storage.datastore 一处
4. 轻量统计接口 get_stats / get_summary 与真实数据一致
5. 接口层（search / dashboard / builder）都改走统一读取层

说明：本测试**不要求**先跑 scripts/build_datastore.py——datastore 不存在时相关用例自动跳过，
回退路径的用例反而必须照常通过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage import datastore as ds

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_JSON = _PROJECT_ROOT / "output" / "cityflow_results.json"

_has_datastore = ds.datastore_available()
_has_json = _RESULTS_JSON.is_file()

requires_datastore = pytest.mark.skipif(not _has_datastore, reason="output/datastore 未构建")
requires_json = pytest.mark.skipif(not _has_json, reason="cityflow_results.json 不存在")


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个用例前后都清缓存，避免用例之间互相影响"""
    ds.reset_cache()
    yield
    ds.reset_cache()


# ============================================================
# 1. datastore 与源 JSON 等价
# ============================================================

@pytest.fixture(scope="module")
def both():
    """(datastore 还原结果, 源 JSON) —— 大文件只读一次"""
    if not (_has_datastore and _has_json):
        pytest.skip("datastore 或 cityflow_results.json 不存在")
    from src.storage.datastore import read_json_file

    rebuilt = ds._read_datastore(ds.datastore_dir())
    assert rebuilt is not None, "datastore 还原失败"
    return rebuilt, read_json_file(str(_RESULTS_JSON))


@requires_datastore
@requires_json
class TestDatastoreEquivalence:
    """datastore 必须与源 JSON 逐字段等价（否则就是换了套数据）"""

    def test_detections_equal(self, both):
        rebuilt, source = both
        assert len(rebuilt["detections"]) == len(source["detections"])
        assert rebuilt["detections"] == source["detections"]

    def test_tracks_equal(self, both):
        rebuilt, source = both
        assert rebuilt["tracks"] == source["tracks"]

    def test_det_to_track_map_equal(self, both):
        rebuilt, source = both
        assert rebuilt["det_to_track_map"] == source["det_to_track_map"]

    def test_summary_equal(self, both):
        rebuilt, source = both
        assert rebuilt["summary"] == source["summary"]

    def test_top_level_keys(self, both):
        rebuilt, _ = both
        assert set(rebuilt.keys()) == {"detections", "tracks", "summary", "det_to_track_map"}

    def test_sparse_key_presence_preserved(self, both):
        """源 JSON 里"根本没有某个键"与"键值为 null"是两回事，还原时必须保住这个区别"""
        rebuilt, source = both
        for row, src_row in zip(rebuilt["detections"], source["detections"]):
            assert set(row.keys()) == set(src_row.keys())


# ============================================================
# 2. 回退路径（叠加式设计的核心保证）
# ============================================================

class TestJsonFallback:
    """datastore 不可用时必须回退 JSON，且结果与之前完全一致"""

    def test_missing_datastore_dir_is_not_available(self, tmp_path):
        assert ds.datastore_available(tmp_path / "not_here") is False

    def test_corrupt_datastore_falls_back(self, tmp_path):
        """目录在但文件是坏的 → 视为不可用，不能抛异常"""
        corrupt = tmp_path / "datastore"
        corrupt.mkdir()
        (corrupt / ds.META_SQLITE).write_text("not a sqlite file", encoding="utf-8")
        (corrupt / ds.DETECTIONS_PARQUET).write_bytes(b"junk")
        (corrupt / ds.TRACKS_PARQUET).write_bytes(b"junk")
        assert ds.datastore_available(corrupt) is False

    def test_schema_version_mismatch_falls_back(self, tmp_path):
        """schema 版本不匹配 → 回退而非报错"""
        import sqlite3

        mismatched = tmp_path / "datastore"
        mismatched.mkdir()
        for name in (ds.DETECTIONS_PARQUET, ds.TRACKS_PARQUET):
            (mismatched / name).write_bytes(b"")
        conn = sqlite3.connect(str(mismatched / ds.META_SQLITE))
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (ds.SCHEMA_VERSION + 1,))
        conn.commit()
        conn.close()
        assert ds.datastore_available(mismatched) is False

    def test_small_json_roundtrip(self, tmp_path):
        """自定义路径：datastore 不存在时直读 JSON（小样本，不走大数据）"""
        payload = {
            "detections": [{"target_id": "CF3_c001_V0034_000001", "camera_id": "c001"}],
            "tracks": [{"track_id": "CF3_TRACK_c001_V0034"}],
            "summary": {"total_detections": 1},
            "det_to_track_map": {"CF3_c001_V0034_000001": "CF3_TRACK_c001_V0034"},
        }
        path = tmp_path / "cityflow_results.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert ds.datastore_available(tmp_path / "datastore") is False
        loaded = ds.load_results(str(path))
        assert loaded == payload

    def test_missing_everything_returns_none(self, tmp_path):
        """JSON 与 datastore 都没有 → 返回 None（调用方按原有语义处理），不抛异常"""
        assert ds.load_results(str(tmp_path / "nope.json")) is None
        assert ds.get_stats(str(tmp_path / "nope.json"))["source"] == "none"

    @requires_json
    def test_default_path_falls_back_to_json(self, monkeypatch):
        """默认路径下把 datastore 指到不存在的位置 → 必须回退 JSON 且数据完整"""
        monkeypatch.setattr(ds, "_DEFAULT_DATASTORE_DIR", Path("Z:/definitely/not/here"))
        ds.reset_cache()
        assert ds.data_source() == "json"
        data = ds.load_results()
        assert data is not None
        assert len(data["detections"]) > 0

    @requires_datastore
    def test_data_source_reports_datastore(self):
        assert ds.data_source() == "datastore"


# ============================================================
# 3. 读取点收敛
# ============================================================

class TestConvergence:
    """散落的 JSON 读取必须收敛到 src/storage/datastore.py 一处"""

    # frontend/（Streamlit）已于 2026-09-21 下线，其读取点随之消失；
    # 幸存的前端 webapp/ 是 TypeScript，不直接读数据文件（只调后端接口）。
    _SCOPES = ("api", "src")

    def _iter_py(self):
        for scope in self._SCOPES:
            for path in (_PROJECT_ROOT / scope).rglob("*.py"):
                if "__pycache__" not in path.parts:
                    yield path

    def test_only_datastore_opens_cityflow_json(self):
        """全仓（api/src）只有 src/storage/datastore.py 会打开 cityflow_results.json"""
        offenders = []
        for path in self._iter_py():
            if path.name == "datastore.py" and path.parent.name == "storage":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if "open(" in line and "cityflow_results" in line:
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{lineno}")
                if "json.load(" in line and "cityflow" in line:
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{lineno}")
        assert offenders == [], f"仍存在散落的 JSON 读取点: {offenders}"

    def test_consumers_use_datastore(self):
        """原先 5 处读取点都已改走统一读取层"""
        checks = {
            "api/routes/search.py": "from src.storage.datastore import",
            "api/routes/dashboard.py": "from src.storage.datastore import",
            "src/trajectory/builder.py": "from src.storage.datastore import",
        }
        for rel, marker in checks.items():
            text = (_PROJECT_ROOT / rel).read_text(encoding="utf-8")
            assert marker in text, f"{rel} 未改走统一读取层"

    def test_datastore_module_is_single_json_loader(self):
        """回退分支里确实存在唯一的 json.load"""
        text = (_PROJECT_ROOT / "src/storage/datastore.py").read_text(encoding="utf-8")
        assert "json.load(f)" in text


# ============================================================
# 4. 轻量统计
# ============================================================

class TestStats:
    """get_stats / get_summary 与真实数据一致"""

    @requires_json
    def test_stats_match_json(self):
        """无论走哪条路径，统计口径必须一致（检测数/轨迹数/摄像头数）"""
        from src.storage.datastore import read_json_file

        source = read_json_file(str(_RESULTS_JSON))
        detections = source["detections"]
        expected = {
            "detections": len(detections),
            "tracks": len(source["tracks"]),
            "cameras": len({d.get("camera_id", "") for d in detections if d.get("camera_id")}),
        }
        stats = ds.get_stats()
        assert stats["detections"] == expected["detections"]
        assert stats["tracks"] == expected["tracks"]
        assert stats["cameras"] == expected["cameras"]

    @requires_json
    def test_summary_matches_json(self):
        from src.storage.datastore import read_json_file

        source = read_json_file(str(_RESULTS_JSON))
        assert ds.get_summary() == source["summary"]

    def test_stats_never_raises_without_data(self, tmp_path):
        stats = ds.get_stats(str(tmp_path / "none.json"))
        assert stats == {"detections": 0, "tracks": 0, "cameras": 0, "source": "none"}


# ============================================================
# 5. 接口层仍走原有字段（不回归）
# ============================================================

@pytest.fixture(scope="module")
def client():
    """FastAPI 测试客户端（不启真实端口）"""
    if not _has_json:
        pytest.skip("cityflow_results.json 不存在")
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@requires_json
class TestApiStillWorks:
    """接口层字段不因数据来源切换而变化"""

    def test_search_query_ok(self, client):
        resp = client.post("/api/v1/search/query", json={"query_text": "黑色轿车", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"query_id", "candidates", "total_count"}
        assert body["total_count"] == len(body["candidates"])

    def test_dashboard_stats_ok(self, client):
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("camera_count", "instance_count", "tracklet_count", "confidence_distribution"):
            assert key in body

    def test_dashboard_health_reports_source(self, client):
        resp = client.get("/api/v1/dashboard/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results_loaded"] is True
        assert body["data_source"] in ("datastore", "json")

    def test_builder_data_available(self):
        from src.trajectory.builder import get_trajectory_builder

        assert get_trajectory_builder().data_available is True
