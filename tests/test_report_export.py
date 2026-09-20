"""
tests.test_report_export - 研判报告导出（PLAN3-A2）

背景：`frontend/pages/confirm.py` 的「导出确认报告」按钮此前是
`st.info("导出功能预留接口")` —— 一个占位。而 `RESUME_PROJECT.md` 声称
"研判报告支持 PDF/Excel 格式导出，满足公安留痕存档要求"。
现在后端 `api/routes/report.py` + `src/reporting/report_builder.py` 把它做出来了。

本文件守住三条不变量（都来自项目红线）：
1. 报告**只用会话里已有的数据**，不计算、不推断、不补全；
2. **没有回溯结果时不得编造观测链** —— 报告要如实说"尚未执行回溯"；
3. 推断段必须与观测段**明确区分**，且报告里带免责说明。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.common.session_store import get_session_store, reset_session_store
from src.reporting.report_builder import (
    UNAVAILABLE_FIELDS,
    build_report_payload,
    render_pdf,
    render_xlsx,
)

_QID = "Q_RPT_TEST"

_STITCHED_RESULT = {
    "query_id": _QID,
    "camera_sequence": ["c002", "c004", "c005"],
    "overall_confidence": 0.89,
    "identity": {"mode": "strong"},
    "observation_segments": [
        {"camera_id": "c002", "start_time": "00:01:02", "end_time": "00:01:11",
         "direction": "由南向北"},
    ],
    "inference_segments": [
        {"source_camera_id": "c002", "target_camera_id": "c004",
         "actual_travel_time": 27.0, "estimated_travel_time": 41.0, "confidence": 0.86},
    ],
    "observation_nodes": [],
    "evidence": {"source": "src.stitching"},
}


@pytest.fixture
def session_searched():
    """只有检索、尚未回溯的会话"""
    reset_session_store()
    store = get_session_store()
    store.create(_QID, "黑色轿车", candidate_count=6)
    yield store
    reset_session_store()


@pytest.fixture
def session_backtracked():
    """走完 检索 → 确认 → 回溯 的会话"""
    reset_session_store()
    store = get_session_store()
    store.create(_QID, "黑色轿车", candidate_count=6)
    store.confirm(_QID, "CF3_c001_V0034_000001")
    store.set_backtracked(_QID, _STITCHED_RESULT)
    yield store
    reset_session_store()


# ============================================================
# payload：只用真实数据
# ============================================================


class TestPayloadUsesOnlySessionData:
    def test_payload_echoes_session_fields(self, session_searched):
        sess = session_searched.get(_QID)
        payload = build_report_payload(sess)

        assert payload["query_id"] == _QID
        assert payload["query"]["text"] == "黑色轿车"
        assert payload["query"]["candidate_count"] == 6
        assert payload["confirmation"]["state"] == "searched"

    def test_no_backtrack_does_not_fabricate_chain(self, session_searched):
        """尚未回溯时，报告不得凭空造出摄像头序列或置信度"""
        payload = build_report_payload(session_searched.get(_QID))

        assert payload["has_backtrack"] is False
        assert payload["chain"]["camera_sequence"] == []
        assert payload["chain"]["overall_confidence"] is None
        assert payload["chain"]["observation_segments"] == []
        assert payload["chain"]["inference_segments"] == []

    def test_backtracked_payload_carries_real_chain(self, session_backtracked):
        payload = build_report_payload(session_backtracked.get(_QID))

        assert payload["has_backtrack"] is True
        assert payload["chain"]["camera_sequence"] == ["c002", "c004", "c005"]
        assert payload["chain"]["overall_confidence"] == pytest.approx(0.89)
        assert len(payload["chain"]["observation_segments"]) == 1
        assert len(payload["chain"]["inference_segments"]) == 1

    def test_unavailable_fields_are_declared(self, session_searched):
        """没有数据来源的项必须在报告里显式声明，而不是留空"""
        payload = build_report_payload(session_searched.get(_QID))
        assert payload["unavailable"] == UNAVAILABLE_FIELDS
        assert "camera_online" in payload["unavailable"]

    def test_disclaimer_mentions_inference(self, session_searched):
        """免责说明必须点明"推断段不是实拍" —— 这是报告可用于留痕的前提"""
        payload = build_report_payload(session_searched.get(_QID))
        assert "推断" in payload["disclaimer"]
        assert "不可观测" in payload["disclaimer"]

    def test_empty_session_raises(self):
        with pytest.raises(ValueError):
            build_report_payload(None)


# ============================================================
# 渲染：两种格式都是真文件
# ============================================================


class TestRendering:
    def test_pdf_has_pdf_magic(self, session_backtracked):
        pdf = render_pdf(build_report_payload(session_backtracked.get(_QID)))
        assert pdf[:4] == b"%PDF"
        assert len(pdf) > 1000, "PDF 太小，疑似渲染失败"

    def test_xlsx_has_zip_magic(self, session_backtracked):
        xlsx = render_xlsx(build_report_payload(session_backtracked.get(_QID)))
        assert xlsx[:4] == b"PK\x03\x04", "xlsx 应为 zip 容器"
        assert len(xlsx) > 1000

    def test_renders_without_backtrack_too(self, session_searched):
        """没回溯也要能出报告（如实写"尚未执行"），而不是报错"""
        payload = build_report_payload(session_searched.get(_QID))
        assert render_pdf(payload)[:4] == b"%PDF"
        assert render_xlsx(payload)[:4] == b"PK\x03\x04"


# ============================================================
# 接口
# ============================================================


class TestReportEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    @pytest.mark.parametrize("fmt,magic", [("pdf", b"%PDF"), ("xlsx", b"PK\x03\x04")])
    def test_export_returns_file(self, client, session_backtracked, fmt, magic):
        resp = client.get(f"/api/v1/report/{_QID}", params={"fmt": fmt})
        assert resp.status_code == 200
        assert resp.content[:4] == magic
        assert "attachment" in resp.headers["content-disposition"]

    def test_missing_session_returns_404(self, client, session_searched):
        resp = client.get("/api/v1/report/NOT_EXIST", params={"fmt": "pdf"})
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_invalid_format_is_rejected(self, client, session_searched):
        """格式白名单：不支持的 fmt 应被 422 拒绝，而不是落回默认格式"""
        resp = client.get(f"/api/v1/report/{_QID}", params={"fmt": "docx"})
        assert resp.status_code == 422
