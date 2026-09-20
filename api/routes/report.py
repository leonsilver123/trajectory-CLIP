"""
api.routes.report - 研判报告导出 API

把一条「检索 → 确认 → 回溯」会话导出为可留痕的报告文件。

- GET /api/v1/report/{query_id}?fmt=pdf    研判报告 PDF
- GET /api/v1/report/{query_id}?fmt=xlsx   研判报告 Excel

数据来源**只有会话本身**（`SessionStore` 的会话副本，其中已含回溯结果）。
本模块不做任何计算、不推断、不补全 —— 报告里的每个字段都能在会话里找到出处。
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from src.common.logger import get_logger
from src.common.session_store import get_session_store
from src.reporting.report_builder import build_report_payload, render_pdf, render_xlsx

logger = get_logger("api.routes.report")

router = APIRouter()

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@router.get("/{query_id}")
async def export_report(
    query_id: str,
    fmt: str = Query("pdf", pattern="^(pdf|xlsx)$", description="导出格式"),
) -> Response:
    """
    导出指定会话的研判报告

    Args:
        query_id: 会话 ID（检索接口返回的 query_id）
        fmt: 导出格式，pdf 或 xlsx

    Returns:
        报告文件字节流

    Raises:
        404: 会话不存在（已被 LRU 淘汰或 query_id 有误）
        500: 报告渲染失败
    """
    session = get_session_store().get(query_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"会话 {query_id} 不存在或已过期（会话有 LRU 上限，"
                f"长期未访问的历史会话会被淘汰）"
            ),
        )

    try:
        payload = build_report_payload(session)
        content = render_pdf(payload) if fmt == "pdf" else render_xlsx(payload)
    except Exception as e:  # 渲染失败要显式报错，不能返回一个空文件
        logger.error("生成研判报告失败: query_id=%s fmt=%s err=%s", query_id, fmt, e)
        raise HTTPException(status_code=500, detail=f"报告生成失败: {e}") from e

    filename = f"研判报告_{query_id}.{fmt}"
    logger.info("研判报告已生成: query_id=%s fmt=%s bytes=%d", query_id, fmt, len(content))

    return Response(
        content=content,
        media_type=_MEDIA_TYPES[fmt],
        headers={
            # RFC 5987：中文文件名必须用 filename* 才能被浏览器正确解码
            "Content-Disposition": (
                f"attachment; filename=\"{query_id}.{fmt}\"; "
                f"filename*=UTF-8''{quote(filename)}"
            )
        },
    )
