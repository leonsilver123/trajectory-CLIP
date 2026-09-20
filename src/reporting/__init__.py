"""
src.reporting - 研判报告生成

对外只暴露两件事：把会话整理成 payload、把 payload 渲染成文件。
"""

from src.reporting.report_builder import (
    UNAVAILABLE_FIELDS,
    build_report_payload,
    render_pdf,
    render_xlsx,
)

__all__ = [
    "UNAVAILABLE_FIELDS",
    "build_report_payload",
    "render_pdf",
    "render_xlsx",
]
