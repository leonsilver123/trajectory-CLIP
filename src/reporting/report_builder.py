"""
src.reporting - 研判报告生成

把一次「检索 → 确认 → 回溯」会话渲染成可留痕的报告文件（PDF / Excel）。

设计原则（延续项目红线）
------------------------
1. **只用会话里已有的真实数据**。本模块不做任何计算、不推断、不补全 ——
   它拿到的 `session` 是什么就渲染什么。
2. **没有事实来源的项显式标注为"无数据来源"**，而不是留空让人误以为"没有"。
   例如摄像头在线状态在离线视频数据集里根本不存在，报告里要写明这一点。
3. **区分观测段与推断段**。报告必须让读者一眼看出哪一段是拍到的、
   哪一段是推出来的，以及各自置信度 —— 这是本项目相对普通检索系统的核心差异，
   在一份要签字留痕的报告里尤其不能含糊。
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============================================================
# 报告里显式声明"无数据来源"的项
# ============================================================

UNAVAILABLE_FIELDS: Dict[str, str] = {
    "camera_online": "摄像头在线状态 —— 数据集为离线视频，不存在该事实",
    "abnormal_vehicles": "异常车辆 —— 无行为分析模块，系统不产生该结论",
    "pending_review": "待审核轨迹 —— 无审核流程定义",
}

_REPORT_TITLE = "交通风险感知子系统 · 目标轨迹研判报告"


def build_report_payload(
    session: Dict[str, Any],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    把一条会话整理成报告所需的结构

    Args:
        session: `SessionStore` 返回的会话副本（含 query_text / state / backtrack_result）
        generated_at: 生成时间字符串；缺省用当前时间

    Returns:
        报告 payload。**不含任何计算或推断结果** —— 全部字段都来自 session。
    """
    if session is None:
        raise ValueError("session 不能为空")

    result = session.get("backtrack_result") or {}

    return {
        "title": _REPORT_TITLE,
        "query_id": session.get("query_id"),
        "generated_at": generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # ── 检索条件 ──
        "query": {
            "text": session.get("query_text") or "",
            "type": session.get("query_type") or "",
            "target_type": session.get("target_type"),
            "candidate_count": session.get("candidate_count", 0),
        },

        # ── 人工确认结论 ──
        "confirmation": {
            "state": session.get("state"),
            "instance_id": session.get("confirmed_instance_id"),
            "excluded_instance_id": session.get("excluded_instance_id"),
            "suspect_instance_id": session.get("suspect_instance_id"),
        },

        # ── 观测链（全部来自回溯结果，未做二次计算）──
        "chain": {
            "camera_sequence": result.get("camera_sequence") or [],
            "overall_confidence": result.get("overall_confidence"),
            "observation_segments": result.get("observation_segments") or [],
            "inference_segments": result.get("inference_segments") or [],
            "observation_nodes": result.get("observation_nodes") or [],
            "identity": result.get("identity") or {},
            "evidence": result.get("evidence") or {},
        },

        "has_backtrack": bool(result),

        # ── 诚实边界：这些项没有数据来源，报告里必须写明 ──
        "unavailable": UNAVAILABLE_FIELDS,

        "disclaimer": (
            "本报告由「交通风险感知子系统」依据会话记录自动生成。"
            "报告中标注为「推断段」的片段是系统基于时空约束的概率推断，"
            "并非摄像头实际拍摄内容；段间空白因摄像头点状覆盖而不可观测。"
            "所有结论需经人工研判确认后方可作为依据。"
        ),
    }


# ============================================================
# 内部：把 payload 摊平成「标签 / 值」行，供两种渲染共用
# ============================================================


def _fmt(value: Any, fallback: str = "—") -> str:
    """渲染单个值；None 一律显示为 '—'（不显示为 0 或空，避免与"真的是 0"混淆）"""
    if value is None or value == "":
        return fallback
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _summary_rows(payload: Dict[str, Any]) -> List[tuple]:
    """报告概览行（标签, 值）"""
    q = payload["query"]
    c = payload["confirmation"]
    chain = payload["chain"]

    rows = [
        ("报告标题", payload["title"]),
        ("查询 ID", _fmt(payload["query_id"])),
        ("生成时间", _fmt(payload["generated_at"])),
        ("", ""),
        ("【检索条件】", ""),
        ("查询内容", _fmt(q["text"])),
        ("查询类型", _fmt(q["type"])),
        ("候选数量", _fmt(q["candidate_count"])),
        ("", ""),
        ("【人工确认】", ""),
        ("会话状态", _fmt(c["state"])),
        ("确认目标实例", _fmt(c["instance_id"])),
        ("排除实例", _fmt(c["excluded_instance_id"])),
        ("存疑实例", _fmt(c["suspect_instance_id"])),
    ]

    rows += [
        ("", ""),
        ("【跨镜观测链】", ""),
    ]
    if not payload["has_backtrack"]:
        rows.append(("状态", "尚未执行轨迹回溯（会话中无回溯结果）"))
    else:
        rows += [
            ("经过摄像头数", _fmt(len(chain["camera_sequence"]))),
            ("摄像头序列", _fmt(" → ".join(chain["camera_sequence"]))),
            ("整链置信度", _fmt(chain["overall_confidence"])),
            ("观测段数（实拍）", _fmt(len(chain["observation_segments"]))),
            ("推断段数（概率推断）", _fmt(len(chain["inference_segments"]))),
            ("匹配方式", _fmt(chain["identity"].get("mode"))),
        ]

    rows += [("", ""), ("【无数据来源的项（不参与研判）】", "")]
    for key, reason in payload["unavailable"].items():
        rows.append((key, reason))

    return rows


def _segment_rows(payload: Dict[str, Any]) -> List[tuple]:
    """观测段与推断段明细"""
    header = ("类型", "摄像头", "开始", "结束", "置信度/说明")
    rows: List[tuple] = [header]

    for seg in payload["chain"]["observation_segments"]:
        rows.append((
            "观测段（实拍）",
            _fmt(seg.get("camera_id")),
            _fmt(seg.get("start_time")),
            _fmt(seg.get("end_time")),
            _fmt(seg.get("direction")),
        ))

    for seg in payload["chain"]["inference_segments"]:
        rows.append((
            "推断段（概率）",
            f"{_fmt(seg.get('source_camera_id'))} → {_fmt(seg.get('target_camera_id'))}",
            _fmt(seg.get("actual_travel_time")),
            _fmt(seg.get("estimated_travel_time")),
            _fmt(seg.get("confidence")),
        ))

    return rows


# ============================================================
# PDF 渲染
# ============================================================


def render_pdf(payload: Dict[str, Any]) -> bytes:
    """
    渲染 PDF（中文，使用 reportlab 内置 CID 字体 STSong-Light）

    Args:
        payload: `build_report_payload()` 的返回值

    Returns:
        PDF 字节流
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    font = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    except Exception:
        # 重复注册会抛异常，忽略即可
        pass

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left, right = 50, width - 50
    y = height - 60

    def new_page() -> None:
        nonlocal y
        pdf.showPage()
        pdf.setFont(font, 10)
        y = height - 60

    def line(text: str, size: int = 10, indent: int = 0, gap: int = 16) -> None:
        nonlocal y
        if y < 70:
            new_page()
        pdf.setFont(font, size)
        # 简单截断，避免超宽字符跑出页面
        max_chars = int((right - left - indent) / (size * 0.55))
        shown = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
        pdf.drawString(left + indent, y, shown)
        y -= gap

    # 标题
    line(payload["title"], size=16, gap=14)
    pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
    pdf.line(left, y + 4, right, y + 4)
    y -= 12

    # 概览
    for label, value in _summary_rows(payload):
        if not label and not value:
            y -= 6
            continue
        if label.startswith("【"):
            line(label, size=11.5, gap=15)
            continue
        text = f"{label}：{value}" if value else label
        line(text, indent=8)

    # 段明细
    rows = _segment_rows(payload)
    y -= 8
    if y < 120:
        new_page()
    line("【片段明细】", size=11.5, gap=15)
    for i, row in enumerate(rows):
        if i == 0:
            line("  | ".join(str(c) for c in row), size=9.5, gap=14, indent=8)
            continue
        line("  | ".join(str(c) for c in row), size=9, gap=13, indent=8)

    # 免责声明
    y -= 10
    if y < 120:
        new_page()
    line("【说明】", size=11.5, gap=15)
    for chunk in _wrap(payload["disclaimer"], 44):
        line(chunk, size=9.5, gap=13, indent=8)

    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _wrap(text: str, width: int) -> List[str]:
    """按字符数粗暴折行（中文没有空格，不能用 textwrap）"""
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


# ============================================================
# Excel 渲染
# ============================================================


def render_xlsx(payload: Dict[str, Any]) -> bytes:
    """
    渲染 Excel（概览页 + 片段明细页 + 说明页）

    Args:
        payload: `build_report_payload()` 的返回值

    Returns:
        xlsx 字节流
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()

    # ── Sheet 1：概览 ──
    ws = wb.active
    ws.title = "研判概览"
    ws.append(["项目", "内容"])
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)
    for label, value in _summary_rows(payload):
        if not label and not value:
            continue
        ws.append([label, value])
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 72
    for row in ws.iter_rows(min_col=2, max_col=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    # ── Sheet 2：片段明细 ──
    ws2 = wb.create_sheet("片段明细")
    for row in _segment_rows(payload):
        ws2.append(list(row))
    for cell in ws2[1]:
        cell.font = Font(bold=True)
    for col, w in zip("ABCDE", (18, 30, 14, 14, 22)):
        ws2.column_dimensions[col].width = w

    # ── Sheet 3：说明 ──
    ws3 = wb.create_sheet("说明")
    ws3.append(["说明"])
    ws3["A1"].font = Font(bold=True)
    ws3.append([payload["disclaimer"]])
    ws3.column_dimensions["A"].width = 100
    ws3["A2"].alignment = Alignment(wrap_text=True, vertical="top")

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
