"""
frontend.components - 可复用 UI 组件

包含：
- PhaseNavigator      阶段式导航栏
- ProgressBar         优雅进度条
- LoadingSpinner      加载动画
- render_icon         SVG 图标渲染
- render_data_table   统一表格组件
- render_status_tag   统一状态标签
- render_detail_panel 统一弹窗/面板
- render_empty_state  统一空状态
- render_success_toast / render_error_toast   操作提示
- render_success_message / render_error_message  操作提示（别名）
- render_loading_state                        加载状态
- render_loading_spinner                      加载动画
- render_confirm_dialog                       确认对话框
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Optional

import streamlit as st

# ============================================================
# 本地颜色常量（浅色主题）
# ============================================================

class _C:
    """组件库本地颜色常量，不依赖 frontend.styles"""
    PRIMARY_BLUE   = "#0B4EC2"
    BUTTON_BLUE    = "#1677FF"
    PAGE_BG        = "#F3F6FA"
    PANEL_WHITE    = "#FFFFFF"
    TABLE_HEADER   = "#E8F1FF"
    TABLE_BORDER   = "#D9E2EF"
    TEXT_PRIMARY   = "#1F2D3D"
    TEXT_SECONDARY = "#667085"
    RED_ALERT      = "#E53935"
    ORANGE_WARN    = "#F59E0B"
    GREEN_OK       = "#2E7D32"
    EXCLUDED_GRAY  = "#9AA4B2"
    BORDER_LIGHT   = "#EEF2F7"
    ROW_ALT        = "#F8FAFD"

# ============================================================
# 常量
# ============================================================

_ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

# 阶段定义 — icon 对应 assets/icons/*.svg 文件名
PHASES = [
    {"key": "search",     "label": "目标检索",   "icon": "search"},
    {"key": "confirm",    "label": "目标确认",   "icon": "check-square"},
    {"key": "trajectory", "label": "轨迹回溯",   "icon": "route"},
    {"key": "timeline",   "label": "时间轴回放", "icon": "film"},
    {"key": "export",     "label": "数据导出",   "icon": "file-output"},
]

_PHASE_TO_INDEX = {p["key"]: i for i, p in enumerate(PHASES)}


# ============================================================
# SVG 图标渲染
# ============================================================

# 图标名称 → SVG 文件名的别名映射（兼容旧名称）
_ICON_ALIASES = {
    "confirm": "check-square",
    "map": "route",
    "timeline": "film",
    "export": "file-output",
}

# 简单 SVG 缓存，避免重复磁盘 IO
_icon_cache: dict[str, str] = {}

# 缺失图标的内联 SVG fallback（Feather Icons 风格）
_FALLBACK_SVGS: dict[str, str] = {
    "x-circle": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/>'
        '<line x1="9" y1="9" x2="15" y2="15"/></svg>'
    ),
    "refresh": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'
    ),
    "ban": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "target": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
    ),
    "sun": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>'
        '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/>'
        '<line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>'
        '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
    ),
    "moon": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
    ),
    "circle": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/></svg>'
    ),
    "star": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
    ),
}


def render_icon(icon_name: str, size: int = 18, color: Optional[str] = None) -> str:
    """
    读取 assets/icons/*.svg 并返回内联 HTML。
    如果 SVG 文件不存在，优先使用内置 Feather 风格 fallback；
    若无内置 fallback，返回文字占位符。

    参数:
        icon_name: 图标名称（对应 SVG 文件名，不含扩展名）
        size:      渲染尺寸（px），默认 18
        color:     stroke 颜色，默认 None 保留 currentColor
    """
    # 解析别名
    resolved = _ICON_ALIASES.get(icon_name, icon_name)
    cache_key = f"{resolved}_{size}_{color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    svg_path = _ICONS_DIR / f"{resolved}.svg"
    if svg_path.exists():
        svg_text = svg_path.read_text(encoding="utf-8")
    elif resolved in _FALLBACK_SVGS:
        svg_text = _FALLBACK_SVGS[resolved]
    else:
        # 文字 fallback
        _fallback = {
            "search": "[搜]", "check-square": "[确]", "route": "[地]",
            "film": "[时]", "file-output": "[导]", "dashboard": "[表]",
            "arrow-right": ">", "loading": "[...]",
        }
        return _fallback.get(resolved, f"[{icon_name}]")
    # 替换尺寸
    svg_text = svg_text.replace('width="24"', f'width="{size}"')
    svg_text = svg_text.replace('height="24"', f'height="{size}"')
    # 替换颜色
    if color:
        svg_text = svg_text.replace('stroke="currentColor"', f'stroke="{color}"')
    # 添加 display inline 样式
    svg_text = svg_text.replace(
        "<svg ",
        f'<svg style="vertical-align:middle;display:inline-block;" ',
    )
    _icon_cache[cache_key] = svg_text
    return svg_text


# ============================================================
# PhaseNavigator — 阶段式导航条
# ============================================================

_PHASE_NAV_CSS = """
<style>
.phase-nav-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px 0;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #0B4EC2 0%, #083B91 100%);
    border: 1px solid #083B91;
}
.phase-step {
    display: flex;
    align-items: center;
    padding: 4px 10px;
    font-size: 12px;
    color: #5A6A80;
    transition: all 0.25s ease;
}
.phase-step.active {
    color: #FFFFFF;
    background-color: rgba(26,60,110,0.6);
    border: 1px solid #1A3C6E;
}
.phase-step.done {
    color: #28A745;
}
.phase-step .phase-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px; height: 20px;
    border: 1px solid currentColor;
    font-size: 11px;
    font-weight: 700;
    margin-right: 5px;
}
.phase-step.active .phase-num {
    background-color: #1A3C6E;
    border-color: #FFB800;
    color: #FFB800;
}
.phase-step.done .phase-num {
    background-color: #28A745;
    border-color: #28A745;
    color: #FFFFFF;
}
.phase-arrow {
    color: #3A4A5F;
    font-size: 14px;
    padding: 0 2px;
}
</style>
"""


def render_phase_navigator(current_page: str):
    """渲染阶段式导航条"""
    st.markdown(_PHASE_NAV_CSS, unsafe_allow_html=True)

    # 首页不显示阶段导航
    if current_page == "home":
        return

    current_idx = _PHASE_TO_INDEX.get(current_page, 0)

    # 阶段图标文字映射
    _PHASE_ICON_TEXT = {
        "search": "[搜]", "confirm": "[确]", "trajectory": "[轨]",
        "timeline": "[时]", "export": "[导]",
    }

    steps_html = ""
    for i, phase in enumerate(PHASES):
        icon_html = _PHASE_ICON_TEXT.get(phase["key"], f"[{phase['key']}]")
        if i < current_idx:
            cls = "phase-step done"
            num_content = "[✓]"
        elif i == current_idx:
            cls = "phase-step active"
            num_content = str(i + 1)
        else:
            cls = "phase-step"
            num_content = str(i + 1)

        steps_html += f"""
        <div class="{cls}">
            <span class="phase-num">{num_content}</span>
            {icon_html}&nbsp;{phase['label']}
        </div>
        """
        if i < len(PHASES) - 1:
            steps_html += '<div class="phase-arrow">&rarr;</div>'

    st.markdown(
        f'<div class="phase-nav-wrap">{steps_html}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ProgressBar — 优雅进度条（浅色主题）
# ============================================================

_PROGRESS_CSS = """
<style>
.prog-wrap {
    margin: 8px 0;
}
.prog-label {
    color: #667085;
    font-size: 11px;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
}
.prog-track {
    width: 100%;
    height: 6px;
    background-color: #E8F1FF;
    border-radius: 3px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: linear-gradient(90deg, #0B4EC2 0%, #1677FF 50%, #0B4EC2 100%);
    border-radius: 3px;
    transition: width 0.4s ease;
}
.prog-fill.indeterminate {
    width: 40% !important;
    animation: prog-slide 1.5s ease-in-out infinite;
}
@keyframes prog-slide {
    0%   { margin-left: -40%; }
    100% { margin-left: 100%; }
}
</style>
"""


def render_progress_bar(
    progress: float = -1,
    label: str = "处理中...",
    show_pct: bool = True,
):
    """
    渲染进度条。
    progress: 0.0 ~ 1.0，负数表示 indeterminate 模式。
    """
    st.markdown(_PROGRESS_CSS, unsafe_allow_html=True)

    if progress < 0:
        fill_cls = "prog-fill indeterminate"
        pct_text = ""
        width = "0"
    else:
        fill_cls = "prog-fill"
        pct = int(progress * 100)
        pct_text = f"{pct}%" if show_pct else ""
        width = f"{pct}%"

    st.markdown(f"""
    <div class="prog-wrap">
        <div class="prog-label">
            <span>{label}</span>
            <span>{pct_text}</span>
        </div>
        <div class="prog-track">
            <div class="{fill_cls}" style="width:{width};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LoadingSpinner — 加载动画（浅色主题）
# ============================================================

_SPINNER_CSS = """
<style>
.spinner-overlay {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px 0;
    gap: 10px;
}
.spinner-ring {
    width: 28px; height: 28px;
    border: 3px solid #E8F1FF;
    border-top-color: #0B4EC2;
    animation: spin-anim 0.8s linear infinite;
}
@keyframes spin-anim {
    to { transform: rotate(360deg); }
}
.spinner-text {
    color: #667085;
    font-size: 12px;
}
</style>
"""


def render_loading_spinner(message: str = "加载中..."):
    """渲染加载动画"""
    st.markdown(_SPINNER_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="spinner-overlay">
        <div class="spinner-ring"></div>
        <span class="spinner-text">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# render_status_tag — 统一状态标签
# ============================================================

_STATUS_TAG_MAP = {
    # 中文状态
    "待确认": {"bg": _C.RED_ALERT,    "color": "#FFFFFF", "border": "none"},
    "已确认": {"bg": _C.GREEN_OK,     "color": "#FFFFFF", "border": "none"},
    "疑似":   {"bg": _C.ORANGE_WARN,  "color": "#FFFFFF", "border": "none"},
    "已排除": {"bg": "#E8ECF0",       "color": _C.EXCLUDED_GRAY, "border": "none"},
    "低置信": {"bg": "transparent",   "color": _C.RED_ALERT, "border": f"1px solid {_C.RED_ALERT}"},
    "已导出": {"bg": _C.BUTTON_BLUE,  "color": "#FFFFFF", "border": "none"},
    # 英文状态别名
    "pending":        {"bg": _C.RED_ALERT,    "color": "#FFFFFF", "border": "none"},
    "confirmed":      {"bg": _C.GREEN_OK,     "color": "#FFFFFF", "border": "none"},
    "suspected":      {"bg": _C.ORANGE_WARN,  "color": "#FFFFFF", "border": "none"},
    "excluded":       {"bg": "#E8ECF0",       "color": _C.EXCLUDED_GRAY, "border": "none"},
    "low_confidence": {"bg": "transparent",   "color": _C.RED_ALERT, "border": f"1px solid {_C.RED_ALERT}"},
    "exported":       {"bg": _C.BUTTON_BLUE,  "color": "#FFFFFF", "border": "none"},
}


def render_status_tag(status: str) -> str:
    """返回 HTML 状态标签。

    支持的 status 值:
        待确认 / 已确认 / 疑似 / 已排除 / 低置信 / 已导出

    未识别的状态会以灰色标签展示原始文字。
    """
    cfg = _STATUS_TAG_MAP.get(status)
    if cfg is None:
        # 未知状态：灰色 fallback
        return (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;'
            f'font-size:12px;font-weight:500;line-height:18px;'
            f'background-color:#E8ECF0;color:#667085;">{status}</span>'
        )
    bg = cfg["bg"]
    c = cfg["color"]
    border = cfg["border"]
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;'
        f'font-size:12px;font-weight:500;line-height:18px;'
        f'background-color:{bg};color:{c};border:{border};">{status}</span>'
    )


# ============================================================
# render_data_table — 统一表格组件
# ============================================================

_DATA_TABLE_CSS = """
<style>
.dt-wrap {
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    overflow: hidden;
    background: #FFFFFF;
    margin: 8px 0;
}
.dt-title {
    padding: 10px 14px;
    font-size: 14px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    background: #FFFFFF;
}
.dt-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}
.dt-table thead th {
    background-color: #E8F1FF;
    color: #0B4EC2;
    font-size: 12px;
    font-weight: 600;
    padding: 0 12px;
    height: 36px;
    line-height: 36px;
    border-bottom: 1px solid #D9E2EF;
    text-align: left;
    white-space: nowrap;
    position: relative;
}
.dt-table thead th.dt-sortable {
    cursor: pointer;
    user-select: none;
}
.dt-table thead th.dt-sortable:hover {
    background-color: #D6E6FF;
}
.dt-sort-icon {
    margin-left: 4px;
    font-size: 11px;
    color: #0B4EC2;
    vertical-align: middle;
}
.dt-table tbody td {
    padding: 0 12px;
    height: 38px;
    line-height: 38px;
    font-size: 13px;
    color: #1F2D3D;
    border-bottom: 1px solid #EEF2F7;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dt-table tbody tr:nth-child(even) {
    background-color: #F8FAFD;
}
.dt-table tbody tr:hover {
    background-color: #E8F1FF;
    transition: background-color 0.15s;
}
.dt-table tbody tr.dt-clickable {
    cursor: pointer;
}
.dt-table thead th.dt-actions-col,
.dt-table tbody td.dt-actions-cell {
    position: sticky;
    right: 0;
    background: inherit;
    z-index: 1;
    border-left: 1px solid #D9E2EF;
    width: 120px;
    text-align: center;
}
.dt-table thead th.dt-actions-col {
    background-color: #E8F1FF;
}
.dt-action-btn {
    display: inline-block;
    padding: 2px 8px;
    margin: 0 2px;
    border-radius: 3px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    line-height: 20px;
}
.dt-action-primary {
    background-color: #1677FF;
    color: #FFFFFF;
}
.dt-action-primary:hover {
    background-color: #0B4EC2;
}
.dt-action-default {
    background-color: #FFFFFF;
    color: #0B4EC2;
    border: 1px solid #0B4EC2;
}
.dt-action-default:hover {
    background-color: #E8F1FF;
}
.dt-empty {
    text-align: center;
    padding: 40px 0;
    color: #667085;
    font-size: 13px;
}
.dt-empty-icon {
    font-size: 32px;
    color: #D9E2EF;
    margin-bottom: 8px;
}
</style>
"""

# 用于状态标签自动识别的列 key 集合
_STATUS_COLUMN_KEYS = {"status", "状态", "state", "confirm_status", "确认状态"}


def render_data_table(
    data: list[dict],
    columns: list[dict],
    title: str = "",
    empty_text: str = "暂无数据",
    on_row_click: callable = None,
    actions: list[dict] = None,
    sort_key: str = None,
    sort_order: str = "asc",
):
    """渲染统一业务表格。

    Parameters
    ----------
    data : list[dict]
        行数据列表，每行 dict 的 key 对应 columns 中的 key。
    columns : list[dict]
        列定义。每项: {"key": str, "label": str, "width": str, "sortable": bool}
    title : str
        表格标题，为空则不渲染标题栏。
    empty_text : str
        空数据时的提示文字。
    on_row_click : callable, optional
        行点击回调，接收行 dict 参数（当前仅标记行可点击样式）。
    actions : list[dict], optional
        操作列按钮定义: [{"label": "查看", "key": "view", "color": "primary"}]
        color 可选 "primary" / "default"。
    sort_key : str, optional
        当前排序列的 key，用于显示排序箭头。
    sort_order : str
        排序方向 "asc" / "desc"。
    """
    st.markdown(_DATA_TABLE_CSS, unsafe_allow_html=True)

    # ── 标题 ──
    title_html = f'<div class="dt-title">{title}</div>' if title else ""

    # ── 空状态 ──
    if not data:
        st.markdown(f"""
        <div class="dt-wrap">
            {title_html}
            <div class="dt-empty">
                <div class="dt-empty-icon">[文件]</div>
                <div>{empty_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── 表头 ──
    head_cells = ""
    for col in columns:
        w = f' style="width:{col["width"]}"' if col.get("width") else ""
        sortable = col.get("sortable", False)
        cls = "dt-sortable" if sortable else ""
        sort_icon = ""
        if sortable and sort_key == col["key"]:
            sort_icon = " ↑" if sort_order == "asc" else " ↓"
        head_cells += f'<th class="{cls}"{w}>{col["label"]}<span class="dt-sort-icon">{sort_icon}</span></th>'

    # 操作列
    if actions:
        head_cells += '<th class="dt-actions-col" style="width:120px">操作</th>'

    # ── 表体 ──
    body_rows = ""
    for row in data:
        clickable_cls = "dt-clickable" if on_row_click else ""
        cells = ""
        for col in columns:
            val = row.get(col["key"], "")
            safe_val = html.escape(str(val))
            # 自动识别状态列 → 渲染状态标签
            if col["key"] in _STATUS_COLUMN_KEYS and isinstance(val, str):
                cell_html = render_status_tag(val)
            else:
                cell_html = safe_val
            cells += f'<td title="{safe_val}">{cell_html}</td>'

        # 操作列
        if actions:
            btns = ""
            for act in actions:
                color_cls = "dt-action-primary" if act.get("color") == "primary" else "dt-action-default"
                btns += f'<span class="dt-action-btn {color_cls}">{act["label"]}</span>'
            cells += f'<td class="dt-actions-cell">{btns}</td>'

        body_rows += f'<tr class="{clickable_cls}">{cells}</tr>'

    st.markdown(f"""
    <div class="dt-wrap">
        {title_html}
        <table class="dt-table">
            <thead><tr>{head_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# render_detail_panel — 统一弹窗/面板
# ============================================================

_DETAIL_PANEL_CSS = """
<style>
.dp-wrap {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 8px 0;
    overflow: hidden;
}
.dp-header {
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    background: #F8FAFD;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dp-body {
    padding: 16px;
    color: #1F2D3D;
    font-size: 13px;
    line-height: 1.6;
}
.dp-img {
    max-width: 100%;
    border-radius: 4px;
    margin-bottom: 12px;
    border: 1px solid #D9E2EF;
}
.dp-footer {
    padding: 10px 16px;
    border-top: 1px solid #D9E2EF;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #F8FAFD;
}
.dp-footer-left {
    display: flex;
    gap: 8px;
}
.dp-footer-right {
    display: flex;
    gap: 8px;
}
.dp-btn {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid transparent;
    line-height: 20px;
}
.dp-btn-primary {
    background-color: #1677FF;
    color: #FFFFFF;
    border-color: #1677FF;
}
.dp-btn-primary:hover {
    background-color: #0B4EC2;
}
.dp-btn-secondary {
    background-color: #FFFFFF;
    color: #0B4EC2;
    border-color: #0B4EC2;
}
.dp-btn-secondary:hover {
    background-color: #E8F1FF;
}
</style>
"""


def render_detail_panel(
    title: str,
    content_html: str,
    actions: list[dict] = None,
    image_path: str = None,
):
    """渲染统一详情面板。

    Parameters
    ----------
    title : str
        面板标题。
    content_html : str
        内容区 HTML。
    actions : list[dict], optional
        按钮列表。每项: {"label": str, "key": str, "type": "primary"|"secondary"}
        type 为 "primary" 的按钮放右侧，其余放左侧。
    image_path : str, optional
        可选图片路径（相对或绝对），在内容区顶部渲染。
    """
    st.markdown(_DETAIL_PANEL_CSS, unsafe_allow_html=True)

    # 图片
    img_html = ""
    if image_path:
        img_html = f'<img class="dp-img" src="{image_path}" alt="detail image" />'

    # 按钮分区
    left_btns = ""
    right_btns = ""
    if actions:
        for act in actions:
            btn_type = act.get("type", "secondary")
            cls = "dp-btn-primary" if btn_type == "primary" else "dp-btn-secondary"
            btn_html = f'<span class="dp-btn {cls}">{act["label"]}</span>'
            if btn_type == "primary":
                right_btns += btn_html
            else:
                left_btns += btn_html

    footer_html = ""
    if actions:
        footer_html = f"""
        <div class="dp-footer">
            <div class="dp-footer-left">{left_btns}</div>
            <div class="dp-footer-right">{right_btns}</div>
        </div>
        """

    st.markdown(f"""
    <div class="dp-wrap">
        <div class="dp-header">
            <span>{title}</span>
        </div>
        <div class="dp-body">
            {img_html}
            {content_html}
        </div>
        {footer_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# render_empty_state — 统一空状态
# ============================================================

_EMPTY_STATE_CSS = """
<style>
.es-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
}
.es-icon {
    font-size: 48px;
    color: #D9E2EF;
    margin-bottom: 12px;
    line-height: 1;
}
.es-message {
    color: #667085;
    font-size: 14px;
    line-height: 1.6;
    max-width: 400px;
    margin-bottom: 16px;
}
.es-action-btn {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    background-color: #1677FF;
    color: #FFFFFF;
    border: none;
    line-height: 20px;
}
.es-action-btn:hover {
    background-color: #0B4EC2;
}
</style>
"""


def render_empty_state(
    message: str,
    action_label: str = None,
    action_key: str = None,
):
    """渲染统一空状态。

    Parameters
    ----------
    message : str
        居中显示的说明文字。
    action_label : str, optional
        可选操作按钮文字（如"前往检索"）。
    action_key : str, optional
        按钮的 Streamlit key。
    """
    st.markdown(_EMPTY_STATE_CSS, unsafe_allow_html=True)

    btn_html = ""
    if action_label:
        btn_html = f'<span class="es-action-btn">{action_label}</span>'

    st.markdown(f"""
    <div class="es-wrap">
        <div class="es-icon">[搜索]</div>
        <div class="es-message">{message}</div>
        {btn_html}
    </div>
    """, unsafe_allow_html=True)

    # 如果提供了 action_key，用 Streamlit 按钮覆盖（让调用方可以检测点击）
    if action_label and action_key:
        if st.button(action_label, key=action_key):
            return True
    return False


# ============================================================
# render_success_toast — 成功操作提示（绿色）
# ============================================================

_TOAST_CSS = """
<style>
.toast-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    margin: 6px 0;
    animation: toast-fadein 0.3s ease;
}
@keyframes toast-fadein {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.toast-success {
    background-color: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #A5D6A7;
}
.toast-error {
    background-color: #FFEBEE;
    color: #E53935;
    border: 1px solid #EF9A9A;
}
.toast-icon {
    font-size: 16px;
    flex-shrink: 0;
}
</style>
"""


def render_success_toast(message: str):
    """成功操作提示（绿色）"""
    st.markdown(_TOAST_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="toast-wrap toast-success">
        <span class="toast-icon">[✓]</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# render_error_toast — 错误提示（红色）
# ============================================================


def render_error_toast(message: str):
    """错误提示（红色）"""
    st.markdown(_TOAST_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="toast-wrap toast-error">
        <span class="toast-icon">[✗]</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 交互反馈提示 — 别名函数
# ============================================================


def render_success_message(message: str):
    """成功操作提示（绿色背景提示条），render_success_toast 的别名。"""
    render_success_toast(message)


def render_error_message(message: str):
    """错误操作提示（红色背景提示条），render_error_toast 的别名。"""
    render_error_toast(message)


# ============================================================
# render_loading_state — 加载状态
# ============================================================

_LOADING_STATE_CSS = """
<style>
.ls-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px 0;
    gap: 12px;
}
.ls-ring {
    width: 24px; height: 24px;
    border: 3px solid #E8F1FF;
    border-top-color: #0B4EC2;
    border-radius: 50%;
    animation: spin-anim 0.8s linear infinite;
}
@keyframes spin-anim {
    to { transform: rotate(360deg); }
}
.ls-text {
    color: #667085;
    font-size: 13px;
}
</style>
"""


def render_loading_state(message: str = "处理中..."):
    """加载状态"""
    st.markdown(_LOADING_STATE_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ls-wrap">
        <div class="ls-ring"></div>
        <span class="ls-text">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# render_confirm_dialog — 确认对话框
# ============================================================

_CONFIRM_DIALOG_CSS = """
<style>
.cd-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.35);
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: cd-fadein 0.2s ease;
}
@keyframes cd-fadein {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.cd-box {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    min-width: 360px;
    max-width: 480px;
    overflow: hidden;
}
.cd-header {
    padding: 14px 20px 10px;
    font-size: 15px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #EEF2F7;
}
.cd-body {
    padding: 16px 20px;
    font-size: 13px;
    color: #667085;
    line-height: 1.6;
}
.cd-footer {
    padding: 10px 20px 14px;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    border-top: 1px solid #EEF2F7;
}
.cd-btn {
    display: inline-block;
    padding: 5px 18px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid transparent;
    line-height: 20px;
}
.cd-btn-cancel {
    background-color: #FFFFFF;
    color: #667085;
    border-color: #D9E2EF;
}
.cd-btn-cancel:hover {
    background-color: #F3F6FA;
}
.cd-btn-confirm {
    background-color: #1677FF;
    color: #FFFFFF;
    border-color: #1677FF;
}
.cd-btn-confirm:hover {
    background-color: #0B4EC2;
}
</style>
"""


def render_confirm_dialog(
    title: str,
    message: str,
    confirm_label: str = "确定",
):
    """渲染确认对话框 HTML。

    注意：纯 HTML 展示，按钮不具备 Streamlit 交互。
    如需交互确认，建议结合 st.dialog 或 st.session_state 控制。

    Parameters
    ----------
    title : str
        对话框标题。
    message : str
        对话框正文。
    confirm_label : str
        确认按钮文字。
    """
    st.markdown(_CONFIRM_DIALOG_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cd-overlay">
        <div class="cd-box">
            <div class="cd-header">{title}</div>
            <div class="cd-body">{message}</div>
            <div class="cd-footer">
                <span class="cd-btn cd-btn-cancel">取消</span>
                <span class="cd-btn cd-btn-confirm">{confirm_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
