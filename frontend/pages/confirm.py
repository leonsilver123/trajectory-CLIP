"""
frontend.pages.confirm - 目标复核台（浅色主题）

三栏式布局：左侧原始抓拍图 / 中间候选目标列表 / 右侧属性与确认操作
底部展示确认记录和操作日志。
"""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

import html as _html

from frontend.utils import (
    api_confirm, api_backtrack, api_trajectory, _convert_trajectory_response,
    type_label, get_target_type_icon, format_attributes,
    resolve_image_path, get_no_image_placeholder,
    confidence_color, confidence_label, format_number,
)
from frontend.styles import Colors


# ── 页面级 CSS ──
_CONFIRM_CSS = f"""
<style>
/* ── 复核台顶部标题栏 ── */
.review-header {{
    background: linear-gradient(135deg, {Colors.PRIMARY_BLUE} 0%, {Colors.PRIMARY_DARK} 100%);
    padding: 12px 20px;
    margin: 0 -14px 12px -14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(11, 78, 194, 0.18);
}}
.review-header .title-text {{
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 2px;
}}
.review-header .sub-text {{
    color: rgba(255,255,255,0.75);
    font-size: 12px;
}}

/* ── 面板卡片 ── */
.review-panel {{
    background-color: #FFFFFF;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.review-panel-title {{
    color: {Colors.TEXT_PRIMARY};
    font-size: 14px;
    font-weight: 600;
    border-bottom: 2px solid {Colors.PRIMARY_BLUE};
    padding-bottom: 6px;
    margin-bottom: 10px;
}}

/* ── 抓拍图区域 ── */
.capture-area {{
    background-color: #F5F7FA;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 10px;
    text-align: center;
}}
.capture-info {{
    font-size: 12px;
    color: {Colors.TEXT_SECONDARY};
    margin-top: 6px;
    line-height: 1.8;
}}
.capture-info b {{
    color: {Colors.TEXT_PRIMARY};
}}

/* ── 候选卡片 ── */
.candidate-card {{
    background-color: #FFFFFF;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
}}
.candidate-card:hover {{
    border-color: {Colors.PRIMARY_BLUE};
    box-shadow: 0 2px 8px rgba(11,78,194,0.12);
    transform: translateY(-1px);
}}
.candidate-card.selected {{
    border-color: {Colors.PRIMARY_BLUE};
    border-width: 2px;
    background-color: {Colors.TABLE_HEADER};
}}
.cc-title {{
    font-size: 13px;
    font-weight: 600;
    color: {Colors.TEXT_PRIMARY};
}}
.cc-meta {{
    font-size: 11px;
    color: {Colors.TEXT_SECONDARY};
    margin-top: 2px;
}}
.cc-score {{
    font-size: 13px;
    font-weight: 700;
    color: {Colors.BUTTON_BLUE};
    font-family: 'Consolas', monospace;
    margin-top: 3px;
}}

/* ── 属性表格 ── */
.attr-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 8px;
}}
.attr-table th {{
    background-color: {Colors.TABLE_HEADER};
    color: {Colors.PRIMARY_BLUE};
    padding: 5px 8px;
    text-align: left;
    border: 1px solid {Colors.TABLE_BORDER};
    font-weight: 600;
    width: 80px;
}}
.attr-table td {{
    padding: 4px 8px;
    border: 1px solid {Colors.TABLE_BORDER};
    color: {Colors.TEXT_PRIMARY};
}}
.attr-table tr:nth-child(even) {{
    background-color: #F8FAFD;
}}

/* ── 操作区 ── */
.action-section {{
    background-color: #F8FAFD;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 10px;
    margin-top: 8px;
}}
.action-section-title {{
    font-size: 13px;
    font-weight: 600;
    color: {Colors.TEXT_PRIMARY};
    margin-bottom: 8px;
    border-bottom: 1px solid {Colors.TABLE_BORDER};
    padding-bottom: 4px;
}}

/* ── 状态标签 ── */
.status-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
}}
.status-confirmed {{ background-color: {Colors.GREEN_OK}; color: #fff; }}
.status-excluded {{ background-color: {Colors.EXCLUDED_GRAY}; color: #fff; }}
.status-suspect {{ background-color: {Colors.ORANGE_WARN}; color: #fff; }}
.status-review {{ background-color: {Colors.BUTTON_BLUE}; color: #fff; }}
.status-misident {{ background-color: {Colors.RED_ALERT}; color: #fff; }}

/* ── 空状态 ── */
.empty-state {{
    text-align: center;
    padding: 60px 20px;
    color: {Colors.TEXT_SECONDARY};
}}
.empty-state .msg {{
    font-size: 15px;
    margin-bottom: 20px;
}}

/* ── 顶部信息栏 ── */
.review-top-bar {{
    background-color: #FFFFFF;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.review-top-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}}
.review-top-row + .review-top-row {{
    margin-top: 8px;
    border-top: 1px solid {Colors.TABLE_BORDER};
    padding-top: 8px;
}}
.top-section-label {{
    font-size: 12px;
    font-weight: 600;
    color: {Colors.PRIMARY_BLUE};
    white-space: nowrap;
    min-width: 70px;
}}
.query-tag {{
    display: inline-block;
    background-color: {Colors.TABLE_HEADER};
    color: {Colors.TEXT_PRIMARY};
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 2px;
}}
.query-tag b {{
    color: {Colors.PRIMARY_BLUE};
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}}
.status-pill-pending {{
    background-color: #FFF3E0;
    color: {Colors.ORANGE_WARN};
}}
.status-pill-confirmed {{
    background-color: #E8F5E9;
    color: {Colors.GREEN_OK};
}}
.status-pill-excluded {{
    background-color: #F0F0F0;
    color: {Colors.EXCLUDED_GRAY};
}}
.status-pill-suspect {{
    background-color: #FFF8E1;
    color: {Colors.ORANGE_WARN};
}}
.status-pill-misident {{
    background-color: #FFEBEE;
    color: {Colors.RED_ALERT};
}}

/* ── 多选模式 ── */
.candidate-card.multi-select {{
    cursor: pointer;
}}
.candidate-card.multi-select.checked {{
    border-color: {Colors.BUTTON_BLUE};
    background-color: {Colors.TABLE_HEADER};
}}
.multi-select-badge {{
    position: absolute;
    top: 6px;
    right: 6px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: {Colors.BUTTON_BLUE};
    color: #fff;
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* ── 底部批量操作栏 ── */
.batch-action-bar {{
    background-color: #FFFFFF;
    border: 1px solid {Colors.TABLE_BORDER};
    border-radius: 6px;
    padding: 10px 16px;
    margin-top: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.batch-info {{
    font-size: 13px;
    color: {Colors.TEXT_PRIMARY};
    font-weight: 600;
}}
.batch-info span {{
    color: {Colors.BUTTON_BLUE};
}}
</style>
"""


# ── 状态标签映射 ──
_STATUS_LABELS = {
    "confirmed": "确认为同一目标",
    "excluded": "排除该目标",
    "suspect": "标记为疑似目标",
    "review": "需人工复核",
    "misident": "误识别反馈",
}

_STATUS_BADGE_HTML = {
    "confirmed": '<span class="status-badge status-confirmed">已确认</span>',
    "excluded": '<span class="status-badge status-excluded">已排除</span>',
    "suspect": '<span class="status-badge status-suspect">疑似</span>',
    "review": '<span class="status-badge status-review">待复核</span>',
    "misident": '<span class="status-badge status-misident">误识别</span>',
}

_STATUS_ICONS = {
    "confirmed": "[✓]", "excluded": "[✗]", "suspect": "[!]",
    "review": "[↻]", "misident": "[⊘]",
}


# ============================================================
# session_state 初始化
# ============================================================

def _init_session_state():
    """初始化确认页所需的 session_state 变量"""
    if "confirm_records" not in st.session_state:
        st.session_state["confirm_records"] = []
    if "candidate_statuses" not in st.session_state:
        st.session_state["candidate_statuses"] = {}
    if "selected_candidate_idx" not in st.session_state:
        st.session_state["selected_candidate_idx"] = 0
    if "current_frame_idx" not in st.session_state:
        st.session_state["current_frame_idx"] = 0
    # ── 新增：查询条件 & 操作模式 ──
    if "confirm_query_conditions" not in st.session_state:
        st.session_state["confirm_query_conditions"] = {}
    if "confirm_action_mode" not in st.session_state:
        st.session_state["confirm_action_mode"] = "normal"
    if "confirm_multi_selected" not in st.session_state:
        st.session_state["confirm_multi_selected"] = set()


# ============================================================
# 数据获取
# ============================================================

def _get_candidates() -> List[Dict[str, Any]]:
    """从 session_state 获取候选列表"""
    # 优先从检索结果获取
    search_res = st.session_state.get("search_results")
    if search_res and search_res.get("candidates"):
        return search_res["candidates"]
    # 如果单个确认候选
    cand = st.session_state.get("confirmed_candidate")
    if cand:
        return [cand]
    return []


def _add_confirm_record(instance_id: str, cand: Dict, action: str,
                        operator: str = "系统快捷", basis: str = "",
                        remarks: str = ""):
    """添加一条确认记录到 session_state"""
    t_type = cand.get("target_type", "unknown")
    type_cn = type_label(t_type)
    plate = cand.get("plate_number") or "未识别"
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operator": operator,
        "target_id": instance_id,
        "target_desc": f"{type_cn} · {plate}",
        "conclusion": _STATUS_LABELS.get(action, action),
        "basis": basis,
        "remarks": remarks,
    }
    st.session_state["confirm_records"].append(record)


# ============================================================
# 渲染：顶部标题栏
# ============================================================

def _render_header(candidates: List[Dict] | None = None):
    """渲染页面顶部：标题栏 + 查询条件 + 状态汇总 + 操作按钮"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    action_mode = st.session_state.get("confirm_action_mode", "normal")

    # ── 第一行：标题 ──
    st.markdown(f"""
    <div class="review-header">
        <div>
            <span class="title-text">目标复核台</span>
        </div>
        <div class="sub-text">确认操作将记录留痕 &nbsp;|&nbsp; {now}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 第二行：查询条件 + 状态汇总 + 操作按钮 ──
    st.markdown('<div class="review-top-bar">', unsafe_allow_html=True)

    # 查询条件
    qc = st.session_state.get("confirm_query_conditions", {})
    if qc:
        tags_html = ""
        for k, v in qc.items():
            tags_html += f'<span class="query-tag"><b>{_html.escape(str(k))}:</b> {_html.escape(str(v))}</span>'
    else:
        tags_html = '<span class="query-tag" style="opacity:0.5;">暂无查询条件</span>'

    st.markdown(f"""
    <div class="review-top-row">
        <div class="top-section-label">查询条件</div>
        <div style="flex:1;">{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # 状态汇总 + 操作按钮
    if candidates:
        statuses = st.session_state.get("candidate_statuses", {})
        total = len(candidates)
        n_confirmed = sum(1 for s in statuses.values() if s == "confirmed")
        n_excluded = sum(1 for s in statuses.values() if s == "excluded")
        n_suspect = sum(1 for s in statuses.values() if s == "suspect")
        n_misident = sum(1 for s in statuses.values() if s == "misident")
        n_pending = total - n_confirmed - n_excluded - n_suspect - n_misident
    else:
        n_confirmed = n_excluded = n_suspect = n_misident = n_pending = 0

    pills_html = (
        f'<span class="status-pill status-pill-pending">待确认 {n_pending}</span>'
        f'<span class="status-pill status-pill-confirmed">已确认 {n_confirmed}</span>'
        f'<span class="status-pill status-pill-excluded">已排除 {n_excluded}</span>'
        f'<span class="status-pill status-pill-suspect">疑似 {n_suspect}</span>'
        f'<span class="status-pill status-pill-misident">误识别 {n_misident}</span>'
    )

    st.markdown(f"""
    <div class="review-top-row">
        <div class="top-section-label">状态汇总</div>
        <div style="flex:1;">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 第三行：操作按钮（5 等分） ──
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns(5)
    with btn_c1:
        if st.button("导出确认报告", key="hdr_export",
                     use_container_width=True):
            st.info("导出功能预留接口")
    with btn_c2:
        if action_mode != "normal":
            if candidates:
                if st.button("全选", key="hdr_select_all",
                             use_container_width=True):
                    st.session_state["confirm_multi_selected"] = set(
                        range(len(candidates))
                    )
                    st.rerun()
    with btn_c3:
        if action_mode == "normal":
            if st.button("批量确认", key="hdr_batch",
                         use_container_width=True):
                st.session_state["confirm_action_mode"] = "multi_select"
                st.session_state["confirm_multi_selected"] = set()
                st.rerun()
        else:
            if st.button("完成", key="hdr_done", use_container_width=True,
                         type="primary"):
                selected_set = st.session_state.get(
                    "confirm_multi_selected", set()
                )
                for idx_in_list in selected_set:
                    if candidates and idx_in_list < len(candidates):
                        c = candidates[idx_in_list]
                        cid = c.get("instance_id", "")
                        st.session_state["candidate_statuses"][cid] = \
                            "confirmed"
                        _add_confirm_record(cid, c, "confirmed",
                                            operator="批量确认")
                st.session_state["confirm_action_mode"] = "normal"
                st.session_state["confirm_multi_selected"] = set()
                st.rerun()
    with btn_c4:
        if action_mode != "normal":
            if st.button("取消", key="hdr_cancel", use_container_width=True):
                st.session_state["confirm_action_mode"] = "normal"
                st.session_state["confirm_multi_selected"] = set()
                st.rerun()
    with btn_c5:
        if st.button("返回检索", key="hdr_back_search",
                     use_container_width=True):
            st.session_state.pop("confirmed_candidate", None)
            st.session_state["current_page"] = "search"
            st.rerun()


# ============================================================
# 渲染：空状态
# ============================================================

def _render_empty_state():
    """无待确认目标时的空状态"""
    st.markdown("""
    <div class="empty-state">
        <div class="msg">当前没有待确认的目标。请先在目标检索页查找目标。</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("前往检索", use_container_width=True):
            st.session_state["current_page"] = "search"
            st.rerun()


# ============================================================
# 渲染：左侧 — 原始抓拍图
# ============================================================

def _render_left_panel(candidates: List[Dict], selected_idx: int):
    """左侧面板：大图展示 + 图片信息 + 相邻帧浏览"""
    st.markdown('<div class="review-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="review-panel-title">[抓拍] 原始抓拍图</div>',
        unsafe_allow_html=True,
    )

    cand = candidates[selected_idx] if selected_idx < len(candidates) else None
    if cand is None:
        st.caption("无选中目标")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── 帧浏览：从 track_frames 读取当前帧数据 ──
    frame_idx = st.session_state.get("current_frame_idx", 0)
    track_frames = cand.get("track_frames", [])

    if track_frames and 0 <= frame_idx < len(track_frames):
        current_frame = track_frames[frame_idx]
        raw_img_path = current_frame.get("crop_path", "")
        timestamp = current_frame.get("timestamp", cand.get("timestamp", ""))
        frame_id = current_frame.get("frame_id", 0)
        camera_name = current_frame.get("camera_name", cand.get("camera_name", ""))
    else:
        # 回退到候选的默认帧
        raw_img_path = cand.get("keyframe_path", cand.get("crop_path", ""))
        timestamp = cand.get("timestamp", "")
        frame_id = 0
        camera_name = cand.get("camera_name", cand.get("camera_id", ""))

    img_path = resolve_image_path(raw_img_path) if raw_img_path else None

    # ── 大图展示（支持点击放大） ──
    st.markdown('<div class="capture-area">', unsafe_allow_html=True)
    if img_path:
        st.image(img_path, use_container_width=True)
    else:
        st.image(get_no_image_placeholder(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 相邻帧浏览 ──
    if track_frames:
        total_frames = len(track_frames)
        nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
        with nav_c1:
            if st.button("上一帧", key="prev_frame", use_container_width=True):
                new_idx = max(0, frame_idx - 1)
                if new_idx != frame_idx:
                    st.session_state["current_frame_idx"] = new_idx
                    st.rerun()
        with nav_c2:
            st.caption(f"帧 {frame_idx + 1} / {total_frames}")
        with nav_c3:
            if st.button("下一帧", key="next_frame", use_container_width=True):
                new_idx = min(total_frames - 1, frame_idx + 1)
                if new_idx != frame_idx:
                    st.session_state["current_frame_idx"] = new_idx
                    st.rerun()
    else:
        st.caption("此目标只有单帧检测记录")

    # ── 图片信息 ──
    conf = cand.get("quality_score", cand.get("combined_score", 0))
    conf_pct = f"{conf * 100:.1f}%" if conf else "N/A"
    conf_clr = confidence_color(conf) if conf else Colors.TEXT_SECONDARY
    conf_lbl = confidence_label(conf) if conf else ""

    info_html = f"""
    <div class="capture-info">
        <div><b>摄像头：</b>{_html.escape(str(camera_name))}</div>
        <div><b>抓拍时间：</b>{_html.escape(str(timestamp))}</div>
    """
    if frame_id > 0:
        info_html += f'        <div><b>帧ID：</b>{frame_id}</div>\n'
    info_html += f"""        <div><b>检测置信度：</b>
            <span style="color:{conf_clr};font-weight:600;">
                {conf_pct}（{conf_lbl}）
            </span>
        </div>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 渲染：中间 — 候选目标列表
# ============================================================

def _render_middle_panel(candidates: List[Dict], selected_idx: int,
                         multi_select: bool = False,
                         multi_selected: set | None = None):
    """中间面板：候选目标卡片列表。

    multi_select=True 时显示复选框，不显示选中/确认/排除按钮。
    """
    if multi_selected is None:
        multi_selected = set()

    st.markdown('<div class="review-panel">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="review-panel-title">'
        f'[列表] 候选目标列表（{len(candidates)} 个）</div>',
        unsafe_allow_html=True,
    )

    statuses = st.session_state.get("candidate_statuses", {})

    for i, cand in enumerate(candidates):
        instance_id = cand.get("instance_id", f"#{i}")
        plate = cand.get("plate_number") or "未识别"
        t_type = cand.get("target_type", "unknown")
        type_cn = type_label(t_type)
        attrs = cand.get("attributes", {})
        color = attrs.get("颜色", "未知")
        camera_name = cand.get("camera_name", cand.get("camera_id", ""))
        ts = cand.get("timestamp", "")
        score = cand.get("combined_score", 0)
        status = statuses.get(instance_id, "")

        # 状态标签 HTML
        status_html = _STATUS_BADGE_HTML.get(status, "")

        # 卡片样式
        if multi_select:
            checked_cls = " checked" if i in multi_selected else ""
            selected_cls = f" multi-select{checked_cls}"
        else:
            selected_cls = " selected" if i == selected_idx else ""

        # 多选复选框或单选指示
        if multi_select:
            check_mark = "[✓]" if i in multi_selected else ""
            check_style = (
                f'<div class="multi-select-badge">{check_mark}</div>'
                if i in multi_selected else
                f'<div style="position:absolute;top:6px;right:6px;'
                f'width:20px;height:20px;border-radius:50%;'
                f'border:2px solid {Colors.TABLE_BORDER};'
                f'background:#fff;"></div>'
            )
        else:
            check_style = ""

        # ── 卡片渲染（使用 st.columns 分块，避免复杂 HTML 暴露） ──
        with st.container():
            st.markdown(f'<div class="candidate-card{selected_cls}" style="position:relative;">{check_style}</div>', unsafe_allow_html=True)
            col_title, col_status = st.columns([3, 1])
            with col_title:
                icon_html = get_target_type_icon(t_type)
                st.markdown(f'{icon_html} **{_html.escape(str(type_cn))}** · **{_html.escape(str(plate))}**')
            with col_status:
                if status_html:
                    st.markdown(status_html, unsafe_allow_html=True)
            st.caption(f"{_html.escape(str(color))} | {_html.escape(str(camera_name))} | {_html.escape(str(ts))}")
            st.markdown(f'<div class="cc-score">相似度 {format_number(score, ".1%")}</div>', unsafe_allow_html=True)

        if multi_select:
            # 多选模式：只显示复选框
            is_checked = i in multi_selected
            label = "已选中" if is_checked else "选择"
            btn_type = "primary" if is_checked else "secondary"
            if st.button(label, key=f"chk_{i}", use_container_width=True,
                         type=btn_type):
                if is_checked:
                    st.session_state["confirm_multi_selected"].discard(i)
                else:
                    st.session_state["confirm_multi_selected"].add(i)
                st.rerun()
        else:
            # 普通模式：选中查看按钮
            is_selected = (i == selected_idx)
            btn_label = "当前选中" if is_selected else "选中查看"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(btn_label, key=f"sel_{i}", use_container_width=True,
                         type=btn_type):
                if not is_selected:
                    st.session_state["selected_candidate_idx"] = i
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 渲染：右侧 — 目标属性与确认操作
# ============================================================

def _render_right_panel(candidates: List[Dict], selected_idx: int):
    """右侧面板：完整属性表 + 确认操作按钮 + 确认信息表单"""
    st.markdown('<div class="review-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="review-panel-title">[属性] 目标属性与确认操作</div>',
        unsafe_allow_html=True,
    )

    cand = candidates[selected_idx] if selected_idx < len(candidates) else None
    if cand is None:
        st.caption("请先选择候选目标")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    instance_id = cand.get("instance_id", "")
    t_type = cand.get("target_type", "unknown")
    type_cn = type_label(t_type)
    attrs = cand.get("attributes", {})
    plate = cand.get("plate_number") or "未识别"
    camera_name = cand.get("camera_name", cand.get("camera_id", ""))
    timestamp = cand.get("timestamp", "")
    quality = cand.get("quality_score", 0)

    # ── 完整属性表格（只显示业务字段，隐藏技术字段） ──
    BUSINESS_KEYS = {"颜色", "车型", "摄像头", "场景", "发现位置", "发现时间"}
    attr_rows = ""
    for k, v in attrs.items():
        if k not in BUSINESS_KEYS and k not in {"color", "vehicle_type"}:
            continue
        attr_rows += f"<tr><th>{_html.escape(str(k))}</th><td>{_html.escape(str(v))}</td></tr>"
    attr_rows += (
        f"<tr><th>车牌号</th>"
        f"<td style='color:{Colors.BUTTON_BLUE};font-weight:600;'>{_html.escape(str(plate))}</td></tr>"
    )
    attr_rows += f"<tr><th>目标类型</th><td>{_html.escape(str(type_cn))}</td></tr>"
    attr_rows += f"<tr><th>发现位置</th><td>{_html.escape(str(camera_name))}</td></tr>"
    attr_rows += f"<tr><th>发现时间</th><td>{_html.escape(str(timestamp))}</td></tr>"
    if quality:
        clr = confidence_color(quality)
        attr_rows += (
            f"<tr><th>图像质量</th>"
            f"<td style='color:{clr};font-weight:600;'>{quality:.1%}</td></tr>"
        )

    st.markdown(
        f'<table class="attr-table">{attr_rows}</table>',
        unsafe_allow_html=True,
    )

    # ── 确认操作区 ──
    st.markdown("""
    <div class="action-section">
        <div class="action-section-title">确认操作</div>
    </div>
    """, unsafe_allow_html=True)

    # 操作按钮 — 第一行
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("确认该目标", key="act_confirm",
                     use_container_width=True):
            st.session_state["candidate_statuses"][instance_id] = "confirmed"
            _add_confirm_record(instance_id, cand, "confirmed")
            st.rerun()
    with b2:
        if st.button("排除该目标", key="act_exclude",
                     use_container_width=True):
            st.session_state["candidate_statuses"][instance_id] = "excluded"
            _add_confirm_record(instance_id, cand, "excluded")
            st.rerun()
    with b3:
        if st.button("标记疑似", key="act_suspect",
                     use_container_width=True):
            st.session_state["candidate_statuses"][instance_id] = "suspect"
            _add_confirm_record(instance_id, cand, "suspect")
            st.rerun()

    # 操作按钮 — 第二行
    b4, b5 = st.columns(2)
    with b4:
        if st.button("需人工复核", key="act_review",
                     use_container_width=True):
            st.session_state["candidate_statuses"][instance_id] = "review"
            _add_confirm_record(instance_id, cand, "review")
            st.rerun()
    with b5:
        if st.button("误识别反馈", key="act_misident",
                     use_container_width=True):
            st.session_state["candidate_statuses"][instance_id] = "misident"
            _add_confirm_record(instance_id, cand, "misident")
            st.rerun()

    # ── 确认信息表单 ──
    st.markdown("---")
    st.markdown(
        '<div class="action-section-title">确认信息表单</div>',
        unsafe_allow_html=True,
    )

    with st.form(key=f"confirm_form_{instance_id}", clear_on_submit=False):
        operator = st.text_input("操作人", placeholder="请输入姓名或警号")
        conclusion = st.selectbox(
            "确认结论",
            ["确认为同一目标", "排除该目标", "标记为疑似目标",
             "需人工复核", "误识别反馈"],
        )
        basis = st.text_area(
            "确认依据", placeholder="请说明确认或排除的依据...", height=80,
        )
        remarks = st.text_area(
            "备注", placeholder="其他需要记录的信息...", height=68,
        )
        submitted = st.form_submit_button(
            "提交确认记录", use_container_width=True, type="primary",
        )

        if submitted:
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": operator or "未填写",
                "target_id": instance_id,
                "target_desc": f"{type_cn} · {plate}",
                "conclusion": conclusion,
                "basis": basis,
                "remarks": remarks,
            }
            st.session_state["confirm_records"].append(record)
            st.success("确认记录已提交！")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 渲染：底部 — 确认记录和操作日志
# ============================================================

def _render_bottom_records():
    """底部面板：表格展示历史确认记录"""
    st.markdown("---")
    st.markdown(
        '<div class="review-panel-title">'
        '[记录] 确认记录与操作日志</div>',
        unsafe_allow_html=True,
    )

    records = st.session_state.get("confirm_records", [])
    if not records:
        st.info("暂无确认记录。对候选目标执行确认操作后，记录将显示在此处。")
        return

    # 构建 DataFrame（最新记录在最上方）
    df_data = []
    for r in reversed(records):
        df_data.append({
            "时间": r.get("time", ""),
            "操作人": r.get("operator", ""),
            "目标": r.get("target_desc", ""),
            "结论": r.get("conclusion", ""),
            "依据": r.get("basis", "") or "—",
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True, height=220)


# ============================================================
# 主渲染入口
# ============================================================

def render():
    """渲染确认页面（目标复核台）"""
    st.markdown(_CONFIRM_CSS, unsafe_allow_html=True)
    _init_session_state()

    candidates = _get_candidates()

    _render_header(candidates)

    if not candidates:
        _render_empty_state()
        return

    # 确保选中索引有效
    selected_idx = st.session_state.get("selected_candidate_idx", 0)
    if selected_idx >= len(candidates):
        selected_idx = 0
        st.session_state["selected_candidate_idx"] = 0

    # ── 切换候选时重置帧索引 ──
    current_cand = candidates[selected_idx] if candidates else None
    if current_cand:
        cand_id = current_cand.get("instance_id", "")
        last_id = st.session_state.get("_last_candidate_id", "")
        if cand_id != last_id:
            st.session_state["current_frame_idx"] = 0
            st.session_state["_last_candidate_id"] = cand_id

    # ── 目标快速切换按钮 ──
    sel_cols = st.columns(len(candidates))
    statuses = st.session_state.get("candidate_statuses", {})
    for i, col in enumerate(sel_cols):
        with col:
            label = f"目标 {i + 1}"
            status = statuses.get(
                candidates[i].get("instance_id", ""), ""
            )
            if status:
                icon = _STATUS_ICONS.get(status, "")
                label = f"{icon} 目标 {i + 1}"
            is_selected = (i == selected_idx)
            if st.button(
                label, key=f"tab_{i}", use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                if not is_selected:
                    st.session_state["selected_candidate_idx"] = i
                    st.rerun()

    st.markdown("")

    # ── 判断操作模式 ──
    action_mode = st.session_state.get("confirm_action_mode", "normal")
    multi_select = (action_mode == "multi_select")
    multi_selected = st.session_state.get("confirm_multi_selected", set())

    # ── 三栏布局 ──
    col_left, col_mid, col_right = st.columns([3, 4, 4])

    with col_left:
        _render_left_panel(candidates, selected_idx)

    with col_mid:
        _render_middle_panel(candidates, selected_idx,
                             multi_select=multi_select,
                             multi_selected=multi_selected)

    with col_right:
        _render_right_panel(candidates, selected_idx)

    # ── 底部确认记录 ──
    _render_bottom_records()

    # ── 底部操作栏 ──
    st.markdown("---")
    foot_c1, foot_c2, foot_c3 = st.columns([1, 1, 2])
    with foot_c1:
        if st.button("返回检索页", use_container_width=True):
            st.session_state.pop("confirmed_candidate", None)
            st.session_state["current_page"] = "search"
            st.rerun()
    with foot_c2:
        if st.button("启动轨迹回溯", type="primary",
                     use_container_width=True):
            # 找到已确认的目标
            confirmed_ids = [
                cid for cid, status in
                st.session_state.get("candidate_statuses", {}).items()
                if status == "confirmed"
            ]
            if confirmed_ids:
                # 从候选列表中找到对应的完整候选对象
                target_cand = None
                for c in candidates:
                    if c.get("instance_id", "") in confirmed_ids:
                        target_cand = c
                        break
                if target_cand is None and candidates:
                    target_cand = candidates[selected_idx]
            elif candidates:
                target_cand = candidates[selected_idx]
            else:
                target_cand = None

            if target_cand:
                instance_id = target_cand.get("instance_id", "")
                track_id = target_cand.get("track_id", "")

                with st.spinner("正在加载跨镜轨迹..."):
                    try:
                        raw_traj = api_trajectory(
                            instance_id=instance_id,
                            track_id=track_id,
                        )

                        if raw_traj and raw_traj.get("success"):
                            traj_data = _convert_trajectory_response(
                                raw_traj.get("trajectory", {}),
                                target_cand,
                            )

                            # 设置所有必要的 session_state
                            st.session_state["selected_track_id"] = track_id
                            st.session_state["selected_instance_id"] = instance_id
                            st.session_state["trajectory_data"] = traj_data
                            st.session_state["confirmed_candidate"] = target_cand
                            st.session_state["current_page"] = "trajectory"

                            st.success(
                                f"已加载跨镜轨迹："
                                f"{traj_data.get('total_cameras', 0)}个摄像头，"
                                f"{traj_data.get('total_detections', 0)}帧"
                            )
                            st.rerun()
                        else:
                            # 尝试从本地数据构建
                            from frontend.pages.search import _build_traj_from_local
                            traj_data = _build_traj_from_local(target_cand)
                            if traj_data and traj_data.get("observation_nodes"):
                                st.session_state["selected_track_id"] = track_id
                                st.session_state["selected_instance_id"] = instance_id
                                st.session_state["trajectory_data"] = traj_data
                                st.session_state["confirmed_candidate"] = target_cand
                                st.session_state["current_page"] = "trajectory"
                                st.success("已从本地数据构建跨镜轨迹")
                                st.rerun()
                            else:
                                st.error("无法获取跨镜轨迹数据")
                    except Exception as e:
                        # 异常时回退到本地数据构建
                        try:
                            from frontend.pages.search import _build_traj_from_local
                            traj_data = _build_traj_from_local(target_cand)
                            if traj_data and traj_data.get("observation_nodes"):
                                st.session_state["selected_track_id"] = track_id
                                st.session_state["selected_instance_id"] = instance_id
                                st.session_state["trajectory_data"] = traj_data
                                st.session_state["confirmed_candidate"] = target_cand
                                st.session_state["current_page"] = "trajectory"
                                st.success("已从本地数据构建跨镜轨迹")
                                st.rerun()
                            else:
                                st.error(f"加载轨迹失败: {str(e)}")
                        except Exception:
                            st.error(f"加载轨迹失败: {str(e)}")
            else:
                st.warning("请先选择一个目标")

    # ── 批量操作底栏（多选模式时显示） ──
    if multi_select:
        n_sel = len(multi_selected)
        st.markdown(f"""
        <div class="batch-action-bar">
            <div class="batch-info">已选择 <span>{n_sel}</span> 个目标</div>
        </div>
        """, unsafe_allow_html=True)
