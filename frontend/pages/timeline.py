"""
frontend.pages.timeline - 证据链式时间轴页面（浅色主题 · 交警研判风格）

横向时间轴 + 证据标记 + 时间控制 + 证据摘要
"""

from __future__ import annotations

import os
from datetime import datetime

import html as _html

import streamlit as st

from frontend.utils import (
    type_label, get_target_type_icon, resolve_image_path,
    get_no_image_placeholder,
    confidence_color, confidence_label,
    CAMERA_NAME_MAP,
)


# ── 时间戳解析工具 ──
_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y%m%d %H%M%S",
]


def _parse_timestamp(ts: str) -> datetime | None:
    """尝试多种格式解析时间戳，兼容 CityFlow 毫秒格式"""
    if not ts:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


# ── 页面级 CSS（浅色主题） ──
_TIMELINE_CSS = """
<style>
/* ── 横向时间轴 ─────────────────────────────── */
.tl-container {
    overflow-x: auto;
    padding: 16px 0;
}
.tl-track {
    display: flex;
    align-items: flex-start;
    position: relative;
    min-width: max-content;
    padding: 0 20px;
}
.tl-track::before {
    content: '';
    position: absolute;
    top: 58px;
    left: 20px;
    right: 20px;
    height: 3px;
    background: #D9E2EF;
    z-index: 0;
}
.tl-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 140px;
    max-width: 160px;
    position: relative;
    z-index: 1;
    cursor: pointer;
    margin-right: 8px;
}
.tl-node:last-child {
    margin-right: 0;
}
.tl-node-thumb {
    width: 100px;
    height: 68px;
    border-radius: 4px;
    border: 2px solid #D9E2EF;
    object-fit: cover;
    background: #F3F6FA;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.tl-node:hover .tl-node-thumb {
    border-color: #0B4EC2;
    box-shadow: 0 2px 8px rgba(11,78,194,0.15);
}
.tl-node-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #0B4EC2;
    border: 3px solid #FFFFFF;
    box-shadow: 0 0 0 2px #0B4EC2;
    margin: 6px 0;
    flex-shrink: 0;
}
.tl-node.tl-selected .tl-node-dot {
    background: #F59E0B;
    box-shadow: 0 0 0 2px #F59E0B;
}
.tl-node.tl-evidence-star .tl-node-dot {
    background: #F59E0B;
    box-shadow: 0 0 0 2px #F59E0B, 0 0 6px rgba(245,158,11,0.4);
}
.tl-node-time {
    font-size: 11px;
    font-weight: 600;
    color: #0B4EC2;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    margin-top: 2px;
    white-space: nowrap;
}
.tl-node-cam {
    font-size: 10px;
    color: #667085;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 130px;
    text-align: center;
}
.tl-node-dir {
    font-size: 10px;
    color: #9AA4B2;
}
.tl-node-conf {
    font-size: 10px;
    font-weight: 600;
    margin-top: 1px;
}
.tl-node-star {
    position: absolute;
    top: -4px;
    right: 12px;
    font-size: 16px;
    color: #F59E0B;
    z-index: 2;
}

/* ── 证据状态标签 ───────────────────────────── */
.tl-evidence-tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 500;
    margin-top: 2px;
}
.tl-tag-confirmed {
    background: #2E7D32;
    color: #FFFFFF;
}
.tl-tag-pending {
    background: #F59E0B;
    color: #FFFFFF;
}
.tl-tag-excluded {
    background: #E8ECF0;
    color: #9AA4B2;
}

/* ── 节点详情展开 ───────────────────────────── */
.tl-detail-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-top: 8px;
}
.tl-detail-title {
    font-size: 13px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.tl-detail-meta {
    font-size: 13px;
    color: #1F2D3D;
    line-height: 1.8;
}
.tl-detail-label {
    color: #667085;
    font-size: 12px;
}

/* ── 控制栏 ─────────────────────────────────── */
.tl-control-bar {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.tl-control-label {
    font-size: 12px;
    color: #667085;
    white-space: nowrap;
}

/* ── 证据摘要 ───────────────────────────────── */
.tl-summary-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-top: 12px;
}
.tl-summary-title {
    font-size: 13px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.tl-summary-stats {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.tl-summary-box {
    flex: 1;
    min-width: 100px;
    background: #F3F6FA;
    border: 1px solid #D9E2EF;
    border-radius: 4px;
    padding: 8px 12px;
    text-align: center;
}
.tl-summary-value {
    font-size: 20px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}
.tl-summary-label {
    font-size: 11px;
    color: #667085;
    margin-top: 2px;
}

/* ── 空状态 ─────────────────────────────────── */
.tl-empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #667085;
    font-size: 14px;
}
.tl-empty-state-icon {
    font-size: 36px;
    color: #D9E2EF;
    margin-bottom: 12px;
}
</style>
"""


def render():
    """渲染证据链式时间轴页面"""
    st.markdown(_TIMELINE_CSS, unsafe_allow_html=True)

    st.markdown("#### 证据链时间轴")

    traj_data = st.session_state.get("trajectory_data")
    if traj_data is None:
        st.markdown("""
        <div class="tl-empty-state">
            <div class="tl-empty-state-icon">[∅]</div>
            <div>暂无时间轴数据。请先完成目标轨迹回溯。</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("返回检索"):
            st.session_state["current_page"] = "search"
            st.rerun()
        return

    obs_nodes = traj_data.get("observation_nodes", [])
    inf_segs = traj_data.get("inference_segments", [])
    target = traj_data.get("target_instance", {})
    obs_segs = traj_data.get("observation_segments", [])

    if not obs_nodes:
        st.markdown("""
        <div class="tl-empty-state">
            <div class="tl-empty-state-icon">[∅]</div>
            <div>暂无时间轴数据。请先完成目标轨迹回溯。</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── 初始化 session state ──
    if "tl_selected_idx" not in st.session_state:
        st.session_state["tl_selected_idx"] = 0
    if "tl_evidence_marks" not in st.session_state:
        st.session_state["tl_evidence_marks"] = {}  # {idx: "confirmed"|"pending"|"excluded"}
    if "tl_zoom_level" not in st.session_state:
        st.session_state["tl_zoom_level"] = "按小时"

    selected_idx = st.session_state["tl_selected_idx"]

    # ── 目标简要信息 ──
    icon = get_target_type_icon(target.get("target_type", ""))
    type_cn = type_label(target.get("target_type", ""))
    st.markdown(f"""
    <div style="font-size:12px;color:#667085;margin-bottom:8px;">
        目标: <span style="color:#1F2D3D;font-weight:600;">{icon} {type_cn}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp; 证据节点: <span style="color:#0B4EC2;font-weight:600;">{len(obs_nodes)}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 时间轴控制栏 ──
    _render_controls(obs_nodes)

    # ── 横向时间轴 ──
    _render_horizontal_timeline(obs_nodes, obs_segs, selected_idx)

    # ── 当前节点详情 ──
    _render_node_detail(obs_nodes, obs_segs, inf_segs, selected_idx)

    # ── 证据摘要 ──
    _render_evidence_summary(obs_nodes)


# ============================================================
# 控制栏
# ============================================================

def _render_controls(obs_nodes):
    """渲染时间轴控制栏"""
    ctrl_cols = st.columns([1, 1, 1, 1, 1])

    with ctrl_cols[0]:
        if st.button("上一帧", key="btn_prev_frame", use_container_width=True):
            idx = st.session_state["tl_selected_idx"]
            if idx > 0:
                st.session_state["tl_selected_idx"] = idx - 1
                st.rerun()

    with ctrl_cols[1]:
        if st.button("后一帧", key="btn_next_frame", use_container_width=True):
            idx = st.session_state["tl_selected_idx"]
            if idx < len(obs_nodes) - 1:
                st.session_state["tl_selected_idx"] = idx + 1
                st.rerun()

    with ctrl_cols[2]:
        zoom = st.selectbox(
            "缩放级别",
            ["按小时", "按天"],
            index=0 if st.session_state["tl_zoom_level"] == "按小时" else 1,
            key="sel_zoom",
        )
        st.session_state["tl_zoom_level"] = zoom

    with ctrl_cols[3]:
        # 时间范围筛选
        timestamps = [n.get("timestamp", "") for n in obs_nodes if n.get("timestamp")]
        if len(timestamps) >= 2:
            st.markdown('<div class="tl-control-label">时间范围</div>', unsafe_allow_html=True)
            try:
                t_start = _parse_timestamp(timestamps[0])
                t_end = _parse_timestamp(timestamps[-1])
                if t_start and t_end:
                    st.markdown(
                        f'<div style="font-size:11px;color:#1F2D3D;font-family:monospace;">'
                        f'{t_start.strftime("%H:%M")} — {t_end.strftime("%H:%M")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:11px;color:#1F2D3D;">{timestamps[0]} — {timestamps[-1]}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.markdown(
                    f'<div style="font-size:11px;color:#1F2D3D;">{timestamps[0]} — {timestamps[-1]}</div>',
                    unsafe_allow_html=True,
                )

    with ctrl_cols[4]:
        st.markdown('<div class="tl-control-label">当前帧</div>', unsafe_allow_html=True)
        idx = st.session_state["tl_selected_idx"]
        st.markdown(
            f'<div style="font-size:14px;font-weight:700;color:#0B4EC2;">'
            f'{idx + 1} / {len(obs_nodes)}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 横向时间轴
# ============================================================

def _render_horizontal_timeline(obs_nodes, obs_segs, selected_idx):
    """使用 Streamlit 原生组件渲染横向时间轴"""
    evidence_marks = st.session_state.get("tl_evidence_marks", {})

    # 使用 st.columns 渲染每个节点，避免复杂HTML泄露
    nodes_per_row = min(len(obs_nodes), 6)
    if nodes_per_row == 0:
        return

    for row_start in range(0, len(obs_nodes), nodes_per_row):
        row_end = min(row_start + nodes_per_row, len(obs_nodes))
        node_cols = st.columns(row_end - row_start)

        for i in range(row_start, row_end):
            col_idx = i - row_start
            node = obs_nodes[i]

            with node_cols[col_idx]:
                cam_name = str(CAMERA_NAME_MAP.get(
                    node.get("camera_id", ""),
                    node.get("camera_name", node.get("camera_id", "")),
                ))
                ts = str(node.get("timestamp", ""))
                dt = _parse_timestamp(ts)
                time_short = dt.strftime("%H:%M:%S") if dt else ts

                conf = node.get("confidence", 0.9)
                conf_hex = confidence_color(conf)

                direction = ""
                if i < len(obs_segs):
                    direction = str(obs_segs[i].get("direction", ""))

                is_selected = (i == selected_idx)
                mark_status = evidence_marks.get(i, "")

                # 证据标记状态文字
                mark_text = ""
                if mark_status == "confirmed":
                    mark_text = "[已确认]"
                elif mark_status == "pending":
                    mark_text = "[待确认]"
                elif mark_status == "excluded":
                    mark_text = "[已排除]"

                # 缩略图
                keyframe = node.get("keyframe_path")
                img_path = resolve_image_path(keyframe) if keyframe else None
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="background:#F3F6FA;border:1px solid #D9E2EF;'
                        'border-radius:4px;padding:12px;text-align:center;'
                        'color:#9AA4B2;font-size:11px;">无图片</div>',
                        unsafe_allow_html=True,
                    )

                # 节点信息
                st.markdown(
                    f'<div style="text-align:center;font-size:11px;font-weight:600;'
                    f'color:#0B4EC2;font-family:monospace;">{_html.escape(time_short)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="text-align:center;font-size:10px;color:#667085;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                    f'{_html.escape(cam_name)}</div>',
                    unsafe_allow_html=True,
                )
                if direction:
                    st.markdown(
                        f'<div style="text-align:center;font-size:10px;color:#9AA4B2;">'
                        f'{_html.escape(direction)}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="text-align:center;font-size:10px;font-weight:600;'
                    f'color:{conf_hex};">{conf:.0%}</div>',
                    unsafe_allow_html=True,
                )
                if mark_text:
                    st.markdown(
                        f'<div style="text-align:center;font-size:10px;font-weight:600;'
                        f'color:#F59E0B;">{_html.escape(mark_text)}</div>',
                        unsafe_allow_html=True,
                    )

                # 选择按钮
                btn_label = f"选择 #{i + 1}"
                if st.button(btn_label, key=f"tl_node_{i}", use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    st.session_state["tl_selected_idx"] = i
                    st.rerun()

    st.markdown(
        '<div style="font-size:11px;color:#667085;margin-top:4px;">'
        '点击上方按钮选择节点，或使用上方控制栏切换</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 节点详情
# ============================================================

def _render_node_detail(obs_nodes, obs_segs, inf_segs, idx):
    """渲染当前选中节点的详情"""
    if idx >= len(obs_nodes):
        return

    node = obs_nodes[idx]
    camera_id = _html.escape(str(node.get("camera_id", "")))
    camera_name = _html.escape(str(CAMERA_NAME_MAP.get(
        node.get("camera_id", ""),
        node.get("camera_name", node.get("camera_id", "")),
    )))
    timestamp = _html.escape(str(node.get("timestamp", "")))
    conf = node.get("confidence", 0.9)
    conf_hex = confidence_color(conf)

    direction = ""
    entry_desc = ""
    exit_desc = ""
    if idx < len(obs_segs):
        direction = _html.escape(str(obs_segs[idx].get("direction", "")))
        entry_desc = _html.escape(str(obs_segs[idx].get("entry_description", "")))
        exit_desc = _html.escape(str(obs_segs[idx].get("exit_description", "")))

    col_img, col_info = st.columns([1, 2])

    with col_img:
        keyframe = node.get("keyframe_path")
        img_path = resolve_image_path(keyframe) if keyframe else None
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.image(get_no_image_placeholder(), use_container_width=True)

    with col_info:
        # 证据标记操作
        mark_cols = st.columns([1, 1, 1, 2])
        evidence_marks = st.session_state.get("tl_evidence_marks", {})
        current_mark = evidence_marks.get(idx, "")

        with mark_cols[0]:
            if st.button("标记关键证据", key=f"btn_mark_ev_{idx}", use_container_width=True):
                evidence_marks[idx] = "pending"
                st.session_state["tl_evidence_marks"] = evidence_marks
                st.rerun()
        with mark_cols[1]:
            if st.button("确认", key=f"btn_confirm_ev_{idx}", use_container_width=True):
                evidence_marks[idx] = "confirmed"
                st.session_state["tl_evidence_marks"] = evidence_marks
                st.rerun()
        with mark_cols[2]:
            if st.button("排除", key=f"btn_exclude_ev_{idx}", use_container_width=True):
                evidence_marks[idx] = "excluded"
                st.session_state["tl_evidence_marks"] = evidence_marks
                st.rerun()

        # 当前证据状态
        mark_label = {"confirmed": "已确认", "pending": "待确认", "excluded": "已排除"}.get(current_mark, "未标记")
        mark_color = {"confirmed": "#2E7D32", "pending": "#F59E0B", "excluded": "#9AA4B2"}.get(current_mark, "#667085")

        st.markdown(f"""
        <div class="tl-detail-card">
            <div class="tl-detail-title">节点 {idx + 1} / {len(obs_nodes)} &nbsp;
                <span style="font-size:11px;color:{mark_color};font-weight:500;">[{mark_label}]</span>
            </div>
            <div class="tl-detail-meta">
                <span class="tl-detail-label">摄像头名称：</span>
                <strong>{camera_name}</strong><br>
                <span class="tl-detail-label">摄像头 ID：</span>{camera_id}<br>
                <span class="tl-detail-label">抓拍时间：</span>
                <strong>{timestamp}</strong><br>
                <span class="tl-detail-label">行驶方向：</span>{direction}<br>
                <span class="tl-detail-label">进入描述：</span>{entry_desc}<br>
                <span class="tl-detail-label">离开描述：</span>{exit_desc}<br>
                <span class="tl-detail-label">置信度：</span>
                <span style="color:{conf_hex};font-weight:600;">{conf:.1%}</span>
                &nbsp;({confidence_label(conf)})
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 推断段信息
    if idx < len(inf_segs):
        inf = inf_segs[idx]
        inf_conf = inf.get("confidence", 0.7)
        inf_hex = confidence_color(inf_conf)
        est_t = inf.get("estimated_travel_time", 0)
        st.markdown(f"""
        <div style="background:#FFF8E1;border:1px solid #F59E0B;border-radius:4px;
                    padding:8px 12px;margin-top:6px;font-size:12px;">
            <strong style="color:#F59E0B;">[→] 下一段为推断行驶</strong><br>
            {_html.escape(str(CAMERA_NAME_MAP.get(inf.get('source_camera_id', ''), inf.get('source_camera_name', ''))))} → {_html.escape(str(CAMERA_NAME_MAP.get(inf.get('target_camera_id', ''), inf.get('target_camera_name', ''))))}
            &nbsp;|&nbsp; 置信度: <span style="color:{inf_hex};font-weight:600;">{inf_conf:.1%}</span>
            &nbsp;|&nbsp; 预估行驶时间: {est_t:.0f}秒
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 证据摘要
# ============================================================

def _render_evidence_summary(obs_nodes):
    """渲染底部证据摘要"""
    evidence_marks = st.session_state.get("tl_evidence_marks", {})
    total = len(obs_nodes)
    confirmed = sum(1 for v in evidence_marks.values() if v == "confirmed")
    pending = sum(1 for v in evidence_marks.values() if v == "pending")
    excluded = sum(1 for v in evidence_marks.values() if v == "excluded")
    unmarked = total - confirmed - pending - excluded

    st.markdown(f"""
    <div class="tl-summary-card">
        <div class="tl-summary-title">证据摘要</div>
        <div class="tl-summary-stats">
            <div class="tl-summary-box">
                <div class="tl-summary-value" style="color:#0B4EC2;">{total}</div>
                <div class="tl-summary-label">证据节点总数</div>
            </div>
            <div class="tl-summary-box">
                <div class="tl-summary-value" style="color:#2E7D32;">{confirmed}</div>
                <div class="tl-summary-label">已确认</div>
            </div>
            <div class="tl-summary-box">
                <div class="tl-summary-value" style="color:#F59E0B;">{pending}</div>
                <div class="tl-summary-label">待确认</div>
            </div>
            <div class="tl-summary-box">
                <div class="tl-summary-value" style="color:#9AA4B2;">{excluded}</div>
                <div class="tl-summary-label">已排除</div>
            </div>
            <div class="tl-summary-box">
                <div class="tl-summary-value" style="color:#667085;">{unmarked}</div>
                <div class="tl-summary-label">未标记</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 导出按钮
    st.markdown("")
    col_export, _ = st.columns([1, 3])
    with col_export:
        icon_export = "[导出]"
        if st.button(f"{icon_export}  导出时间轴截图", key="btn_export_timeline", use_container_width=True):
            st.info("截图导出功能需要配合浏览器截图工具使用。请使用系统截图功能保存当前时间轴。")
