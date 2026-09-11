"""
frontend.pages.trajectory - 轨迹回溯页面（浅色主题 · 交警研判风格）

布局：
  上部 — 轨迹概览卡片（目标信息 + 轨迹统计）
  中部 — 轨迹详情（摄像头表格 + 推断路段 + 断点标注）
  右侧 — 研判摘要（结论 / 驶离方向 / 可信度 / 建议操作）
"""

from __future__ import annotations

import html as _html
import io
import os
from datetime import datetime

import streamlit as st

from frontend.utils import (
    confidence_color, confidence_label, type_label,
    get_target_type_icon, CAMERA_NAME_MAP, resolve_image_path,
)
from frontend.components import render_icon
from frontend.components.map_view import render_trajectory_plot


# ── 页面级 CSS（浅色主题） ──
_TRAJ_CSS = """
<style>
/* ── 概览卡片行 ─────────────────────────────── */
.traj-overview-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}
.traj-overview-title {
    font-size: 13px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.traj-target-row {
    display: flex;
    gap: 24px;
    align-items: center;
    margin-bottom: 8px;
}
.traj-target-label {
    font-size: 12px;
    color: #667085;
}
.traj-target-value {
    font-size: 14px;
    font-weight: 600;
    color: #1F2D3D;
}
.traj-stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.traj-stat-box {
    flex: 1;
    min-width: 120px;
    background: #F3F6FA;
    border: 1px solid #D9E2EF;
    border-radius: 4px;
    padding: 8px 12px;
    text-align: center;
}
.traj-stat-value {
    font-size: 18px;
    font-weight: 700;
    color: #0B4EC2;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}
.traj-stat-label {
    font-size: 11px;
    color: #667085;
    margin-top: 2px;
}

/* ── 摄像头表格 ─────────────────────────────── */
.traj-cam-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.traj-cam-table th {
    background: #E8F1FF;
    color: #0B4EC2;
    padding: 6px 10px;
    text-align: left;
    border: 1px solid #D9E2EF;
    font-weight: 600;
    font-size: 12px;
}
.traj-cam-table td {
    padding: 5px 10px;
    border: 1px solid #D9E2EF;
    color: #1F2D3D;
    font-size: 13px;
}
.traj-cam-table tr:nth-child(even) {
    background: #F8FAFD;
}
.traj-cam-table tr:hover {
    background: #EAF6FF;
}

/* ── 推断路段说明 ───────────────────────────── */
.traj-infer-note {
    background: #FFF8E1;
    border: 1px solid #F59E0B;
    border-radius: 4px;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 12px;
    color: #1F2D3D;
}
.traj-break-note {
    background: #FFEBEE;
    border: 1px solid #E53935;
    border-radius: 4px;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 12px;
    color: #1F2D3D;
}

/* ── 研判摘要面板 ───────────────────────────── */
.traj-analysis-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 10px;
}
.traj-analysis-title {
    font-size: 13px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.traj-analysis-row {
    margin-bottom: 8px;
}
.traj-analysis-label {
    font-size: 11px;
    color: #667085;
}
.traj-analysis-value {
    font-size: 13px;
    color: #1F2D3D;
    font-weight: 500;
}
.traj-rating-high {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    background: #2E7D32;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
}
.traj-rating-mid {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    background: #F59E0B;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
}
.traj-rating-low {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    background: #E53935;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
}

/* ── 图例 ───────────────────────────────────── */
.traj-legend {
    display: flex;
    gap: 18px;
    padding: 8px 0;
    font-size: 12px;
    color: #667085;
}
.traj-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
}
.traj-legend-line {
    display: inline-block;
    width: 22px;
    height: 3px;
    border-radius: 2px;
}

/* ── 空状态 ─────────────────────────────────── */
.traj-empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #667085;
    font-size: 14px;
}
.traj-empty-state-icon {
    font-size: 36px;
    color: #D9E2EF;
    margin-bottom: 12px;
}
</style>
"""


def render():
    """渲染轨迹回溯页面"""
    st.markdown(_TRAJ_CSS, unsafe_allow_html=True)

    st.markdown("#### 轨迹回溯")

    traj_data = st.session_state.get("trajectory_data")
    if traj_data is None:
        # 检查是否有 selected_track_id 或 selected_instance_id，尝试调用 API
        selected_track_id = st.session_state.get("selected_track_id", "")
        selected_instance_id = st.session_state.get("selected_instance_id", "")
        if selected_track_id or selected_instance_id:
            _load_and_render_trajectory(selected_track_id, selected_instance_id)
            return
        st.markdown("""
        <div class="traj-empty-state">
            <div class="traj-empty-state-icon">\u2298</div>
            <div>\u5c1a\u672a\u9009\u62e9\u76ee\u6807\u8fdb\u884c\u8f68\u8ff9\u56de\u6eaf\u3002\u8bf7\u5148\u5b8c\u6210\u76ee\u6807\u68c0\u7d22\u5e76\u786e\u8ba4\u76ee\u6807\u3002</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("\u8fd4\u56de\u68c0\u7d22"):
            st.session_state["current_page"] = "search"
            st.rerun()
        return

    # ── 解构数据 ──
    target = traj_data.get("target_instance", {})
    overall_conf = traj_data.get("overall_confidence", 0)
    obs_nodes = traj_data.get("observation_nodes", [])
    obs_segs = traj_data.get("observation_segments", [])
    inf_segs = traj_data.get("inference_segments", [])
    cand_paths = traj_data.get("candidate_paths", [])
    cross_traj = traj_data.get("cross_camera_trajectory")

    # ── 上部：轨迹概览卡片 ──
    _render_overview_card(target, overall_conf, obs_nodes, obs_segs, inf_segs, cross_traj)

    # ── 轨迹地图 ──
    if cross_traj and cross_traj.get("camera_sequence"):
        st.markdown("##### 轨迹时空分布图")
        try:
            fig = render_trajectory_plot(cross_traj["camera_sequence"])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <div style="display:flex; gap:24px; align-items:center; margin-top:4px; font-size:13px; color:#555;">
                  <span><b>跨镜间隙置信度：</b></span>
                  <span style="color:#00FF00;">┅┅ 绿色虚线 &lt;15s 高置信度</span>
                  <span style="color:#FFD700;">┅┅ 黄色虚线 15~60s 中置信度</span>
                  <span style="color:#FF4444;">···· 红色点线 ≥60s 低置信度</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"轨迹图渲染失败: {e}")
            st.write(f"错误详情: {str(e)}")

    # ── 中部 + 右侧 ──
    col_main, col_side = st.columns([3, 2])

    with col_main:
        _render_trajectory_details(obs_nodes, obs_segs, inf_segs, cross_traj)

    with col_side:
        _render_analysis_summary(target, overall_conf, obs_nodes, inf_segs, cand_paths)

    # ── 导出 ──
    _render_export_section(traj_data)


# ============================================================
# 跨镜摄像头序列渲染（含帧缩略图）
# ============================================================

def _render_cross_camera_sequence(cross_traj: dict):
    """渲染跨镜摄像头序列，每个摄像头显示帧缩略图"""
    camera_seq = cross_traj.get("camera_sequence", [])
    if not camera_seq:
        st.info("无摄像头通过记录")
        return

    for idx, cam in enumerate(camera_seq, 1):
        cam_name = cam.get("camera_name", cam.get("camera_id", ""))
        arrival = cam.get("arrival_time", "--")
        departure = cam.get("departure_time", "--")
        duration = cam.get("duration_seconds", 0)
        det_count = cam.get("detection_count", 0)
        direction = cam.get("direction", "")
        frames = cam.get("frames", [])

        # 摄像头标题 - 使用Streamlit原生组件
        duration_str = f"{duration}秒" if duration > 0 else "--"
        st.markdown(
            f'<div style="background:#E8F1FF;border:1px solid #D9E2EF;border-radius:4px;'
            f'padding:8px 12px;margin:8px 0 4px 0;">'
            f'<span style="color:#0B4EC2;font-weight:600;font-size:13px;">'
            f'{idx}. {_html.escape(str(cam_name))}</span>'
            f'<span style="color:#667085;font-size:11px;margin-left:12px;">'
            f'{_html.escape(str(arrival))} → {_html.escape(str(departure))}</span>'
            f'<span style="color:#667085;font-size:11px;margin-left:12px;">'
            f'停留 {_html.escape(duration_str)} | {det_count}帧</span>'
            f'<span style="color:#667085;font-size:11px;margin-left:12px;">'
            f'{_html.escape(str(direction))}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 帧缩略图（每行最多5张，整齐网格布局）
        if frames:
            max_per_row = 5
            for row_start in range(0, len(frames), max_per_row):
                row_end = min(row_start + max_per_row, len(frames))
                thumb_cols = st.columns(row_end - row_start)
                for j in range(row_start, row_end):
                    frame = frames[j]
                    col_idx = j - row_start
                    with thumb_cols[col_idx]:
                        crop_path = frame.get("crop_path", "")
                        img_path = resolve_image_path(crop_path) if crop_path else None
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.markdown(
                                '<div style="background:#F3F6FA;border:1px solid #D9E2EF;'
                                'border-radius:4px;padding:8px;text-align:center;'
                                'color:#9AA4B2;font-size:10px;">无图片</div>',
                                unsafe_allow_html=True,
                            )
                        ts = frame.get("timestamp", "")
                        ts_short = ts.split(" ")[-1] if " " in ts else ts
                        st.caption(f"帧{frame.get('frame_id', '')} {ts_short}")
            if len(frames) > 8:
                st.caption(f"...共 {len(frames)} 帧")
        st.markdown("")  # 间距


def _load_and_render_trajectory(track_id: str, instance_id: str):
    """从 API 加载跨镜轨迹数据并渲染"""
    from frontend.utils import api_trajectory

    # 使用 container 包裹加载状态，避免 DOM 冲突
    load_container = st.container()
    with load_container:
        st.info("正在加载跨镜轨迹数据...")

    try:
        resp = api_trajectory(track_id=track_id, instance_id=instance_id)
        if resp and resp.get("success"):
            traj = resp["trajectory"]
            # 构建 trajectory_data 格式
            from frontend.utils import _convert_trajectory_response
            cand = st.session_state.get("confirmed_candidate", {})
            traj_data = _convert_trajectory_response(traj, cand)
            st.session_state["trajectory_data"] = traj_data
            st.rerun()
        else:
            load_container.warning("无法加载跨镜轨迹数据，请确认目标信息是否正确。")
    except Exception as e:
        load_container.warning(f"加载轨迹数据失败: {e}")

    if st.button("返回检索"):
        st.session_state["current_page"] = "search"
        st.rerun()


# ============================================================
# 上部 — 轨迹概览卡片
# ============================================================

def _render_overview_card(target, overall_conf, obs_nodes, obs_segs, inf_segs, cross_traj=None):
    """渲染轨迹概览卡片"""
    icon = get_target_type_icon(target.get("target_type", ""))
    type_cn = type_label(target.get("target_type", ""))
    plate = target.get("plate_number", "") or "--"
    color_attr = target.get("attributes", {}).get("颜色", "--")

    # 如果有跨镜轨迹数据，优先使用
    if cross_traj:
        first_time = cross_traj.get("first_appearance", "--")
        last_time = cross_traj.get("last_appearance", "--")
        num_cameras = cross_traj.get("total_cameras", 0)
        total_duration = cross_traj.get("total_duration_seconds", 0)
        total_detections = cross_traj.get("total_detections", 0)
        vehicle_id = cross_traj.get("vehicle_id", "")
    else:
        # 计算统计指标
        timestamps = [n.get("timestamp", "") for n in obs_nodes if n.get("timestamp")]
        first_time = timestamps[0] if timestamps else "--"
        last_time = timestamps[-1] if timestamps else "--"
        num_cameras = len(obs_nodes)
        total_duration = 0
        total_detections = 0
        vehicle_id = ""
        if len(timestamps) >= 2:
            try:
                t0 = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
                t1 = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
                total_duration = (t1 - t0).total_seconds()
            except Exception:
                pass

    # 轨迹总距离（简单估算：推断段距离之和）
    total_distance = sum(
        seg.get("estimated_travel_time", 0) * 8  # 粗略估算：8m/s
        for seg in inf_segs
    )

    # 最大时间空档
    max_gap = 0
    check_timestamps = [c.get("arrival_time", "") for c in cross_traj.get("camera_sequence", [])] if cross_traj else [n.get("timestamp", "") for n in obs_nodes if n.get("timestamp")]
    for i in range(len(check_timestamps) - 1):
        try:
            ta = datetime.strptime(check_timestamps[i], "%Y-%m-%d %H:%M:%S")
            tb = datetime.strptime(check_timestamps[i + 1], "%Y-%m-%d %H:%M:%S")
            gap = (tb - ta).total_seconds()
            max_gap = max(max_gap, gap)
        except Exception:
            pass

    duration_str = f"{total_duration / 60:.0f} 分钟" if total_duration >= 60 else f"{total_duration:.0f} 秒"
    distance_str = f"{total_distance / 1000:.1f} km" if total_distance >= 1000 else f"{total_distance:.0f} m"
    gap_str = f"{max_gap / 60:.0f} 分钟" if max_gap >= 60 else f"{max_gap:.0f} 秒"
    detections_str = str(total_detections) if total_detections else "--"

    # 跨摄像头序列展示 - 使用Streamlit原生组件避免HTML泄露
    camera_ids = st.session_state.get("trajectory_camera_ids", [])
    if cross_traj:
        camera_ids = [c["camera_id"] for c in cross_traj.get("camera_sequence", [])]

    plate_display = plate if plate != "--" else "--"

    # 概览卡片 - 使用st.container + st.columns替代大段HTML
    with st.container():
        st.markdown('<div class="traj-overview-card"><div class="traj-overview-title">轨迹概览</div></div>', unsafe_allow_html=True)

        # 目标信息行
        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            st.markdown(f'<span style="font-size:11px;color:#667085;">目标类型</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size:14px;font-weight:600;color:#1F2D3D;">{icon} {_html.escape(type_cn)}</span>', unsafe_allow_html=True)
        with info_c2:
            st.markdown(f'<span style="font-size:11px;color:#667085;">车牌号码</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size:14px;font-weight:600;color:#0B4EC2;">{_html.escape(plate_display)}</span>', unsafe_allow_html=True)
        with info_c3:
            st.markdown(f'<span style="font-size:11px;color:#667085;">车身颜色</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size:14px;font-weight:600;color:#1F2D3D;">{_html.escape(str(color_attr))}</span>', unsafe_allow_html=True)

        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        # 统计指标行
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        with s1:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{_html.escape(str(first_time))}</div><div class="traj-stat-label">首次出现</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{_html.escape(str(last_time))}</div><div class="traj-stat-label">最后出现</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{num_cameras}</div><div class="traj-stat-label">经过摄像头数</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{_html.escape(duration_str)}</div><div class="traj-stat-label">轨迹总时长</div></div>', unsafe_allow_html=True)
        with s5:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{_html.escape(detections_str)}</div><div class="traj-stat-label">检测帧数</div></div>', unsafe_allow_html=True)
        with s6:
            st.markdown(f'<div class="traj-stat-box"><div class="traj-stat-value">{_html.escape(gap_str)}</div><div class="traj-stat-label">最大时间空档</div></div>', unsafe_allow_html=True)

        # 跨摄像头序列 - 使用columns分块渲染
        if camera_ids:
            st.divider()
            st.caption("跨摄像头序列：")
            cam_cols = st.columns(min(len(camera_ids), 6))
            for ci, cid in enumerate(camera_ids):
                col = cam_cols[ci % len(cam_cols)]
                cam_display_name = CAMERA_NAME_MAP.get(cid, cid)
                with col:
                    st.markdown(
                        f'<div style="background:#E8F1FF;color:#0B4EC2;font-size:11px;font-weight:600;'
                        f'padding:2px 8px;border-radius:3px;text-align:center;margin:2px 0;">'
                        f'{_html.escape(cam_display_name)}</div>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# 中部 — 轨迹详情
# ============================================================

def _render_trajectory_details(obs_nodes, obs_segs, inf_segs, cross_traj=None):
    """渲染轨迹详情：摄像头表格 + 推断路段 + 断点 + 跨镜帧展示"""
    st.markdown('<div class="traj-analysis-title">经过摄像头列表</div>', unsafe_allow_html=True)

    # 如果有跨镜轨迹数据，显示增强的摄像头列表（含帧缩略图）
    if cross_traj and cross_traj.get("camera_sequence"):
        _render_cross_camera_sequence(cross_traj)
    elif obs_nodes:
        rows = []
        for i, node in enumerate(obs_nodes):
            cam_name = _html.escape(str(CAMERA_NAME_MAP.get(node.get("camera_id", ""), node.get("camera_name", node.get("camera_id", "")))))
            ts = _html.escape(str(node.get("timestamp", "")))
            direction = ""
            enter_time = ""
            leave_time = ""
            if i < len(obs_segs):
                direction = _html.escape(str(obs_segs[i].get("direction", "")))
                enter_time = _html.escape(str(obs_segs[i].get("start_time", "")))
                leave_time = _html.escape(str(obs_segs[i].get("end_time", "")))
            conf = node.get("confidence", 0)
            conf_col = confidence_color(conf)
            rows.append(
                f"<tr>"
                f"<td style='text-align:center;'>{i + 1}</td>"
                f"<td>{cam_name}</td>"
                f"<td style='text-align:center;'>{enter_time or ts}</td>"
                f"<td style='text-align:center;'>{leave_time or '--'}</td>"
                f"<td style='text-align:center;'>{direction}</td>"
                f"<td style='text-align:center;color:{conf_col};font-weight:600;'>{conf:.0%}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<table class="traj-cam-table">'
            f'<tr><th style="text-align:center;">序号</th><th>摄像头名称</th>'
            f'<th style="text-align:center;">到达时间</th><th style="text-align:center;">离开时间</th>'
            f'<th style="text-align:center;">方向</th><th style="text-align:center;">置信度</th></tr>'
            f'{"".join(rows)}</table>',
            unsafe_allow_html=True,
        )
    else:
        st.info("无摄像头通过记录")

    # 图例
    st.markdown("""
    <div class="traj-legend">
        <div class="traj-legend-item">
            <span class="traj-legend-line" style="background:#2E7D32;"></span>
            <span>已确认轨迹段</span>
        </div>
        <div class="traj-legend-item">
            <span class="traj-legend-line" style="background:#F59E0B;"></span>
            <span>推断轨迹段</span>
        </div>
        <div class="traj-legend-item">
            <span class="traj-legend-line" style="background:#E53935;border-top:2px dashed #E53935;height:0;"></span>
            <span>低置信轨迹段</span>
        </div>
        <div class="traj-legend-item">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#0B4EC2;"></span>
            <span>摄像头点位</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 推断路段说明
    if inf_segs:
        infer_items = []
        for seg in inf_segs:
            src = _html.escape(str(CAMERA_NAME_MAP.get(seg.get("source_camera_id", ""), seg.get("source_camera_name", ""))))
            tgt = _html.escape(str(CAMERA_NAME_MAP.get(seg.get("target_camera_id", ""), seg.get("target_camera_name", ""))))
            conf = seg.get("confidence", 0)
            conf_tag = confidence_label(conf)
            est_t = seg.get("estimated_travel_time", 0)
            low_class = ' style="color:#E53935;font-weight:600;"' if conf < 0.5 else ""
            infer_items.append(
                f"<li{low_class}>{src} → {tgt}（置信度 {conf:.0%}，{conf_tag}，预估 {est_t:.0f}s）</li>"
            )
        st.markdown(f"""
        <div class="traj-infer-note">
            <strong>推断路段说明</strong>（以下路段为系统推断，非直接观测）
            <ul style="margin:4px 0 0 0;padding-left:18px;">{"".join(infer_items)}</ul>
        </div>
        """, unsafe_allow_html=True)

    # 轨迹断点标注
    if len(obs_nodes) >= 2:
        timestamps = [n.get("timestamp", "") for n in obs_nodes]
        gaps = []
        for i in range(len(timestamps) - 1):
            try:
                ta = datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
                tb = datetime.strptime(timestamps[i + 1], "%Y-%m-%d %H:%M:%S")
                gap = (tb - ta).total_seconds()
                if gap > 300:  # 超过 5 分钟视为断点
                    cam_a = _html.escape(str(obs_nodes[i].get("camera_name", "")))
                    cam_b = _html.escape(str(obs_nodes[i + 1].get("camera_name", "")))
                    gaps.append(f"<li>{cam_a} → {cam_b}：空档 {gap / 60:.0f} 分钟</li>")
            except Exception:
                pass
        if gaps:
            st.markdown(f"""
            <div class="traj-break-note">
                <strong>轨迹断点标注</strong>
                <ul style="margin:4px 0 0 0;padding-left:18px;">{"".join(gaps)}</ul>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 右侧 — 研判摘要
# ============================================================

def _render_analysis_summary(target, overall_conf, obs_nodes, inf_segs, cand_paths):
    """渲染右侧研判摘要面板"""
    st.markdown('<div class="traj-analysis-title">研判摘要</div>', unsafe_allow_html=True)

    # 系统判断结论
    if overall_conf >= 0.7:
        conclusion = "目标轨迹连续性良好，多卡口确认经过，系统判断为同一目标。"
    elif overall_conf >= 0.4:
        conclusion = "目标轨迹存在部分推断段，整体可信度中等，建议补充周边卡口证据。"
    else:
        conclusion = "目标轨迹断点较多，推断段置信度低，需人工复核确认。"

    # 可能驶离方向
    last_direction = "未知"
    if obs_nodes:
        # 从最后一个推断段或候选路径获取方向
        if cand_paths:
            last_direction = cand_paths[0].get("description", "").split("→")[-1].strip() or "未知"
        elif inf_segs:
            last_direction = CAMERA_NAME_MAP.get(
                inf_segs[-1].get("target_camera_id", ""),
                inf_segs[-1].get("target_camera_name", "未知"),
            )

    # 可信度评级
    if overall_conf >= 0.7:
        rating_label = "高"
        rating_class = "traj-rating-high"
    elif overall_conf >= 0.4:
        rating_label = "中"
        rating_class = "traj-rating-mid"
    else:
        rating_label = "低"
        rating_class = "traj-rating-low"

    st.markdown(f"""
    <div class="traj-analysis-card">
        <div class="traj-analysis-row">
            <div class="traj-analysis-label">系统判断结论</div>
            <div class="traj-analysis-value">{_html.escape(conclusion)}</div>
        </div>
        <div class="traj-analysis-row">
            <div class="traj-analysis-label">可能驶离方向</div>
            <div class="traj-analysis-value">{_html.escape(last_direction)}</div>
        </div>
        <div class="traj-analysis-row">
            <div class="traj-analysis-label">轨迹可信度评级</div>
            <div><span class="{rating_class}">{rating_label}</span>
                 &nbsp;<span style="font-size:12px;color:#667085;">({overall_conf:.0%})</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 建议操作
    st.markdown('<div class="traj-analysis-label" style="margin-bottom:6px;">建议操作</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("确认轨迹", key="btn_confirm_traj", use_container_width=True):
            st.success("轨迹已确认")
    with col2:
        if st.button("补充证据", key="btn_supplement", use_container_width=True):
            st.info("请在时间轴页面标记关键证据节点")
    with col3:
        if st.button("标记异常", key="btn_mark_abnormal", use_container_width=True):
            st.warning("已标记为异常轨迹，待人工复核")


# ============================================================
# 导出功能
# ============================================================

_EXPORT_CSS = """
<style>
.export-section {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.export-title {
    color: #1F2D3D;
    font-size: 13px;
    font-weight: 600;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.export-desc {
    color: #667085;
    font-size: 11px;
    margin-top: 6px;
}
</style>
"""


def _generate_pdf_report(traj_data: dict) -> bytes:
    """使用 reportlab 生成轨迹报告 PDF，返回字节数据"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase.pdfmetrics import registerFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER

    font_name = "Helvetica"
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in font_paths:
        try:
            registerFont(TTFont("CNFont", fp))
            font_name = "CNFont"
            break
        except Exception:
            continue

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    color_primary = HexColor("#0B4EC2")
    color_text = HexColor("#1F2D3D")
    color_light = HexColor("#667085")

    style_title = ParagraphStyle(
        "Title", fontName=font_name, fontSize=18,
        textColor=color_primary, alignment=TA_CENTER, spaceAfter=12,
    )
    style_h2 = ParagraphStyle(
        "H2", fontName=font_name, fontSize=13,
        textColor=color_primary, spaceBefore=14, spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "Body", fontName=font_name, fontSize=10,
        textColor=color_text, leading=16,
    )
    style_small = ParagraphStyle(
        "Small", fontName=font_name, fontSize=9,
        textColor=color_light, leading=14,
    )
    style_footer = ParagraphStyle(
        "Footer", fontName=font_name, fontSize=8,
        textColor=color_light, alignment=TA_CENTER,
    )

    elements = []

    target = traj_data.get("target_instance", {})
    overall_conf = traj_data.get("overall_confidence", 0)
    obs_nodes = traj_data.get("observation_nodes", [])
    obs_segs = traj_data.get("observation_segments", [])
    inf_segs = traj_data.get("inference_segments", [])
    cand_paths = traj_data.get("candidate_paths", [])

    elements.append(Paragraph("轨迹回溯报告", style_title))
    elements.append(Paragraph(
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        style_small,
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("一、轨迹概要", style_h2))
    type_cn = type_label(target.get("target_type", ""))
    plate = target.get("plate_number", "") or "--"
    summary_text = (
        f"目标类型: {type_cn}  |  "
        f"车牌号码: {plate}  |  "
        f"综合可信度: {overall_conf:.0%}<br/>"
        f"确认经过卡口: {len(obs_nodes)} 个  |  "
        f"推断行驶段: {len(inf_segs)} 段  |  "
        f"可能路线: {len(cand_paths)} 条"
    )
    elements.append(Paragraph(summary_text, style_body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("二、卡口通过记录", style_h2))
    if obs_nodes:
        table_data = [["序号", "卡口名称", "到达时间", "离开时间", "方向", "置信度"]]
        for i, node in enumerate(obs_nodes):
            cam_name = CAMERA_NAME_MAP.get(node.get("camera_id", ""), node.get("camera_name", node.get("camera_id", "")))
            ts = node.get("timestamp", "")
            enter_t = leave_t = ""
            direction = ""
            if i < len(obs_segs):
                enter_t = obs_segs[i].get("start_time", "")
                leave_t = obs_segs[i].get("end_time", "")
                direction = obs_segs[i].get("direction", "")
            conf = node.get("confidence", 0)
            table_data.append([str(i + 1), cam_name, enter_t or ts, leave_t or "--", direction, f"{conf:.0%}"])

        t = Table(table_data, colWidths=[30, 140, 80, 80, 70, 50])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), color_primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#E8F1FF"), HexColor("#FFFFFF")]),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("无卡口通过记录", style_small))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("三、推断行驶段", style_h2))
    if inf_segs:
        inf_table = [["路段", "置信度", "预估行驶时间", "实际行驶时间"]]
        for seg in inf_segs:
            src = CAMERA_NAME_MAP.get(seg.get("source_camera_id", ""), seg.get("source_camera_name", ""))
            tgt = CAMERA_NAME_MAP.get(seg.get("target_camera_id", ""), seg.get("target_camera_name", ""))
            conf = seg.get("confidence", 0)
            est_t = seg.get("estimated_travel_time", 0)
            act_t = seg.get("actual_travel_time", 0)
            inf_table.append([
                f"{src} -> {tgt}", f"{conf:.0%}",
                f"{est_t:.0f}秒", f"{act_t:.0f}秒",
            ])
        t2 = Table(inf_table, colWidths=[180, 60, 80, 80])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), color_primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#E8F1FF"), HexColor("#FFFFFF")]),
        ]))
        elements.append(t2)
    else:
        elements.append(Paragraph("无推断行驶段", style_small))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("四、可能路线", style_h2))
    if cand_paths:
        path_table = [["编号", "路线描述", "可信度", "距离"]]
        for i, path in enumerate(cand_paths):
            conf = path.get("confidence", 0)
            desc = path.get("description", "")
            dist = path.get("distance_meters", 0)
            path_table.append([f"路线{i + 1}", desc, f"{conf:.0%}", f"{dist:.0f}m"])
        t3 = Table(path_table, colWidths=[50, 200, 60, 60])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), color_primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#E8F1FF"), HexColor("#FFFFFF")]),
        ]))
        elements.append(t3)
    else:
        elements.append(Paragraph("无可能路线", style_small))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "--- 交通风险感知集成指挥平台 ---", style_footer,
    ))

    doc.build(elements)
    return buf.getvalue()


def _generate_excel_report(traj_data: dict) -> bytes:
    """使用 openpyxl 生成轨迹数据 Excel，返回字节数据"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()

    header_font = Font(name="Microsoft YaHei", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="0B4EC2", end_color="0B4EC2", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")
    cell_font = Font(name="Microsoft YaHei", size=10)
    cell_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="CCCCCC"),
        top=Side(style="thin", color="CCCCCC"),
        bottom=Side(style="thin", color="CCCCCC"),
    )
    alt_fill = PatternFill(start_color="F0F4F8", end_color="F0F4F8", fill_type="solid")

    target = traj_data.get("target_instance", {})
    overall_conf = traj_data.get("overall_confidence", 0)
    obs_nodes = traj_data.get("observation_nodes", [])
    obs_segs = traj_data.get("observation_segments", [])
    inf_segs = traj_data.get("inference_segments", [])
    cand_paths = traj_data.get("candidate_paths", [])

    def _style_header(ws, row_idx, col_count):
        for col in range(1, col_count + 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = thin_border

    def _style_data_row(ws, row_idx, col_count, is_alt=False):
        for col in range(1, col_count + 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.font = cell_font
            cell.alignment = cell_align
            cell.border = thin_border
            if is_alt:
                cell.fill = alt_fill

    ws1 = wb.active
    ws1.title = "检测记录"

    ws1.merge_cells("A1:F1")
    title_cell = ws1["A1"]
    title_cell.value = "轨迹回溯 - 检测记录"
    title_cell.font = Font(name="Microsoft YaHei", size=14, bold=True, color="0B4EC2")
    title_cell.alignment = Alignment(horizontal="center")

    ws1.merge_cells("A2:F2")
    ws1["A2"].value = f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws1["A2"].font = Font(name="Microsoft YaHei", size=9, color="888888")
    ws1["A2"].alignment = Alignment(horizontal="center")

    info_rows = [
        ("目标类型", type_label(target.get("target_type", ""))),
        ("车牌号码", target.get("plate_number", "--") or "--"),
        ("综合可信度", f"{overall_conf:.0%}"),
        ("确认卡口数", str(len(obs_nodes))),
        ("推断行驶段", str(len(inf_segs))),
        ("可能路线", str(len(cand_paths))),
    ]
    for i, (k, v) in enumerate(info_rows):
        r = i + 4
        ws1.cell(row=r, column=1, value=k).font = Font(
            name="Microsoft YaHei", size=10, bold=True,
        )
        ws1.cell(row=r, column=2, value=v).font = cell_font

    start_row = 12
    headers = ["序号", "卡口名称", "到达时间", "离开时间", "方向", "置信度"]
    for col, h in enumerate(headers, 1):
        ws1.cell(row=start_row, column=col, value=h)
    _style_header(ws1, start_row, len(headers))

    for i, node in enumerate(obs_nodes):
        r = start_row + 1 + i
        cam_name = CAMERA_NAME_MAP.get(node.get("camera_id", ""), node.get("camera_name", node.get("camera_id", "")))
        ts = node.get("timestamp", "")
        enter_t = leave_t = ""
        direction = ""
        if i < len(obs_segs):
            enter_t = obs_segs[i].get("start_time", "")
            leave_t = obs_segs[i].get("end_time", "")
            direction = obs_segs[i].get("direction", "")
        conf = node.get("confidence", 0)
        ws1.cell(row=r, column=1, value=i + 1)
        ws1.cell(row=r, column=2, value=cam_name)
        ws1.cell(row=r, column=3, value=enter_t or ts)
        ws1.cell(row=r, column=4, value=leave_t or "--")
        ws1.cell(row=r, column=5, value=direction)
        ws1.cell(row=r, column=6, value=f"{conf:.0%}")
        _style_data_row(ws1, r, len(headers), is_alt=(i % 2 == 1))

    ws1.column_dimensions["A"].width = 8
    ws1.column_dimensions["B"].width = 28
    ws1.column_dimensions["C"].width = 14
    ws1.column_dimensions["D"].width = 14
    ws1.column_dimensions["E"].width = 14
    ws1.column_dimensions["F"].width = 10

    ws2 = wb.create_sheet("轨迹数据")

    ws2.merge_cells("A1:E1")
    ws2["A1"].value = "推断行驶段"
    ws2["A1"].font = Font(name="Microsoft YaHei", size=12, bold=True, color="0B4EC2")

    inf_headers = ["起点卡口", "终点卡口", "可信度", "预估时间(秒)", "实际时间(秒)"]
    for col, h in enumerate(inf_headers, 1):
        ws2.cell(row=2, column=col, value=h)
    _style_header(ws2, 2, len(inf_headers))

    for i, seg in enumerate(inf_segs):
        r = 3 + i
        ws2.cell(row=r, column=1, value=CAMERA_NAME_MAP.get(seg.get("source_camera_id", ""), seg.get("source_camera_name", "")))
        ws2.cell(row=r, column=2, value=CAMERA_NAME_MAP.get(seg.get("target_camera_id", ""), seg.get("target_camera_name", "")))
        ws2.cell(row=r, column=3, value=f"{seg.get('confidence', 0):.0%}")
        ws2.cell(row=r, column=4, value=f"{seg.get('estimated_travel_time', 0):.0f}")
        ws2.cell(row=r, column=5, value=f"{seg.get('actual_travel_time', 0):.0f}")
        _style_data_row(ws2, r, len(inf_headers), is_alt=(i % 2 == 1))

    path_start = 3 + len(inf_segs) + 2
    ws2.merge_cells(f"A{path_start}:E{path_start}")
    ws2.cell(row=path_start, column=1, value="可能路线")
    ws2.cell(row=path_start, column=1).font = Font(
        name="Microsoft YaHei", size=12, bold=True, color="0B4EC2",
    )

    path_headers = ["编号", "路线描述", "可信度", "距离(米)", "预估时间(秒)"]
    for col, h in enumerate(path_headers, 1):
        ws2.cell(row=path_start + 1, column=col, value=h)
    _style_header(ws2, path_start + 1, len(path_headers))

    for i, path in enumerate(cand_paths):
        r = path_start + 2 + i
        ws2.cell(row=r, column=1, value=f"路线{i + 1}")
        ws2.cell(row=r, column=2, value=path.get("description", ""))
        ws2.cell(row=r, column=3, value=f"{path.get('confidence', 0):.0%}")
        ws2.cell(row=r, column=4, value=f"{path.get('distance_meters', 0):.0f}")
        ws2.cell(row=r, column=5, value=f"{path.get('estimated_time', 0):.0f}")
        _style_data_row(ws2, r, len(path_headers), is_alt=(i % 2 == 1))

    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 32
    ws2.column_dimensions["C"].width = 10
    ws2.column_dimensions["D"].width = 14
    ws2.column_dimensions["E"].width = 14

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _render_export_section(traj_data: dict):
    """渲染导出轨迹功能区域"""
    st.markdown(_EXPORT_CSS, unsafe_allow_html=True)

    icon_export = render_icon("export", size=16, color="#0B4EC2")

    st.markdown(f"""
    <div class="export-section">
        <div class="export-title">
            {icon_export} &nbsp;导出轨迹数据
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_pdf, col_xlsx = st.columns(2)

    with col_pdf:
        try:
            pdf_bytes = _generate_pdf_report(traj_data)
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="导出 PDF 报告",
                data=pdf_bytes,
                file_name=f"trajectory_report_{ts_tag}.pdf",
                mime="application/pdf",
                key="btn_export_pdf",
            )
        except ImportError:
            st.warning("reportlab 未安装，无法生成 PDF")

    with col_xlsx:
        try:
            xlsx_bytes = _generate_excel_report(traj_data)
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="导出 Excel 表格",
                data=xlsx_bytes,
                file_name=f"trajectory_data_{ts_tag}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_export_xlsx",
            )
        except ImportError:
            st.warning("openpyxl 未安装，无法生成 Excel")

    st.markdown("""
    <div class="export-desc">
        PDF 报告包含轨迹概要、卡口列表、时间线与置信度；Excel 表格包含检测记录与轨迹数据。
    </div>
    """, unsafe_allow_html=True)
