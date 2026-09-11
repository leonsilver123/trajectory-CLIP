"""
frontend.pages.dashboard - 仪表盘页面（公安交管浅色风格）

展示系统运行状态、检索统计、回溯统计和摄像头覆盖热力图。
图表使用 plotly 浅色主题，配色匹配警蓝体系。
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd

from frontend.utils import (
    api_dashboard_stats, api_camera_list,
    cameras_to_pydeck_layer, make_pydeck_view_state, MAP_CENTER,
    load_cityflow_results, has_cityflow_results,
    CAMERA_NAME_MAP,
)


def _build_dashboard_stats_from_cityflow() -> dict:
    """从 CityFlow 数据构建仪表盘统计数据"""
    data = load_cityflow_results()
    if data is None:
        return {
            "camera_count": 0, "camera_online": 0,
            "instance_count": 0, "tracklet_count": 0,
            "today_search_count": 0, "today_backtrack_count": 0,
            "search_by_type": {}, "search_frequency": [],
            "confidence_distribution": [],
        }
    detections = data.get("detections", [])
    camera_ids = set()
    type_counts = {}
    conf_scores = []
    for det in detections:
        cam = det.get("camera_id", "")
        if cam:
            camera_ids.add(cam)
        t = det.get("target_type", "vehicle")
        type_counts[t] = type_counts.get(t, 0) + 1
        c = det.get("confidence", 0)
        if c > 0:
            conf_scores.append(c)
    conf_dist = []
    for lo, hi, label in [
        (0.0, 0.2, "0.0-0.2"), (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"), (0.6, 0.8, "0.6-0.8"), (0.8, 1.01, "0.8-1.0"),
    ]:
        cnt = sum(1 for c in conf_scores if lo <= c < hi)
        conf_dist.append({"range": label, "count": cnt})
    return {
        "camera_count": len(camera_ids),
        "camera_online": len(camera_ids),
        "instance_count": len(detections),
        "tracklet_count": len(data.get("tracks", [])),
        "today_search_count": 0,
        "today_backtrack_count": 0,
        "search_by_type": type_counts,
        "search_frequency": [],
        "confidence_distribution": conf_dist,
    }


# ── Plotly 浅色主题布局（符合UI规范 #F3F6FA 背景） ──
_PLOTLY_LIGHT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(color="#1F2D3D", size=11, family="Microsoft YaHei, sans-serif"),
    xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", tickfont=dict(color="#64748B")),
    yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", tickfont=dict(color="#64748B")),
    margin=dict(l=40, r=20, t=30, b=30),
    showlegend=False,
)

# 警蓝体系配色
_PALETTE = ["#0B4EC2", "#F5A623", "#28A745", "#DC3545", "#4285F4"]


# ── 页面级 CSS ──
_DASH_CSS = """
<style>
.dash-metric-row {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
}
.dash-metric-box {
    flex: 1;
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 4px;
    padding: 8px 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.dash-metric-value {
    color: #0B4EC2;
    font-size: 20px;
    font-weight: 700;
}
.dash-metric-label {
    color: #667085;
    font-size: 11px;
    margin-top: 2px;
}
.dash-chart-panel {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.dash-chart-title {
    color: #1F2D3D;
    font-size: 12px;
    font-weight: 600;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 3px;
    margin-bottom: 6px;
}
.dash-legend {
    display: flex;
    gap: 16px;
    padding: 6px 0;
    font-size: 11px;
    color: #667085;
}
.dash-legend-dot {
    display: inline-block;
    width: 10px; height: 10px;
    vertical-align: middle;
    margin-right: 4px;
}
</style>
"""


def render():
    """渲染仪表盘页面"""
    st.markdown(_DASH_CSS, unsafe_allow_html=True)

    st.markdown("#### 系统仪表盘")

    # ── 获取数据 ──
    stats = api_dashboard_stats()
    if stats is None:
        stats = _build_dashboard_stats_from_cityflow()

    cameras = api_camera_list()
    if cameras is None:
        cameras = []

    # ── 系统概览指标 ──
    online = stats.get("camera_online", stats.get("camera_count", 0))
    st.markdown(f"""
    <div class="dash-metric-row">
        <div class="dash-metric-box">
            <div class="dash-metric-value" style="color:#28A745;">{online}</div>
            <div class="dash-metric-label">在线摄像头</div>
        </div>
        <div class="dash-metric-box">
            <div class="dash-metric-value">{stats.get('instance_count', 0):,}</div>
            <div class="dash-metric-label">今日过车数</div>
        </div>
        <div class="dash-metric-box">
            <div class="dash-metric-value">{stats.get('instance_count', 0):,}</div>
            <div class="dash-metric-label">发现目标数</div>
        </div>
        <div class="dash-metric-box">
            <div class="dash-metric-value">{stats.get('tracklet_count', 0):,}</div>
            <div class="dash-metric-label">追踪记录数</div>
        </div>
        <div class="dash-metric-box">
            <div class="dash-metric-value">{stats.get('today_search_count', 0)}</div>
            <div class="dash-metric-label">今日检索</div>
        </div>
        <div class="dash-metric-box">
            <div class="dash-metric-value">{stats.get('today_backtrack_count', 0)}</div>
            <div class="dash-metric-label">今日回溯</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 图表区域 ──
    col_left, col_right = st.columns(2)

    with col_left:
        _render_search_type_pie(stats)

    with col_right:
        _render_search_frequency_line(stats)

    # ── 回溯统计 ──
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        _render_backtrack_stats(stats)

    with col_right2:
        _render_confidence_histogram(stats)

    # ── 摄像头覆盖分布 ──
    st.markdown("---")
    st.markdown("#### 摄像头覆盖分布")
    _render_camera_heatmap(cameras)


def _render_search_type_pie(stats: dict):
    """目标类型统计饼图"""
    st.markdown('<div class="dash-chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dash-chart-title">目标类型统计</div>', unsafe_allow_html=True)

    by_type = stats.get("search_by_type", {})
    if by_type:
        df = pd.DataFrame([{"类型": k, "次数": v} for k, v in by_type.items()])
        fig = px.pie(
            df, values="次数", names="类型", hole=0.4,
            color_discrete_sequence=_PALETTE[:len(by_type)],
        )
        layout_kwargs = {**_PLOTLY_LIGHT_LAYOUT, "height": 260, "showlegend": True,
                         "legend": dict(font=dict(color="#1F2D3D"))}
        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无检索数据")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_search_frequency_line(stats: dict):
    """系统使用统计折线图"""
    st.markdown('<div class="dash-chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dash-chart-title">系统使用统计</div>', unsafe_allow_html=True)

    freq = stats.get("search_frequency", [])
    if freq:
        df = pd.DataFrame(freq)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["count"],
            mode="lines+markers",
            line=dict(color="#0B4EC2", width=2),
            marker=dict(color="#F5A623", size=5),
            fill="tozeroy",
            fillcolor="rgba(11,78,194,0.1)",
        ))
        fig.update_layout(**_PLOTLY_LIGHT_LAYOUT, height=260)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无使用统计数据")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_backtrack_stats(stats: dict):
    """轨迹研判统计"""
    st.markdown('<div class="dash-chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dash-chart-title">轨迹研判统计</div>', unsafe_allow_html=True)

    backtrack_count = stats.get("today_backtrack_count", 0)
    edge_count = stats.get("edge_count", 0)
    st.markdown(f"""
    <div style="display:flex;gap:12px;padding:12px 0;">
        <div style="flex:1;text-align:center;">
            <div style="color:#0B4EC2;font-size:28px;font-weight:700;">{backtrack_count}</div>
            <div style="color:#667085;font-size:11px;margin-top:4px;">今日轨迹回溯</div>
        </div>
        <div style="flex:1;text-align:center;">
            <div style="color:#28A745;font-size:28px;font-weight:700;">{edge_count:,}</div>
            <div style="color:#667085;font-size:11px;margin-top:4px;">路径匹配记录</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _render_confidence_histogram(stats: dict):
    """轨迹研判统计直方图"""
    st.markdown('<div class="dash-chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dash-chart-title">轨迹研判统计</div>', unsafe_allow_html=True)

    dist = stats.get("confidence_distribution", [])
    if dist:
        df = pd.DataFrame(dist)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["range"], y=df["count"],
            marker=dict(
                color=df["count"],
                colorscale=[[0, "#0B4EC2"], [0.5, "#F5A623"], [1, "#28A745"]],
                showscale=False,
            ),
        ))
        fig.update_layout(**_PLOTLY_LIGHT_LAYOUT, height=260)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无研判统计数据")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_camera_heatmap(cameras: list):
    """摄像头覆盖热力图（pydeck HeatmapLayer）"""
    if not cameras:
        st.info("暂无摄像头数据")
        return

    heat_data = []
    for cam in cameras:
        weight = cam.get("today_detections", 500)
        heat_data.append({
            "position": [cam["longitude"], cam["latitude"]],
            "weight": min(weight / 100, 10),
            "name": CAMERA_NAME_MAP.get(cam.get("camera_id", ""), cam.get("name", cam.get("camera_id", ""))),
            "status": cam.get("status", "online"),
        })

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position="position",
        get_weight="weight",
        radius_pixels=80,
        intensity=1.5,
        threshold=0.1,
    )

    scatter_layer = cameras_to_pydeck_layer(cameras)

    view_state = pdk.ViewState(
        latitude=MAP_CENTER[0],
        longitude=MAP_CENTER[1],
        zoom=13,
        pitch=20,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[heat_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{name}</b><br/>状态: {status}"},
        map_style="mapbox://styles/mapbox/light-v11",
    )

    st.pydeck_chart(deck)

    # 图例
    online_count = sum(1 for c in cameras if c.get("status") == "online")
    offline_count = len(cameras) - online_count
    st.markdown(f"""
    <div class="dash-legend">
        <span><span class="dash-legend-dot" style="background:#28A745;"></span> 在线 ({online_count})</span>
        <span><span class="dash-legend-dot" style="background:#DC3545;"></span> 离线 ({offline_count})</span>
        <span><span class="dash-legend-dot" style="background:rgba(245,166,35,0.5);"></span> 检测热力密度</span>
    </div>
    """, unsafe_allow_html=True)
