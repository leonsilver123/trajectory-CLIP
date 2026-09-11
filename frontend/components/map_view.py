"""
frontend.components.map_view - 轨迹时空可视化组件

使用 Plotly 绘制基于真实坐标的轨迹散点图，支持时序演化动画。
不依赖在线地图服务，仅使用坐标系。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import yaml
import pandas as pd
import plotly.graph_objects as go


# ── 摄像头坐标缓存 ──
_camera_coords_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _load_camera_coords(
    config_path: str = "configs/cityflow_camera_metadata.yaml",
) -> Dict[str, Dict[str, Any]]:
    """从 cityflow_camera_metadata.yaml 加载摄像头坐标（带缓存）"""
    global _camera_coords_cache
    if _camera_coords_cache is not None:
        return _camera_coords_cache

    candidates = [
        Path(__file__).parent.parent.parent / config_path,
        Path(config_path),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            coords: Dict[str, Dict[str, Any]] = {}
            for cam in config.get("cameras", []):
                cam_id = cam["camera_id"]
                coords[cam_id] = {
                    "lat": cam["latitude"],
                    "lon": cam["longitude"],
                    "name": cam.get("name", cam.get("camera_name", cam_id)),
                    "scene": cam.get("scene", ""),
                }
            _camera_coords_cache = coords
            return coords

    return {}


def _build_enriched_sequence(
    camera_sequence: list,
    coords: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """为摄像头序列补充经纬度信息，返回 DataFrame"""
    enriched = []
    for idx, cam in enumerate(camera_sequence):
        cam_id = cam.get("camera_id", "")
        meta = coords.get(cam_id)

        if meta:
            lat, lon = meta["lat"], meta["lon"]
            cam_name = cam.get("camera_name", meta.get("name", cam_id))
        else:
            # 回退：如果没有坐标，用前一个点 + 小偏移
            if enriched:
                lat = enriched[-1]["lat"] + 0.0003
                lon = enriched[-1]["lon"] + 0.0003
            else:
                lat = 31.3050 + idx * 0.0003
                lon = 120.5850 + idx * 0.0003
            cam_name = cam.get("camera_name", cam_id)

        enriched.append({
            **cam,
            "lat": lat,
            "lon": lon,
            "order": idx,
            "camera_id": cam_id,
            "camera_name": cam_name,
            "time_label": cam.get("arrival_time", "--"),
        })

    return pd.DataFrame(enriched)


def render_trajectory_plot(camera_sequence: list) -> go.Figure:
    """
    使用 Plotly 绘制基于真实坐标的轨迹时空分布图（含动画帧）。

    Args:
        camera_sequence: 摄像头序列列表，每个元素包含：
            - camera_id: 摄像头ID (如 "c001")
            - camera_name: 摄像头名称
            - arrival_time: 到达时间
            - departure_time: 离开时间
            - duration_seconds: 停留时长
            - frames: 帧列表
            - direction: 方向

    Returns:
        Plotly Figure 对象（含动画帧，可在 Streamlit 中用 st.plotly_chart 渲染）
    """
    coords = _load_camera_coords()
    df = _build_enriched_sequence(camera_sequence, coords)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无轨迹数据", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#999"))
        return fig

    n = len(df)

    # ── 颜色方案：蓝→红渐变表示时间顺序 ──
    colorscale = [
        [0.0, "#1a6eff"],
        [0.25, "#6e4eff"],
        [0.5, "#c84eff"],
        [0.75, "#ff4e8a"],
        [1.0, "#ff2e2e"],
    ]

    # ── 基础静态层：所有点 + 不确定性编码连线 ──
    fig = go.Figure()

    # 绘制带不确定性编码的轨迹连线
    for i in range(len(df) - 1):
        # 解析时间字符串计算间隔
        try:
            dep_time_str = df.iloc[i]['departure_time']
            arr_time_str = df.iloc[i+1]['arrival_time']

            # 处理可能的时间格式（可能是 "HH:MM:SS" 或完整时间戳）
            if ':' in dep_time_str and len(dep_time_str.split(':')) == 3:
                if ' ' in dep_time_str:
                    dep_time = datetime.strptime(dep_time_str, '%Y-%m-%d %H:%M:%S')
                    arr_time = datetime.strptime(arr_time_str, '%Y-%m-%d %H:%M:%S')
                else:
                    dep_time = datetime.strptime(dep_time_str, '%H:%M:%S')
                    arr_time = datetime.strptime(arr_time_str, '%H:%M:%S')
            else:
                dep_time = datetime.strptime(dep_time_str, '%Y-%m-%d %H:%M:%S')
                arr_time = datetime.strptime(arr_time_str, '%Y-%m-%d %H:%M:%S')

            time_gap = (arr_time - dep_time).seconds
        except Exception:
            # 如果时间解析失败，使用默认中等置信度
            time_gap = 30

        # 根据时间间隔选择线型和颜色
        if time_gap < 15:
            line_style = dict(color='#00FF00', width=2.5, dash='dash')  # 绿色虚线
            confidence_label = '高'
        elif time_gap < 60:
            line_style = dict(color='#FFD700', width=2.5, dash='dash')  # 黄色虚线
            confidence_label = '中'
        else:
            line_style = dict(color='#FF4444', width=2.5, dash='dot')   # 红色点线
            confidence_label = '低'

        # 绘制虚线连接
        fig.add_trace(go.Scatter(
            x=[df.iloc[i]['lon'], df.iloc[i+1]['lon']],
            y=[df.iloc[i]['lat'], df.iloc[i+1]['lat']],
            mode='lines',
            line=line_style,
            name=f'{df.iloc[i]["camera_id"]}→{df.iloc[i+1]["camera_id"]} ({time_gap}s)',
            hoverinfo='text',
            text=f'时间间隔: {time_gap}秒<br/>置信度: {confidence_label}<br/>{df.iloc[i]["camera_name"]} → {df.iloc[i+1]["camera_name"]}'
        ))

        # 在连线中点添加时间间隔标注
        mid_lon = (df.iloc[i]['lon'] + df.iloc[i+1]['lon']) / 2
        mid_lat = (df.iloc[i]['lat'] + df.iloc[i+1]['lat']) / 2

        fig.add_annotation(
            x=mid_lon,
            y=mid_lat,
            text=f'{time_gap}s',
            showarrow=False,
            font=dict(size=9, color=line_style['color'], family="Arial Black"),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=line_style['color'],
            borderwidth=1.5,
            borderpad=3,
            opacity=0.9
        )

    # 带颜色的散点（按时间顺序着色）
    fig.add_trace(go.Scatter(
        x=df["lon"], y=df["lat"],
        mode="markers+text",
        marker=dict(
            size=14,
            color=df["order"],
            colorscale=colorscale,
            colorbar=dict(
                title="时间顺序",
                tickvals=list(range(n)),
                ticktext=[f"{i+1}" for i in range(n)],
                len=0.5,
                thickness=12,
                x=1.02,
            ),
            line=dict(width=1.5, color="white"),
            symbol="circle",
        ),
        text=df["camera_id"],
        textposition="top center",
        textfont=dict(size=10, color="#333"),
        name="摄像头",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "名称: %{customdata[0]}<br>"
            "到达: %{customdata[1]}<br>"
            "离开: %{customdata[2]}<br>"
            "经度: %{x:.4f}<br>"
            "纬度: %{y:.4f}"
            "<extra></extra>"
        ),
        customdata=df[["camera_name", "time_label", "departure_time"]].values,
    ))

    # 起点标注
    fig.add_annotation(
        x=df.iloc[0]["lon"], y=df.iloc[0]["lat"],
        text=" 起点",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#2ECC71",
        font=dict(size=12, color="#2ECC71", family="Arial Black"),
        ax=20, ay=-25,
    )

    # 终点标注
    if n > 1:
        fig.add_annotation(
            x=df.iloc[-1]["lon"], y=df.iloc[-1]["lat"],
            text=" 终点",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor="#E74C3C",
            font=dict(size=12, color="#E74C3C", family="Arial Black"),
            ax=20, ay=25,
        )

    # ── 动画帧：逐步显示轨迹演化 ──
    frames = []
    for k in range(1, n + 1):
        sub = df.iloc[:k]
        frame_traces = []

        # 已走过的连线（实线）
        if k > 1:
            frame_traces.append(go.Scatter(
                x=sub["lon"], y=sub["lat"],
                mode="lines",
                line=dict(color="#0B4EC2", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            ))
        else:
            # 第一帧：空线占位
            frame_traces.append(go.Scatter(
                x=[sub.iloc[0]["lon"]], y=[sub.iloc[0]["lat"]],
                mode="lines",
                line=dict(color="#0B4EC2", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            ))

        # 当前已到达的点
        frame_traces.append(go.Scatter(
            x=sub["lon"], y=sub["lat"],
            mode="markers+text",
            marker=dict(
                size=14,
                color=sub["order"],
                colorscale=colorscale,
                line=dict(width=1.5, color="white"),
            ),
            text=sub["camera_id"],
            textposition="top center",
            textfont=dict(size=10, color="#333"),
            showlegend=False,
        ))

        # 当前点高亮（最新到达的点脉冲效果）
        last = sub.iloc[-1]
        frame_traces.append(go.Scatter(
            x=[last["lon"]], y=[last["lat"]],
            mode="markers",
            marker=dict(
                size=22,
                color="rgba(11,78,194,0.25)",
                symbol="circle",
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        time_text = last["time_label"]
        frames.append(go.Frame(
            data=frame_traces,
            name=f"frame_{k}",
            layout=dict(
                title=dict(text=f"轨迹演化  |  第 {k}/{n} 个摄像头  |  {time_text}"),
            ),
        ))

    fig.frames = frames

    # ── 播放/暂停按钮 ──
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": -0.18,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"r": 8, "t": 8},
            "buttons": [
                {
                    "label": "▶ 播放",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 800, "redraw": True},
                        "from_current": True,
                        "transition": {"duration": 400},
                        "mode": "immediate",
                    }],
                },
                {
                    "label": "⏸ 暂停",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
    )

    # ── 时间轴滑块 ──
    sliders = [{
        "active": 0,
        "y": -0.25,
        "x": 0.0,
        "len": 1.0,
        "xanchor": "left",
        "yanchor": "top",
        "pad": {"t": 20, "b": 10},
        "currentvalue": {
            "visible": True,
            "prefix": "摄像头 ",
            "font": {"color": "#0B4EC2", "size": 12},
        },
        "transition": {"duration": 400},
        "steps": [
            {
                "method": "animate",
                "label": f"{i+1}. {df.iloc[i]['camera_id']}",
                "args": [
                    [f"frame_{i+1}"],
                    {
                        "frame": {"duration": 600, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
            }
            for i in range(n)
        ],
    }]

    fig.update_layout(
        sliders=sliders,
    )

    # ── 整体布局 ──
    lon_range = df["lon"].max() - df["lon"].min()
    lat_range = df["lat"].max() - df["lat"].min()
    margin_x = max(lon_range * 0.15, 0.001)
    margin_y = max(lat_range * 0.15, 0.001)

    fig.update_layout(
        title=dict(
            text="跨镜轨迹时空分布图",
            font=dict(size=16, color="#1F2D3D"),
            x=0.5,
        ),
        xaxis=dict(
            title="经度",
            scaleanchor="y",
            scaleratio=1,
            range=[df["lon"].min() - margin_x, df["lon"].max() + margin_x],
            gridcolor="#E8ECF0",
            zerolinecolor="#E8ECF0",
        ),
        yaxis=dict(
            title="纬度",
            scaleanchor="x",
            scaleratio=1,
            range=[df["lat"].min() - margin_y, df["lat"].max() + margin_y],
            gridcolor="#E8ECF0",
            zerolinecolor="#E8ECF0",
        ),
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="white",
        font=dict(size=12, family="Microsoft YaHei, Arial"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=80, t=60, b=100),
        height=520,
    )

    return fig


# ── 向后兼容：保留旧函数名作为别名 ──
def render_trajectory_map(camera_sequence: list, width: int = 700, height: int = 400):
    """
    已废弃 - 请使用 render_trajectory_plot()。
    保留此函数仅为向后兼容。
    """
    import streamlit as st
    fig = render_trajectory_plot(camera_sequence)
    st.plotly_chart(fig, use_container_width=True)
