"""
frontend.home - 交通态势工作台（浅色主题）

公安平台风格：浅色背景 + 警蓝体系配色，与全局浅色主题一致。
让交警打开系统就知道"哪里有事、有什么待办、点哪里处理"。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# CityFlow 数据统一走 src.storage.datastore（优先 Parquet + SQLite，缺失时回退 JSON 直读），
# 本模块先前单独开文件读 107MB JSON，现已收敛。
from src.storage.datastore import get_stats as _cityflow_stats

# render_icon 不再需要（快捷操作按钮改用 Unicode 图标，避免 SVG 在 st.button 中泄露 HTML）
# from frontend.components import render_icon

# ============================================================
# 数据加载（预留 API 接口）
# ============================================================


def _load_dashboard_stats() -> dict:
    """加载仪表盘核心统计数据

    预留接口:
        stats = api_get_dashboard_stats()
    """
    stats = {
        "today_detections": 0,
        "active_tracks": 0,
        "pending_confirm": 0,
        "cameras_online": 0,
        "cameras_total": 0,
        "detections_yesterday": 0,
    }

    # 尝试从后端 API 获取
    try:
        import requests
        api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        r = requests.get(f"{api_url}/api/v1/dashboard/stats", timeout=3)
        if r.status_code == 200:
            data = r.json()
            stats["today_detections"] = data.get("instance_count", 0)
            stats["active_tracks"] = data.get("tracklet_count", 0)
            stats["cameras_total"] = data.get("camera_count", 0)
            stats["cameras_online"] = data.get("camera_online", 0)
            return stats
    except Exception:
        pass

    # 后端不可用 → 从本地数据统计（datastore 在位时为毫秒级元数据查询，不解析大 JSON）
    try:
        local = _cityflow_stats()
        stats["today_detections"] = local.get("detections", 0)
        stats["active_tracks"] = local.get("tracks", 0)
        stats["cameras_total"] = local.get("cameras", 0)
        stats["cameras_online"] = stats["cameras_total"]
    except Exception:
        pass

    return stats


def _load_todo_items() -> dict:
    """加载待办事项数据

    预留接口:
        todos = api_get_todo_items()
    """
    return {
        "pending_confirm": 23,
        "pending_review": 8,
        "abnormal_vehicles": 5,
        "offline_cameras": 0,
        "low_confidence": 12,
    }


def _load_recent_activities() -> list:
    """加载最近活动列表

    预留接口:
        activities = api_get_recent_activities()
    """
    now = datetime.now()
    return [
        {
            "time": (now - timedelta(minutes=5)).strftime("%H:%M"),
            "type": "检索",
            "summary": "黑色轿车 京A·8832 — 匹配 6 条结果",
            "action": "查看",
        },
        {
            "time": (now - timedelta(minutes=18)).strftime("%H:%M"),
            "type": "确认",
            "summary": "INST_a3f2c1 — 确认为目标车辆",
            "action": "查看",
        },
        {
            "time": (now - timedelta(minutes=35)).strftime("%H:%M"),
            "type": "检索",
            "summary": "蓝色面包车 无车牌 — 匹配 3 条结果",
            "action": "查看",
        },
        {
            "time": (now - timedelta(minutes=52)).strftime("%H:%M"),
            "type": "导出",
            "summary": "每日工作简报_20260628.pdf",
            "action": "下载",
        },
        {
            "time": (now - timedelta(minutes=78)).strftime("%H:%M"),
            "type": "告警",
            "summary": "c003 检测到异常停留车辆 (>30min)",
            "action": "处理",
        },
        {
            "time": (now - timedelta(minutes=95)).strftime("%H:%M"),
            "type": "确认",
            "summary": "INST_b7e4d9 — 排除嫌疑",
            "action": "查看",
        },
        {
            "time": (now - timedelta(minutes=120)).strftime("%H:%M"),
            "type": "检索",
            "summary": "红色外套女性行人 — 匹配 4 条结果",
            "action": "查看",
        },
    ]


# ============================================================
# 页面样式 — 浅色主题（与全局设计系统一致）
# ============================================================

_HOME_PAGE_CSS = """
<style>
/* ── 页面背景 ────────────────────────────────── */
.stApp {
    background-color: #F3F6FA !important;
}

/* ── 标题区 ──────────────────────────────────── */
.hw-title-area {
    background: linear-gradient(135deg, #0B4EC2 0%, #083B91 100%);
    border: none;
    padding: 10px 16px;
    margin-bottom: 10px;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(11, 78, 194, 0.18);
}
.hw-title {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 1px;
    margin: 0 0 4px 0;
}
.hw-subtitle {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.75);
    display: flex;
    align-items: center;
    gap: 8px;
}
.hw-subtitle .hw-status-ok {
    color: #10B981;
    font-weight: 500;
}
.hw-subtitle .hw-status-warn {
    color: #EF4444;
    font-weight: 500;
}
.hw-subtitle .hw-sep {
    color: rgba(255, 255, 255, 0.4);
}

/* ── 统计卡片行 — 白色卡片 ────────────────── */
.hw-stat-cell {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    padding: 10px 12px;
    text-align: center;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
.hw-stat-icon-row {
    margin-bottom: 4px;
}
.hw-stat-value {
    color: #0B4EC2;
    font-size: 22px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.hw-stat-value.red {
    color: #EF4444;
}
.hw-stat-value.green {
    color: #10B981;
}
.hw-stat-label {
    color: #667085;
    font-size: 11px;
    margin-top: 2px;
}
.hw-stat-trend {
    font-size: 10px;
    margin-top: 3px;
}
.hw-stat-trend.up {
    color: #10B981;
}
.hw-stat-trend.down {
    color: #EF4444;
}
.hw-stat-trend.neutral {
    color: #667085;
}

/* ── 面板卡片 — 白色 ────────────────────────── */
.hw-panel {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    margin-bottom: 8px;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
.hw-panel-header {
    padding: 8px 12px 6px;
    font-size: 12px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hw-panel-header .hw-badge {
    background: #EF4444;
    color: #FFFFFF;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    min-width: 18px;
    text-align: center;
}
.hw-panel-body {
    padding: 8px 12px;
}

/* ── 待办事项列表 ────────────────────────────── */
.hw-todo-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #E8F1FF;
    cursor: pointer;
    transition: background-color 0.15s;
}
.hw-todo-item:last-child {
    border-bottom: none;
}
.hw-todo-item:hover {
    background-color: #F3F6FA;
    margin: 0 -12px;
    padding: 6px 12px;
}
.hw-todo-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #667085;
}
.hw-todo-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.hw-todo-dot.red    { background: #EF4444; }
.hw-todo-dot.orange { background: #F59E0B; }
.hw-todo-dot.gray   { background: #9AA4B2; }
.hw-todo-dot.yellow { background: #F59E0B; }
.hw-todo-count {
    font-size: 14px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.hw-todo-count.red    { color: #EF4444; }
.hw-todo-count.orange { color: #F59E0B; }
.hw-todo-count.gray   { color: #9AA4B2; }
.hw-todo-count.yellow { color: #F59E0B; }
.hw-todo-unit {
    font-size: 11px;
    color: #667085;
    margin-left: 2px;
    font-weight: 400;
}

/* ── 快捷操作 ────────────────────────────────── */
.hw-qa-panel {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    margin-bottom: 8px;
}
.hw-qa-panel-header {
    padding: 8px 12px 6px;
    font-size: 12px;
    font-weight: 600;
    color: #1F2D3D;
    border-bottom: 1px solid #D9E2EF;
}

/* ── 最近活动表格 ────────────────────────────── */
.hw-activity-wrap {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
}
.hw-activity-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.hw-activity-table thead th {
    background-color: #E8F1FF;
    color: #0B4EC2;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 10px;
    text-align: left;
    border-bottom: 1px solid #D9E2EF;
    white-space: nowrap;
}
.hw-activity-table tbody td {
    padding: 4px 10px;
    color: #1F2D3D;
    border-bottom: 1px solid #E8F1FF;
    font-size: 12px;
}
.hw-activity-table tbody tr:hover td {
    background-color: #F3F6FA;
    color: #1F2D3D;
}
.hw-activity-table tbody tr:last-child td {
    border-bottom: none;
}
.hw-activity-type {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
    line-height: 16px;
}
.hw-activity-type.search  { background: #E8F1FF; color: #0B4EC2; }
.hw-activity-type.confirm { background: rgba(16,185,129,0.12); color: #10B981; }
.hw-activity-type.export  { background: #E8F1FF; color: #1677FF; }
.hw-activity-type.alert   { background: rgba(239,68,68,0.12); color: #EF4444; }
.hw-activity-action {
    color: #1677FF;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
}
.hw-activity-action:hover {
    color: #0B4EC2;
    text-decoration: underline;
}
.hw-activity-time {
    color: #667085;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    white-space: nowrap;
}
</style>
"""


# ============================================================
# 渲染函数
# ============================================================

def render():
    """渲染交通态势工作台（指挥工作台首页）"""

    st.markdown(_HOME_PAGE_CSS, unsafe_allow_html=True)

    # ── 加载数据 ──
    stats = _load_dashboard_stats()
    todos = _load_todo_items()
    activities = _load_recent_activities()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cam_online = stats["cameras_online"]
    cam_total = stats["cameras_total"]

    # ══════════════════════════════════════════
    # 1. 页面标题区（蓝色渐变标题栏）
    # ══════════════════════════════════════════
    status_cls = "hw-status-ok" if cam_online == cam_total else "hw-status-warn"
    status_text = "系统运行正常" if cam_online == cam_total else "部分设备离线"
    st.markdown(f"""
    <div class="hw-title-area">
        <div class="hw-title">交通态势工作台</div>
        <div class="hw-subtitle">
            <span>{now_str}</span>
            <span class="hw-sep">|</span>
            <span class="{status_cls}">{status_text}</span>
            <span class="hw-sep">|</span>
            <span>在线摄像头 {cam_online}/{cam_total}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # 2. 主体左右布局：左侧 2/3 核心数据，右侧 1/3 快捷操作
    # ══════════════════════════════════════════
    col_left, col_right = st.columns([2, 1])

    # ────────────────────────────────────────
    # 左侧：待办事项 + 最近活动
    # ────────────────────────────────────────
    with col_left:
        total_todo = (todos["pending_confirm"] + todos["pending_review"]
                      + todos["abnormal_vehicles"] + todos["low_confidence"])

        # 待办事项面板 — 完整HTML块（避免div开闭分离导致泄露）
        _todo_rows = [
            ("red", "待确认目标", todos["pending_confirm"], "条", "red"),
            ("orange", "待审核轨迹", todos["pending_review"], "条", "orange"),
            ("red", "异常车辆", todos["abnormal_vehicles"], "条", "red"),
            ("gray", "离线摄像头", todos["offline_cameras"], "台", "gray"),
            ("yellow", "低置信识别", todos["low_confidence"], "条", "yellow"),
        ]
        _todo_items_html = ""
        for dot_color, label, count, unit, count_color in _todo_rows:
            _todo_items_html += (
                f'<div class="hw-todo-item">'
                f'<div class="hw-todo-left">'
                f'<span class="hw-todo-dot {dot_color}"></span>'
                f'<span>{label}</span></div>'
                f'<div><span class="hw-todo-count {count_color}">{count}</span>'
                f'<span class="hw-todo-unit">{unit}</span></div></div>'
            )

        st.markdown(
            f'<div class="hw-panel">'
            f'<div class="hw-panel-header">'
            f'<span>待办事项</span>'
            f'<span class="hw-badge">{total_todo}</span>'
            f'</div>'
            f'<div class="hw-panel-body">'
            f'{_todo_items_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # 待确认目标跳转按钮
        if st.button("前往处理待确认目标", key="home_goto_confirm", use_container_width=True):
            st.session_state["current_page"] = "confirm"
            st.rerun()

        # ── 最近活动列表 ──
        activity_data = []
        for act in activities:
            activity_data.append({
                "时间": act["time"],
                "类型": act["type"],
                "内容摘要": act["summary"],
                "操作": act["action"],
            })
        df_activity = pd.DataFrame(activity_data)

        st.markdown(
            '<div class="hw-panel"><div class="hw-panel-header">最近活动</div></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df_activity,
            use_container_width=True,
            hide_index=True,
            column_config={
                "时间": st.column_config.TextColumn("时间", width="small"),
                "类型": st.column_config.TextColumn("类型", width="small"),
                "内容摘要": st.column_config.TextColumn("内容摘要", width="large"),
                "操作": st.column_config.TextColumn("操作", width="small"),
            },
        )

    # ────────────────────────────────────────
    # 右侧：快捷操作面板
    # ────────────────────────────────────────
    with col_right:
        # 快捷操作面板标题
        st.markdown(
            '<div class="hw-qa-panel">'
            '<div class="hw-qa-panel-header">快捷操作</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # 每个快捷操作按钮：使用 Unicode 图标 + 标题（避免 SVG 在 st.button 中泄露原始 HTML）
        if st.button(
            "\u25b8  新建目标检索",
            key="home_qa_search",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "search"
            st.rerun()
        if st.button(
            "\u25b8  轨迹快速回溯",
            key="home_qa_trajectory",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "trajectory"
            st.rerun()
        if st.button(
            "\u25b8  导出工作报告",
            key="home_qa_export",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "data"
            st.rerun()
        if st.button(
            "\u25b8  查看数据仪表盘",
            key="home_qa_dashboard",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "data"
            st.rerun()

