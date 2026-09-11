"""
frontend.pages.search - 业务查询台（公安交警浅色主题）

支持自然语言查询和车牌精确查询，以专业业务表格展示候选目标。
包含时间范围、区域/路口、车辆类型、车辆颜色、行驶方向等业务化筛选条件。
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timedelta, date, time

import yaml
import streamlit as st

from frontend.components import render_icon, render_progress_bar, render_loading_spinner
from frontend.utils import (
    api_search, api_backtrack, api_trajectory,
    _convert_trajectory_response,
    type_label, get_target_type_icon, format_attributes,
    resolve_image_path, get_no_image_placeholder, confidence_color,
    CAMERA_NAME_MAP, API_BASE,
)
from src.common.ids import extract_vehicle_id

logger = logging.getLogger(__name__)


def _load_camera_name_map_local():
    """备用加载：从 cityflow_camera_metadata.yaml 加载摄像头名称映射"""
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "configs", "cityflow_camera_metadata.yaml")
    name_map = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for cam in data.get("cameras", []):
            cid = cam.get("camera_id", "")
            cname = cam.get("name", cid)
            name_map[cid] = cname
    return name_map


# 确保 CAMERA_NAME_MAP 可用（兼容 utils 未加载的情况）
if not CAMERA_NAME_MAP:
    CAMERA_NAME_MAP.update(_load_camera_name_map_local())


# ── 页面级 CSS（浅色主题） ──
_SEARCH_CSS = """
<style>
.search-query-panel {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.search-panel-title {
    color: #1F2D3D;
    font-size: 14px;
    font-weight: 600;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.search-result-bar {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px 6px 0 0;
    padding: 8px 14px;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.search-result-count {
    color: #1F2D3D;
    font-size: 13px;
}
.search-detail-panel {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 12px 14px;
    margin-top: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.search-detail-title {
    color: #1F2D3D;
    font-size: 14px;
    font-weight: 600;
    border-bottom: 1px solid #D9E2EF;
    padding-bottom: 6px;
    margin-bottom: 8px;
}
.search-attr-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.search-attr-table th {
    background-color: #E8F1FF;
    color: #0B4EC2;
    padding: 4px 8px;
    text-align: left;
    border: 1px solid #D9E2EF;
    font-weight: 600;
}
.search-attr-table td {
    padding: 4px 8px;
    border: 1px solid #D9E2EF;
    color: #1F2D3D;
}
.filter-section-title {
    color: #0B4EC2;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 4px;
}
.filter-tag {
    display: inline-block;
    background-color: #E8F1FF;
    color: #0B4EC2;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    margin: 2px 3px 2px 0;
    border: 1px solid #D9E2EF;
}
.srch-thumb {
    width: 48px; height: 48px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #D9E2EF;
}
.srch-thumb-placeholder {
    width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    background: #F3F6FA;
    border-radius: 4px;
    border: 1px solid #D9E2EF;
    color: #9AA4B2;
    font-size: 10px;
}
.srch-btn-sm {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid #D9E2EF;
    background: #FFFFFF;
    color: #0B4EC2;
    margin-right: 4px;
    text-decoration: none;
    white-space: nowrap;
}
.srch-btn-sm:hover {
    background: #E8F1FF;
    border-color: #0B4EC2;
}
.srch-match-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    line-height: 18px;
}
.srch-status-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
    line-height: 18px;
}
</style>
"""


# ── 筛选选项常量 ──

SCENE_OPTIONS = {
    "全部场景": "",
    "S01 (train)": "S01",
    "S02 (validation)": "S02",
    "S03 (train)": "S03",
    "S04 (train)": "S04",
    "S05 (validation)": "S05",
    "S06 (test)": "S06",
}

CAMERA_OPTIONS = {"全部摄像头": ""}
for _i in range(1, 47):
    _cid = f"c{str(_i).zfill(3)}"
    _display_name = CAMERA_NAME_MAP.get(_cid, _cid)
    CAMERA_OPTIONS[_display_name] = _cid

VEHICLE_TYPES = [
    "轿车", "SUV", "面包车", "两厢车", "MPV", "皮卡",
    "公交车", "卡车", "旅行车", "跑车", "房车",
]

VEHICLE_COLORS = [
    "黄色", "橙色", "绿色", "灰色", "红色", "蓝色",
    "白色", "金色", "棕色", "黑色", "紫色", "粉色",
]

DIRECTION_OPTIONS = ["东", "南", "西", "北", "东北", "西北", "东南", "西南"]
MOVEMENT_OPTIONS = ["直行", "左转", "右转", "掉头"]

# ── 不可查询类型 → 相似可用车型推荐 ──
_NON_MOTOR_SUGGESTIONS = {
    "三轮车": ["面包车", "皮卡"],
    "小电驴": ["轿车", "SUV"],
    "电动车": ["轿车", "SUV"],
    "电瓶车": ["轿车", "SUV"],
    "摩托车": ["轿车", "跑车"],
    "拖拉机": ["卡车", "面包车"],
    "自行车": ["轿车", "面包车"],
}
_NON_MOTOR_KEYWORDS = ["三轮车", "小电驴", "拖拉机", "电动车", "摩托车", "电瓶车", "自行车"]
_PEDESTRIAN_KEYWORDS = ["行人", "穿", "背", "男人", "女人", "男的", "女的"]


def _build_non_motor_hint(query: str) -> str:
    """为非机动车查询构建带建议的友好提示。"""
    matched = [kw for kw in _NON_MOTOR_KEYWORDS if kw in query]
    suggestions: list[str] = []
    for kw in matched:
        suggestions.extend(_NON_MOTOR_SUGGESTIONS.get(kw, []))
    # 去重保留顺序
    seen: set[str] = set()
    unique_suggestions: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)
    if unique_suggestions:
        sug_str = "、".join(unique_suggestions)
        return (
            f"当前数据集不包含非机动车目标（如「{matched[0]}」），"
            f"仅包含机动车类型。您可以尝试搜索相似车型：{sug_str}。"
        )
    return "当前数据集不包含非机动车目标，仅包含常见汽车类型（轿车、SUV、卡车、公交车、面包车等）。"


def _detect_query_hint(query: str) -> tuple[str, str, str] | None:
    """根据查询文本检测是否应显示特殊提示，返回 (message, icon, border_color) 或 None。"""
    if any(t in query for t in _NON_MOTOR_KEYWORDS):
        return (_build_non_motor_hint(query), "🚲", "#F5A623")
    if any(t in query for t in _PEDESTRIAN_KEYWORDS):
        return ("当前数据集仅包含车辆目标，暂无行人数据。", "🚶", "#F5A623")
    return None


def _render_warning_block(message: str, icon: str = "⚠️", border_color: str = "#F5A623") -> None:
    """渲染醒目的警告提示块（st.warning + 自定义 HTML 双重显示，确保醒目）。"""
    st.warning(message)
    st.markdown(
        f'<div style="border-left:5px solid {border_color};background:#FFF8E1;'
        f'border-radius:0 8px 8px 0;padding:16px 20px;margin:4px 0 12px 0;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
        f'<span style="font-size:20px;margin-right:12px;vertical-align:middle;">{icon}</span>'
        f'<span style="color:#1F2D3D;font-size:14px;font-weight:600;vertical-align:middle;">{message}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _match_level(score: float) -> str:
    if score >= 0.7:
        return "高"
    elif score >= 0.4:
        return "中"
    return "低"


def _match_level_color(score: float) -> str:
    if score >= 0.7:
        return "#2E7D32"
    elif score >= 0.4:
        return "#F59E0B"
    return "#E53935"


def _match_level_bg(score: float) -> str:
    if score >= 0.7:
        return "#E8F5E9"
    elif score >= 0.4:
        return "#FFF8E1"
    return "#FFEBEE"


def _confirm_status_tag(cand: dict) -> str:
    """根据候选目标状态返回 HTML 标签"""
    status = cand.get("confirm_status", "pending")
    mapping = {
        "pending": ('<span class="srch-status-tag" style="background:#FFEBEE;color:#E53935;">待确认</span>'),
        "confirmed": ('<span class="srch-status-tag" style="background:#E8F5E9;color:#2E7D32;">已确认</span>'),
        "suspect": ('<span class="srch-status-tag" style="background:#FFF8E1;color:#F59E0B;">疑似</span>'),
        "excluded": ('<span class="srch-status-tag" style="background:#F0F3F8;color:#9AA4B2;">已排除</span>'),
    }
    return mapping.get(status, mapping["pending"])


def _build_filter_summary(filters: dict) -> list[str]:
    tags = []
    if filters.get("time_preset") and filters["time_preset"] not in ("自定义", "全部时间"):
        tags.append(f"时间: {filters['time_preset']}")
    if filters.get("cf_scene") and filters["cf_scene"] != "全部场景":
        tags.append(f"场景: {filters['cf_scene']}")
    if filters.get("cf_camera") and filters["cf_camera"] != "全部摄像头":
        tags.append(f"摄像头: {filters['cf_camera']}")
    if filters.get("cf_types"):
        tags.append(f"车型: {', '.join(filters['cf_types'])}")
    if filters.get("cf_colors"):
        tags.append(f"颜色: {', '.join(filters['cf_colors'])}")
    if filters.get("directions"):
        tags.append(f"方向: {', '.join(filters['directions'])}")
    if filters.get("movements"):
        tags.append(f"行驶: {', '.join(filters['movements'])}")
    if filters.get("min_confidence") and filters["min_confidence"] > 0:
        tags.append(f"置信度≥{filters['min_confidence']:.2f}")
    if filters.get("confirmed_only"):
        tags.append("仅已确认")
    return tags


def _render_filter_panel() -> dict:
    """渲染高级业务筛选条件面板（可折叠），返回当前筛选值 dict。"""
    filter_values: dict = {}

    with st.expander("高级筛选条件", expanded=False):
        # ── 第一行：时间范围 + 场景 + 摄像头 ──
        col_time, col_scene, col_camera = st.columns([2, 1, 1])

        with col_time:
            st.markdown('<div class="filter-section-title">时间范围</div>', unsafe_allow_html=True)
            time_preset = st.selectbox(
                "时间快捷",
                options=["全部时间", "最近1小时", "今天", "最近7天", "自定义"],
                index=0, label_visibility="collapsed",
            )
            filter_values["time_preset"] = time_preset
            if time_preset == "全部时间":
                pass
            elif time_preset == "自定义":
                col_s, col_e = st.columns(2)
                with col_s:
                    sd = st.date_input("开始日期", value=date.today() - timedelta(days=7), label_visibility="collapsed")
                    st_t = st.time_input("开始时间", value=time(0, 0), label_visibility="collapsed")
                with col_e:
                    ed = st.date_input("结束日期", value=date.today(), label_visibility="collapsed")
                    en_t = st.time_input("结束时间", value=time(23, 59), label_visibility="collapsed")
                filter_values["custom_start"] = datetime.combine(sd, st_t)
                filter_values["custom_end"] = datetime.combine(ed, en_t)
            else:
                now = datetime.now()
                preset_map = {
                    "最近1小时": (now - timedelta(hours=1), now),
                    "今天": (datetime.combine(now.date(), time(0, 0)), now),
                    "最近7天": (now - timedelta(days=7), now),
                }
                filter_values["custom_start"], filter_values["custom_end"] = preset_map[time_preset]

        with col_scene:
            st.markdown('<div class="filter-section-title">场景</div>', unsafe_allow_html=True)
            scene = st.selectbox("选择场景", options=list(SCENE_OPTIONS.keys()), index=0, label_visibility="collapsed")
            filter_values["cf_scene"] = scene

        with col_camera:
            st.markdown('<div class="filter-section-title">摄像头</div>', unsafe_allow_html=True)
            camera = st.selectbox("选择摄像头", options=list(CAMERA_OPTIONS.keys()), index=0, label_visibility="collapsed")
            filter_values["cf_camera"] = camera

        # ── 第二行：车辆类型 + 颜色 ──
        col_type, col_color = st.columns([1, 1])
        with col_type:
            st.markdown('<div class="filter-section-title">车辆类型</div>', unsafe_allow_html=True)
            vehicle_types = st.multiselect("车辆类型", options=VEHICLE_TYPES, default=[],
                                           placeholder="选择车辆类型（可多选）", label_visibility="collapsed")
            filter_values["cf_types"] = vehicle_types
        with col_color:
            st.markdown('<div class="filter-section-title">车辆颜色</div>', unsafe_allow_html=True)
            colors = st.multiselect("车辆颜色", options=VEHICLE_COLORS, default=[],
                                    placeholder="选择颜色（可多选）", label_visibility="collapsed")
            filter_values["cf_colors"] = colors

        # ── 第三行：行驶方向 + 行驶动作 ──
        col_dir, col_move = st.columns([1, 1])
        with col_dir:
            st.markdown('<div class="filter-section-title">行驶方向</div>', unsafe_allow_html=True)
            directions = st.multiselect("行驶方向", options=DIRECTION_OPTIONS, default=[],
                                        placeholder="选择方向（可多选）", label_visibility="collapsed")
            filter_values["directions"] = directions
        with col_move:
            st.markdown('<div class="filter-section-title">行驶动作</div>', unsafe_allow_html=True)
            movements = st.multiselect("行驶动作", options=MOVEMENT_OPTIONS, default=[],
                                       placeholder="选择动作（可多选）", label_visibility="collapsed")
            filter_values["movements"] = movements

        # ── 第四行：置信度 + 已确认 + 重置 ──
        col_conf, col_confirmed, col_reset = st.columns([2, 1, 1])
        with col_conf:
            st.markdown('<div class="filter-section-title">置信度下限</div>', unsafe_allow_html=True)
            min_conf = st.slider("置信度下限", min_value=0.0, max_value=1.0, value=0.0,
                                 step=0.05, label_visibility="collapsed")
            filter_values["min_confidence"] = min_conf
        with col_confirmed:
            st.markdown('<div class="filter-section-title">仅已确认目标</div>', unsafe_allow_html=True)
            confirmed_only = st.toggle("只看已确认", value=False, label_visibility="collapsed")
            filter_values["confirmed_only"] = confirmed_only
        with col_reset:
            st.markdown('<div class="filter-section-title">&nbsp;</div>', unsafe_allow_html=True)
            reset_clicked = st.button("重置筛选", use_container_width=True, key="reset_filters")

        if reset_clicked:
            st.session_state.pop("_filter_state", None)
            st.rerun()

        st.session_state["_filter_state"] = filter_values

    return filter_values


def _render_active_filters(filters: dict):
    tags = _build_filter_summary(filters)
    if not tags:
        return
    tag_html = "".join(f'<span class="filter-tag">{html.escape(t)}</span>' for t in tags)
    st.markdown(f'<div style="margin-bottom:6px;">{tag_html}</div>', unsafe_allow_html=True)


def _do_search(query: str, top_k: int, filters: dict | None = None) -> dict | None:
    """执行检索 —— 唯一真源是后端检索接口（POST /api/v1/search/query）。

    前端不再本地加载 CLIP/BLIP 打分：后端在 CLIP 不可用时已会自行退化为属性分数
    排序（「少一步」），因此这里不做任何本地替身实现。
    后端不可用时**如实提示用户并返回 None**，不静默切换成另一套算法。

    返回：与后端 SearchResponse 兼容的 dict；后端不可用时为 None。
    """
    progress_placeholder = st.empty()

    with progress_placeholder.container():
        render_progress_bar(progress=0.2, label="正在连接检索服务...", show_pct=False)
        render_loading_spinner("正在连接检索服务...")

    results = api_search(query, top_k=top_k)
    if results is not None:
        st.session_state.pop("_search_backend_down", None)
        with progress_placeholder.container():
            render_progress_bar(progress=1.0, label="检索完成", show_pct=True)
        return _apply_filters(results, filters) if filters else results

    # 后端不可用 / 返回非 200：如实告知，不用本地算法兜底
    logger.warning(f"检索服务不可用（{API_BASE}/api/v1/search/query），本次检索未产生结果")
    st.session_state["_search_backend_down"] = True
    with progress_placeholder.container():
        render_progress_bar(progress=1.0, label="检索失败", show_pct=True)
    _render_warning_block(
        f"检索服务不可用，本次未返回任何结果。请确认后端服务已启动（{API_BASE}）；"
        "本页面不再提供本地替代检索。",
        "⚠️", "#E53935",
    )
    return None


def _apply_filters(results: dict, filters: dict | None) -> dict:
    """客户端业务筛选。"""
    if not filters or not results:
        return results
    candidates = results.get("candidates", [])
    if not candidates:
        return results

    filtered = candidates

    # 时间
    time_start = filters.get("custom_start")
    time_end = filters.get("custom_end")
    if time_start and time_end:
        def _in_time(c):
            ts_str = c.get("timestamp", "")
            if not ts_str:
                return True
            try:
                return time_start <= datetime.fromisoformat(ts_str) <= time_end
            except (ValueError, TypeError):
                return True
        filtered = [c for c in filtered if _in_time(c)]


    # 车型
    vtypes = filters.get("cf_types", [])
    if vtypes:
        def _mt(c):
            attrs = c.get("attributes", {})
            tl = type_label(c.get("target_type", ""))
            for av in attrs.values():
                if any(vt in str(av) for vt in vtypes):
                    return True
            return any(vt in tl for vt in vtypes)
        filtered = [c for c in filtered if _mt(c)]

    # 颜色
    colors = filters.get("cf_colors", [])
    if colors:
        def _mc(c):
            for av in c.get("attributes", {}).values():
                if any(cl in str(av) for cl in colors):
                    return True
            return False
        filtered = [c for c in filtered if _mc(c)]

    # 方向
    directions = filters.get("directions", [])
    if directions:
        def _md(c):
            for av in c.get("attributes", {}).values():
                if any(d in str(av) for d in directions):
                    return True
            return False
        filtered = [c for c in filtered if _md(c)]

    # 动作
    movements = filters.get("movements", [])
    if movements:
        def _mm(c):
            for av in c.get("attributes", {}).values():
                if any(m in str(av) for m in movements):
                    return True
            return False
        filtered = [c for c in filtered if _mm(c)]

    # 置信度下限
    min_conf = filters.get("min_confidence", 0)
    if min_conf and min_conf > 0:
        filtered = [c for c in filtered if c.get("combined_score", 0) >= min_conf]

    # 仅已确认
    if filters.get("confirmed_only"):
        filtered = [c for c in filtered if c.get("confirm_status") == "confirmed"]

    for i, c in enumerate(filtered, 1):
        c["rank"] = i
    results["candidates"] = filtered
    results["total_count"] = len(filtered)
    return results


def _has_active_filters(filters: dict) -> bool:
    """判断是否有激活的业务筛选条件。"""
    if not filters:
        return False
    if filters.get("cf_scene", "全部场景") != "全部场景":
        return True
    if filters.get("cf_camera", "全部摄像头") != "全部摄像头":
        return True
    if filters.get("cf_types"):
        return True
    if filters.get("cf_colors"):
        return True
    if filters.get("directions"):
        return True
    if filters.get("movements"):
        return True
    if filters.get("min_confidence") and filters["min_confidence"] > 0:
        return True
    if filters.get("confirmed_only"):
        return True
    if filters.get("time_preset") and filters["time_preset"] not in ("自定义", "全部时间"):
        return True
    if filters.get("custom_start") and filters.get("custom_end"):
        return True
    return False


def _build_thumb_html(cand: dict) -> str:
    """为表格行构建缩略图 HTML。"""
    keyframe = cand.get("keyframe_path")
    img_path = resolve_image_path(keyframe) if keyframe else None
    if img_path:
        return f'<img src="{html.escape(str(img_path))}" class="srch-thumb" />'
    return '<div class="srch-thumb-placeholder">无图</div>'





def _build_traj_from_local(cand: dict) -> dict:
    """从本地 cityflow_results.json 构建跨镜轨迹数据（当 API 不可用时的兑底逻辑）"""
    from collections import defaultdict
    from frontend.utils import load_cityflow_results

    instance_id = cand.get("instance_id", "")
    track_id = cand.get("track_id", "")

    # 从 instance_id 提取 vehicle_id：统一走 src.common.ids，勿在此重复实现反解。
    # 保留 "_V" 前置判断——只有复合 ID（target_id / track_id）才需要反解，
    # 裸 vehicle_id 不作为 instance_id 使用，与改造前行为一致。
    vehicle_id = ""
    if instance_id and "_V" in instance_id:
        vehicle_id = extract_vehicle_id(instance_id)

    empty_traj = {
        "target_instance": cand,
        "observation_nodes": [],
        "observation_segments": [],
        "inference_segments": [],
        "candidate_paths": [],
        "overall_confidence": 0,
    }

    if not vehicle_id:
        return empty_traj

    results = load_cityflow_results()
    if not results:
        return empty_traj

    detections = results.get("detections", [])

    # 查找该 vehicle_id 的所有 detections
    vehicle_dets = []
    for det in detections:
        det_target = det.get("target_id", "")
        if f"_{vehicle_id}_" in det_target or det_target.endswith(f"_{vehicle_id}"):
            vehicle_dets.append(det)

    if not vehicle_dets:
        return empty_traj

    # 按时间排序
    vehicle_dets.sort(key=lambda x: (x.get("frame_id", 0), x.get("camera_id", "")))

    # 按摄像头分组
    camera_groups = defaultdict(list)
    for det in vehicle_dets:
        cam_id = det.get("camera_id", "")
        camera_groups[cam_id].append(det)

    # 构建简化的跨镜轨迹数据
    camera_sequence = []
    for cam_id, cam_dets in sorted(camera_groups.items(), key=lambda x: x[1][0].get("frame_id", 0)):
        first_det = cam_dets[0]
        last_det = cam_dets[-1]
        camera_sequence.append({
            "camera_id": cam_id,
            "camera_name": CAMERA_NAME_MAP.get(cam_id, cam_id),
            "arrival_time": first_det.get("timestamp", ""),
            "departure_time": last_det.get("timestamp", ""),
            "detection_count": len(cam_dets),
            "frames": [{"crop_path": d.get("crop_path", d.get("keyframe_path", "")), "frame_id": d.get("frame_id")} for d in cam_dets[:10]],
        })

    # 构建与 api_trajectory 响应兼容的结构
    raw_traj = {
        "track_id": track_id,
        "first_appearance": camera_sequence[0]["arrival_time"] if camera_sequence else "",
        "last_appearance": camera_sequence[-1]["departure_time"] if camera_sequence else "",
        "total_cameras": len(camera_sequence),
        "total_detections": len(vehicle_dets),
        "camera_sequence": camera_sequence,
        "attributes": vehicle_dets[0].get("attributes", {}) if vehicle_dets else {},
    }

    # 使用 utils 中的统一转换函数（传入 cand 以丰富 target_instance 字段）
    from frontend.utils import _convert_trajectory_response
    return _convert_trajectory_response(raw_traj, cand)


def _get_cityflow_trajectory_cams(cand: dict) -> list[str]:
    """
    从 CityFlow 数据中查找同一车辆在不同摄像头的出现记录。
    返回摄像头 ID 列表（按场景/时间排序）。
    """
    from frontend.utils import load_cityflow_results
    instance_id = cand.get("instance_id", "")
    if not instance_id or "_" not in instance_id:
        return []
    # 检查是否以 CF 开头（兼容 CF、CF3、CF4等）
    if not instance_id.split("_")[0].startswith("CF"):
        return []
    # 从 target_id 提取 vehicle_id: CF_{cam_id}_V{vid:04d}_{det_id}
    parts = instance_id.split("_")
    if len(parts) < 4:
        return []
    # 查找 tracks 中相同 vehicle_id 的记录
    data = load_cityflow_results()
    if not data:
        return []
    # 提取 cam_id 和 vid
    src_cam = parts[1]  # e.g. "c001"
    vid_part = parts[2]  # e.g. "V0001"
    vid_num = vid_part.lstrip("V")
    # 在所有 tracks 中查找同一 vehicle_id
    cameras_seen: list[str] = []
    for track in data.get("tracks", []):
        if track.get("vehicle_id") and str(track["vehicle_id"]) == vid_num:
            for cam_id in track.get("camera_ids", []):
                if cam_id not in cameras_seen:
                    cameras_seen.append(cam_id)
    # 如果 tracks 中没找到，从 detections 中查找
    if not cameras_seen:
        for det in data.get("detections", []):
            det_id = det.get("target_id", "")
            if f"_V{vid_num}_" in det_id:
                cam = det.get("camera_id", "")
                if cam and cam not in cameras_seen:
                    cameras_seen.append(cam)
    return sorted(cameras_seen)


def render():
    """渲染检索页面 — 业务查询台"""
    st.markdown(_SEARCH_CSS, unsafe_allow_html=True)

    st.markdown("#### 业务查询台")

    # ── 查询条件面板 ──
    _hint_icon = render_icon("scan", size=14, color="#0B4EC2")

    with st.container():
        st.markdown(
            '<div class="search-panel-title">请输入要查找的车辆或目标描述</div>',
            unsafe_allow_html=True,
        )
        
        # ── 全库自然语言查询输入 ──
        query_text = st.text_input(
            "目标描述",
            placeholder="例如：白色SUV、蓝色出租车、红色三轮车、行人、骑电动车的人",
            help="输入车辆颜色、类型或目标特征进行全库智能检索，支持中英文描述",
            label_visibility="collapsed",
        )
                
        # ── 搜索提示 ──
        _hint_html = (
            '<div style="background:#F8FAFC;border:1px solid #E8F1FF;border-radius:4px;'
            'padding:8px 12px;margin:4px 0 8px 0;font-size:12px;color:#667085;">'
            f'<span style="color:#0B4EC2;font-weight:600;">{_hint_icon} 示例：</span><br/>'
            '&bull; <b>按颜色+车型</b>：白色SUV、蓝色轿车、黑色面包车<br/>'
            '&bull; <b>按目标类型</b>：行人、骑电动车的人、三轮车<br/>'
            '&bull; <b>按场景描述</b>：逆行的红色卡车、左转的白色轿车'
            '</div>'
        )
        st.markdown(_hint_html, unsafe_allow_html=True)
        
        # ── 高级业务筛选条件 ──
        filters = _render_filter_panel()
        
        # 显示已激活的筛选标签
        _render_active_filters(filters)
        
        # ── 按钮行 ──
        col_search, col_page = st.columns([3, 1])
        with col_search:
            search_clicked = st.button("检索", use_container_width=True, type="primary")
        with col_page:
            page_size = st.selectbox("每页显示", [10, 20, 50], index=1, label_visibility="collapsed")

    # ── 执行检索 ──
    if search_clicked:
        results = None
        top_k = page_size
        st.session_state.pop("_search_hint", None)

        if query_text.strip():
            results = _do_search(query_text.strip(), top_k, filters=filters)
        else:
            st.warning("请输入目标描述后点击检索")
            return

        if results:
            st.session_state["search_results"] = results
            st.session_state["search_query"] = query_text or "全库检索"
        else:
            # 搜索无结果时，检测是否需要显示特殊提示并持久化。
            # 后端不可用时不再叠加该提示：那只是服务没起来，推断「数据集不含该目标」会误导用户。
            if not st.session_state.get("_search_backend_down"):
                _q = query_text or ""
                _hint = _detect_query_hint(_q)
                if _hint:
                    st.session_state["_search_hint"] = {
                        "message": _hint[0], "icon": _hint[1], "border_color": _hint[2],
                    }
            st.session_state.pop("search_results", None)

    # ── 展示结果 ──
    results = st.session_state.get("search_results")
    if results:
        _render_results_table(results)
    elif st.session_state.get("_search_hint"):
        # 持久化的特殊提示（非机动车/行人查询）
        _h = st.session_state["_search_hint"]
        _render_warning_block(_h["message"], _h["icon"], _h["border_color"])
    elif not search_clicked:
        st.info("请输入查询条件或设置筛选条件后点击检索")


def _render_results_table(results: dict):
    """以专业业务表格渲染检索结果（含缩略图和操作按钮）。"""
    candidates = results.get("candidates", [])
    total = results.get("total_count", len(candidates))
    query_id = results.get("query_id", "")

    # 结果计数条
    st.markdown(
        f'<div class="search-result-bar">'
        f'<span class="search-result-count">共 <strong style="color:#0B4EC2;">{total}</strong> 条记录</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not candidates:
        # 行人/非机动车查询返回 0 条时给出友好提示
        _query = st.session_state.get("search_query", "")
        _hint = _detect_query_hint(_query)
        if _hint:
            _render_warning_block(_hint[0], _hint[1], _hint[2])
        else:
            st.info("未找到匹配的目标，请调整查询条件后重试。")
        return

    # ── 表头行（8 列：缩略图 | 序号 | 车型+颜色 | 摄像头 | 时间 | 方向 | 匹配度+状态 | 操作） ──
    _col_widths = [0.8, 0.7, 1.0, 1.4, 1.4, 0.6, 0.7, 1.5]
    h_thumb, h_id, h_type_color, h_cam, h_time, h_dir, h_match, h_act = st.columns(_col_widths)
    _hdr_style = "background:#E8F1FF;color:#0B4EC2;font-weight:600;font-size:12px;padding:6px 4px;"
    with h_thumb:
        st.markdown(f"<div style='{_hdr_style}'>缩略图</div>", unsafe_allow_html=True)
    with h_id:
        st.markdown(f"<div style='{_hdr_style}'>序号</div>", unsafe_allow_html=True)
    with h_type_color:
        st.markdown(f"<div style='{_hdr_style}'>车型 / 颜色</div>", unsafe_allow_html=True)
    with h_cam:
        st.markdown(f"<div style='{_hdr_style}'>摄像头</div>", unsafe_allow_html=True)
    with h_time:
        st.markdown(f"<div style='{_hdr_style}'>抓拍时间</div>", unsafe_allow_html=True)
    with h_dir:
        st.markdown(f"<div style='{_hdr_style}'>方向</div>", unsafe_allow_html=True)
    with h_match:
        st.markdown(f"<div style='{_hdr_style}'>匹配 / 状态</div>", unsafe_allow_html=True)
    with h_act:
        st.markdown(f"<div style='{_hdr_style}'>操作</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0;border-color:#0B4EC2;'/>", unsafe_allow_html=True)

    # 用 st.columns 逐行渲染，支持缩略图和操作按钮
    for cand in candidates:
        target_type = cand.get("target_type", "unknown")
        type_cn = type_label(target_type)
        attrs = cand.get("attributes", {})
        camera_name = cand.get("camera_name", cand.get("camera_id", ""))
        timestamp = cand.get("timestamp", "")
        score = cand.get("combined_score", 0)
        rank = cand.get("rank", "?")

        # HTML 转义所有动态数据
        rank_safe = html.escape(str(rank))
        type_safe = html.escape(type_cn)
        camera_safe = html.escape(camera_name)
        timestamp_safe = html.escape(str(timestamp))

        # 颜色：从属性中提取
        color_val = attrs.get("颜色", "—")
        color_safe = html.escape(color_val)
        # 方向
        dir_val = attrs.get("方向", attrs.get("行驶方向", "—"))
        dir_safe = html.escape(dir_val)
        # 匹配度标签
        level = _match_level(score)
        level_color = _match_level_color(score)
        level_bg = _match_level_bg(score)
        match_html = (
            f'<span class="srch-match-tag" style="background:{level_bg};color:{level_color};">'
            f'{level} {score:.0%}</span>'
        )
        status_html = _confirm_status_tag(cand)

        # 缩略图：优先 keyframe_path，回退 image_path
        keyframe = cand.get("keyframe_path")
        img_path = resolve_image_path(keyframe) if keyframe else None
        if not img_path:
            logger.warning(f"图片路径解析失败: keyframe={keyframe}")
            # 回退：尝试 image_path 字段
            fallback_path = cand.get("image_path")
            if fallback_path:
                img_path = resolve_image_path(fallback_path)
                if img_path:
                    logger.info(f"使用 image_path 回退成功: {fallback_path} -> {img_path}")
                else:
                    logger.warning(f"image_path 也失败: {fallback_path}")

        # 每行用 8 列布局
        c_thumb, c_id, c_type_color, c_cam, c_time, c_dir, c_match, c_act = st.columns(_col_widths)

        with c_thumb:
            if img_path:
                st.image(img_path, width=48)
            else:
                st.markdown(
                    '<div class="srch-thumb-placeholder">无图</div>',
                    unsafe_allow_html=True,
                )
        with c_id:
            st.markdown(
                f"<div style='padding-top:12px;text-align:center;'>"
                f"<span style='color:#0B4EC2;font-weight:600;font-size:14px;'>#{rank_safe}</span></div>",
                unsafe_allow_html=True,
            )
        with c_type_color:
            st.markdown(
                f"<div style='padding-top:6px;font-size:12px;'>"
                f"{type_safe}<br/>"
                f"<span style='color:#667085;font-size:11px;'>{color_safe}</span></div>",
                unsafe_allow_html=True,
            )
        with c_cam:
            st.markdown(
                f"<div style='padding-top:8px;font-size:11px;color:#667085;"
                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                f"{camera_safe}</div>",
                unsafe_allow_html=True,
            )
        with c_time:
            st.markdown(
                f"<div style='padding-top:8px;font-size:11px;color:#667085;'>"
                f"{timestamp_safe}</div>",
                unsafe_allow_html=True,
            )
        with c_dir:
            st.markdown(
                f"<div style='padding-top:8px;font-size:13px;'>{dir_safe}</div>",
                unsafe_allow_html=True,
            )
        with c_match:
            st.markdown(
                f"<div style='padding-top:4px;'>{match_html}<br/>{status_html}</div>",
                unsafe_allow_html=True,
            )
        with c_act:
            ab1, ab2 = st.columns(2)
            with ab1:
                if st.button("详情", key=f"det_{rank}", use_container_width=True):
                    st.session_state[f"_show_detail_{rank}"] = True
            with ab2:
                action_label = "确认"
                if cand.get("data_source") == "cityflow":
                    action_label = "回溯"
                if st.button(action_label, key=f"act_{rank}", use_container_width=True):
                    if action_label == "确认":
                        st.session_state["confirmed_candidate"] = cand
                        st.session_state["query_id"] = query_id
                        st.session_state["current_page"] = "confirm"
                        st.rerun()
                    else:
                        instance_id = cand.get("instance_id", "")
                        st.session_state["backtrack_instance"] = instance_id
                        st.session_state["trajectory_camera_ids"] = _get_cityflow_trajectory_cams(cand)
                        # 传递 track_id 和 instance_id 供跨镜轨迹 API 使用
                        st.session_state["selected_track_id"] = cand.get("track_id", "")
                        st.session_state["selected_instance_id"] = instance_id

                        # 调用跨镜轨迹 API
                        try:
                            traj_resp = api_trajectory(
                                track_id=cand.get("track_id", ""),
                                instance_id=instance_id,
                            )
                            if traj_resp and traj_resp.get("success"):
                                # 将新 API 响应转换为 trajectory 页面期望的格式
                                traj = traj_resp["trajectory"]
                                st.session_state["trajectory_data"] = _convert_trajectory_response(traj, cand)
                            else:
                                # 尝试从本地数据构建跨镜轨迹
                                traj_data = _build_traj_from_local(cand)
                                st.session_state["trajectory_data"] = traj_data
                        except Exception:
                            traj_data = _build_traj_from_local(cand)
                            st.session_state["trajectory_data"] = traj_data
                        st.session_state["current_page"] = "trajectory"
                        st.rerun()

        # 分隔线
        st.markdown("<hr style='margin:2px 0;border-color:#EEF2F7;'/>", unsafe_allow_html=True)

        # 展开详情
        if st.session_state.get(f"_show_detail_{rank}"):
            _render_detail(cand, query_id)
            if st.button("收起详情", key=f"hide_{rank}"):
                st.session_state[f"_show_detail_{rank}"] = False
                st.rerun()


def _render_detail(cand: dict, query_id: str):
    """渲染选中目标的详细信息面板。"""
    target_type = cand.get("target_type", "unknown")
    type_cn = type_label(target_type)
    attrs = cand.get("attributes", {})
    score = cand.get("combined_score", 0)
    camera_name = cand.get("camera_name", cand.get("camera_id", ""))
    timestamp = cand.get("timestamp", "")
    plate = cand.get("plate_number")
    rank = cand.get("rank", "?")

    level = _match_level(score)
    level_color = _match_level_color(score)

    # HTML 转义所有动态数据
    rank_safe = html.escape(str(rank))
    type_safe = html.escape(type_cn)
    camera_safe = html.escape(camera_name)
    timestamp_safe = html.escape(str(timestamp))

    # 使用 st.container 包裹详情面板，避免 div 开闭标签分离导致代码泄露
    with st.container():
        st.markdown(
            f'<div class="search-detail-panel">'
            f'<div class="search-detail-title">'
            f'第 {rank_safe} 条详情 &nbsp; '
            f'<span style="color:{level_color};font-size:12px;">匹配度: {level} ({score:.0%})</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        col_img, col_info = st.columns([1, 1])

        with col_img:
            keyframe = cand.get("keyframe_path")
            img_path = resolve_image_path(keyframe) if keyframe else None
            if not img_path:
                # 回退：尝试 image_path 字段
                fallback_path = cand.get("image_path")
                if fallback_path:
                    img_path = resolve_image_path(fallback_path)
            if img_path:
                st.image(img_path, use_container_width=True)
            else:
                st.image(get_no_image_placeholder(), use_container_width=True)

        with col_info:
            attr_rows = ""
            for k, v in attrs.items():
                k_safe = html.escape(str(k))
                v_safe = html.escape(str(v))
                attr_rows += f"<tr><th style='width:80px;'>{k_safe}</th><td>{v_safe}</td></tr>"
            attr_rows += f"<tr><th>类型</th><td>{type_safe}</td></tr>"
            attr_rows += f"<tr><th>发现位置</th><td>{camera_safe}</td></tr>"
            attr_rows += f"<tr><th>发现时间</th><td>{timestamp_safe}</td></tr>"

            st.markdown(
                f'<table class="search-attr-table">{attr_rows}</table>',
                unsafe_allow_html=True,
            )

        # ── 轨迹回溯组件 ──
        track_frames = cand.get("track_frames", [])
        track_cameras = cand.get("track_cameras", [])
        track_frame_count = cand.get("track_frame_count", 0)

        if track_frames and track_frame_count > 1:
            st.divider()
            st.subheader(f"轨迹回溯 ({track_frame_count}帧 / {len(track_cameras)}个摄像头)")

            # 显示摄像头路径
            cam_names = [cand.get("camera_name", c) for c in track_cameras[:5]]
            st.caption(f"经过摄像头: {' → '.join(cam_names)}")

            # 显示轨迹时间线（最多展示10帧缩略图）
            max_display = min(10, len(track_frames))
            cols = st.columns(max_display)
            for idx, frame in enumerate(track_frames[:max_display]):
                with cols[idx]:
                    frame_img = frame.get("crop_path", "")
                    if frame_img and os.path.exists(frame_img):
                        st.image(frame_img, use_container_width=True)
                    else:
                        st.caption("(无图片)")
                    cam_label = frame.get("camera_name", "")[:8]
                    st.caption(f"{cam_label} 帧{frame.get('frame_id', '')}")

            # 如果有更多帧，显示提示
            if len(track_frames) > max_display:
                st.caption(f"...还有 {len(track_frames) - max_display} 帧未显示")

    if st.button("确认此目标", key=f"confirm_full_{cand.get('instance_id', rank)}", use_container_width=True):
        st.session_state["confirmed_candidate"] = cand
        st.session_state["query_id"] = query_id
        # 传递 track_id 和 instance_id 供跨镜轨迹 API 使用
        st.session_state["selected_track_id"] = cand.get("track_id", "")
        st.session_state["selected_instance_id"] = cand.get("instance_id", "")
        st.session_state["trajectory_camera_ids"] = _get_cityflow_trajectory_cams(cand)

        # 调用跨镜轨迹 API
        try:
            traj_resp = api_trajectory(
                track_id=cand.get("track_id", ""),
                instance_id=cand.get("instance_id", ""),
            )
            if traj_resp and traj_resp.get("success"):
                traj = traj_resp["trajectory"]
                st.session_state["trajectory_data"] = _convert_trajectory_response(traj, cand)
            else:
                # 尝试从本地数据构建跨镜轨迹
                traj_data = _build_traj_from_local(cand)
                st.session_state["trajectory_data"] = traj_data
        except Exception:
            traj_data = _build_traj_from_local(cand)
            st.session_state["trajectory_data"] = traj_data
        st.session_state["current_page"] = "trajectory"
        st.rerun()
