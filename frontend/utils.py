"""
frontend.utils - 前端工具函数

封装后端 API 调用、地图数据转换、颜色映射等通用功能。
地图可视化使用 pydeck。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import requests
import torch

from src.common.ids import extract_vehicle_id

logger = logging.getLogger(__name__)

# ============================================================
# API 配置
# ============================================================

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 10  # 秒


def _api_available() -> bool:
    """检查后端 API 是否可用"""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


API_AVAILABLE = _api_available()


# ============================================================
# API 调用封装
# ============================================================

def api_search(query_text: str, top_k: int = 20) -> Optional[Dict[str, Any]]:
    """调用文本检索 API"""
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/search/query",
            json={"query_text": query_text, "top_k": top_k},
            timeout=API_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        logger.warning(f"检索 API 返回 {r.status_code}: {r.text}")
        return None
    except Exception as e:
        logger.warning(f"检索 API 调用失败: {e}")
        return None


def api_plate_search(plate_number: str) -> Optional[Dict[str, Any]]:
    """调用车牌检索 API"""
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/search/plate",
            json={"plate_number": plate_number},
            timeout=API_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        logger.warning(f"车牌检索 API 调用失败: {e}")
        return None


def api_confirm(query_id: str, instance_id: str) -> Optional[Dict[str, Any]]:
    """调用目标确认 API"""
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/confirm/target",
            json={"query_id": query_id, "instance_id": instance_id},
            timeout=API_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        logger.warning(f"确认 API 调用失败: {e}")
        return None


def api_backtrack(instance_id: str) -> Optional[Dict[str, Any]]:
    """调用轨迹回溯 API"""
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/backtrack/trace",
            json={"instance_id": instance_id},
            timeout=API_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        logger.warning(f"回溯 API 调用失败: {e}")
        return None


def api_trajectory(track_id: str = "", instance_id: str = "") -> Optional[Dict[str, Any]]:
    """调用跨镜轨迹 API"""
    try:
        payload = {}
        if track_id:
            payload["track_id"] = track_id
        if instance_id:
            payload["instance_id"] = instance_id
        r = requests.post(
            f"{API_BASE}/api/v1/backtrack/trajectory",
            json=payload,
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        logger.warning(f"跨镜轨迹 API 返回 {r.status_code}: {r.text}")
        return None
    except Exception as e:
        logger.warning(f"跨镜轨迹 API 调用失败: {e}")
        return None


def _convert_trajectory_response(traj: Dict[str, Any], cand: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """将跨镜轨迹 API 响应转换为 trajectory 页面期望的格式。

    统一版本：合并 search.py 和 utils.py 两个版本的优点。
    - cand 参数可选，提供候选目标信息以丰富 target_instance 字段。
    - 始终包含 total_cameras / total_detections 等统计字段。
    """
    if cand is None:
        cand = {}
    camera_seq = traj.get("camera_sequence", [])
    attrs = traj.get("attributes", {})

    # 构建 target_instance（合并两版本字段，确保完整）
    target_instance = {
        "instance_id": cand.get("instance_id", "") or traj.get("track_id", ""),
        "target_id": cand.get("target_id", cand.get("instance_id", "")),
        "camera_id": camera_seq[0]["camera_id"] if camera_seq else cand.get("camera_id", ""),
        "timestamp": traj.get("first_appearance", cand.get("timestamp", "")),
        "target_type": cand.get("target_type", "vehicle"),
        "attributes": attrs,
        "plate_number": cand.get("plate_number"),
        "quality_score": cand.get("confidence", cand.get("combined_score", cand.get("quality_score", 0.8))),
        "keyframe_path": cand.get("keyframe_path", ""),
    }

    # 构建 observation_nodes（每个摄像头一个节点）
    observation_nodes = []
    for cam in camera_seq:
        first_frame = cam["frames"][0] if cam.get("frames") else {}
        observation_nodes.append({
            "camera_id": cam["camera_id"],
            "camera_name": cam.get("camera_name", cam["camera_id"]),
            "tracklet_id": f"TRK_{cam['camera_id']}",
            "timestamp": cam.get("arrival_time", ""),
            "latitude": None,
            "longitude": None,
            "keyframe_path": first_frame.get("crop_path", ""),
            "confidence": first_frame.get("confidence", 0.85),
        })

    # 构建 observation_segments
    observation_segments = []
    for cam in camera_seq:
        arrival = cam.get("arrival_time", "")
        departure = cam.get("departure_time", "")
        observation_segments.append({
            "tracklet_id": f"TRK_{cam['camera_id']}",
            "camera_id": cam["camera_id"],
            "camera_name": cam.get("camera_name", cam["camera_id"]),
            "start_time": arrival.split(" ")[-1] if " " in arrival else arrival,
            "end_time": departure.split(" ")[-1] if " " in departure else departure,
            "entry_description": "从画面进入",
            "exit_description": "从画面离开",
            "direction": cam.get("direction", ""),
        })

    # 构建 inference_segments（跨摄像头推断）
    inference_segments = []
    for i in range(len(camera_seq) - 1):
        src = camera_seq[i]
        tgt = camera_seq[i + 1]
        src_name = src.get("camera_name", src["camera_id"])
        tgt_name = tgt.get("camera_name", tgt["camera_id"])
        inference_segments.append({
            "source_camera_id": src["camera_id"],
            "target_camera_id": tgt["camera_id"],
            "source_camera_name": src_name,
            "target_camera_name": tgt_name,
            "source_latitude": None,
            "source_longitude": None,
            "target_latitude": None,
            "target_longitude": None,
            "confidence": 0.80,
            "estimated_travel_time": 120.0,
            "actual_travel_time": 120.0,
            "route_description": f"{src_name} → {tgt_name}",
        })

    # 构建 candidate_paths
    cam_names = [c.get("camera_name", c["camera_id"]) for c in camera_seq]
    candidate_paths = [{
        "path_id": "PATH_001",
        "road_segments": [f"SEG_{c['camera_id']}" for c in camera_seq],
        "confidence": 0.80,
        "distance_meters": len(camera_seq) * 250.0,
        "estimated_time": len(camera_seq) * 95.0,
        "description": " → ".join(cam_names),
    }] if camera_seq else []

    # 计算 total_detections（如果 API 未提供则从帧数据汇总）
    total_detections = traj.get("total_detections", 0)
    if not total_detections and camera_seq:
        total_detections = sum(len(cam.get("frames", [])) for cam in camera_seq)

    return {
        "target_instance": target_instance,
        "camera_sequence": [c["camera_id"] for c in camera_seq],
        "observation_nodes": observation_nodes,
        "observation_segments": observation_segments,
        "inference_segments": inference_segments,
        "candidate_paths": candidate_paths,
        "overall_confidence": 0.82,
        "total_cameras": traj.get("total_cameras", len(camera_seq)),
        "total_detections": total_detections,
        "total_duration_seconds": traj.get("total_duration_seconds", 0),
        "cross_camera_trajectory": traj,
    }


def api_dashboard_stats() -> Optional[Dict[str, Any]]:
    """获取仪表盘统计"""
    try:
        r = requests.get(f"{API_BASE}/api/v1/dashboard/stats", timeout=API_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        logger.warning(f"仪表盘 API 调用失败: {e}")
        return None


def api_camera_list() -> Optional[List[Dict[str, Any]]]:
    """获取摄像头列表"""
    try:
        r = requests.get(f"{API_BASE}/api/v1/dashboard/cameras", timeout=API_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return data.get("cameras", [])
        return None
    except Exception as e:
        logger.warning(f"摄像头列表 API 调用失败: {e}")
        return None


# ============================================================
# 颜色映射工具
# ============================================================

# 目标类型颜色
TYPE_COLORS = {
    "vehicle": "#1A3C6E",        # 警蓝
    "pedestrian": "#DC3545",     # 危险红
    "non_motor_vehicle": "#F5A623",  # 警示黄
}

# 轨迹段颜色
OBSERVATION_COLOR = "#28A745"    # 状态绿 - 观测段
INFERENCE_COLOR = "#F5A623"      # 警示黄 - 推断段
CANDIDATE_PATH_COLORS = [
    "#1A3C6E",  # 警蓝
    "#DC3545",  # 危险红
    "#F5A623",  # 警示黄
    "#28A745",  # 状态绿
    "#4285F4",  # 亮蓝
]

# 置信度颜色映射
CONFIDENCE_COLORS = {
    "high": "#28A745",    # > 0.7  状态绿
    "medium": "#F5A623",  # 0.4 - 0.7  警示黄
    "low": "#DC3545",     # < 0.4  危险红
}


def confidence_color(score: float) -> str:
    """根据置信度返回颜色"""
    if score >= 0.7:
        return CONFIDENCE_COLORS["high"]
    elif score >= 0.4:
        return CONFIDENCE_COLORS["medium"]
    return CONFIDENCE_COLORS["low"]


def confidence_label(score: float) -> str:
    """根据置信度返回标签"""
    if score >= 0.7:
        return "高"
    elif score >= 0.4:
        return "中"
    return "低"


def type_label(target_type: str) -> str:
    """目标类型中文标签"""
    mapping = {
        "vehicle": "车辆",
        "pedestrian": "行人",
        "non_motor_vehicle": "非机动车",
    }
    return mapping.get(target_type, target_type)


# ============================================================
# pydeck 地图数据转换
# ============================================================

# 苏州相城区默认中心（匹配摄像头坐标区域）
MAP_CENTER = [31.3050, 120.5850]
MAP_ZOOM = 13


def trajectory_to_pydeck_layers(traj_data: Dict[str, Any]) -> list:
    """
    将轨迹回溯结果转换为 pydeck 图层列表

    参数:
        traj_data: 回溯结果字典

    返回:
        pydeck 图层列表
    """
    import pydeck as pdk

    layers = []
    obs_nodes = traj_data.get("observation_nodes", [])
    inf_segs = traj_data.get("inference_segments", [])
    cand_paths = traj_data.get("candidate_paths", [])

    # --- 摄像头节点: CircleMarker ---
    if obs_nodes:
        node_data = []
        for node in obs_nodes:
            node_data.append({
                "position": [node["longitude"], node["latitude"]],
                "camera_id": node["camera_id"],
                "camera_name": node.get("camera_name", node["camera_id"]),
                "timestamp": node["timestamp"],
                "confidence": node.get("confidence", 0.9),
                "is_observation": True,
            })
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=node_data,
                get_position="position",
                get_radius=60,
                get_fill_color=[40, 167, 69, 220],  # 状态绿
                get_line_color=[255, 255, 255, 200],
                line_width_min_pixels=2,
                pickable=True,
                radius_min_pixels=8,
            )
        )

    # --- 观测段: 实线（绿色）---
    if len(obs_nodes) >= 2:
        obs_lines = []
        for i in range(len(obs_nodes) - 1):
            src = obs_nodes[i]
            tgt = obs_nodes[i + 1]
            obs_lines.append({
                "source": [src["longitude"], src["latitude"]],
                "target": [tgt["longitude"], tgt["latitude"]],
                "segment_type": "观测段",
                "confidence": src.get("confidence", 0.9),
            })
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=obs_lines,
                get_source_position="source",
                get_target_position="target",
                get_color=[40, 167, 69, 230],  # 状态绿
                get_width=5,
                pickable=True,
            )
        )

    # --- 推断段: 虚线效果（橙色）---
    if inf_segs:
        inf_lines = []
        for seg in inf_segs:
            inf_lines.append({
                "source": [seg["source_longitude"], seg["source_latitude"]],
                "target": [seg["target_longitude"], seg["target_latitude"]],
                "segment_type": "推断段",
                "confidence": seg.get("confidence", 0.7),
                "route": seg.get("route_description", ""),
            })
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=inf_lines,
                get_source_position="source",
                get_target_position="target",
                get_color=[245, 166, 35, 200],  # 警示黄
                get_width=3,
                pickable=True,
            )
        )

    # --- 候选路径: 不同颜色半透明线 ---
    # 候选路径用偏移线来区分（pydeck 不支持 dash，用颜色区分）
    for idx, path in enumerate(cand_paths):
        color = CANDIDATE_PATH_COLORS[idx % len(CANDIDATE_PATH_COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # 候选路径使用与推断段相同的节点但稍微偏移
        if obs_nodes and len(obs_nodes) >= 2:
            path_lines = []
            offset = 0.0003 * (idx + 1)  # 微小偏移以区分
            for i in range(len(obs_nodes) - 1):
                src = obs_nodes[i]
                tgt = obs_nodes[i + 1]
                path_lines.append({
                    "source": [src["longitude"] + offset, src["latitude"] + offset],
                    "target": [tgt["longitude"] + offset, tgt["latitude"] + offset],
                    "path_name": path.get("description", f"路径 {idx+1}"),
                    "confidence": path.get("confidence", 0.5),
                })
            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=path_lines,
                    get_source_position="source",
                    get_target_position="target",
                    get_color=[r, g, b, 120],
                    get_width=2,
                    pickable=True,
                )
            )

    return layers


def cameras_to_pydeck_layer(cameras: List[Dict[str, Any]]) -> "pdk.Layer":
    """将摄像头列表转换为 pydeck ScatterplotLayer"""
    import pydeck as pdk

    data = []
    for cam in cameras:
        status = cam.get("status", "online")
        color = [40, 167, 69] if status == "online" else [220, 53, 69]
        data.append({
            "position": [cam["longitude"], cam["latitude"]],
            "camera_id": cam["camera_id"],
            "name": cam["name"],
            "status": status,
            "color": color + [200],
            "detections": cam.get("today_detections", 0),
        })

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_radius=50,
        get_fill_color="color",
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1,
        pickable=True,
        radius_min_pixels=6,
    )


def make_pydeck_view_state(center: List[float] = None, zoom: int = None) -> "pdk.ViewState":
    """创建 pydeck ViewState"""
    import pydeck as pdk

    return pdk.ViewState(
        latitude=center[0] if center else MAP_CENTER[0],
        longitude=center[1] if center else MAP_CENTER[1],
        zoom=zoom if zoom else MAP_ZOOM,
        pitch=20,
        bearing=0,
    )


def make_pydeck_map(layers: list, view_state=None) -> "pdk.Deck":
    """创建 pydeck Deck 对象"""
    import pydeck as pdk

    if view_state is None:
        view_state = make_pydeck_view_state()

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"html": "<b>{camera_name}</b><br/>时间: {timestamp}<br/>置信度: {confidence}"},
        map_style="mapbox://styles/mapbox/dark-v10",
    )


# ============================================================
# 图片加载工具
# ============================================================

# 内联 SVG 占位图（无外部依赖，浅色风格）
_NO_IMAGE_SVG = (
    "data:image/svg+xml;utf8,"
    "%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22320%22%20height%3D%22180%22%3E"
    "%3Crect%20width%3D%22320%22%20height%3D%22180%22%20fill%3D%22%23F3F6FA%22%2F%3E"
    "%3Ctext%20x%3D%22160%22%20y%3D%2290%22%20fill%3D%22%23667085%22%20font-size%3D%2213%22"
    "%20text-anchor%3D%22middle%22%20dominant-baseline%3D%22middle%22"
    "%20font-family%3D%22Microsoft%20YaHei%22%3E"
    "%E6%97%A0%E6%A3%80%E6%B5%8B%E5%9B%BE%E7%89%87"
    "%3C%2Ftext%3E%3C%2Fsvg%3E"
)


def resolve_image_path(keyframe_path: str) -> Optional[str]:
    """
    解析 keyframe_path 并返回可用的图片路径/URL。

    优先级：
    1. 绝对路径且文件存在 → 直接返回
    2. 相对于 output/ 的路径且文件存在 → 返回绝对路径
    3. 相对于项目根目录的路径（cityflow/、data/ 等）且文件存在 → 返回绝对路径
    4. HTTP/HTTPS URL → 直接返回
    5. 以上均不满足 → 返回 None
    """
    if not keyframe_path:
        return None

    # HTTP URL（优先检查，避免 Path 解析干扰）
    if keyframe_path.startswith(("http://", "https://")):
        return keyframe_path

    p = Path(keyframe_path)

    # 绝对路径
    if p.is_absolute() and p.exists():
        return str(p)

    # 相对于项目 output/ 目录（兼容路径以 output/ 或 output\\ 开头的情况）
    rel = keyframe_path.replace("/", os.sep)
    if rel.lower().startswith(("output" + os.sep, "output/")):
        # 去掉 output/ 前缀
        rel_no_output = rel.split(os.sep, 1)[-1] if os.sep in rel else rel.split("/", 1)[-1]
        output_path = _OUTPUT_DIR / rel_no_output
        if output_path.exists():
            return str(output_path)

    # 相对于项目根目录（cityflow/、data/ 等路径）
    project_rel_path = _PROJECT_ROOT / rel
    if project_rel_path.exists():
        return str(project_rel_path)

    return None


def get_no_image_placeholder() -> str:
    """返回内联 SVG 占位图（无 picsum 等外部依赖）"""
    return _NO_IMAGE_SVG


def format_timestamp(ts_str: str) -> str:
    """格式化时间戳显示"""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts_str


def format_attributes(attrs: Dict[str, Any]) -> str:
    """格式化属性字典为可读字符串"""
    parts = []
    for k, v in attrs.items():
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


# ============================================================
# 通用 UI 辅助
# ============================================================

def get_target_type_icon(target_type: str) -> str:
    """目标类型文字标签（无 emoji）"""
    icons = {
        "vehicle": "[车]",
        "pedestrian": "[人]",
        "non_motor_vehicle": "[非]",
    }
    return icons.get(target_type, "[目]")


# ============================================================
# 摄像头名称映射（从 cityflow_camera_metadata.yaml 加载）
# ============================================================

def _load_camera_name_map() -> Dict[str, str]:
    """从 cityflow_camera_metadata.yaml 加载摄像头ID到名称的映射"""
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "cityflow_camera_metadata.yaml",
    )
    name_map: Dict[str, str] = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for cam in data.get("cameras", []):
                cid = cam.get("camera_id", "")
                cname = cam.get("name", cid)
                name_map[cid] = cname
        except Exception as e:
            logger.warning(f"加载摄像头元数据失败: {e}")
    return name_map


CAMERA_NAME_MAP = _load_camera_name_map()


# ============================================================
# CityFlow 数据加载与检索
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_CITYFLOW_RESULTS_JSON = _OUTPUT_DIR / "cityflow_results.json"
_CITYFLOW_CROPS_DIR = _OUTPUT_DIR / "cityflow_crops"

# 说明：颜色/车型 ID→中文名的映射表原为本地检索管线所用，检索改为调用后端后已无用例，
# 故随之删除（后端 api/routes/search.py 自带 _filter_business_attributes 做同样的中文化）。


_cityflow_cache: Optional[Dict[str, Any]] = None
_cityflow_cache_mtime: float = 0.0


def load_cityflow_results() -> Optional[Dict[str, Any]]:
    """加载 CityFlow 预处理结果（带内存缓存，避免每次检索重复读取 90MB+ JSON）"""
    global _cityflow_cache, _cityflow_cache_mtime
    if not _CITYFLOW_RESULTS_JSON.exists():
        _cityflow_cache = None
        return None
    try:
        current_mtime = _CITYFLOW_RESULTS_JSON.stat().st_mtime
        if _cityflow_cache is not None and current_mtime == _cityflow_cache_mtime:
            return _cityflow_cache
        t0 = __import__('time').time()
        with open(_CITYFLOW_RESULTS_JSON, "r", encoding="utf-8") as f:
            _cityflow_cache = json.load(f)
        _cityflow_cache_mtime = current_mtime
        logger.info(f"CityFlow 数据加载完成，共 {len(_cityflow_cache.get('detections', []))} 条检测，耗时 {__import__('time').time()-t0:.1f}s")
        return _cityflow_cache
    except Exception as e:
        logger.warning(f"加载 CityFlow 结果失败: {e}")
        _cityflow_cache = None
        return None


def has_cityflow_results() -> bool:
    """检查是否有 CityFlow 数据"""
    return _CITYFLOW_RESULTS_JSON.exists()


# 颜色同义词扩展
_COLOR_SYNONYMS = {
    "银灰色": "银色", "银色": "灰色", "金黄色": "黄色", "天蓝色": "蓝色",
    "深蓝色": "蓝色", "浅蓝色": "蓝色", "深绿色": "绿色", "浅绿色": "绿色",
    "暗红色": "红色", "鲜红色": "红色", "乳白色": "白色", "米白色": "白色",
    "灰白色": "白色", "墨绿色": "绿色", "桔黄色": "橙色", "橘色": "橙色",
    "土黄色": "黄色", "藏蓝色": "蓝色", "咖啡色": "棕色", "驼色": "棕色",
    # 新增同义词
    "银白": "白色", "珍珠白": "白色", "象牙白": "白色", "雪白": "白色",
    "亮白": "白色", "奶白": "白色",
    "深灰": "灰色", "浅灰": "灰色", "银灰": "灰色", "炭灰": "灰色",
    "烟灰": "灰色", "铁灰": "灰色",
    "深蓝": "蓝色", "湖蓝": "蓝色", "宝蓝": "蓝色", "天蓝": "蓝色",
    "靛蓝": "蓝色", "海军蓝": "蓝色",
    "深红": "红色", "酒红": "红色", "枣红": "红色", "玫红": "红色",
    "玫瑰红": "红色", "朱红": "红色",
    "深绿": "绿色", "橄榄绿": "绿色", "草绿": "绿色", "翠绿": "绿色",
    "柠檬黄": "黄色", "杏黄": "黄色", "鹅黄": "黄色",
    "橙红": "橙色", "橙黄": "橙色",
    "紫红": "红色", "粉红": "红色", "桃红": "红色",
    "古铜色": "棕色", "栗色": "棕色", "褐色": "棕色",
    "香槟色": "金色", "土豪金": "金色",
}

# 车型同义词扩展
_TYPE_SYNONYMS = {
    "越野车": "SUV", "吉普车": "SUV", "货车": "卡车", "大货车": "卡车",
    "小货车": "皮卡", "微型车": "面包车", "商务车": "MPV",
    "旅行轿车": "旅行车", "两厢轿车": "两厢车", "敞篷车": "跑车",
    "客车": "公交车", "大巴": "公交车", "中巴": "公交车",
    # 新增同义词
    "小轿车": "轿车", "三厢": "轿车", "三厢车": "轿车", "家用轿车": "轿车",
    "厢式货车": "卡车", "重型卡车": "卡车", "半挂车": "卡车",
    "小面包": "面包车", "微型面包": "面包车", "五菱": "面包车",
    " SUV ": "SUV", "城市越野": "SUV",
    "皮卡车": "皮卡",
    "轻型客车": "公交车", "大型客车": "公交车",
    "房车": "房车", "露营车": "房车",
    "的士": "轿车", "出租车": "轿车",
}

# 方向关键词映射
_DIRECTION_MAP = {
    "左转": "left_turn", "向左转": "left_turn", "左拐": "left_turn",
    "右转": "right_turn", "向右转": "right_turn", "右拐": "right_turn",
    "直行": "straight", "向前": "straight", "直走": "straight",
    "掉头": "u_turn", "调头": "u_turn", "回转": "u_turn",
    "向北": "north", "向南": "south", "向东": "east", "向西": "west",
    "往北": "north", "往南": "south", "往东": "east", "往西": "west",
}

# 动作关键词映射
_MOTION_MAP = {
    "转弯": "turning", "拐弯": "turning", "转向": "turning",
    "停车": "stopped", "停靠": "stopped", "等待": "stopped",
    "加速": "accelerating", "减速": "decelerating",
    "变道": "lane_change", "超车": "overtaking",
    "倒车": "reversing", "逆行": "wrong_way",
}


def _parse_cityflow_query(query: str) -> Dict[str, List[str]]:
    """
    使用 QueryParser 解析查询，提取颜色、车型、方向、动作等关键词。

    Returns:
        {"colors": [...], "types": [...], "directions": [...], "motions": [...], "keywords": [...]}
    """
    result: Dict[str, List[str]] = {"colors": [], "types": [], "directions": [], "motions": [], "keywords": []}
    q = query.strip()

    try:
        from src.retrieval.query_parser import QueryParser
        parser = QueryParser()
        parsed = parser.parse(q)
        if parsed.color:
            result["colors"].append(parsed.color)
        if parsed.vehicle_type:
            # 出租车/的士 → 映射到轿车（CityFlow 数据中出租车以轿车类型存储）
            if parsed.vehicle_type in ("出租车", "的士"):
                result["types"].append("轿车")
            else:
                result["types"].append(parsed.vehicle_type)
        if parsed.target_type:
            result["keywords"].append(parsed.target_type)
        if parsed.gender:
            result["keywords"].append("pedestrian")
        if parsed.clothing_color:
            result["keywords"].append("pedestrian_attr")
    except Exception:
        # fallback to simple keyword matching
        all_colors_cn = list(_EN_COLOR_TO_CN.values())
        for c in all_colors_cn:
            if c in q:
                result["colors"].append(c)

        # 检查颜色同义词
        for syn, base in _COLOR_SYNONYMS.items():
            if syn in q and base not in result["colors"]:
                result["colors"].append(base)

        # 非机动车关键词 → 标记 target_type 为 non_motor_vehicle
        non_motor_keywords = ["三轮车", "小电驴", "拖拉机", "电动车", "摩托车", "电瓶车", "自行车"]
        for t in non_motor_keywords:
            if t in q:
                result["keywords"].append("non_motor_vehicle")
                break

        # 出租车/的士 → 映射到轿车
        if "出租车" in q or "的士" in q:
            result["types"].append("轿车")

        # 其他 CityFlow 已有车型
        all_types_cn = list(_EN_TYPE_TO_CN.values())
        for t in all_types_cn:
            if t in q and t not in result["types"]:
                result["types"].append(t)

        # 检查车型同义词
        for syn, base in _TYPE_SYNONYMS.items():
            if syn in q and base not in result["types"]:
                result["types"].append(base)

        if "人" in q or "行人" in q or "男" in q or "女" in q:
            result["keywords"].append("pedestrian")
        if "穿" in q or "背" in q or "衣服" in q or "书包" in q:
            result["keywords"].append("pedestrian_attr")

    # 方向识别
    for kw, direction in _DIRECTION_MAP.items():
        if kw in q and direction not in result["directions"]:
            result["directions"].append(direction)

    # 动作识别
    for kw, motion in _MOTION_MAP.items():
        if kw in q and motion not in result["motions"]:
            result["motions"].append(motion)

    return result


# ============================================================
# CLIP 模型懒加载
# ============================================================

_clip_extractor = None


def _check_cuda() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_clip_extractor():
    """获取 CLIP 特征提取器（懒加载单例）"""
    global _clip_extractor
    if _clip_extractor is None:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from src.perception.feature_extractor import FeatureExtractor
        device = "cuda" if _check_cuda() else "cpu"
        _clip_extractor = FeatureExtractor(
            clip_model="CN-CLIP-ViT-L-14",
            clip_dim=768,
            device=device,
        )
    return _clip_extractor


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# BLIP 模型懒加载（图文精排）
# ============================================================

_blip_extractor = None


def _get_blip_extractor():
    """获取 BLIP 图文检索模型单例（延迟加载）。
    使用 HuggingFace transformers 的 BlipForImageTextRetrieval。
    如果加载失败则返回 {"available": False}。
    """
    global _blip_extractor
    if _blip_extractor is None:
        try:
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "Salesforce/blip-itm-base-coco"

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
            model.eval()

            _blip_extractor = {
                "model": model,
                "processor": processor,
                "device": device,
                "available": True,
            }
            logger.info(f"✅ BLIP 模型加载完成 (device={device})")
        except Exception as e:
            logger.warning(f"⚠️ BLIP 模型加载失败，将使用纯 OpenCLIP: {e}")
            _blip_extractor = {"available": False}
    return _blip_extractor


def _blip_score_image_text(blip_ext: dict, image_path: str, text: str) -> float:
    """使用 BLIP 对单张图片-文本对打分，返回 0-1 之间的相似度分数。"""
    try:
        from PIL import Image
        processor = blip_ext["processor"]
        model = blip_ext["model"]
        device = blip_ext["device"]

        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=raw_image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # itm_logits: [1, 2] — 第 1 列是匹配分数
            itm_logits = outputs.itm_logits
            score = torch.softmax(itm_logits, dim=-1)[0, 1].item()

        return float(score)
    except Exception as e:
        logger.debug(f"BLIP 评分失败 ({image_path}): {e}")
        return 0.5  # 中性分数


# ============================================================
# 属性一致性评分
# ============================================================

def _compute_attr_consistency(det_attrs: Dict[str, Any], query_attrs: Dict[str, List[str]]) -> float:
    """计算检测属性与查询属性的一致性评分 (0-1)。"""
    score = 0.0
    weight_sum = 0.0

    # 颜色匹配
    if query_attrs.get("colors"):
        weight_sum += 0.4
        det_color = det_attrs.get("颜色", det_attrs.get("color", "")).lower()
        for q_color in query_attrs["colors"]:
            qc = q_color.lower()
            if qc in det_color or det_color in qc:
                score += 0.4
                break

    # 车型匹配
    if query_attrs.get("types"):
        weight_sum += 0.3
        det_type = det_attrs.get("车型", det_attrs.get("vehicle_type", "")).lower()
        for q_type in query_attrs["types"]:
            qt = q_type.lower()
            if qt in det_type or det_type in qt:
                score += 0.3
                break

    # 方向匹配（简化版：预留接口）
    if query_attrs.get("directions"):
        weight_sum += 0.2
        # 需要轨迹起终点信息，暂给中性分
        score += 0.1

    # 归一化
    if weight_sum > 0:
        score /= weight_sum

    return max(0.0, min(1.0, score))


def _batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算查询向量与矩阵中所有行的余弦相似度"""
    query_vec = np.asarray(query_vec, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-8:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms[matrix_norms < 1e-8] = 1e-8
    similarities = matrix @ query_vec / (matrix_norms * query_norm)
    return similarities


def _cityflow_det_to_candidate(det: Dict[str, Any], rank: int, score: float) -> Dict[str, Any]:
    """将 CityFlow detection 转换为搜索结果的 candidate 格式"""
    attrs_raw = det.get("attributes", {})

    # 解析颜色
    color_id = attrs_raw.get("color_id", -1)
    color_en = attrs_raw.get("color", "unknown")
    if color_id >= 0 and color_id in _CITYFLOW_COLOR_CN:
        color_cn = _CITYFLOW_COLOR_CN[color_id]
    elif color_en in _EN_COLOR_TO_CN:
        color_cn = _EN_COLOR_TO_CN[color_en]
    else:
        color_cn = color_en if color_en != "unknown" else "未知"

    # 解析车型
    type_id = attrs_raw.get("type_id", -1)
    type_en = attrs_raw.get("vehicle_type", "unknown")
    if type_id >= 0 and type_id in _CITYFLOW_TYPE_CN:
        type_cn = _CITYFLOW_TYPE_CN[type_id]
    elif type_en.lower() in _EN_TYPE_TO_CN:
        type_cn = _EN_TYPE_TO_CN[type_en.lower()]
    else:
        type_cn = type_en if type_en != "unknown" else "未知"

    camera_id = det.get("camera_id", "")
    scene_id = det.get("scene_id", "")
    keyframe_path = det.get("keyframe_path", "")
    target_id = det.get("target_id", "")

    # 从 target_id 提取 vehicle_id 并构造 track_id
    # 格式示例: "CF3_c001_V0034_000001"
    vehicle_id = extract_vehicle_id(target_id)
    track_id = f"CF3_TRACK_{camera_id}_{vehicle_id}" if vehicle_id else ""

    # 构建属性字典（同时包含中文键和英文键，兼容不同消费方）
    attributes = {
        "颜色": color_cn,
        "color": color_cn,
        "车型": type_cn,
        "vehicle_type": type_cn,
        "摄像头": CAMERA_NAME_MAP.get(camera_id, camera_id),
        "场景": scene_id,
    }

    return {
        "instance_id": target_id,
        "target_id": target_id,
        "track_id": track_id,
        "target_type": det.get("target_type", "vehicle"),
        "camera_id": camera_id,
        "camera_name": CAMERA_NAME_MAP.get(camera_id, camera_id),
        "timestamp": det.get("timestamp", ""),
        "attributes": attributes,
        "plate_number": None,
        "confidence": det.get("confidence", 0.5),
        "quality_score": det.get("confidence", 0.5),
        "text_score": score,
        "attribute_match_score": score,
        "combined_score": score,
        "score": score,
        "rank": rank,
        "keyframe_path": keyframe_path,
        "image_path": det.get("image_path", ""),
        "detection_bbox": det.get("bbox"),
        "scene_id": scene_id,
        "data_source": "cityflow",
        "source": det.get("source", "cityflow"),
        "has_trajectory": bool(track_id),
        "track_frame_count": 0,
        "track_cameras": [],
        "track_frames": [],
    }
def build_search_results_from_cityflow(
    query: str = "", top_k: int = 20,
    scene_filter: str = "",
    camera_filter: str = "",
    color_filter: list | None = None,
    type_filter: list | None = None,
) -> Optional[Dict[str, Any]]:
    """
    基于 CityFlow 数据构建检索结果。

    检索策略：属性硬过滤 + CLIP 向量相似度排序。
    若 CLIP 不可用则回退到文本匹配评分。

    Args:
        query: 自然语言查询文本
        top_k: 返回结果数量
        scene_filter: 场景筛选 (如 "S01")
        camera_filter: 摄像头筛选 (如 "c001")
        color_filter: 颜色筛选列表
        type_filter: 车型筛选列表

    Returns:
        与现有检索结果格式兼容的字典，或 None（无数据时）
    """
    data = load_cityflow_results()
    if data is None:
        return None

    detections = data.get("detections", [])
    if not detections:
        return None

    # 解析查询
    parsed = _parse_cityflow_query(query) if query else {"colors": [], "types": [], "directions": [], "motions": [], "keywords": []}
    query_colors = set(parsed["colors"])
    query_types = set(parsed["types"])
    is_pedestrian_query = "pedestrian" in parsed["keywords"]
    is_non_motor_query = "non_motor_vehicle" in parsed["keywords"]

    # 外部筛选
    ext_colors = set(color_filter) if color_filter else set()
    ext_types = set(type_filter) if type_filter else set()
    match_colors = query_colors | ext_colors
    match_types = query_types | ext_types

    # ── 步骤 1：属性硬过滤 ──
    filtered_dets = []
    for det in detections:
        if scene_filter and det.get("scene_id", "") != scene_filter:
            continue
        if camera_filter and det.get("camera_id", "") != camera_filter:
            continue

        attrs = det.get("attributes", {})
        color_id = attrs.get("color_id", -1)
        type_id = attrs.get("type_id", -1)

        det_color = (
            _CITYFLOW_COLOR_CN.get(color_id)
            or _EN_COLOR_TO_CN.get(attrs.get("color", "").lower(), "")
            or attrs.get("color", "")
        )
        det_type = (
            _CITYFLOW_TYPE_CN.get(type_id)
            or _EN_TYPE_TO_CN.get(attrs.get("vehicle_type", "").lower(), "")
            or attrs.get("vehicle_type", "")
        )

        # 硬过滤
        if match_colors and det_color not in match_colors:
            continue
        if match_types and det_type not in match_types:
            continue

        # 行人/非机动车查询
        if is_pedestrian_query or is_non_motor_query:
            continue  # CityFlow 只有车辆数据

        filtered_dets.append(det)

    if not filtered_dets:
        return {
            "query_id": "CF_Q_001",
            "candidates": [],
            "total_count": 0,
            "total": 0,
            "source": "cityflow",
            "message": f"未找到与「{query}」匹配的结果",
        }

    # ── 步骤 2：CLIP 向量相似度排序 + BLIP 精排融合 ──
    has_clip = any("clip_image_vector" in det for det in filtered_dets)

    if has_clip and query:
        try:
            import time as _time
            t_start = _time.time()
            extractor = _get_clip_extractor()
            # 编码查询文本
            query_text_vec = extractor.extract_text_clip(query)

            # 计算 CLIP 相似度
            scored = []
            for det in filtered_dets:
                clip_img_vec = det.get("clip_image_vector")
                clip_txt_vec = det.get("clip_text_vector")

                img_sim = _cosine_similarity(query_text_vec, clip_img_vec) if clip_img_vec else 0.0
                txt_sim = _cosine_similarity(query_text_vec, clip_txt_vec) if clip_txt_vec else 0.0

                # 双路加权：图像相似度 0.65 + 文本描述相似度 0.35
                clip_score = 0.65 * img_sim + 0.35 * txt_sim
                # 归一化到 0-1 范围（余弦相似度范围 -1 到 1）
                clip_score = max(0.0, min(1.0, (clip_score + 1.0) / 2.0))

                scored.append((det, clip_score))

            # 按 CLIP 相似度降序排序
            scored.sort(key=lambda x: x[1], reverse=True)

            # ── BLIP 精排 + 属性一致性融合 ──
            # 取 Top-50 候选进行精排
            top_n = min(50, len(scored))
            top_candidates = scored[:top_n]

            blip_ext = _get_blip_extractor()
            blip_available = blip_ext.get("available", False)

            final_scored = []
            for det, openclip_score in top_candidates:
                # 属性一致性评分
                det_attrs_raw = det.get("attributes", {})
                color_id = det_attrs_raw.get("color_id", -1)
                type_id = det_attrs_raw.get("type_id", -1)
                det_color = (
                    _CITYFLOW_COLOR_CN.get(color_id)
                    or _EN_COLOR_TO_CN.get(det_attrs_raw.get("color", "").lower(), "")
                    or det_attrs_raw.get("color", "")
                )
                det_type = (
                    _CITYFLOW_TYPE_CN.get(type_id)
                    or _EN_TYPE_TO_CN.get(det_attrs_raw.get("vehicle_type", "").lower(), "")
                    or det_attrs_raw.get("vehicle_type", "")
                )
                det_attr_dict = {"颜色": det_color, "color": det_color, "车型": det_type, "vehicle_type": det_type}
                attr_score = _compute_attr_consistency(det_attr_dict, parsed)

                # ReID 评分（暂用固定值，后续迭代优化）
                reid_score = 0.5

                if blip_available:
                    # BLIP 精排
                    keyframe_path = det.get("keyframe_path", "")
                    img_path = resolve_image_path(keyframe_path)
                    if img_path and os.path.exists(img_path):
                        blip_score = _blip_score_image_text(blip_ext, img_path, query)
                    else:
                        blip_score = 0.5  # 无图片时给中性分

                    # 融合打分: 0.45*OpenCLIP + 0.30*BLIP + 0.15*Attr + 0.10*ReID
                    fused = (
                        0.45 * openclip_score
                        + 0.30 * blip_score
                        + 0.15 * attr_score
                        + 0.10 * reid_score
                    )
                    final_scored.append((det, fused))
                else:
                    # Fallback: OpenCLIP + 属性 + ReID
                    fused = 0.60 * openclip_score + 0.25 * attr_score + 0.15 * reid_score
                    final_scored.append((det, fused))

            # 按融合分数降序排序
            final_scored.sort(key=lambda x: x[1], reverse=True)

            t_elapsed = _time.time() - t_start
            logger.info(
                f"🔍 双模型融合检索完成: top_n={top_n}, "
                f"BLIP={'✅' if blip_available else '❌'}, "
                f"耗时 {t_elapsed:.2f}s"
            )

            # 构建 candidates
            candidates = []
            for rank, (det, score) in enumerate(final_scored[:top_k], 1):
                cand = _cityflow_det_to_candidate(det, rank, round(score, 3))
                candidates.append(cand)

            return {
                "query_id": "CF_Q_001",
                "candidates": candidates,
                "total_count": len(candidates),
                "total": len(candidates),
                "source": "cityflow",
                "clip_retrieval": True,
                "blip_refined": blip_available,
                "fusion_scores": True,
            }

        except Exception as e:
            logger.warning(f"CLIP 检索失败，回退到文本匹配: {e}")
            # fall through to text-based scoring

    # ── 步骤 3：Fallback - 文本匹配评分 ──
    scored_candidates = []
    for det in filtered_dets:
        attrs = det.get("attributes", {})
        color_id = attrs.get("color_id", -1)
        type_id = attrs.get("type_id", -1)

        det_color = (
            _CITYFLOW_COLOR_CN.get(color_id)
            or _EN_COLOR_TO_CN.get(attrs.get("color", "").lower(), "")
            or attrs.get("color", "")
        )
        det_type = (
            _CITYFLOW_TYPE_CN.get(type_id)
            or _EN_TYPE_TO_CN.get(attrs.get("vehicle_type", "").lower(), "")
            or attrs.get("vehicle_type", "")
        )

        color_matched = bool(match_colors) and det_color in match_colors
        type_matched = bool(match_types) and det_type in match_types
        has_color_req = bool(match_colors)
        has_type_req = bool(match_types)

        # 关键词匹配评分（替代逐字符匹配）
        if has_color_req and has_type_req:
            if color_matched and type_matched:
                score = 0.95  # 组合匹配最高分
            elif color_matched:
                score = 0.9   # 颜色完全匹配
            elif type_matched:
                score = 0.85  # 车型完全匹配
            else:
                score = 0.05  # 无匹配基线
        elif has_color_req:
            score = 0.9 if color_matched else 0.05
        elif has_type_req:
            score = 0.85 if type_matched else 0.05
        else:
            score = 0.5

        # 关键词匹配加分（替代逐字符匹配）
        if query:
            query_keywords = set()
            for kw_list in [list(_EN_COLOR_TO_CN.keys()), list(_EN_TYPE_TO_CN.keys()),
                            list(_EN_COLOR_TO_CN.values()), list(_EN_TYPE_TO_CN.values())]:
                for kw in kw_list:
                    if kw in query:
                        query_keywords.add(kw)
            # 额外同义词关键词
            for syn in list(_COLOR_SYNONYMS.keys()) + list(_TYPE_SYNONYMS.keys()):
                if syn in query:
                    query_keywords.add(syn)
            if query_keywords:
                desc = det.get("description", "")
                matched_kw_count = sum(1 for kw in query_keywords if kw in desc or kw in str(attrs))
                if matched_kw_count > 0:
                    score += min(0.05 * matched_kw_count, 0.15)

        score += det.get("confidence", 0.5) * 0.05
        if score < 0.45:
            continue
        score = max(0.05, min(score, 0.99))
        scored_candidates.append((det, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    candidates = []
    for rank, (det, score) in enumerate(scored_candidates[:top_k], 1):
        cand = _cityflow_det_to_candidate(det, rank, round(score, 3))
        candidates.append(cand)
        # 调试日志：验证第一条结果的 keyframe_path
        if rank == 1:
            logger.info(f"CityFlow 检索结果[1]: instance_id={cand.get('instance_id')}, keyframe_path={cand.get('keyframe_path')}")

    if not candidates:
        return {
            "query_id": "CF_Q_001",
            "candidates": [],
            "total_count": 0,
            "total": 0,
            "source": "cityflow",
            "message": f"未找到与「{query}」匹配的结果",
        }

    return {
        "query_id": "CF_Q_001",
        "candidates": candidates,
        "total_count": len(candidates),
        "total": len(candidates),
        "source": "cityflow",
        "clip_retrieval": False,
    }



