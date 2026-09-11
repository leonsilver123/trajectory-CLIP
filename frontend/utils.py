"""
frontend.utils - 前端工具函数

封装后端 API 调用、地图数据转换、颜色映射等通用功能。
地图可视化使用 pydeck。
"""

from __future__ import annotations

import logging
import numbers
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import requests
# torch 目前在本模块内已无使用点（原 BLIP 精排所依赖），但按部署约定保留导入：
# Dockerfile.frontend 已装 torch(CPU)，去掉导入会让镜像依赖与代码不一致。
import torch

from src.storage.datastore import (
    get_stats as datastore_get_stats,
    has_data as datastore_has_data,
    load_results,
)

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
# 数值安全格式化（None 兜底）
# ============================================================
# 背景：后端在「无真实依据」时会如实返回 null。例如两侧观测时间窗口重叠时，
# inference_segments[].actual_travel_time 就是 null（此时实际行程时间无法确定）。
# 这类字段不能直接用 dict.get(k, 0) 兜底——键存在、值为 None，取出来仍是 None，
# 直接 f"{v:.0f}" 会抛 TypeError: unsupported format string passed to NoneType.__format__。
# 下面两个函数在展示层统一兜底：缺依据时显示占位符，而不是编造一个 0。


def is_number(value: Any) -> bool:
    """判断是否为可用于格式化/比较的实数。

    用 numbers.Real 而不是 (int, float)：numpy 标量（如 np.int64）不是内置 int 的子类，
    但已注册到 numbers.Real；bool 是 int 子类，需要显式排除。
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def safe_number(value: Any, default: float = 0.0) -> float:
    """把可能为 None / 非数值的字段安全转成 float。

    仅用于「聚合求和」与「阈值比较」这类必须有数值的场合；
    纯展示场景请用 format_number()，以免把「无依据」显示成 0。
    """
    return float(value) if is_number(value) else default


def format_number(value: Any, spec: str = ".0f", unit: str = "", fallback: str = "--") -> str:
    """格式化可空数值字段；缺依据（None / 非数值）时返回占位符。

    参数:
        value: 原始字段值，可能为 None
        spec:  Python 格式说明符，如 ".0f" / ".0%" / ".1%"
        unit:  单位后缀，如 "秒" / "m"
        fallback: 缺依据时的占位文案
    """
    if not is_number(value):
        return fallback
    return f"{value:{spec}}{unit}"


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
    "unknown": "#98A2B3",  # 无真实依据（后端如实返回 None），中性灰
}


def confidence_color(score: Any) -> str:
    """根据置信度返回颜色；无真实依据时返回中性灰，不冒充「低置信度」"""
    if not is_number(score):
        return CONFIDENCE_COLORS["unknown"]
    if score >= 0.7:
        return CONFIDENCE_COLORS["high"]
    elif score >= 0.4:
        return CONFIDENCE_COLORS["medium"]
    return CONFIDENCE_COLORS["low"]


def confidence_label(score: Any) -> str:
    """根据置信度返回标签；无真实依据时返回「未知」，不冒充「低」"""
    if not is_number(score):
        return "未知"
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

# 说明：颜色/车型 ID→中文名的映射表原为本地检索管线所用，检索改为调用后端后已无用例，
# 故随之删除（后端 api/routes/search.py 自带 _filter_business_attributes 做同样的中文化）。

# 说明：本模块原先自己开文件读 107MB 的 cityflow_results.json（另有 mtime 缓存），
# 与 api/、src/ 里的几处读取各自独立。现统一收敛到 src.storage.datastore：
# 优先读 output/datastore/ 的 Parquet + SQLite，缺失时自动回退 JSON 直读。
# 这里只保留薄封装，供前端页面沿用原有函数名调用。


def load_cityflow_results() -> Optional[Dict[str, Any]]:
    """加载 CityFlow 预处理结果（统一走 src.storage.datastore，带进程内缓存）"""
    return load_results()


def has_cityflow_results() -> bool:
    """检查是否有 CityFlow 数据（datastore 或 JSON 任一在位）"""
    return datastore_has_data()


def get_cityflow_stats() -> Dict[str, Any]:
    """轻量统计（检测数/轨迹数/摄像头数）——datastore 在位时不解析大文件"""
    return datastore_get_stats()


# 说明：本地查询解析（_parse_cityflow_query）及其专用的颜色/车型同义词表、
# 方向/动作关键词表已删除——它们只服务于已移除的本地检索管线。
# 查询文本的解析现在完全由后端 api/routes/search.py 的 _extract_query_features 负责。


# ============================================================
# 说明：本地 CLIP/BLIP 打分逻辑已删除
# ============================================================
# 原先这里有一套前端自有的「OpenCLIP CN-CLIP-ViT-L-14 + BLIP 精排」实现
# （_get_clip_extractor / _get_blip_extractor / _blip_score_image_text /
#   _compute_attr_consistency / _batch_cosine_similarity / _cosine_similarity /
#   _cityflow_det_to_candidate），与后端 api/routes/search.py 的
# 「Chinese-CLIP ViT-B-16 + 融合排序」模型版本和权重都不一致。
# 现已整体删除：检索只走后端 api_search()，模型加载与打分的唯一实现在后端。
# 说明：本地检索管线（build_search_results_from_cityflow：属性硬过滤 → OpenCLIP
# CN-CLIP-ViT-L-14 → BLIP 精排 → 属性重排）已整体删除。检索唯一真源是后端
# api/routes/search.py（属性粗筛 → Chinese-CLIP ViT-B-16 → 融合排序），前端只调用
# api_search()；后端不可用时如实提示用户，不再用另一套本地算法给出「看起来一样」的结果。



