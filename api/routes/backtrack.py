"""
api.routes.backtrack - 回溯 API

提供轨迹回溯接口:
- POST /api/v1/backtrack/trace  以确认目标为锚点进行轨迹回溯
- GET  /api/v1/backtrack/result/{query_id}  获取回溯结果
- POST /api/v1/backtrack/trajectory  获取完整跨镜轨迹（多摄像头时间线拼接）
"""

from __future__ import annotations

import json
import os
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.ids import extract_vehicle_id, parse_track_id
from src.common.logger import get_logger
from src.common.session_store import SessionNotFoundError, get_session_store

logger = get_logger("api.routes.backtrack")

router = APIRouter()

# 内存存储最近的回溯结果
_backtrack_results: Dict[str, Dict[str, Any]] = {}

# ============================================================
# 加载 CityFlow 真实轨迹数据
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_PATH = _PROJECT_ROOT / "output" / "cityflow_results.json"

# target_id → 该 target 的所有 detections（按 timestamp 排序）
_target_tracks: Dict[str, List[Dict[str, Any]]] = {}
# camera_id -> 摄像头元数据（name, scene, lat, lon）
_camera_meta: Dict[str, Dict[str, Any]] = {}
# 摄像头方向映射（从 YAML 加载）
_camera_direction_map: Dict[str, str] = {}
# 全量 detections 列表（用于跨镜查询）
_all_detections: List[Dict[str, Any]] = []
# vehicle_id -> 该 vehicle 在所有摄像头的 detections 列表
_vehicle_detections: Dict[str, List[Dict[str, Any]]] = {}
# 真实数据文件是否已成功加载（决定查不到目标时返回 404 还是演示数据）
_data_loaded: bool = False


def _load_cityflow_data() -> None:
    """从 cityflow_results.json 加载真实轨迹数据（仅首次调用时加载）"""
    global _target_tracks, _camera_meta, _data_loaded
    if _target_tracks:  # 已加载
        return

    if not _RESULTS_PATH.exists():
        logger.warning(f"CityFlow 结果文件不存在: {_RESULTS_PATH}，将使用 fallback mock 数据")
        return

    try:
        with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载 CityFlow 结果文件失败: {e}")
        return

    detections = data.get("detections", [])
    tracks_list = data.get("tracks", [])
    _data_loaded = True
    logger.info(f"从 cityflow_results.json 加载了 {len(detections)} 条检测记录, {len(tracks_list)} 条轨迹")

    # 保存全量检测列表（用于跨镜查询）
    _all_detections.clear()
    _all_detections.extend(detections)

    # 按 vehicle_id 分组 detections（用于跨镜轨迹构建）
    _vehicle_detections.clear()
    for det in detections:
        # 从 target_id 提取 vehicle_id: CF3_c001_V0034_000001 -> V0034
        target_id = det.get("target_id", "")
        vid = extract_vehicle_id(target_id)
        if vid:
            _vehicle_detections.setdefault(vid, []).append(det)

    # 按 timestamp 排序每个 vehicle 的 detections
    for vid in _vehicle_detections:
        _vehicle_detections[vid].sort(key=lambda d: d.get("timestamp", ""))

    logger.info(f"构建了 {len(_vehicle_detections)} 个 vehicle_id 的跨镜索引")

    # 按 target_id 分组（用于单 target 轨迹）
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in detections:
        tid = det.get("target_id", "")
        if tid:
            grouped[tid].append(det)

    # 对每个 target 的检测按 timestamp 排序，并提取摄像头元数据
    for tid, dets in grouped.items():
        dets_sorted = sorted(dets, key=lambda d: d.get("timestamp", ""))
        _target_tracks[tid] = dets_sorted
        for det in dets_sorted:
            cid = det.get("camera_id", "")
            if cid and cid not in _camera_meta:
                _camera_meta[cid] = {
                    "camera_id": cid,
                    "name": f"{det.get('scene_id', 'Unknown')}-{cid}",
                    "scene": det.get("scene_id", ""),
                    "latitude": det.get("latitude"),
                    "longitude": det.get("longitude"),
                }

    # 从 YAML 加载摄像头名称和方向
    _load_camera_metadata_from_yaml()

    logger.info(f"解析出 {len(_target_tracks)} 条目标轨迹，覆盖 {len(_camera_meta)} 个摄像头")


def _load_camera_metadata_from_yaml():
    """从 cityflow_camera_metadata.yaml 加载摄像头名称和方向映射"""
    global _camera_meta, _camera_direction_map
    yaml_path = _PROJECT_ROOT / "configs" / "cityflow_camera_metadata.yaml"
    if not yaml_path.exists():
        return
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        lane_dir_to_cn = {
            "northbound": "由南向北",
            "southbound": "由北向南",
            "eastbound": "由西向东",
            "westbound": "由东向西",
        }
        for cam in data.get("cameras", []):
            cid = cam.get("camera_id", "")
            cname = cam.get("name", cid)
            lane_dir = cam.get("lane_direction", "")
            # 更新 _camera_meta 中的 name
            if cid in _camera_meta:
                _camera_meta[cid]["name"] = cname
            else:
                _camera_meta[cid] = {
                    "camera_id": cid,
                    "name": cname,
                    "scene": cam.get("scene", ""),
                    "latitude": cam.get("latitude"),
                    "longitude": cam.get("longitude"),
                }
            # 方向映射
            _camera_direction_map[cid] = lane_dir_to_cn.get(lane_dir, "")
        logger.info(f"从 YAML 加载了 {len(_camera_direction_map)} 个摄像头方向信息")
    except Exception as e:
        logger.warning(f"加载摄像头 YAML 失败: {e}")


def _find_target_id(instance_id: str) -> Optional[str]:
    """
    根据确认的 instance_id 定位真实的 target_id，只做直接匹配。

    instance_id 可能是:
    1. 直接的 target_id（如 CF3_c001_V0034_000001）→ 命中 _target_tracks 则返回
    2. 轨迹 ID（如 CF3_TRACK_c001_V0034）→ 解析出 vehicle_id，
       定位到该车辆的真实 target_id（优先同摄像头的检测）

    找不到时返回 None（调用方转 404），不再随机抽取任何目标。
    """
    if not instance_id:
        return None

    # 直接匹配
    if instance_id in _target_tracks:
        return instance_id

    # track_id 形式：按 vehicle_id 定位真实目标
    vehicle_id = extract_vehicle_id(instance_id)
    if not vehicle_id:
        return None

    vehicle_dets = _vehicle_detections.get(vehicle_id, [])
    if not vehicle_dets:
        return None

    # track_id 里带了摄像头信息时，优先返回该摄像头下的检测
    camera_id = parse_track_id(instance_id).get("camera_id")
    if camera_id:
        for det in vehicle_dets:
            if det.get("camera_id") == camera_id:
                return det.get("target_id") or None

    return vehicle_dets[0].get("target_id") or None


def _path_distance_meters(camera_sequence: List[str]) -> float:
    """
    按摄像头序列累加相邻摄像头的真实球面距离（米）

    经纬度取自摄像头元数据；缺坐标时该段按 0 计，不再用「每段 250 米」之类的估算值。
    """
    from src.common.utils import haversine_distance

    total = 0.0
    for src_cid, tgt_cid in zip(camera_sequence, camera_sequence[1:]):
        src_info = _camera_meta.get(src_cid, {})
        tgt_info = _camera_meta.get(tgt_cid, {})
        src_lat, src_lon = src_info.get("latitude"), src_info.get("longitude")
        tgt_lat, tgt_lon = tgt_info.get("latitude"), tgt_info.get("longitude")
        if None in (src_lat, src_lon, tgt_lat, tgt_lon):
            continue
        total += haversine_distance(src_lat, src_lon, tgt_lat, tgt_lon)
    return round(total, 1)


def _build_trajectory_from_target(instance_id: str, target_id: str) -> Dict[str, Any]:
    """
    从真实轨迹数据构建回溯结果

    时间、置信度、方向全部取自 cityflow_results.json 中的真实检测记录：
      - 观测节点的 timestamp 用该摄像头首次检测的真实时间
      - 观测段的 start_time / end_time 用该摄像头首末检测的真实时间
      - 方向取自摄像头元数据中的 lane_direction

    无法从现有数据得出的量（跨镜推断段的置信度与旅行时间、路径可信度、综合评分、
    证据分项）一律留空（None / 0.0）并如实标注，由评分子系统填充，不使用随机数。
    """
    dets = _target_tracks.get(target_id, [])
    if not dets:
        return {}

    # query_id 由 (instance_id, target_id) 确定性派生：同一目标重复回溯，结果完全一致
    query_id = f"BT_{uuid.uuid5(uuid.NAMESPACE_URL, f'{instance_id}|{target_id}').hex[:8]}"

    # 按摄像头分组（dets 已按 timestamp 排序），保留每个摄像头的全部真实检测
    camera_dets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in dets:
        cid = det.get("camera_id", "")
        if cid:
            camera_dets[cid].append(det)

    # 摄像头按各段首次出现的真实时间排序
    sorted_cams = sorted(
        camera_dets.keys(), key=lambda c: camera_dets[c][0].get("timestamp", "")
    )

    observation_nodes = []
    observation_segments = []
    inference_segments = []

    for cid in sorted_cams:
        cam_dets = camera_dets[cid]
        enter_det = cam_dets[0]
        exit_det = cam_dets[-1]
        cam_info = _camera_meta.get(cid, {})
        # tracklet_id 由 target_id + 摄像头确定性派生（原实现用 uuid4，无法复现）
        tracklet_id = f"TRK_{uuid.uuid5(uuid.NAMESPACE_URL, f'{target_id}|{cid}').hex[:8]}"

        observation_nodes.append({
            "camera_id": cid,
            "camera_name": cam_info.get("name", cid),
            "tracklet_id": tracklet_id,
            # 真实时间：该摄像头首次检测的时间戳
            "timestamp": _format_ts_full(enter_det.get("timestamp", "")),
            "latitude": cam_info.get("latitude"),
            "longitude": cam_info.get("longitude"),
            "keyframe_path": enter_det.get("keyframe_path"),
            "confidence": round(enter_det.get("confidence", 0.0), 3),
        })

        observation_segments.append({
            "tracklet_id": tracklet_id,
            "camera_id": cid,
            "camera_name": cam_info.get("name", cid),
            # 真实时间：该摄像头首末检测的时间戳
            "start_time": _format_ts(enter_det.get("timestamp", "")),
            "end_time": _format_ts(exit_det.get("timestamp", "")),
            # 真实方向取自摄像头元数据，无标注则为空串
            "direction": _camera_direction_map.get(cid, ""),
            # 进出画面的方位无真实标注，留空
            "entry_description": "",
            "exit_description": "",
        })

    # 构建跨摄像头推断段
    for i in range(len(sorted_cams) - 1):
        src_cid = sorted_cams[i]
        tgt_cid = sorted_cams[i + 1]
        src_info = _camera_meta.get(src_cid, {})
        tgt_info = _camera_meta.get(tgt_cid, {})
        inference_segments.append({
            "source_camera_id": src_cid,
            "target_camera_id": tgt_cid,
            "source_camera_name": src_info.get("name", src_cid),
            "target_camera_name": tgt_info.get("name", tgt_cid),
            "source_latitude": src_info.get("latitude"),
            "source_longitude": src_info.get("longitude"),
            "target_latitude": tgt_info.get("latitude"),
            "target_longitude": tgt_info.get("longitude"),
            # 推断段无真实依据（缺道路拓扑与行程速度模型），留空待评分子系统填充
            "confidence": None,
            "estimated_travel_time": None,
            "actual_travel_time": None,
            "route_description": f"{src_info.get('name', src_cid)} → {tgt_info.get('name', tgt_cid)}",
        })

    # 构建候选路径
    first_det = dets[0]
    candidate_paths = [{
        "path_id": f"PATH_{uuid.uuid5(uuid.NAMESPACE_URL, f'{target_id}|path').hex[:6]}",
        "road_segments": [f"SEG_{cid}" for cid in sorted_cams],
        # 可信度与预估耗时无真实依据，置 0 待评分子系统填充（保持数值型，
        # 前端报表直接按 {:.0%}/{:.0f} 格式化，不能用 null）
        "confidence": 0.0,
        "distance_meters": _path_distance_meters(sorted_cams),
        "estimated_time": 0.0,
        "description": " → ".join(
            _camera_meta.get(cid, {}).get("name", cid) for cid in sorted_cams
        ),
    }]

    # 获取首个检测的属性信息
    target_attrs = first_det.get("attributes", {})
    color = target_attrs.get("color", target_attrs.get("color_en", "未知"))
    vtype = target_attrs.get("vehicle_type", target_attrs.get("vehicle_type_en", "未知"))

    result = {
        "query_id": query_id,
        "target_instance": {
            "instance_id": instance_id,
            "target_id": target_id,
            "camera_id": first_det.get("camera_id", ""),
            # 真实时间：首个检测的时间戳
            "timestamp": _format_ts_full(first_det.get("timestamp", "")),
            "target_type": first_det.get("target_type", "vehicle"),
            "attributes": {"颜色": color, "类型": vtype},
            "plate_number": None,
            "quality_score": round(first_det.get("confidence", 0.0), 3),
            "keyframe_path": first_det.get("keyframe_path"),
        },
        "camera_sequence": sorted_cams,
        "observation_nodes": observation_nodes,
        "observation_segments": observation_segments,
        "inference_segments": inference_segments,
        "candidate_paths": candidate_paths,
        # 综合置信度需要外观/时空多维评分，当前无实现，置 0 待评分子系统填充
        "overall_confidence": 0.0,
        "evidence": {
            # 以下各项均需真实评分依据（车牌一致性、CLIP 外观相似度、属性一致性、
            # 时空可行性），当前无实现，一律置 0，不使用随机数
            "plate_consistency": 0.0,
            "appearance_similarity": 0.0,
            "attribute_consistency": 0.0,
            "temporal_feasibility": 0.0,
            "spatial_feasibility": 0.0,
            "source": "cityflow_results",
            "target_id": target_id,
            "detection_count": len(dets),
            "camera_count": len(sorted_cams),
        },
    }

    _backtrack_results[query_id] = result
    return result


# ============================================================
# Fallback mock（仅在数据文件不存在时使用）
# ============================================================

_FALLBACK_CAMERAS = [
    {"camera_id": "c001", "name": "Scene01-Camera01", "scene": "S01", "latitude": 42.526, "longitude": -90.7236},
    {"camera_id": "c010", "name": "Scene03-Camera01", "scene": "S03", "latitude": 42.4988, "longitude": -90.6874},
    {"camera_id": "c016", "name": "Scene04-Camera01", "scene": "S04", "latitude": 42.4978, "longitude": -90.6874},
]


def _generate_fallback_mock(instance_id: str) -> Dict[str, Any]:
    """
    演示用回溯结果（**非真实回溯**）

    仅在 output/cityflow_results.json 完全不存在、系统无任何真实数据可用时调用，
    内容为硬编码摄像头 + 随机数生成的演示轨迹，evidence.source 标记为 "fallback_mock"。
    数据文件存在但目标查不到时，接口返回 404，不落到这里。
    """
    base_time = datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)
    query_id = f"BT_{uuid.uuid4().hex[:8]}"

    observation_nodes = []
    observation_segments = []
    inference_segments = []

    for i, cam in enumerate(_FALLBACK_CAMERAS):
        enter_t = base_time + timedelta(seconds=i * 180)
        exit_t = enter_t + timedelta(seconds=random.randint(4, 12))
        node_ts = enter_t + timedelta(seconds=2)

        observation_nodes.append({
            "camera_id": cam["camera_id"],
            "camera_name": cam["name"],
            "tracklet_id": f"TRK_{uuid.uuid4().hex[:8]}",
            "timestamp": node_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": cam["latitude"],
            "longitude": cam["longitude"],
            "keyframe_path": None,
            "confidence": round(random.uniform(0.85, 0.99), 3),
        })
        observation_segments.append({
            "tracklet_id": f"TRK_{uuid.uuid4().hex[:8]}",
            "camera_id": cam["camera_id"],
            "camera_name": cam["name"],
            "start_time": enter_t.strftime("%H:%M:%S"),
            "end_time": exit_t.strftime("%H:%M:%S"),
            "entry_description": "从画面左侧进入",
            "exit_description": "从画面右侧离开",
            "direction": "由西向东" if cam.get("scene") == "S01" else "由北向南",
        })

    for i in range(len(_FALLBACK_CAMERAS) - 1):
        src = _FALLBACK_CAMERAS[i]
        tgt = _FALLBACK_CAMERAS[i + 1]
        inference_segments.append({
            "source_camera_id": src["camera_id"],
            "target_camera_id": tgt["camera_id"],
            "source_camera_name": src["name"],
            "target_camera_name": tgt["name"],
            "source_latitude": src["latitude"],
            "source_longitude": src["longitude"],
            "target_latitude": tgt["latitude"],
            "target_longitude": tgt["longitude"],
            "confidence": round(random.uniform(0.65, 0.92), 3),
            "estimated_travel_time": round(random.uniform(60, 200), 1),
            "actual_travel_time": round(random.uniform(80, 220), 1),
            "route_description": f"{src['name']} → {tgt['name']}",
        })

    candidate_paths = [{
        "path_id": f"PATH_{uuid.uuid4().hex[:6]}",
        "road_segments": ["SEG_001", "SEG_002"],
        "confidence": 0.72,
        "distance_meters": 770.0,
        "estimated_time": 95.0,
        "description": "Scene01-Camera01 → Scene03-Camera01 → Scene04-Camera01",
    }]

    result = {
        "query_id": query_id,
        "target_instance": {
            "instance_id": instance_id,
            "camera_id": _FALLBACK_CAMERAS[0]["camera_id"],
            "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_type": "vehicle",
            "attributes": {"颜色": "未知", "类型": "车辆"},
            "plate_number": None,
            "quality_score": 0.85,
        },
        "camera_sequence": [c["camera_id"] for c in _FALLBACK_CAMERAS],
        "observation_nodes": observation_nodes,
        "observation_segments": observation_segments,
        "inference_segments": inference_segments,
        "candidate_paths": candidate_paths,
        "overall_confidence": 0.78,
        "evidence": {
            "plate_consistency": 0.0,
            "appearance_similarity": 0.75,
            "attribute_consistency": 0.80,
            "temporal_feasibility": 0.79,
            "spatial_feasibility": 0.85,
            "source": "fallback_mock",
        },
    }
    _backtrack_results[query_id] = result
    return result


class BacktrackRequest(BaseModel):
    """回溯请求"""
    instance_id: Optional[str] = None   # 确认的目标实例 ID（缺省时从 session 的确认结果读取）
    query_id: Optional[str] = None      # 检索会话 ID，用于串联「检索→确认→回溯」
    max_upstream: int = 10              # 最大上游回溯深度
    max_downstream: int = 10            # 最大下游回溯深度


class BacktrackResponse(BaseModel):
    """回溯响应"""
    query_id: str
    camera_sequence: List[str]          # 摄像头序列
    observation_nodes: List[Dict[str, Any]]
    observation_segments: List[Dict[str, Any]]
    inference_segments: List[Dict[str, Any]]
    candidate_paths: List[Dict[str, Any]]
    overall_confidence: float


@router.post("/trace", response_model=BacktrackResponse)
async def backtrack_trajectory(request: BacktrackRequest):
    """
    轨迹回溯

    以确认目标为锚点，从 cityflow_results.json 的真实检测记录中聚合其
    经过的摄像头序列（观测段来自真实检测，推断段留空待评分子系统填充）。

    instance_id 缺省时从会话（query_id）中读取用户已确认的目标；
    数据文件存在但目标查不到时返回 404，不再回落到随机目标或演示数据。
    """
    # 确保数据已加载
    _load_cityflow_data()

    store = get_session_store()

    # 1. 确定要回溯的目标：请求体优先，其次取会话中已确认的实例
    instance_id = request.instance_id
    if not instance_id and request.query_id:
        session = store.get(request.query_id)
        if session:
            instance_id = session.get("confirmed_instance_id")
    if not instance_id:
        raise HTTPException(
            status_code=404,
            detail="未指定 instance_id，且会话中没有已确认的目标",
        )

    # 2. 定位真实目标；数据文件存在却找不到该目标 → 404
    target_id = _find_target_id(instance_id)
    if target_id is None:
        if _data_loaded:
            raise HTTPException(
                status_code=404,
                detail=f"未找到目标 {instance_id} 的真实轨迹记录",
            )
        # 数据文件完全不存在：退化为明确标注的演示数据
        logger.warning(f"真实数据不可用，返回演示数据: instance_id={instance_id}")
        result = _generate_fallback_mock(instance_id)
    else:
        result = _build_trajectory_from_target(instance_id, target_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"目标 {instance_id} 没有可用的检测记录",
            )
        logger.info(
            f"回溯成功: instance_id={instance_id}, "
            f"target_id={target_id}, "
            f"cameras={len(result.get('camera_sequence', []))}"
        )

    # 3. 写回会话状态（confirmed → backtracked）
    if request.query_id:
        try:
            store.set_backtracked(request.query_id, result)
        except SessionNotFoundError:
            logger.warning(f"回溯完成但会话不存在，跳过状态写回: {request.query_id}")

    return BacktrackResponse(
        query_id=result["query_id"],
        camera_sequence=result["camera_sequence"],
        observation_nodes=result["observation_nodes"],
        observation_segments=result["observation_segments"],
        inference_segments=result["inference_segments"],
        candidate_paths=result["candidate_paths"],
        overall_confidence=result["overall_confidence"],
    )


@router.get("/result/{query_id}")
async def get_backtrack_result(query_id: str):
    """
    获取回溯结果

    根据查询 ID 获取已完成的回溯结果。
    """
    result = _backtrack_results.get(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"回溯结果 {query_id} 不存在")
    return result


# ============================================================
# 跨镜轨迹回溯 API
# ============================================================

class TrajectoryRequest(BaseModel):
    """跨镜轨迹请求"""
    track_id: Optional[str] = None
    instance_id: Optional[str] = None


def _calc_duration_seconds(ts1: str, ts2: str) -> int:
    """计算两个时间戳之间的秒数差"""
    if not ts1 or not ts2:
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt1 = datetime.strptime(ts1, fmt)
            dt2 = datetime.strptime(ts2, fmt)
            return max(0, int((dt2 - dt1).total_seconds()))
        except (ValueError, TypeError):
            continue
    return 0


def _format_ts(ts_str: str) -> str:
    """将时间戳格式化为 HH:MM:SS 显示"""
    if not ts_str:
        return "--"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            continue
    return ts_str


def _format_ts_full(ts_str: str) -> str:
    """将时间戳格式化为完整显示"""
    if not ts_str:
        return "--"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
    return ts_str


@router.post("/trajectory")
async def get_full_trajectory(request: TrajectoryRequest):
    """
    获取目标的完整跨镜轨迹。

    根据 track_id 或 instance_id，找到同一 vehicle_id 在所有摄像头中的检测记录，
    按时间排序构建跨摄像头时间线。
    """
    _load_cityflow_data()

    if not _all_detections:
        raise HTTPException(status_code=500, detail="数据未加载")

    # 1. 确定 vehicle_id（target_id / track_id 两种格式统一由 src.common.ids 解析）
    source_track_id = request.track_id or ""
    target_vehicle_id = extract_vehicle_id(request.instance_id or "")
    if not target_vehicle_id:
        target_vehicle_id = extract_vehicle_id(source_track_id)

    if not target_vehicle_id:
        raise HTTPException(status_code=404, detail="无法解析 vehicle_id")

    # 2. 获取该 vehicle 在所有摄像头的 detections
    vehicle_dets = _vehicle_detections.get(target_vehicle_id, [])
    if not vehicle_dets:
        raise HTTPException(status_code=404, detail=f"未找到 vehicle {target_vehicle_id} 的检测记录")

    # 3. 按摄像头分组
    camera_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in vehicle_dets:
        cid = det.get("camera_id", "")
        if cid:
            camera_groups[cid].append(det)

    # 4. 每个摄像头内按 timestamp 排序
    for cid in camera_groups:
        camera_groups[cid].sort(key=lambda d: d.get("timestamp", ""))

    # 5. 构建摄像头序列（按首次出现时间排序）
    camera_sequence = []
    for cid, cam_dets in sorted(camera_groups.items(), key=lambda x: x[1][0].get("timestamp", "")):
        first_det = cam_dets[0]
        last_det = cam_dets[-1]
        arrival = first_det.get("timestamp", "")
        departure = last_det.get("timestamp", "")
        cam_info = _camera_meta.get(cid, {})

        frames = []
        for d in cam_dets:
            frames.append({
                "frame_id": d.get("frame_id", 0),
                "timestamp": _format_ts_full(d.get("timestamp", "")),
                "crop_path": d.get("crop_path", d.get("keyframe_path", "")),
                "confidence": d.get("confidence", 0.5),
            })

        camera_sequence.append({
            "camera_id": cid,
            "camera_name": cam_info.get("name", cid),
            "arrival_time": _format_ts_full(arrival),
            "departure_time": _format_ts_full(departure),
            "duration_seconds": _calc_duration_seconds(arrival, departure),
            "detection_count": len(cam_dets),
            "direction": _camera_direction_map.get(cid, ""),
            "frames": frames,
        })

    # 6. 计算总时长
    first_time = camera_sequence[0]["arrival_time"] if camera_sequence else ""
    last_time = camera_sequence[-1]["departure_time"] if camera_sequence else ""
    total_duration = _calc_duration_seconds(
        camera_sequence[0].get("arrival_time", "").replace("--", "") if camera_sequence else "",
        camera_sequence[-1].get("departure_time", "").replace("--", "") if camera_sequence else "",
    )

    # 7. 获取属性
    first_det = vehicle_dets[0]
    attrs_raw = first_det.get("attributes", {})
    # 本地颜色/车型映射（避免导入 frontend 模块）
    _color_cn_map = {0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色", 4: "红色", 5: "蓝色", 6: "白色", 7: "金色", 8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色"}
    _type_cn_map = {0: "轿车", 1: "SUV", 2: "面包车", 3: "两厢车", 4: "MPV", 5: "皮卡", 6: "公交车", 7: "卡车", 8: "旅行车", 9: "跑车", 10: "房车"}
    _en_color = {"yellow": "黄色", "orange": "橙色", "green": "绿色", "gray": "灰色", "red": "红色", "blue": "蓝色", "white": "白色", "golden": "金色", "brown": "棕色", "black": "黑色", "purple": "紫色", "pink": "粉色"}
    _en_type = {"sedan": "轿车", "suv": "SUV", "van": "面包车", "hatchback": "两厢车", "mpv": "MPV", "pickup": "皮卡", "bus": "公交车", "truck": "卡车", "estate": "旅行车", "sportscar": "跑车", "rv": "房车"}
    color_id = attrs_raw.get("color_id", -1)
    type_id = attrs_raw.get("type_id", -1)
    color_cn = (
        attrs_raw.get("颜色", "")
        or _color_cn_map.get(color_id, "")
        or _en_color.get(attrs_raw.get("color", "").lower(), "")
        or attrs_raw.get("color", "未知")
    )
    type_cn = (
        attrs_raw.get("车型", "")
        or _type_cn_map.get(type_id, "")
        or _en_type.get(attrs_raw.get("vehicle_type", "").lower(), "")
        or attrs_raw.get("vehicle_type", "未知")
    )

    # 8. 构建响应
    return {
        "success": True,
        "trajectory": {
            "track_id": source_track_id,
            "vehicle_id": target_vehicle_id,
            "first_appearance": first_time,
            "last_appearance": last_time,
            "total_duration_seconds": total_duration,
            "total_cameras": len(camera_sequence),
            "total_detections": sum(len(c["frames"]) for c in camera_sequence),
            "camera_sequence": camera_sequence,
            "attributes": {
                "颜色": color_cn,
                "车型": type_cn,
            },
        },
    }
