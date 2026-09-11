"""
api.routes.dashboard - 仪表盘 API

提供系统状态和统计接口:
- GET /api/v1/dashboard/stats     系统统计信息
- GET /api/v1/dashboard/cameras   摄像头列表
- GET /api/v1/dashboard/health    服务健康状态

数据来源: output/results.json + configs/camera_metadata.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("api.routes.dashboard")

router = APIRouter()

# 项目路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_RESULTS_JSON = _OUTPUT_DIR / "cityflow_results.json"
_CAMERA_CONFIG = _PROJECT_ROOT / "configs" / "camera_metadata.yaml"


def _load_results() -> Dict[str, Any]:
    """加载 results.json，不存在则返回空结构"""
    if not _RESULTS_JSON.exists():
        return {}
    try:
        with open(_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 results.json 失败: {e}")
        return {}


def _build_images_from_detections(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将扁平的 detections 列表转换为 images 嵌套结构，
    兼容旧版和新版 results.json 格式。
    """
    # 新版格式：扁平 detections 列表
    detections = data.get("detections", [])
    if detections:
        images_map: Dict[str, Dict[str, Any]] = {}
        for det in detections:
            img_path = det.get("image_path", det.get("detection_image_path", ""))
            frame_id = det.get("frame_id", 0)
            key = f"{img_path}_{frame_id}"
            if key not in images_map:
                images_map[key] = {
                    "image_file": img_path,
                    "detection_image": det.get("detection_image_path", img_path),
                    "camera_id": det.get("camera_id", "c001"),
                    "timestamp": det.get("timestamp", ""),
                    "detections": [],
                }
            images_map[key]["detections"].append(det)
        return list(images_map.values())

    # 旧版格式：已有 images 嵌套结构
    return data.get("images", [])


def _load_camera_metadata() -> List[Dict[str, Any]]:
    """加载摄像头元数据"""
    cameras = []
    try:
        import yaml
        if _CAMERA_CONFIG.exists():
            with open(_CAMERA_CONFIG, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                cam_list = data.get("cameras", data.get("camera_list", []))
                if isinstance(cam_list, list):
                    for cam in cam_list:
                        cameras.append({
                            "camera_id": cam.get("camera_id", cam.get("id", "")),
                            "name": cam.get("name", cam.get("camera_name", "")),
                            "latitude": cam.get("latitude", 31.3635),
                            "longitude": cam.get("longitude", 120.6180),
                            "status": cam.get("status", "online"),
                            "direction": cam.get("direction", 0),
                            "today_detections": cam.get("today_detections", 0),
                        })
    except Exception as e:
        logger.warning(f"加载摄像头配置失败: {e}")

    # 如果没有配置文件，从 results.json 推断
    if not cameras:
        data = _load_results()
        detections = data.get("detections", [])
        # 从 image_path 推断摄像头（每个文件对应一个摄像头）
        cam_ids = set()
        for det in detections:
            img_path = det.get("image_path", "")
            if img_path:
                # 取文件名前缀作为摄像头标识
                fname = Path(img_path).stem
                cam_ids.add(fname)
        if not cam_ids:
            cam_ids.add("c001")
        for cid in sorted(cam_ids):
            cameras.append({
                "camera_id": cid,
                "name": cid,
                "latitude": 31.3635,
                "longitude": 120.6180,
                "status": "online",
                "direction": 0,
                "today_detections": 0,
            })

    return cameras


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    获取系统统计信息

    从 output/results.json 读取真实检测统计数据。
    """
    data = _load_results()
    detections = data.get("detections", [])
    summary = data.get("summary", {})
    cameras = _load_camera_metadata()

    total_detections = len(detections)
    type_counts: Dict[str, int] = {"车辆": 0, "行人": 0, "非机动车": 0}
    type_map = {"vehicle": "车辆", "pedestrian": "行人", "non_motor_vehicle": "非机动车"}
    confidence_scores: List[float] = []

    for det in detections:
        t = det.get("target_type", "")
        if t in type_map:
            type_counts[type_map[t]] = type_counts.get(type_map[t], 0) + 1
        conf = det.get("confidence", 0)
        if conf > 0:
            confidence_scores.append(conf)

    # 置信度分布
    conf_dist = []
    for lo, hi, label in [
        (0.0, 0.2, "0.0-0.2"), (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"), (0.6, 0.8, "0.6-0.8"), (0.8, 1.01, "0.8-1.0"),
    ]:
        cnt = sum(1 for c in confidence_scores if lo <= c < hi)
        conf_dist.append({"range": label, "count": cnt})

    cam_count = len(cameras) or 1

    track_count = summary.get("total_tracks", len(data.get("tracks", [])))

    return {
        "camera_count": cam_count,
        "camera_online": cam_count,
        "instance_count": total_detections,
        "tracklet_count": track_count,
        "edge_count": 0,
        "today_search_count": 0,
        "today_backtrack_count": 0,
        "search_by_type": {k: v for k, v in type_counts.items() if v > 0},
        "search_frequency": [],
        "avg_observation_chain_length": 0.0,
        "confidence_distribution": conf_dist,
    }


@router.get("/cameras")
async def list_cameras() -> Dict[str, Any]:
    """
    获取摄像头列表

    从配置文件或检测结果中读取摄像头信息。
    """
    cameras = _load_camera_metadata()
    return {"cameras": cameras}


@router.get("/health")
async def health_status() -> Dict[str, Any]:
    """
    服务健康状态

    返回各组件的运行状态。
    """
    config = get_config()
    results_exist = _RESULTS_JSON.exists()
    return {
        "status": "running",
        "version": config.get("system.version", "1.0.0"),
        "device": config.get("system.device", "cuda"),
        "results_loaded": results_exist,
    }
