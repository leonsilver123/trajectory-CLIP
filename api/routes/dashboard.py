"""
api.routes.dashboard - 仪表盘 API

提供系统状态和统计接口:
- GET /api/v1/dashboard/stats     系统统计信息
- GET /api/v1/dashboard/cameras   摄像头列表
- GET /api/v1/dashboard/health    服务健康状态

数据来源: src.storage.datastore（优先 Parquet + SQLite，缺失时回退 JSON 直读）
          + configs/camera_metadata.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from src.common.config import get_config
from src.common.logger import get_logger
from src.storage.datastore import data_source, has_data, load_results

logger = get_logger("api.routes.dashboard")

router = APIRouter()

# 项目路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CAMERA_CONFIG = _PROJECT_ROOT / "configs" / "camera_metadata.yaml"
# 权威逐摄像头元数据（configs/default.yaml 的 camera.metadata_file 指定的就是这个；
# 另请注意 camera_metadata.yaml 只有 scenes 段、没有 cameras 段，单独用它拿不到 GPS）
_CAMERA_CONFIG_CITYFLOW = _PROJECT_ROOT / "configs" / "cityflow_camera_metadata.yaml"


def _load_results() -> Dict[str, Any]:
    """加载检测结果，不存在则返回空结构（统一走 src.storage.datastore）"""
    return load_results() or {}


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


def _load_camera_metadata(detections: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """
    加载摄像头元数据

    数据源优先级（**全部为真实数据，不编造**）：
      1. `configs/cityflow_camera_metadata.yaml` —— 逐摄像头真实元数据
         （46 个摄像头，含 GPS/方向/场景/分辨率）。这是 `configs/default.yaml`
         的 `camera.metadata_file` 指定的权威来源，`api/routes/search.py` 也用它。
      2. `configs/camera_metadata.yaml` —— 精简索引，**只有 scenes 段没有 cameras 段**，
         故按 `scenes[].cameras` 展开出摄像头 ID（无 GPS）。
      3. 从检测结果里取真实的 `camera_id` 字段（注意：**不是** image_path 的文件名，
         那个是帧号 `000001` 之类，曾被误当摄像头 ID 导致虚报 2384 路）。

    诚实性约定：数据集中**没有**"在线状态"这一事实，故 `status` 一律为 None，
    由前端显示为"未知"；`today_detections` 回填**真实**的检测计数。
    """
    cameras: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append(cam: Dict[str, Any], det_count: Dict[str, int], scene_gps: Dict[str, Any]) -> None:
        cid = (cam.get("camera_id") or cam.get("id") or "").strip()
        if not cid or cid in seen:
            return
        seen.add(cid)
        cameras.append({
            "camera_id": cid,
            "name": cam.get("name") or cam.get("camera_name") or cid,
            # 数据集没有逐摄像头 GPS（AICity22 ReadMe §12），per-camera 坐标为 null，
            # 地图应使用 scene_gps_center（场景近似中心，真实值，指向美国 Iowa Dubuque）。
            "latitude": cam.get("latitude"),
            "longitude": cam.get("longitude"),
            "scene_gps_center": scene_gps.get(cam.get("scene")),
            # 无真实来源的字段一律 None，绝不填默认值冒充真实数据
            "status": cam.get("status"),
            "direction": cam.get("direction"),
            "scene": cam.get("scene"),
            "today_detections": det_count.get(cid, 0),
        })

    try:
        import yaml
    except Exception as e:  # pragma: no cover
        logger.warning(f"yaml 不可用，无法读取摄像头配置: {e}")
        yaml = None

    # 每个摄像头的真实检测计数（用于 today_detections）
    det_count: Dict[str, int] = {}
    for det in detections or []:
        cid = det.get("camera_id")
        if cid:
            det_count[cid] = det_count.get(cid, 0) + 1

    if yaml is not None:
        # 1) 权威来源：逐摄像头真实元数据
        for cfg in (_CAMERA_CONFIG_CITYFLOW, _CAMERA_CONFIG):
            if not cfg.exists():
                continue
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"加载摄像头配置失败 {cfg.name}: {e}")
                continue
            if not isinstance(data, dict):
                continue

            scene_gps = data.get("scene_gps_center", {}) or {}
            cam_list = data.get("cameras", data.get("camera_list", []))
            if isinstance(cam_list, list):
                for cam in cam_list:
                    if isinstance(cam, dict):
                        _append(cam, det_count, scene_gps)

            # 精简索引格式：只有 scenes，按 scenes[].cameras 展开
            if not cam_list:
                for scene in data.get("scenes", []) or []:
                    if not isinstance(scene, dict):
                        continue
                    for cid in scene.get("cameras", []) or []:
                        _append({"camera_id": cid, "scene": scene.get("scene_id")}, det_count, scene_gps)

            if cameras:
                break

    # 3) 兜底：从检测里取真实 camera_id 字段
    if not cameras:
        for cid in sorted(det_count):
            _append({"camera_id": cid}, det_count, {})

    return cameras


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    获取系统统计信息

    从 output/results.json 读取真实检测统计数据。
    """
    data = _load_results()
    detections = data.get("detections", [])
    cameras = _load_camera_metadata(detections)

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

    # 单摄轨迹数：按 det_to_track_map 的值域数（= builder 生成 tracklet 的口径，实测 926）。
    # **不能**用 len(tracks[])（=2070）：缺陷 E3 —— tracks[] 混装了 CF3_TRACK_*/BL_TRACK_*/
    # CF2S_TRACK_* 三个来源，其中 1700 条没有任何对应检测，会虚报近一倍。
    # **也不要**信 data["summary"]：实测它是陈旧的（total_tracks=2070、
    # total_detections=69229，而 detections 数组实际只有 68349 条），已不再作为真源。
    track_count = len({t for t in (data.get("det_to_track_map") or {}).values() if t})

    return {
        "camera_count": cam_count,
        # 数据集里**没有**"在线状态"这一事实（离线视频，非实时流），
        # 原先直接回填 cam_count 等于声称"全部在线"，属编造。无真实来源即 None。
        "camera_online": None,
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
    cameras = _load_camera_metadata(_load_results().get("detections", []))
    return {"cameras": cameras}


@router.get("/health")
async def health_status() -> Dict[str, Any]:
    """
    服务健康状态

    返回各组件的运行状态。
    """
    config = get_config()
    # 数据可用性：datastore 或城市流 JSON 任一在位即视为已加载（两者数据等价）
    results_exist = has_data()
    return {
        "status": "running",
        "version": config.get("system.version", "1.0.0"),
        "device": config.get("system.device", "cuda"),
        "results_loaded": results_exist,
        # 新增（不改动既有字段）：当前生效的数据来源 datastore / json / none
        "data_source": data_source(),
    }
