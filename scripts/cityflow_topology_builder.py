"""
scripts/cityflow_topology_builder.py - CityFlow 摄像头拓扑自动生成脚本

从 CityFlow 数据集原始文件中解析摄像头信息并生成 cityflow_camera_metadata.yaml。

输入文件 (CityFlow 数据集目录结构):
    - list_cam.txt: 摄像头列表 (每行一个摄像头 ID)
    - cam_loc/*.png: 每个摄像头的位置标定图
    - cam_timestamp/*.txt: 每个摄像头的时间戳信息

输出:
    - configs/cityflow_camera_metadata.yaml

使用方式:
    python scripts/cityflow_topology_builder.py --cityflow_root /path/to/cityflow
    python scripts/cityflow_topology_builder.py --cityflow_root /path/to/cityflow --output configs/cityflow_camera_metadata.yaml

如果没有 CityFlow 原始数据，脚本会使用内置的已知场景信息生成配置。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.common.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 已知场景先验信息 (当无原始数据时使用)
# ============================================================

SCENE_INFO = {
    "S01": {
        "cameras": [f"c{i:03d}" for i in range(1, 6)],       # c001-c005
        "type": "intersection",
        "split": "train",
        "gps_center": (42.525678, -90.723601),
        "description": "5 cameras at single intersection",
    },
    "S02": {
        "cameras": [f"c{i:03d}" for i in range(6, 10)],      # c006-c009
        "type": "intersection",
        "split": "validation",
        "gps_center": (42.491916, -90.723723),
        "description": "4 cameras at single intersection",
    },
    "S03": {
        "cameras": [f"c{i:03d}" for i in range(10, 16)],     # c010-c015
        "type": "road_segment",
        "split": "train",
        "gps_center": (42.498780, -90.686393),
        "description": "6 cameras along road segment",
    },
    "S04": {
        "cameras": [f"c{i:03d}" for i in range(16, 41)],     # c016-c040
        "type": "city_level",
        "split": "train",
        "gps_center": (42.498780, -90.686393),
        "description": "25 cameras city-level coverage (5x5 grid)",
    },
    "S05": {
        "cameras": [f"c{i:03d}" for i in [10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36]],
        "type": "city_level",
        "split": "validation",
        "gps_center": (42.498780, -90.686393),
        "description": "Reuses S04 cameras for validation",
    },
    "S06": {
        "cameras": [f"c{i:03d}" for i in range(41, 47)],     # c041-c046
        "type": "intersection",
        "split": "test",
        "gps_center": (42.492448, -90.723343),
        "description": "6 cameras at test intersection",
    },
}

# 场景间跨场景连接 (基于 GPS 距离)
CROSS_SCENE_CONNECTIONS = [
    ("S01", "S02"),  # 距离较近 (~3.7km)
    ("S03", "S04"),  # 共享区域
    ("S02", "S03"),  # S02→S03 桥接 (c009↔c010)
    ("S02", "S06"),  # S02→S06 桥接 (c006↔c041, 距离近)
]

# 方向循环分配
DIRECTION_CYCLE = [0.0, 90.0, 180.0, 270.0]
LANE_DIRECTIONS = ["northbound", "eastbound", "southbound", "westbound"]

# S04 网格布局参数
S04_GRID_ROWS = 5
S04_GRID_COLS = 5
S04_GRID_SPACING_LAT = 0.0005  # ~55m per step
S04_GRID_SPACING_LON = 0.0005  # ~40m per step


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个 GPS 坐标之间的 Haversine 距离 (米)"""
    R = 6371000.0  # 地球半径 (米)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_list_cam(cityflow_root: str) -> List[str]:
    """
    解析 list_cam.txt 获取摄像头 ID 列表

    Args:
        cityflow_root: CityFlow 数据集根目录

    Returns:
        摄像头 ID 列表
    """
    list_cam_path = os.path.join(cityflow_root, "list_cam.txt")
    if not os.path.exists(list_cam_path):
        logger.warning(f"list_cam.txt 不存在: {list_cam_path}")
        return []

    cameras = []
    with open(list_cam_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cameras.append(line)
    logger.info(f"从 list_cam.txt 读取到 {len(cameras)} 个摄像头")
    return cameras


def parse_cam_timestamps(cityflow_root: str, camera_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    解析 cam_timestamp/*.txt 获取每个摄像头的时间信息

    Args:
        cityflow_root: CityFlow 数据集根目录
        camera_ids: 摄像头 ID 列表

    Returns:
        {camera_id: {"start_frame": int, "end_frame": int, "fps": float, "num_frames": int}}
    """
    ts_dir = os.path.join(cityflow_root, "cam_timestamp")
    result = {}

    if not os.path.isdir(ts_dir):
        logger.warning(f"cam_timestamp 目录不存在: {ts_dir}")
        return result

    for cam_id in camera_ids:
        ts_file = os.path.join(ts_dir, f"{cam_id}.txt")
        if not os.path.exists(ts_file):
            continue

        try:
            with open(ts_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) >= 2:
                start_frame = int(lines[0].strip())
                end_frame = int(lines[-1].strip())
                result[cam_id] = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_frames": end_frame - start_frame + 1,
                    "fps": 10.0,  # CityFlow 默认 10fps
                }
        except (ValueError, IOError) as e:
            logger.warning(f"解析 {ts_file} 失败: {e}")

    logger.info(f"从 cam_timestamp 解析到 {len(result)} 个摄像头的时间信息")
    return result


def parse_cam_locations(cityflow_root: str, camera_ids: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    解析 cam_loc/*.png 获取摄像头位置 (如果有 GPS 标注)

    注: CityFlow 的 cam_loc 通常是图片文件，需要额外标定文件才能获取 GPS。
        这里尝试读取同名的 .txt 标定文件 (如果存在)。

    Args:
        cityflow_root: CityFlow 数据集根目录
        camera_ids: 摄像头 ID 列表

    Returns:
        {camera_id: (latitude, longitude)}
    """
    loc_dir = os.path.join(cityflow_root, "cam_loc")
    result = {}

    if not os.path.isdir(loc_dir):
        logger.warning(f"cam_loc 目录不存在: {loc_dir}")
        return result

    for cam_id in camera_ids:
        # 尝试读取同名 txt 标定文件
        loc_txt = os.path.join(loc_dir, f"{cam_id}.txt")
        if os.path.exists(loc_txt):
            try:
                with open(loc_txt, "r", encoding="utf-8") as f:
                    parts = f.read().strip().split()
                if len(parts) >= 2:
                    lat, lon = float(parts[0]), float(parts[1])
                    result[cam_id] = (lat, lon)
            except (ValueError, IOError):
                pass

    logger.info(f"从 cam_loc 解析到 {len(result)} 个摄像头的位置信息")
    return result


def infer_scene_for_camera(cam_id: str) -> Optional[str]:
    """根据摄像头 ID 推断所属场景"""
    num = int(cam_id[1:])  # c001 -> 1
    for scene_id, info in SCENE_INFO.items():
        if scene_id == "S05":
            continue  # S05 复用 S04，跳过
        if cam_id in info["cameras"]:
            return scene_id
    return None


def generate_gps_for_scene(
    scene_id: str,
    cameras: List[str],
    scene_type: str,
    gps_center: Tuple[float, float],
) -> Dict[str, Tuple[float, float]]:
    """
    为场景内的摄像头生成近似 GPS 坐标

    Args:
        scene_id: 场景 ID
        cameras: 摄像头 ID 列表
        scene_type: 场景类型 (intersection/road_segment/city_level)
        gps_center: 场景中心 GPS

    Returns:
        {camera_id: (latitude, longitude)}
    """
    center_lat, center_lon = gps_center
    result = {}

    if scene_type == "intersection":
        # 交叉路口: 摄像头围绕中心小范围分布 (~50m)
        n = len(cameras)
        for i, cam_id in enumerate(cameras):
            angle = 2 * math.pi * i / n
            offset_lat = 0.0003 * math.cos(angle)
            offset_lon = 0.0004 * math.sin(angle)
            result[cam_id] = (
                round(center_lat + offset_lat, 4),
                round(center_lon + offset_lon, 4),
            )

    elif scene_type == "road_segment":
        # 路段: 沿东西方向线性排列
        n = len(cameras)
        start_lon = center_lon - 0.001
        step_lon = 0.0004
        for i, cam_id in enumerate(cameras):
            result[cam_id] = (
                round(center_lat + (i % 2) * 0.0001, 4),  # 轻微南北偏移
                round(start_lon + i * step_lon, 4),
            )

    elif scene_type == "city_level":
        # 城市级: 5x5 网格布局
        n = len(cameras)
        for idx, cam_id in enumerate(cameras):
            row = idx // S04_GRID_COLS
            col = idx % S04_GRID_COLS
            lat = center_lat + (row - S04_GRID_ROWS // 2) * S04_GRID_SPACING_LAT
            lon = center_lon + (col - S04_GRID_COLS // 2) * S04_GRID_SPACING_LON
            result[cam_id] = (round(lat, 4), round(lon, 4))

    return result


def generate_adjacency(
    scene_cameras: Dict[str, List[str]],
    scene_types: Dict[str, str],
    cross_scene_connections: List[Tuple[str, str]],
    gps: Dict[str, Tuple[float, float]],
) -> Dict[str, List[str]]:
    """
    生成摄像头邻接关系

    规则:
        - intersection: 全连接
        - road_segment: 串联
        - city_level: 网格邻接 (上下左右)
        - 跨场景连接: 基于场景对

    Args:
        scene_cameras: {scene_id: [camera_ids]}
        scene_types: {scene_id: type}
        cross_scene_connections: [(scene_a, scene_b), ...]
        gps: {camera_id: (lat, lon)}

    Returns:
        邻接表 {camera_id: [neighbor_ids]}
    """
    adjacency: Dict[str, List[str]] = {}

    for scene_id, cam_list in scene_cameras.items():
        scene_type = scene_types.get(scene_id, "intersection")

        if scene_type == "intersection":
            # 全连接
            for cam_id in cam_list:
                neighbors = [c for c in cam_list if c != cam_id]
                adjacency[cam_id] = neighbors

        elif scene_type == "road_segment":
            # 串联: c010→c011→c012→...
            for i, cam_id in enumerate(cam_list):
                neighbors = []
                if i > 0:
                    neighbors.append(cam_list[i - 1])
                if i < len(cam_list) - 1:
                    neighbors.append(cam_list[i + 1])
                adjacency[cam_id] = neighbors

        elif scene_type == "city_level":
            # 5x5 网格邻接
            n = len(cam_list)
            rows = S04_GRID_ROWS
            cols = S04_GRID_COLS
            for idx, cam_id in enumerate(cam_list):
                row = idx // cols
                col = idx % cols
                neighbors = []
                # 上
                if row > 0:
                    neighbors.append(cam_list[(row - 1) * cols + col])
                # 下
                if row < rows - 1:
                    neighbors.append(cam_list[(row + 1) * cols + col])
                # 左
                if col > 0:
                    neighbors.append(cam_list[row * cols + (col - 1)])
                # 右
                if col < cols - 1:
                    neighbors.append(cam_list[row * cols + (col + 1)])
                adjacency[cam_id] = neighbors

    # 跨场景连接
    for scene_a, scene_b in cross_scene_connections:
        cams_a = scene_cameras.get(scene_a, [])
        cams_b = scene_cameras.get(scene_b, [])
        if not cams_a or not cams_b:
            continue

        if scene_types.get(scene_a) == "road_segment" and scene_types.get(scene_b) == "city_level":
            # S03→S04: 路段摄像头按顺序连接到网格边缘
            for i, cam_a in enumerate(cams_a):
                if cam_a not in adjacency:
                    adjacency[cam_a] = []
                # 连接到 S04 网格中对应的摄像头
                if i < len(cams_b):
                    cam_b = cams_b[i]
                    if cam_b not in adjacency[cam_a]:
                        adjacency[cam_a].append(cam_b)
                    if cam_a not in adjacency.get(cam_b, []):
                        adjacency.setdefault(cam_b, []).append(cam_a)
        else:
            # 通用跨场景连接: 距离最近的摄像头对互相连接
            # 同时确保双向连接
            if gps:
                # 找最近的 N 对连接 (至少 1 对)
                pairs = []
                for ca in cams_a:
                    for cb in cams_b:
                        if ca in gps and cb in gps:
                            d = haversine_distance(
                                gps[ca][0], gps[ca][1],
                                gps[cb][0], gps[cb][1],
                            )
                            pairs.append((d, ca, cb))
                pairs.sort()
                # 添加最近的连接 (最多 3 对)
                added = 0
                for d, ca, cb in pairs:
                    if added >= 3:
                        break
                    if cb not in adjacency.get(ca, []):
                        adjacency.setdefault(ca, []).append(cb)
                    if ca not in adjacency.get(cb, []):
                        adjacency.setdefault(cb, []).append(ca)
                    added += 1

    # 排序邻接表
    for cam_id in adjacency:
        adjacency[cam_id] = sorted(set(adjacency[cam_id]))

    return adjacency


def generate_road_segments(
    scene_cameras: Dict[str, List[str]],
    scene_types: Dict[str, str],
    gps: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """生成路段定义"""
    segments = []

    for scene_id, cam_list in scene_cameras.items():
        scene_type = scene_types.get(scene_id, "intersection")

        if scene_type == "intersection":
            # 交叉路口: 每对对角摄像头一个路段
            seg_id = f"SEG_{scene_id}_001"
            segments.append({
                "segment_id": seg_id,
                "name": f"{scene_id} Intersection Segment 1",
                "start_camera_ids": [cam_list[0]],
                "end_camera_ids": [cam_list[1]] if len(cam_list) > 1 else [],
                "distance_meters": 50.0,
                "direction": "east-west",
            })

        elif scene_type == "road_segment":
            # 路段: 每两个相邻摄像头一个路段
            for i in range(0, len(cam_list) - 1, 2):
                seg_id = f"SEG_{scene_id}_{(i // 2) + 1:03d}"
                start_cam = cam_list[i]
                end_cam = cam_list[i + 1] if i + 1 < len(cam_list) else cam_list[i]
                dist = 200.0
                if gps and start_cam in gps and end_cam in gps:
                    dist = haversine_distance(
                        gps[start_cam][0], gps[start_cam][1],
                        gps[end_cam][0], gps[end_cam][1],
                    )
                segments.append({
                    "segment_id": seg_id,
                    "name": f"{scene_id} Road Segment {(i // 2) + 1}",
                    "start_camera_ids": [start_cam],
                    "end_camera_ids": [end_cam],
                    "distance_meters": round(dist, 1),
                    "direction": "east-west",
                })

        elif scene_type == "city_level":
            # 城市级: 行路段 + 列路段
            rows = S04_GRID_ROWS
            cols = S04_GRID_COLS
            # 行路段
            for r in range(rows):
                row_cams = cam_list[r * cols:(r + 1) * cols]
                seg_id = f"SEG_{scene_id}_R{r}"
                dist = 800.0
                if gps and len(row_cams) >= 2:
                    dist = haversine_distance(
                        gps[row_cams[0]][0], gps[row_cams[0]][1],
                        gps[row_cams[-1]][0], gps[row_cams[-1]][1],
                    )
                segments.append({
                    "segment_id": seg_id,
                    "name": f"{scene_id} Grid Row {r}",
                    "start_camera_ids": [row_cams[0]],
                    "end_camera_ids": [row_cams[-1]],
                    "distance_meters": round(dist, 1),
                    "direction": "east-west",
                })
            # 列路段
            for c in range(cols):
                col_cams = [cam_list[r * cols + c] for r in range(rows)]
                seg_id = f"SEG_{scene_id}_C{c}"
                dist = 800.0
                if gps and len(col_cams) >= 2:
                    dist = haversine_distance(
                        gps[col_cams[0]][0], gps[col_cams[0]][1],
                        gps[col_cams[-1]][0], gps[col_cams[-1]][1],
                    )
                segments.append({
                    "segment_id": seg_id,
                    "name": f"{scene_id} Grid Col {c}",
                    "start_camera_ids": [col_cams[0]],
                    "end_camera_ids": [col_cams[-1]],
                    "distance_meters": round(dist, 1),
                    "direction": "north-south",
                })

    return segments


def build_camera_metadata(
    cam_id: str,
    scene_id: str,
    scene_type: str,
    gps: Dict[str, Tuple[float, float]],
    cam_index_in_scene: int,
    timestamp_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """为单个摄像机构建元数据字典"""
    lat, lon = gps.get(cam_id, (0.0, 0.0))
    direction = DIRECTION_CYCLE[cam_index_in_scene % len(DIRECTION_CYCLE)]
    lane_dir = LANE_DIRECTIONS[cam_index_in_scene % len(LANE_DIRECTIONS)]

    # 根据场景类型调整路段 ID
    if scene_type == "intersection":
        seg_id = f"SEG_{scene_id}_001"
    elif scene_type == "road_segment":
        seg_idx = cam_index_in_scene // 2
        seg_id = f"SEG_{scene_id}_{seg_idx + 1:03d}"
    elif scene_type == "city_level":
        row = cam_index_in_scene // S04_GRID_COLS
        seg_id = f"SEG_{scene_id}_R{row}"
    else:
        seg_id = f"SEG_{scene_id}_001"

    meta = {
        "camera_id": cam_id,
        "name": f"Scene{scene_id[1:]}-Camera{cam_index_in_scene + 1:02d}",
        "latitude": lat,
        "longitude": lon,
        "direction": direction,
        "covered_road_segment": seg_id,
        "lane_direction": lane_dir,
        "scene": scene_id,
        "fps": 10,
        "resolution": "1920x1080",
    }

    if timestamp_info and cam_id in timestamp_info:
        meta["start_frame"] = timestamp_info[cam_id]["start_frame"]
        meta["end_frame"] = timestamp_info[cam_id]["end_frame"]
        meta["num_frames"] = timestamp_info[cam_id]["num_frames"]

    return meta


def generate_full_config(
    cityflow_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成完整的 CityFlow 摄像头元数据配置

    Args:
        cityflow_root: CityFlow 数据集根目录 (可选)

    Returns:
        完整的 YAML 配置字典
    """
    gps_data: Dict[str, Tuple[float, float]] = {}
    timestamp_info: Dict[str, Dict] = {}
    all_camera_ids: List[str] = []

    # 尝试从原始数据解析
    if cityflow_root and os.path.isdir(cityflow_root):
        logger.info(f"从 CityFlow 数据集加载: {cityflow_root}")
        all_camera_ids = parse_list_cam(cityflow_root)
        if all_camera_ids:
            timestamp_info = parse_cam_timestamps(cityflow_root, all_camera_ids)
            gps_data = parse_cam_locations(cityflow_root, all_camera_ids)

    # 如果没有从原始数据获取到摄像头列表，使用内置信息
    if not all_camera_ids:
        logger.info("使用内置场景信息生成配置")
        for scene_id in ["S01", "S02", "S03", "S04", "S06"]:
            all_camera_ids.extend(SCENE_INFO[scene_id]["cameras"])
        # 去重 (S05 复用 S04)
        all_camera_ids = sorted(set(all_camera_ids), key=lambda x: int(x[1:]))

    # 为没有 GPS 数据的摄像头生成近似坐标
    for scene_id in ["S01", "S02", "S03", "S04", "S06"]:
        info = SCENE_INFO[scene_id]
        scene_gps = generate_gps_for_scene(
            scene_id, info["cameras"], info["type"], info["gps_center"]
        )
        for cam_id, (lat, lon) in scene_gps.items():
            if cam_id not in gps_data:
                gps_data[cam_id] = (lat, lon)

    # 构建场景映射
    scene_cameras: Dict[str, List[str]] = {}
    scene_types: Dict[str, str] = {}
    cam_to_scene: Dict[str, str] = {}

    for scene_id in ["S01", "S02", "S03", "S04", "S06"]:
        info = SCENE_INFO[scene_id]
        scene_cameras[scene_id] = info["cameras"]
        scene_types[scene_id] = info["type"]
        for cam_id in info["cameras"]:
            cam_to_scene[cam_id] = scene_id

    # 生成邻接关系
    adjacency = generate_adjacency(
        scene_cameras, scene_types, CROSS_SCENE_CONNECTIONS, gps_data
    )

    # 生成路段定义
    road_segments = generate_road_segments(scene_cameras, scene_types, gps_data)

    # 生成摄像头元数据
    cameras = []
    for scene_id in ["S01", "S02", "S03", "S04", "S06"]:
        info = SCENE_INFO[scene_id]
        for idx, cam_id in enumerate(info["cameras"]):
            meta = build_camera_metadata(
                cam_id, scene_id, info["type"], gps_data, idx, timestamp_info
            )
            cameras.append(meta)

    config = {
        "cameras": cameras,
        "road_segments": road_segments,
        "adjacency": adjacency,
    }

    return config


def write_yaml(config: Dict[str, Any], output_path: str) -> None:
    """将配置写入 YAML 文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ============================================================\n")
        f.write("# CityFlow 摄像头元数据配置 (自动生成)\n")
        f.write(f"# 共 {len(config['cameras'])} 个摄像头\n")
        f.write("# ============================================================\n\n")

        # 使用 yaml.dump 但保持中文可读
        yaml_str = yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
        f.write(yaml_str)

    logger.info(f"配置已写入: {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """验证生成的配置是否完整和一致"""
    errors = []

    cameras = config.get("cameras", [])
    adjacency = config.get("adjacency", {})
    segments = config.get("road_segments", [])

    # 1. 检查摄像头数量
    cam_ids = {c["camera_id"] for c in cameras}
    logger.info(f"摄像头数量: {len(cam_ids)}")
    if len(cam_ids) != 46:
        errors.append(f"期望 46 个摄像头，实际 {len(cam_ids)} 个")

    # 2. 检查邻接表中的摄像头是否都在摄像头列表中
    for cam_id, neighbors in adjacency.items():
        if cam_id not in cam_ids:
            errors.append(f"邻接表中的 {cam_id} 不在摄像头列表中")
        for n in neighbors:
            if n not in cam_ids:
                errors.append(f"邻接表中的邻居 {n} (来自 {cam_id}) 不在摄像头列表中")

    # 3. 检查邻接对称性
    for cam_id, neighbors in adjacency.items():
        for n in neighbors:
            if cam_id not in adjacency.get(n, []):
                errors.append(f"邻接不对称: {cam_id}→{n} 但 {n} 不包含 {cam_id}")

    # 4. 检查路段引用的摄像头
    for seg in segments:
        for cam_id in seg.get("start_camera_ids", []) + seg.get("end_camera_ids", []):
            if cam_id not in cam_ids:
                errors.append(f"路段 {seg['segment_id']} 引用了不存在的摄像头 {cam_id}")

    # 5. 检查场景覆盖
    scenes = {c.get("scene") for c in cameras}
    logger.info(f"覆盖场景: {sorted(scenes)}")
    expected_scenes = {"S01", "S02", "S03", "S04", "S06"}
    if scenes != expected_scenes:
        errors.append(f"场景不匹配: 期望 {expected_scenes}，实际 {scenes}")

    # 6. 检查连通性 (BFS)
    if adjacency:
        start = next(iter(adjacency))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        unreachable = cam_ids - visited
        if unreachable:
            errors.append(f"不可达摄像头: {unreachable}")
        else:
            logger.info("拓扑连通性: 全部连通 ✓")

    if errors:
        logger.warning(f"发现 {len(errors)} 个问题:")
        for err in errors:
            logger.warning(f"  - {err}")
        return False
    else:
        logger.info("配置验证通过 ✓")
        return True


def print_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要"""
    cameras = config.get("cameras", [])
    adjacency = config.get("adjacency", {})
    segments = config.get("road_segments", [])

    print("\n" + "=" * 60)
    print("CityFlow 摄像头拓扑配置摘要")
    print("=" * 60)
    print(f"摄像头总数: {len(cameras)}")
    print(f"路段总数:   {len(segments)}")

    # 邻接边数
    total_edges = sum(len(v) for v in adjacency.values()) // 2
    print(f"邻接边数:   {total_edges}")

    # 按场景统计
    scene_counts = defaultdict(int)
    for cam in cameras:
        scene_counts[cam.get("scene", "unknown")] += 1
    print("\n场景分布:")
    for scene_id in sorted(scene_counts.keys()):
        print(f"  {scene_id}: {scene_counts[scene_id]} 个摄像头")

    # 跨场景连接
    cross_edges = 0
    for cam_id, neighbors in adjacency.items():
        cam_scene = None
        for c in cameras:
            if c["camera_id"] == cam_id:
                cam_scene = c.get("scene")
                break
        for n in neighbors:
            for c in cameras:
                if c["camera_id"] == n:
                    if c.get("scene") != cam_scene:
                        cross_edges += 1
                        break
    print(f"\n跨场景连接: {cross_edges // 2} 条")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CityFlow 摄像头拓扑自动生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用内置场景信息生成 (无需 CityFlow 原始数据)
  python scripts/cityflow_topology_builder.py

  # 从 CityFlow 数据集生成
  python scripts/cityflow_topology_builder.py --cityflow_root /path/to/cityflow

  # 指定输出路径
  python scripts/cityflow_topology_builder.py --output configs/my_config.yaml
        """,
    )
    parser.add_argument(
        "--cityflow_root",
        type=str,
        default=None,
        help="CityFlow 数据集根目录 (包含 list_cam.txt, cam_loc/, cam_timestamp/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/cityflow_camera_metadata.yaml",
        help="输出 YAML 文件路径 (默认: configs/cityflow_camera_metadata.yaml)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证现有配置，不生成新文件",
    )

    args = parser.parse_args()

    if args.validate_only:
        # 仅验证模式
        output_path = args.output
        if not os.path.exists(output_path):
            logger.error(f"文件不存在: {output_path}")
            sys.exit(1)
        with open(output_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        valid = validate_config(config)
        print_summary(config)
        sys.exit(0 if valid else 1)

    # 生成配置
    config = generate_full_config(cityflow_root=args.cityflow_root)

    # 验证
    valid = validate_config(config)

    # 写入文件
    write_yaml(config, args.output)

    # 打印摘要
    print_summary(config)

    if not valid:
        logger.warning("配置存在验证问题，但已写入文件。请检查上述错误。")
        sys.exit(1)
    else:
        logger.info(f"配置生成并验证通过: {args.output}")


if __name__ == "__main__":
    main()
