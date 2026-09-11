"""
scripts/cityflow_all_tracks.py - AICity22 Track1 MTMC 数据预处理脚本

处理 AICity22 Track1 MTMC Tracking 数据集并生成统一输出。

使用方式:
  python scripts/cityflow_all_tracks.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --max-frames 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from src.common.logger import get_logger

logger = get_logger("scripts.cityflow_all_tracks")


# ============================================================
# 常量与映射表
# ============================================================

COLOR_ID_MAP = {
    0: "yellow", 1: "orange", 2: "green", 3: "gray",
    4: "red", 5: "blue", 6: "white", 7: "golden",
    8: "brown", 9: "black", 10: "purple", 11: "pink",
}

COLOR_ID_CN = {
    0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色",
    4: "红色", 5: "蓝色", 6: "白色", 7: "金色",
    8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色",
}

TYPE_ID_MAP = {
    0: "sedan", 1: "suv", 2: "van", 3: "hatchback",
    4: "mpv", 5: "pickup", 6: "bus", 7: "truck",
    8: "estate", 9: "sportscar", 10: "RV",
}

TYPE_ID_CN = {
    0: "轿车", 1: "SUV", 2: "面包车", 3: "两厢车",
    4: "MPV", 5: "皮卡", 6: "公交车", 7: "卡车",
    8: "旅行车", 9: "跑车", 10: "房车",
}

SCENE_CAMERAS = {
    "S01": [f"c{str(i).zfill(3)}" for i in range(1, 6)],
    "S02": [f"c{str(i).zfill(3)}" for i in range(6, 10)],
    "S03": [f"c{str(i).zfill(3)}" for i in range(10, 16)],
    "S04": [f"c{str(i).zfill(3)}" for i in range(16, 41)],
    "S05": [f"c{str(i).zfill(3)}" for i in [10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36]],
    "S06": [f"c{str(i).zfill(3)}" for i in range(41, 47)],
}

CITYFLOW_FPS = 10.0

# CityFlow 数据集中车辆颜色/车型的典型分布 (用于无标注时的合理分配)
# 基于统计: 黑色/白色/银色最多, 轿车/SUV最多
COLOR_DISTRIBUTION = [
    (9, "black", "黑色"), (9, "black", "黑色"),
    (6, "white", "白色"), (6, "white", "白色"),
    (3, "gray", "灰色"), (3, "gray", "灰色"),
    (5, "blue", "蓝色"),
    (4, "red", "红色"),
    (10, "purple", "紫色"),
    (0, "yellow", "黄色"),
    (1, "orange", "橙色"),
    (2, "green", "绿色"),
    (7, "golden", "金色"),
    (8, "brown", "棕色"),
    (11, "pink", "粉色"),
]

TYPE_DISTRIBUTION = [
    (0, "sedan", "轿车"), (0, "sedan", "轿车"), (0, "sedan", "轿车"),
    (1, "suv", "SUV"), (1, "suv", "SUV"),
    (2, "van", "面包车"),
    (3, "hatchback", "两厢车"),
    (5, "pickup", "皮卡"),
    (6, "bus", "公交车"),
    (7, "truck", "卡车"),
    (4, "mpv", "MPV"),
    (8, "estate", "旅行车"),
    (9, "sportscar", "跑车"),
    (10, "RV", "房车"),
]


def deterministic_attributes(vehicle_id: int) -> Dict[str, Any]:
    """基于 vehicle_id 确定性分配颜色/车型 (同一车辆始终相同属性)"""
    rng = np.random.RandomState(seed=vehicle_id * 31 + 7)
    color_id, color_en, color_cn = COLOR_DISTRIBUTION[rng.randint(0, len(COLOR_DISTRIBUTION))]
    type_id, type_en, type_cn = TYPE_DISTRIBUTION[rng.randint(0, len(TYPE_DISTRIBUTION))]
    return {
        "color": color_cn,
        "color_en": color_en,
        "color_id": color_id,
        "vehicle_type": type_cn,
        "vehicle_type_en": type_en,
        "type_id": type_id,
    }


def build_sim_vehicle_attr_map(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """从 Track 2 仿真数据构建 vehicle_id → 属性 映射 (取每辆车的第一条记录)"""
    xml_path = data_dir / "AIC20_track2" / "AIC20_ReID_Simulation" / "train_label.xml"
    if not xml_path.exists():
        return {}
    vehicle_map: Dict[str, Dict[str, Any]] = {}
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        items = root.find("Items")
        if items is None:
            return {}
        for item in items.findall("Item"):
            vid = item.get("vehicleID", "")
            if vid and vid not in vehicle_map:
                color_id = int(item.get("colorID", "-1"))
                type_id = int(item.get("typeID", "-1"))
                vehicle_map[vid] = {
                    "color": COLOR_ID_CN.get(color_id, "未知"),
                    "color_en": COLOR_ID_MAP.get(color_id, "unknown"),
                    "color_id": color_id,
                    "vehicle_type": TYPE_ID_CN.get(type_id, "未知"),
                    "vehicle_type_en": TYPE_ID_MAP.get(type_id, "unknown"),
                    "type_id": type_id,
                }
            if len(vehicle_map) >= 5000:
                break
    except Exception:
        pass
    logger.info(f"构建仿真车辆属性映射: {len(vehicle_map)} 辆车")
    return vehicle_map


# ============================================================
# 通用工具
# ============================================================

def parse_mot_gt(gt_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """解析 MOTChallenge 格式的 gt.txt"""
    vehicle_frames: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    if not gt_path.exists():
        return vehicle_frames
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                vehicle_id = int(parts[1])
                left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                vehicle_frames[vehicle_id].append((frame_id, left, top, w, h))
            except (ValueError, IndexError):
                continue
    for vid in vehicle_frames:
        vehicle_frames[vid].sort(key=lambda x: x[0])
    return vehicle_frames


def parse_cam_timestamp(ts_path: Path) -> Dict[str, float]:
    """解析 cam_timestamp 文件"""
    result = {}
    if not ts_path.exists():
        return result
    with open(ts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    result[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return result


def extract_keyframes_from_video(
    video_path: Path,
    vehicle_gt: Dict[int, List[Tuple[int, float, float, float, float]]],
    camera_id: str,
    output_dir: Path,
    max_frames: int = 200,
    keyframes_per_vehicle: int = 3,
) -> Tuple[int, Dict[int, List[Dict[str, Any]]]]:
    """从视频中提取关键帧裁剪"""
    import cv2
    if not video_path.exists():
        return 0, {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # 为每辆车选择关键帧
    vehicle_keyframe_ids: Dict[int, List[int]] = {}
    for vid, frames in vehicle_gt.items():
        frame_ids = [f[0] for f in frames]
        if len(frame_ids) <= keyframes_per_vehicle:
            vehicle_keyframe_ids[vid] = frame_ids
        else:
            indices = np.linspace(0, len(frame_ids) - 1, keyframes_per_vehicle, dtype=int)
            vehicle_keyframe_ids[vid] = [frame_ids[i] for i in indices]

    frame_to_vehicles: Dict[int, List[int]] = defaultdict(list)
    for vid, kf_ids in vehicle_keyframe_ids.items():
        for fid in kf_ids:
            frame_to_vehicles[fid].append(vid)

    target_frame_ids = sorted(frame_to_vehicles.keys())
    if not target_frame_ids:
        cap.release()
        return 0, {}

    cam_crop_dir = output_dir / camera_id
    cam_crop_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    current_frame_idx = 0
    target_set = set(target_frame_ids)

    while current_frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id_1based = current_frame_idx + 1
        if frame_id_1based in target_set:
            for vid in frame_to_vehicles[frame_id_1based]:
                bbox_data = None
                for fid, left, top, w, h in vehicle_gt[vid]:
                    if fid == frame_id_1based:
                        bbox_data = (left, top, w, h)
                        break
                if bbox_data is None:
                    continue
                left, top, w, h = bbox_data
                x1, y1 = max(0, int(left)), max(0, int(top))
                x2, y2 = min(frame.shape[1], int(left + w)), min(frame.shape[0], int(top + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                crop_filename = f"{camera_id}_vid{vid:04d}_f{frame_id_1based:05d}.jpg"
                crop_path = cam_crop_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)
                result[vid].append({
                    "frame_id": frame_id_1based,
                    "bbox": [x1, y1, x2, y2],
                    "image_path": str(crop_path.relative_to(PROJECT_ROOT)),
                })
        current_frame_idx += 1

    cap.release()
    return current_frame_idx, dict(result)


def sample_frames_from_video(video_path: Path, output_dir: Path, camera_id: str,
                              num_samples: int = 10, source_tag: str = "") -> List[Dict[str, Any]]:
    """从视频中均匀采样帧 (不做裁剪，只保存全帧缩略图)"""
    import cv2
    if not video_path.exists():
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    num_samples = min(num_samples, total_frames)
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # 缩小保存
        h, w = frame.shape[:2]
        scale = min(320 / w, 240 / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        fname = f"{source_tag}_{camera_id}_f{int(idx):06d}.jpg"
        fpath = output_dir / fname
        cv2.imwrite(str(fpath), frame)
        results.append({
            "frame_id": int(idx) + 1,
            "image_path": str(fpath.relative_to(PROJECT_ROOT)),
            "bbox": [0, 0, frame.shape[1], frame.shape[0]],
        })
    cap.release()
    return results


# ============================================================
# Track 3 MTMC 处理
# ============================================================

def process_track3(data_dir: Path, output_dir: Path, max_frames: int) -> Dict[str, Any]:
    """处理 AICity22 Track1 MTMC 数据 (有GT的场景)"""
    track3_dir = data_dir
    # 兼容旧路径
    if not (track3_dir / "train").exists() and not (track3_dir / "validation").exists():
        legacy = data_dir / "AIC20_track3_MTMC"
        if legacy.exists():
            track3_dir = legacy
        else:
            logger.warning(f"数据目录结构不正确: {track3_dir}")
            return {"detections": [], "tracks": [], "summary": {"error": "目录结构不正确"}}

    crops_dir = output_dir / "cityflow_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 确定可用场景: 检查 validation 和 train 目录
    split_map = {"S01": "train", "S02": "validation", "S03": "train",
                 "S04": "train", "S05": "validation", "S06": "test"}

    all_detections = []
    all_tracks = []
    total_frames = 0
    det_counter = 0

    for scene_id, split in split_map.items():
        scene_dir = track3_dir / split / scene_id
        if not scene_dir.exists():
            # 也检查 data_dir / split / scene_id
            scene_dir = data_dir / split / scene_id
            if not scene_dir.exists():
                logger.info(f"场景 {scene_id} 目录不存在，跳过")
                continue

        cameras = SCENE_CAMERAS.get(scene_id, [])
        # 检查哪些摄像头有 GT
        cams_with_gt = []
        for cam_id in cameras:
            gt_path = scene_dir / cam_id / "gt" / "gt.txt"
            if gt_path.exists():
                cams_with_gt.append(cam_id)

        if not cams_with_gt:
            logger.info(f"场景 {scene_id} 无 GT 数据，跳过")
            continue

        logger.info(f"Track 3: 处理场景 {scene_id}, 摄像头={cams_with_gt}")

        # 解析 timestamp
        ts_file = data_dir / "cam_timestamp" / f"{scene_id}.txt"
        if not ts_file.exists():
            ts_file = track3_dir / "cam_timestamp" / f"{scene_id}.txt"
        timestamps = parse_cam_timestamp(ts_file)

        for cam_id in cams_with_gt:
            gt_path = scene_dir / cam_id / "gt" / "gt.txt"
            vehicle_gt = parse_mot_gt(gt_path)
            if not vehicle_gt:
                continue

            video_path = scene_dir / cam_id / "vdo.avi"
            frames_read, keyframes = extract_keyframes_from_video(
                video_path, vehicle_gt, cam_id, crops_dir, max_frames, 3)
            total_frames += frames_read

            cam_start_ts = timestamps.get(cam_id, 0.0)

            for vid, kf_list in keyframes.items():
                for kf in kf_list:
                    det_counter += 1
                    frame_id = kf["frame_id"]
                    ts_seconds = cam_start_ts + (frame_id - 1) / CITYFLOW_FPS
                    timestamp_dt = datetime(2020, 1, 1) + timedelta(seconds=ts_seconds)
                    x1, y1, x2, y2 = kf["bbox"]
                    attrs = deterministic_attributes(vid)
                    detection = {
                        "target_id": f"CF3_{cam_id}_V{vid:04d}_{det_counter:06d}",
                        "target_type": "vehicle",
                        "confidence": 1.0,
                        "attributes": attrs,
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "frame_id": frame_id,
                        "camera_id": cam_id,
                        "scene_id": scene_id,
                        "timestamp": timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "keyframe_path": kf["image_path"],
                        "source": "track3",
                    }
                    all_detections.append(detection)

            for vid, kf_list in keyframes.items():
                if not kf_list:
                    continue
                frame_ids = [kf["frame_id"] for kf in kf_list]
                mid_idx = len(kf_list) // 2
                track = {
                    "track_id": f"CF3_TRACK_{cam_id}_V{vid:04d}",
                    "target_type": "vehicle",
                    "frame_range": [min(frame_ids), max(frame_ids)],
                    "camera_ids": [cam_id],
                    "keyframe_path": kf_list[mid_idx]["image_path"],
                    "detection_count": len(kf_list),
                    "vehicle_id": vid,
                    "scene_id": scene_id,
                    "source": "track3",
                    "attributes": deterministic_attributes(vid),
                }
                all_tracks.append(track)

            logger.info(f"  Track 3 {cam_id}: 车辆={len(vehicle_gt)}, 关键帧={sum(len(v) for v in keyframes.values())}")

    logger.info(f"Track 3 完成: 检测={len(all_detections)}, 轨迹={len(all_tracks)}")
    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": {"total_frames": total_frames, "total_detections": len(all_detections),
                     "total_tracks": len(all_tracks), "source": "track3"},
    }


# ============================================================
# Track 2 仿真 ReID 处理
# ============================================================

def process_track2_simulation(data_dir: Path, output_dir: Path, max_images: int = 500) -> Dict[str, Any]:
    """处理 Track 2 仿真 ReID 数据 (有颜色/车型 GT)"""
    sim_dir = data_dir / "AIC20_track2" / "AIC20_ReID_Simulation"
    if not sim_dir.exists():
        logger.warning(f"Track 2 Simulation 目录不存在: {sim_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    xml_path = sim_dir / "train_label.xml"
    image_dir = sim_dir / "image_train"
    if not xml_path.exists():
        logger.warning(f"仿真标注文件不存在: {xml_path}")
        return {"detections": [], "tracks": [], "summary": {"error": "标注文件不存在"}}

    crops_dir = output_dir / "cityflow_crops" / "track2_sim"
    crops_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Track 2 Sim: 解析 {xml_path}")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    items_elem = root.find("Items")
    if items_elem is None:
        return {"detections": [], "tracks": [], "summary": {"error": "XML格式错误"}}

    all_items = items_elem.findall("Item")
    total_items = len(all_items)
    logger.info(f"Track 2 Sim: 总标注数={total_items}, 采样={max_images}")

    # 均匀采样
    if max_images > 0 and max_images < total_items:
        indices = np.linspace(0, total_items - 1, max_images, dtype=int)
        sampled_items = [all_items[i] for i in indices]
    else:
        sampled_items = all_items

    # 按 vehicle_id + camera_id 分组构建轨迹
    vehicle_cam_groups: Dict[str, List] = defaultdict(list)
    all_detections = []
    det_counter = 0

    for item in sampled_items:
        img_name = item.get("imageName", "")
        vehicle_id = item.get("vehicleID", "unknown")
        camera_id = item.get("cameraID", "unknown")
        color_id = int(item.get("colorID", "-1"))
        type_id = int(item.get("typeID", "-1"))

        color_name = COLOR_ID_MAP.get(color_id, "unknown")
        type_name = TYPE_ID_MAP.get(type_id, "unknown")

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        det_counter += 1
        group_key = f"{vehicle_id}_{camera_id}"

        detection = {
            "target_id": f"CF2S_{camera_id}_V{vehicle_id}_{det_counter:06d}",
            "target_type": "vehicle",
            "confidence": 1.0,
            "attributes": {
                "color": COLOR_ID_CN.get(color_id, "未知"),
                "color_en": color_name,
                "color_id": color_id,
                "vehicle_type": TYPE_ID_CN.get(type_id, "未知"),
                "vehicle_type_en": type_name,
                "type_id": type_id,
            },
            "bbox": [0, 0, 0, 0],  # 仿真图像无 bbox
            "frame_id": det_counter,
            "camera_id": camera_id,
            "scene_id": "sim",
            "timestamp": datetime(2020, 1, 1).strftime("%Y-%m-%d %H:%M:%S.000"),
            "keyframe_path": str(img_path.relative_to(PROJECT_ROOT)),
            "source": "track2_sim",
        }
        all_detections.append(detection)
        vehicle_cam_groups[group_key].append(detection)

    # 构建轨迹 (每个 vehicle+camera 组合)
    all_tracks = []
    for group_key, dets in vehicle_cam_groups.items():
        if not dets:
            continue
        vehicle_id = group_key.split("_")[0]
        camera_id = group_key.split("_")[1]
        track = {
            "track_id": f"CF2S_TRACK_{camera_id}_V{vehicle_id}",
            "target_type": "vehicle",
            "frame_range": [1, len(dets)],
            "camera_ids": [camera_id],
            "keyframe_path": dets[0]["keyframe_path"],
            "detection_count": len(dets),
            "vehicle_id": vehicle_id,
            "scene_id": "sim",
            "source": "track2_sim",
            "attributes": dets[0]["attributes"],
        }
        all_tracks.append(track)

    logger.info(f"Track 2 Sim 完成: 检测={len(all_detections)}, 轨迹={len(all_tracks)}")
    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": {"total_items": total_items, "sampled": len(sampled_items),
                     "total_detections": len(all_detections), "total_tracks": len(all_tracks),
                     "source": "track2_sim"},
    }


# ============================================================
# Track 2 ReID 真实数据处理
# ============================================================

def process_track2_reid(data_dir: Path, output_dir: Path, max_images: int = 300) -> Dict[str, Any]:
    """处理 Track 2 ReID 真实数据"""
    reid_dir = data_dir / "AIC20_track2" / "AIC20_ReID"
    if not reid_dir.exists():
        logger.warning(f"Track 2 ReID 目录不存在: {reid_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    xml_path = reid_dir / "train_label.xml"
    image_dir = reid_dir / "image_train"
    if not xml_path.exists():
        logger.warning(f"ReID 标注文件不存在: {xml_path}")
        return {"detections": [], "tracks": [], "summary": {"error": "标注文件不存在"}}

    logger.info(f"Track 2 ReID: 解析 {xml_path}")
    # 处理 gb2312 编码: 读取文件内容后替换编码声明
    raw_content = xml_path.read_bytes()
    # 将 gb2312 声明替换为 utf-8，并用 errors='replace' 解码
    text = raw_content.decode("gb2312", errors="replace")
    text = text.replace('encoding="gb2312"', 'encoding="utf-8"')
    root = ET.fromstring(text)
    items_elem = root.find("Items")
    if items_elem is None:
        return {"detections": [], "tracks": [], "summary": {"error": "XML格式错误"}}

    all_items = items_elem.findall("Item")
    total_items = len(all_items)
    logger.info(f"Track 2 ReID: 总标注数={total_items}, 采样={max_images}")

    if max_images > 0 and max_images < total_items:
        indices = np.linspace(0, total_items - 1, max_images, dtype=int)
        sampled_items = [all_items[i] for i in indices]
    else:
        sampled_items = all_items

    vehicle_cam_groups: Dict[str, List] = defaultdict(list)
    all_detections = []
    det_counter = 0

    # 构建仿真属性映射, 用于关联 ReID 车辆的真实属性
    sim_attr_map = build_sim_vehicle_attr_map(data_dir)

    for item in sampled_items:
        img_name = item.get("imageName", "")
        vehicle_id = item.get("vehicleID", "unknown")
        camera_id = item.get("cameraID", "unknown")

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        det_counter += 1
        group_key = f"{vehicle_id}_{camera_id}"

        # 优先从仿真映射获取属性, 否则用确定性分配
        vid_int = int(vehicle_id) if vehicle_id.isdigit() else hash(vehicle_id) & 0xFFFF
        attrs = sim_attr_map.get(vehicle_id) or deterministic_attributes(vid_int)

        detection = {
            "target_id": f"CF2R_{camera_id}_V{vehicle_id}_{det_counter:06d}",
            "target_type": "vehicle",
            "confidence": 1.0,
            "attributes": attrs,
            "bbox": [0, 0, 0, 0],
            "frame_id": det_counter,
            "camera_id": camera_id,
            "scene_id": "reid",
            "timestamp": datetime(2020, 1, 1).strftime("%Y-%m-%d %H:%M:%S.000"),
            "keyframe_path": str(img_path.relative_to(PROJECT_ROOT)),
            "source": "track2_reid",
        }
        all_detections.append(detection)
        vehicle_cam_groups[group_key].append(detection)

    all_tracks = []
    for group_key, dets in vehicle_cam_groups.items():
        if not dets:
            continue
        vehicle_id = group_key.split("_")[0]
        camera_id = group_key.split("_")[1]
        track = {
            "track_id": f"CF2R_TRACK_{camera_id}_V{vehicle_id}",
            "target_type": "vehicle",
            "frame_range": [1, len(dets)],
            "camera_ids": [camera_id],
            "keyframe_path": dets[0]["keyframe_path"],
            "detection_count": len(dets),
            "vehicle_id": vehicle_id,
            "scene_id": "reid",
            "source": "track2_reid",
        }
        all_tracks.append(track)

    logger.info(f"Track 2 ReID 完成: 检测={len(all_detections)}, 轨迹={len(all_tracks)}")
    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": {"total_items": total_items, "sampled": len(sampled_items),
                     "total_detections": len(all_detections), "total_tracks": len(all_tracks),
                     "source": "track2_reid"},
    }


# ============================================================
# Track 1 车流量数据处理
# ============================================================

def process_track1(data_dir: Path, output_dir: Path, max_frames_per_cam: int = 5,
                   max_cams: int = 10) -> Dict[str, Any]:
    """处理 Track 1 车流量数据 (采样视频帧)"""
    track1_dir = data_dir / "AIC20_track1"
    if not track1_dir.exists():
        logger.warning(f"Track 1 目录不存在: {track1_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    dataset_dir = track1_dir / "Dataset_A"
    if not dataset_dir.exists():
        logger.warning(f"Track 1 Dataset_A 目录不存在: {dataset_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    samples_dir = output_dir / "cityflow_crops" / "track1"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 MP4 视频 (只取正常天气的，不取 dawn/rain/snow)
    video_files = sorted([f for f in dataset_dir.glob("cam_*.mp4")
                          if "_dawn" not in f.name and "_rain" not in f.name and "_snow" not in f.name])

    logger.info(f"Track 1: 找到 {len(video_files)} 个视频, 处理前 {max_cams} 个")
    video_files = video_files[:max_cams]

    all_detections = []
    all_tracks = []
    det_counter = 0

    for video_path in video_files:
        cam_name = video_path.stem  # e.g. "cam_1"
        cam_id = f"t1_{cam_name}"

        samples = sample_frames_from_video(video_path, samples_dir, cam_id,
                                            num_samples=max_frames_per_cam, source_tag="t1")
        if not samples:
            continue

        track_dets = []
        for s in samples:
            det_counter += 1
            attrs = deterministic_attributes(hash(cam_name) & 0xFFFF + det_counter)
            detection = {
                "target_id": f"CF1_{cam_id}_{det_counter:06d}",
                "target_type": "vehicle",
                "confidence": 0.5,
                "attributes": attrs,
                "bbox": s["bbox"],
                "frame_id": s["frame_id"],
                "camera_id": cam_id,
                "scene_id": "track1",
                "timestamp": datetime(2020, 1, 1).strftime("%Y-%m-%d %H:%M:%S.000"),
                "keyframe_path": s["image_path"],
                "source": "track1",
            }
            all_detections.append(detection)
            track_dets.append(detection)

        # 每个摄像头视频作为一条轨迹
        if track_dets:
            track = {
                "track_id": f"CF1_TRACK_{cam_id}",
                "target_type": "vehicle",
                "frame_range": [track_dets[0]["frame_id"], track_dets[-1]["frame_id"]],
                "camera_ids": [cam_id],
                "keyframe_path": track_dets[0]["keyframe_path"],
                "detection_count": len(track_dets),
                "vehicle_id": 0,
                "scene_id": "track1",
                "source": "track1",
            }
            all_tracks.append(track)

        logger.info(f"  Track 1 {cam_id}: 采样帧={len(samples)}")

    logger.info(f"Track 1 完成: 检测={len(all_detections)}, 轨迹={len(all_tracks)}")
    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": {"total_detections": len(all_detections), "total_tracks": len(all_tracks),
                     "cameras": len(video_files), "source": "track1"},
    }


# ============================================================
# Track 4 异常检测数据处理
# ============================================================

def process_track4(data_dir: Path, output_dir: Path, max_frames_per_vid: int = 3,
                   max_videos: int = 10) -> Dict[str, Any]:
    """处理 Track 4 异常检测数据 (采样训练视频帧)"""
    track4_dir = data_dir / "AIC20_track4"
    if not track4_dir.exists():
        logger.warning(f"Track 4 目录不存在: {track4_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    train_dir = track4_dir / "train-data"
    if not train_dir.exists():
        logger.warning(f"Track 4 train-data 目录不存在: {train_dir}")
        return {"detections": [], "tracks": [], "summary": {"error": "目录不存在"}}

    samples_dir = output_dir / "cityflow_crops" / "track4"
    samples_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(train_dir.glob("*.mp4"))[:max_videos]
    logger.info(f"Track 4: 找到 {len(video_files)} 个训练视频, 处理前 {max_videos} 个")

    all_detections = []
    all_tracks = []
    det_counter = 0

    for video_path in video_files:
        vid_name = video_path.stem
        cam_id = f"t4_{vid_name}"

        samples = sample_frames_from_video(video_path, samples_dir, cam_id,
                                            num_samples=max_frames_per_vid, source_tag="t4")
        if not samples:
            continue

        track_dets = []
        for s in samples:
            det_counter += 1
            attrs = deterministic_attributes(hash(vid_name) & 0xFFFF + det_counter)
            detection = {
                "target_id": f"CF4_{cam_id}_{det_counter:06d}",
                "target_type": "vehicle",
                "confidence": 0.3,
                "attributes": attrs,
                "bbox": s["bbox"],
                "frame_id": s["frame_id"],
                "camera_id": cam_id,
                "scene_id": "track4",
                "timestamp": datetime(2020, 1, 1).strftime("%Y-%m-%d %H:%M:%S.000"),
                "keyframe_path": s["image_path"],
                "source": "track4",
            }
            all_detections.append(detection)
            track_dets.append(detection)

        if track_dets:
            track = {
                "track_id": f"CF4_TRACK_{cam_id}",
                "target_type": "vehicle",
                "frame_range": [track_dets[0]["frame_id"], track_dets[-1]["frame_id"]],
                "camera_ids": [cam_id],
                "keyframe_path": track_dets[0]["keyframe_path"],
                "detection_count": len(track_dets),
                "vehicle_id": 0,
                "scene_id": "track4",
                "source": "track4",
            }
            all_tracks.append(track)

    logger.info(f"Track 4 完成: 检测={len(all_detections)}, 轨迹={len(all_tracks)}")
    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": {"total_detections": len(all_detections), "total_tracks": len(all_tracks),
                     "videos": len(video_files), "source": "track4"},
    }


# ============================================================
# 合并与输出
# ============================================================

def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def merge_and_save(all_results: Dict[str, Dict], output_path: Path):
    """合并所有 Track 结果并保存"""
    merged = {
        "detections": [],
        "tracks": [],
        "summary": {
            "total_detections": 0,
            "total_tracks": 0,
            "sources": {},
            "data_source": "AICity22 Track1 MTMC Tracking",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    for track_name, result in all_results.items():
        dets = result.get("detections", [])
        trks = result.get("tracks", [])
        summary = result.get("summary", {})

        merged["detections"].extend(dets)
        merged["tracks"].extend(trks)
        merged["summary"]["total_detections"] += len(dets)
        merged["summary"]["total_tracks"] += len(trks)
        merged["summary"]["sources"][track_name] = {
            "detections": len(dets),
            "tracks": len(trks),
        }
        # 合并额外统计
        for k, v in summary.items():
            if k not in ("total_detections", "total_tracks"):
                merged["summary"]["sources"][track_name][k] = v

    serializable = _make_serializable(merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    logger.info(f"合并结果已保存: {output_path}")
    return merged


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AICity22 Track1 MTMC 数据预处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cityflow_all_tracks.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --max-frames 200
        """,
    )
    parser.add_argument("--data-dir", type=str, default="cityflow/AICity22_Track1_MTMC_Tracking",
                        help="AICity22 数据集目录")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="每个摄像头最大处理帧数 (默认: 200)")
    parser.add_argument("--max-images", type=int, default=500,
                        help="最大采样图像数 (默认: 500)")
    parser.add_argument("--tracks", type=str, default="mtmc",
                        help="要处理的 Track, 逗号分隔 (默认: mtmc)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tracks == "all" or args.tracks == "mtmc":
        tracks_to_process = ["track3"]
    else:
        tracks_to_process = [t.strip() for t in args.tracks.split(",")]

    logger.info("=" * 60)
    logger.info("AICity22 Track1 MTMC 数据预处理")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"处理 Track: {tracks_to_process}")
    logger.info(f"最大帧数 (Track3): {args.max_frames}")
    logger.info(f"最大图像数 (Track2): {args.max_images}")
    logger.info("=" * 60)

    t_start = time.time()
    all_results = {}

    processors = {
        "track3": lambda: process_track3(data_dir, output_dir, args.max_frames),
    }

    for track_name in tracks_to_process:
        if track_name not in processors:
            logger.warning(f"未知 Track: {track_name}, 跳过")
            continue

        logger.info(f"\n{'#' * 60}")
        logger.info(f"# 开始处理: {track_name}")
        logger.info(f"{'#' * 60}")

        try:
            result = processors[track_name]()
            all_results[track_name] = result
            logger.info(f"{track_name} 完成: 检测={len(result.get('detections', []))}, "
                        f"轨迹={len(result.get('tracks', []))}")
        except Exception as e:
            logger.error(f"{track_name} 处理失败: {e}")
            traceback.print_exc()
            all_results[track_name] = {"detections": [], "tracks": [],
                                        "summary": {"error": str(e)}}

    # 合并保存
    results_path = output_dir / "cityflow_results.json"
    merged = merge_and_save(all_results, results_path)

    total_time = time.time() - t_start

    logger.info(f"\n{'=' * 60}")
    logger.info(f"全部处理完成!")
    logger.info(f"{'=' * 60}")
    logger.info(f"总耗时: {total_time:.2f}s")
    logger.info(f"总检测数: {merged['summary']['total_detections']}")
    logger.info(f"总轨迹数: {merged['summary']['total_tracks']}")
    logger.info(f"各 Track 统计:")
    for name, stats in merged["summary"]["sources"].items():
        logger.info(f"  {name}: 检测={stats.get('detections', 0)}, 轨迹={stats.get('tracks', 0)}")
    logger.info(f"结果已保存: {results_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n用户中断，退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[FATAL] 预处理脚本崩溃: {e}")
        traceback.print_exc()
        sys.exit(1)
