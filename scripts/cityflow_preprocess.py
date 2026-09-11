"""
scripts/cityflow_preprocess.py - AICity22 Track1 MTMC 数据接入与预处理脚本

将 AICity22 Track1 MTMC Tracking 数据集转换为项目标准格式。

功能:
  1. 解析 AICity22 GT 标注 (MOTChallenge 格式)
  2. 解析全局 ground_truth_*.txt (含 camera_id)
  3. 解析 cam_timestamp / cam_framenum
  4. 视频帧提取与关键帧裁剪
  5. 属性映射 (颜色/车型)
  6. 生成项目标准输出 (cityflow_results.json)
  7. (可选) 特征提取 (ReID + CLIP)

使用方式:
  # 快速模式: 只处理 S01, 前 500 帧, 使用 GT
  python scripts/cityflow_preprocess.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --scene S01 --max-frames 500

  # 完整模式: 处理所有场景
  python scripts/cityflow_preprocess.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --all-scenes --max-frames 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 解决 OpenMP 重复加载冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from src.common.logger import get_logger

logger = get_logger("scripts.cityflow_preprocess")


# ============================================================
# 常量与映射表
# ============================================================

# CityFlow 颜色 ID → 中文名
COLOR_ID_MAP = {
    0: "yellow",    # 黄色
    1: "orange",    # 橙色
    2: "green",     # 绿色
    3: "gray",      # 灰色
    4: "red",       # 红色
    5: "blue",      # 蓝色
    6: "white",     # 白色
    7: "golden",    # 金色
    8: "brown",     # 棕色
    9: "black",     # 黑色
    10: "purple",   # 紫色
    11: "pink",     # 粉色
}

COLOR_ID_CN = {
    0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色",
    4: "红色", 5: "蓝色", 6: "白色", 7: "金色",
    8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色",
}

# CityFlow 车型 ID → 英文名
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

# 场景 → 摄像头映射
SCENE_CAMERAS = {
    "S01": [f"c{str(i).zfill(3)}" for i in range(1, 6)],       # c001-c005
    "S02": [f"c{str(i).zfill(3)}" for i in range(6, 10)],      # c006-c009
    "S03": [f"c{str(i).zfill(3)}" for i in range(10, 16)],     # c010-c015
    "S04": [f"c{str(i).zfill(3)}" for i in range(16, 41)],     # c016-c040
    "S05": [f"c{str(i).zfill(3)}" for i in [10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36]],  # c010,c016-c029,c033-c036 (validation)
    "S06": [f"c{str(i).zfill(3)}" for i in range(41, 47)],     # c041-c046 (test)
}

# CityFlow 视频 FPS (标称值)
CITYFLOW_FPS = 10.0


# ============================================================
# 数据解析工具
# ============================================================

def parse_mot_gt(gt_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    解析 MOTChallenge 格式的 gt.txt

    格式: frame_id, vehicle_id, left, top, w, h, 1, -1, -1, -1

    Returns:
        Dict[vehicle_id] = List[(frame_id, left, top, width, height)]
    """
    vehicle_frames: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)

    if not gt_path.exists():
        logger.warning(f"GT 文件不存在: {gt_path}")
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
                left = float(parts[2])
                top = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                vehicle_frames[vehicle_id].append((frame_id, left, top, w, h))
            except (ValueError, IndexError) as e:
                logger.debug(f"跳过无效 GT 行: {line} ({e})")

    # 按 frame_id 排序
    for vid in vehicle_frames:
        vehicle_frames[vid].sort(key=lambda x: x[0])

    logger.info(f"解析 GT: {gt_path.name}, 车辆数={len(vehicle_frames)}, "
                f"总标注帧数={sum(len(v) for v in vehicle_frames.values())}")
    return vehicle_frames


def parse_global_gt(gt_path: Path) -> List[Dict[str, Any]]:
    """
    解析全局 ground_truth_*.txt

    格式: 每行包含 camera_id, frame_id, vehicle_id, left, top, w, h
    具体格式可能为: camera_id frame_id vehicle_id left top w h
    或逗号分隔。

    Returns:
        List of dicts with keys: camera_id, frame_id, vehicle_id, left, top, w, h
    """
    records = []
    if not gt_path.exists():
        logger.warning(f"全局 GT 文件不存在: {gt_path}")
        return records

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 尝试逗号分隔
            parts = line.replace(",", " ").split()
            if len(parts) < 7:
                continue
            try:
                records.append({
                    "camera_id": parts[0],
                    "frame_id": int(parts[1]),
                    "vehicle_id": int(parts[2]),
                    "left": float(parts[3]),
                    "top": float(parts[4]),
                    "w": float(parts[5]),
                    "h": float(parts[6]),
                })
            except (ValueError, IndexError):
                continue

    logger.info(f"解析全局 GT: {gt_path.name}, 记录数={len(records)}")
    return records


def parse_cam_timestamp(ts_path: Path) -> Dict[str, float]:
    """
    解析 cam_timestamp 文件

    格式: 每行 camera_id start_timestamp
    start_timestamp 通常是秒数 (浮点)

    Returns:
        Dict[camera_id] = start_timestamp (秒)
    """
    result = {}
    if not ts_path.exists():
        logger.warning(f"cam_timestamp 文件不存在: {ts_path}")
        return result

    with open(ts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cam_id = parts[0]
                    ts = float(parts[1])
                    result[cam_id] = ts
                except ValueError:
                    continue

    logger.info(f"解析 cam_timestamp: {ts_path.name}, 摄像头数={len(result)}")
    return result


def parse_cam_framenum(fn_path: Path) -> Dict[str, int]:
    """
    解析 cam_framenum 文件

    格式: 每行 camera_id frame_count 或 camera_id start_frame end_frame

    Returns:
        Dict[camera_id] = total_frame_count
    """
    result = {}
    if not fn_path.exists():
        logger.warning(f"cam_framenum 文件不存在: {fn_path}")
        return result

    with open(fn_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cam_id = parts[0]
                    if len(parts) >= 3:
                        # start_frame end_frame 格式
                        start = int(parts[1])
                        end = int(parts[2])
                        result[cam_id] = end - start + 1
                    else:
                        result[cam_id] = int(parts[1])
                except ValueError:
                    continue

    logger.info(f"解析 cam_framenum: {fn_path.name}, 摄像头数={len(result)}")
    return result


def parse_list_cam(list_path: Path) -> Dict[str, List[str]]:
    """
    解析 list_cam.txt

    格式: 每行 camera_id scene_id
    或: scene_id camera_id

    Returns:
        Dict[scene_id] = List[camera_id]
    """
    result: Dict[str, List[str]] = defaultdict(list)
    if not list_path.exists():
        logger.warning(f"list_cam.txt 不存在: {list_path}")
        return dict(result)

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 判断哪个是 scene_id (S开头)
                for p in parts:
                    if p.startswith("S"):
                        scene_id = p
                        break
                else:
                    scene_id = parts[-1]
                cam_id = [p for p in parts if not p.startswith("S")][0] if any(not p.startswith("S") for p in parts) else parts[0]
                result[scene_id].append(cam_id)

    logger.info(f"解析 list_cam.txt: 场景数={len(result)}")
    return dict(result)


# ============================================================
# 视频帧提取
# ============================================================

def extract_keyframes(
    video_path: Path,
    vehicle_gt: Dict[int, List[Tuple[int, float, float, float, float]]],
    camera_id: str,
    output_dir: Path,
    max_frames: int = 0,
    keyframes_per_vehicle: int = 3,
) -> Tuple[int, Dict[int, List[Dict[str, Any]]]]:
    """
    从视频中提取关键帧

    Args:
        video_path: vdo.avi 路径
        vehicle_gt: parse_mot_gt 的输出
        camera_id: 摄像头 ID
        output_dir: 裁剪图输出目录
        max_frames: 最大处理帧数 (0=不限)
        keyframes_per_vehicle: 每辆车取多少关键帧

    Returns:
        (total_frames_read, Dict[vehicle_id] = List[{frame_id, bbox, image_path}])
    """
    import cv2

    if not video_path.exists():
        logger.warning(f"视频文件不存在: {video_path}")
        return 0, {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return 0, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or CITYFLOW_FPS
    logger.info(f"视频信息: {video_path.name}, 总帧数={total_frames}, FPS={fps:.1f}")

    # 限制处理帧数
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # 为每辆车选择关键帧的帧号 (均匀采样)
    vehicle_keyframe_ids: Dict[int, List[int]] = {}
    for vid, frames in vehicle_gt.items():
        frame_ids = [f[0] for f in frames]
        if len(frame_ids) <= keyframes_per_vehicle:
            vehicle_keyframe_ids[vid] = frame_ids
        else:
            # 均匀采样
            indices = np.linspace(0, len(frame_ids) - 1, keyframes_per_vehicle, dtype=int)
            vehicle_keyframe_ids[vid] = [frame_ids[i] for i in indices]

    # 构建 frame_id → 需要提取的车辆列表
    frame_to_vehicles: Dict[int, List[int]] = defaultdict(list)
    for vid, kf_ids in vehicle_keyframe_ids.items():
        for fid in kf_ids:
            frame_to_vehicles[fid].append(vid)

    # 按帧号排序，逐帧读取
    target_frame_ids = sorted(frame_to_vehicles.keys())
    if not target_frame_ids:
        cap.release()
        return 0, {}

    # 输出子目录
    cam_crop_dir = output_dir / camera_id
    cam_crop_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    frames_read = 0
    current_frame_idx = 0
    target_set = set(target_frame_ids)

    while current_frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id_0based = current_frame_idx
        frame_id_1based = frame_id_0based + 1  # MOT 格式 frame_id 从 1 开始

        if frame_id_1based in target_set:
            for vid in frame_to_vehicles[frame_id_1based]:
                # 找到该帧对应的 bbox
                bbox_data = None
                for fid, left, top, w, h in vehicle_gt[vid]:
                    if fid == frame_id_1based:
                        bbox_data = (left, top, w, h)
                        break

                if bbox_data is None:
                    continue

                left, top, w, h = bbox_data
                # 坐标转换: (left, top, w, h) → (x1, y1, x2, y2)
                x1 = max(0, int(left))
                y1 = max(0, int(top))
                x2 = min(frame.shape[1], int(left + w))
                y2 = min(frame.shape[0], int(top + h))

                if x2 <= x1 or y2 <= y1:
                    continue

                # 裁剪并保存
                crop = frame[y1:y2, x1:x2].copy()
                crop_filename = f"{camera_id}_vid{vid:04d}_f{frame_id_1based:05d}.jpg"
                crop_path = cam_crop_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                result[vid].append({
                    "frame_id": frame_id_1based,
                    "bbox": [x1, y1, x2, y2],
                    "image_path": str(crop_path.relative_to(PROJECT_ROOT)),
                })

        frames_read += 1
        current_frame_idx += 1

        # 如果所有目标帧都已处理，提前退出
        if frames_read >= total_frames:
            break

    cap.release()
    logger.info(f"关键帧提取完成: {camera_id}, 读取帧数={frames_read}, "
                f"裁剪数={sum(len(v) for v in result.values())}")
    return frames_read, dict(result)


# ============================================================
# 主处理流程
# ============================================================

def process_scene(
    data_dir: Path,
    scene_id: str,
    output_dir: Path,
    max_frames: int = 500,
    use_gt: bool = True,
    extract_features: bool = False,
) -> Dict[str, Any]:
    """
    处理单个场景

    Args:
        data_dir: AICity22 数据集根目录 (包含 train/ validation/ test/)
        scene_id: 场景 ID (S01-S06)
        output_dir: 输出根目录
        max_frames: 每个摄像头最大处理帧数 (0=不限)
        use_gt: 是否使用 GT 标注
        extract_features: 是否提取特征向量

    Returns:
        处理结果摘要 dict
    """
    import cv2
    import shutil

    # 磁盘空间检查
    total, used, free = shutil.disk_usage(str(output_dir))
    free_gb = free / (1024 ** 3)
    if free_gb < 5:
        logger.warning(f"磁盘可用空间不足: {free_gb:.1f}GB，建议至少保留 5GB 可用空间")

    track3_dir = data_dir
    if not (track3_dir / "train").exists() and not (track3_dir / "validation").exists():
        # 兼容旧路径: 尝试 data_dir / AIC20_track3_MTMC
        legacy_dir = data_dir / "AIC20_track3_MTMC"
        if legacy_dir.exists():
            track3_dir = legacy_dir
        else:
            logger.error(f"数据目录结构不正确: {data_dir}")
            logger.error("期望目录包含 train/ 和 validation/ 子目录")
            return {"error": f"目录结构不正确: {data_dir}"}

    # 确定数据集子目录 (train / validation / test)
    split_map = {
        "S01": "train", "S02": "validation", "S03": "train",
        "S04": "train", "S05": "validation", "S06": "test",
    }
    split = split_map.get(scene_id, "train")
    scene_dir = track3_dir / split / scene_id

    if not scene_dir.exists():
        logger.warning(f"场景目录不存在: {scene_dir}，尝试跳过")
        return {"error": f"场景目录不存在: {scene_dir}"}

    cameras = SCENE_CAMERAS.get(scene_id, [])
    logger.info(f"=" * 60)
    logger.info(f"处理场景: {scene_id} (split={split}), 摄像头数={len(cameras)}")
    logger.info(f"场景目录: {scene_dir}")
    logger.info(f"最大帧数: {max_frames if max_frames > 0 else '不限'}")
    logger.info(f"=" * 60)

    # 解析元数据
    cam_ts_dir = track3_dir / "cam_timestamp"
    cam_fn_dir = track3_dir / "cam_framenum"
    timestamps = {}
    frame_nums = {}

    # 尝试解析场景对应的 timestamp/framenum 文件
    ts_file = cam_ts_dir / f"{scene_id}.txt"
    fn_file = cam_fn_dir / f"{scene_id}.txt"
    if ts_file.exists():
        timestamps = parse_cam_timestamp(ts_file)
    if fn_file.exists():
        frame_nums = parse_cam_framenum(fn_file)

    # 输出目录
    crops_dir = output_dir / "cityflow_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "cityflow_features"

    # 收集所有结果
    all_detections: List[Dict[str, Any]] = []
    all_tracks: List[Dict[str, Any]] = []
    total_frames_read = 0
    total_gt_detections = 0
    vehicle_ids_global: set = set()
    camera_ids_processed: set = set()

    # 特征提取器 (延迟初始化)
    feature_extractor = None
    if extract_features:
        try:
            from src.perception.feature_extractor import FeatureExtractor
            feature_extractor = FeatureExtractor(device="cuda")
            features_dir.mkdir(parents=True, exist_ok=True)
            logger.info("特征提取器已初始化")
        except Exception as e:
            logger.warning(f"特征提取器初始化失败，跳过特征提取: {e}")
            feature_extractor = None

    det_id_counter = 0

    for cam_id in cameras:
        cam_dir = scene_dir / cam_id
        if not cam_dir.exists():
            logger.warning(f"摄像头目录不存在，跳过: {cam_dir}")
            continue

        logger.info(f"--- 处理摄像头: {cam_id} ---")

        # 1. 解析 GT
        gt_path = cam_dir / "gt" / "gt.txt"
        vehicle_gt = {}
        if use_gt and gt_path.exists():
            vehicle_gt = parse_mot_gt(gt_path)
        elif use_gt:
            logger.warning(f"GT 文件不存在: {gt_path}，跳过该摄像头")
            continue

        if not vehicle_gt:
            logger.warning(f"摄像头 {cam_id} 无 GT 数据，跳过")
            continue

        # 2. 提取关键帧
        video_path = cam_dir / "vdo.avi"
        frames_read, keyframes = extract_keyframes(
            video_path=video_path,
            vehicle_gt=vehicle_gt,
            camera_id=cam_id,
            output_dir=crops_dir,
            max_frames=max_frames,
            keyframes_per_vehicle=3,
        )
        total_frames_read += frames_read
        camera_ids_processed.add(cam_id)

        # 3. 计算时间戳
        cam_start_ts = timestamps.get(cam_id, 0.0)

        # 4. 生成 detection 记录
        cam_vehicle_count = 0
        for vid, kf_list in keyframes.items():
            vehicle_ids_global.add(f"{cam_id}_{vid}")
            cam_vehicle_count += 1

            for kf in kf_list:
                det_id_counter += 1
                frame_id = kf["frame_id"]
                bbox = kf["bbox"]

                # 时间戳计算: start_offset + frame_id / FPS
                ts_seconds = cam_start_ts + (frame_id - 1) / CITYFLOW_FPS
                timestamp_dt = datetime(2020, 1, 1) + timedelta(seconds=ts_seconds)

                # 坐标转为 BoundingBox 格式
                x1, y1, x2, y2 = bbox

                detection = {
                    "target_id": f"CF_{cam_id}_V{vid:04d}_{det_id_counter:06d}",
                    "target_type": "vehicle",
                    "confidence": 1.0,  # GT 标注，置信度为 1
                    "attributes": {
                        "color": "unknown",
                        "color_id": -1,
                        "vehicle_type": "unknown",
                        "type_id": -1,
                    },
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "frame_id": frame_id,
                    "camera_id": cam_id,
                    "scene_id": scene_id,
                    "timestamp": timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "image_path": "",
                    "keyframe_path": kf["image_path"],
                }
                all_detections.append(detection)
                total_gt_detections += 1

        # 5. 生成 track 记录 (每个 vehicle_id 在每個攝像頭是一条 track)
        for vid, kf_list in keyframes.items():
            if not kf_list:
                continue

            frame_ids = [kf["frame_id"] for kf in kf_list]
            # 选中间帧作为关键帧
            mid_idx = len(kf_list) // 2

            track = {
                "track_id": f"CF_TRACK_{cam_id}_V{vid:04d}",
                "target_type": "vehicle",
                "frame_range": [min(frame_ids), max(frame_ids)],
                "camera_ids": [cam_id],
                "keyframe_path": kf_list[mid_idx]["image_path"],
                "detection_count": len(kf_list),
                "vehicle_id": vid,
                "scene_id": scene_id,
            }
            all_tracks.append(track)

        logger.info(f"  摄像头 {cam_id}: 车辆数={cam_vehicle_count}, "
                    f"关键帧数={sum(len(v) for v in keyframes.values())}")

    # 6. 特征提取 (可选)
    feature_count = 0
    if feature_extractor and all_detections:
        logger.info(f"开始特征提取, 共 {len(all_detections)} 个检测...")
        for det in all_detections:
            kp = det.get("keyframe_path", "")
            if not kp:
                continue
            full_path = PROJECT_ROOT / kp
            if not full_path.exists():
                continue

            try:
                crop_img = cv2.imread(str(full_path))
                if crop_img is None:
                    continue

                reid_vec = feature_extractor.extract_reid(crop_img)
                clip_vec = feature_extractor.extract_clip(crop_img)

                if reid_vec is not None:
                    det["reid_vector"] = reid_vec.tolist()
                    feature_count += 1
                if clip_vec is not None:
                    det["clip_vector"] = clip_vec.tolist()

                # 保存特征到文件
                if reid_vec is not None or clip_vec is not None:
                    feat_file = features_dir / f"{det['target_id']}_features.npz"
                    np.savez_compressed(
                        str(feat_file),
                        reid=reid_vec if reid_vec is not None else np.zeros(0),
                        clip=clip_vec if clip_vec is not None else np.zeros(0),
                    )
            except Exception as e:
                logger.debug(f"特征提取失败 {det['target_id']}: {e}")

        logger.info(f"特征提取完成: 成功={feature_count}")

    # 7. 汇总
    summary = {
        "scene_id": scene_id,
        "total_frames": total_frames_read,
        "total_detections": total_gt_detections,
        "total_tracks": len(all_tracks),
        "vehicle_count": len(vehicle_ids_global),
        "camera_count": len(camera_ids_processed),
        "cameras_processed": sorted(camera_ids_processed),
    }

    logger.info(f"场景 {scene_id} 处理完成:")
    logger.info(f"  总帧数: {total_frames_read}")
    logger.info(f"  总检测数: {total_gt_detections}")
    logger.info(f"  总 track 数: {len(all_tracks)}")
    logger.info(f"  唯一车辆数: {len(vehicle_ids_global)}")
    logger.info(f"  处理摄像头数: {len(camera_ids_processed)}")

    return {
        "detections": all_detections,
        "tracks": all_tracks,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AICity22 Track1 MTMC 数据预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速模式: 只处理 S01, 前 500 帧
  python scripts/cityflow_preprocess.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --scene S01 --max-frames 500

  # 完整模式: 处理所有场景
  python scripts/cityflow_preprocess.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --all-scenes --max-frames 0

  # 带特征提取
  python scripts/cityflow_preprocess.py --data-dir cityflow/AICity22_Track1_MTMC_Tracking --scene S01 --extract-features
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, default="cityflow/AICity22_Track1_MTMC_Tracking",
        help="AICity22 数据集目录 (默认: cityflow/AICity22_Track1_MTMC_Tracking)",
    )
    parser.add_argument(
        "--scene", type=str, default="S01",
        choices=["S01", "S02", "S03", "S04", "S05", "S06"],
        help="选择场景 (默认: S01)",
    )
    parser.add_argument(
        "--all-scenes", action="store_true",
        help="处理所有场景",
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="每个摄像头最大处理帧数, 0=不限 (默认: 500)",
    )
    parser.add_argument(
        "--use-gt", action="store_true", default=True,
        help="使用 GT 标注而非检测器 (默认: True)",
    )
    parser.add_argument(
        "--no-gt", action="store_true",
        help="不使用 GT 标注，使用检测器",
    )
    parser.add_argument(
        "--extract-features", action="store_true", default=False,
        help="是否提取 ReID/CLIP 特征向量 (需要 GPU)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="输出目录 (默认: output)",
    )

    args = parser.parse_args()

    # 处理 --no-gt
    if args.no_gt:
        args.use_gt = False

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AICity22 Track1 MTMC 数据预处理")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"场景: {'ALL' if args.all_scenes else args.scene}")
    logger.info(f"最大帧数: {args.max_frames if args.max_frames > 0 else '不限'}")
    logger.info(f"使用 GT: {args.use_gt}")
    logger.info(f"提取特征: {args.extract_features}")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 检查数据目录
    track3_dir = data_dir
    if not (track3_dir / "train").exists() and not (track3_dir / "validation").exists():
        legacy_dir = data_dir / "AIC20_track3_MTMC"
        if legacy_dir.exists():
            track3_dir = legacy_dir
        else:
            logger.error(f"数据目录结构不正确: {data_dir}")
            logger.error("请确认 AICity22 数据已解压到正确位置")
            logger.error(f"预期路径: {track3_dir} 包含 train/ validation/ test/ 子目录")
            sys.exit(1)

    t_start = time.time()

    # 确定要处理的场景列表
    if args.all_scenes:
        scenes_to_process = ["S01", "S02", "S03", "S04", "S05", "S06"]
    else:
        scenes_to_process = [args.scene]

    # 逐场景处理
    all_results = {
        "detections": [],
        "tracks": [],
        "summary": {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "vehicle_count": 0,
            "camera_count": 0,
        },
    }

    scene_summaries = []
    for scene_id in scenes_to_process:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# 开始处理场景: {scene_id}")
        logger.info(f"{'#' * 60}")

        try:
            result = process_scene(
                data_dir=data_dir,
                scene_id=scene_id,
                output_dir=output_dir,
                max_frames=args.max_frames,
                use_gt=args.use_gt,
                extract_features=args.extract_features,
            )

            if "error" in result:
                logger.warning(f"场景 {scene_id} 处理失败: {result['error']}")
                scene_summaries.append({"scene_id": scene_id, "error": result["error"]})
                continue

            # 合并结果
            all_results["detections"].extend(result["detections"])
            all_results["tracks"].extend(result["tracks"])
            all_results["summary"]["total_frames"] += result["summary"]["total_frames"]
            all_results["summary"]["total_detections"] += result["summary"]["total_detections"]
            all_results["summary"]["total_tracks"] += result["summary"]["total_tracks"]
            all_results["summary"]["vehicle_count"] += result["summary"]["vehicle_count"]
            all_results["summary"]["camera_count"] += result["summary"]["camera_count"]

            scene_summaries.append(result["summary"])

        except Exception as e:
            logger.error(f"场景 {scene_id} 处理异常: {e}")
            traceback.print_exc()
            scene_summaries.append({"scene_id": scene_id, "error": str(e)})
            continue

    # 添加场景摘要
    all_results["summary"]["scenes"] = scene_summaries
    all_results["summary"]["processing_time_seconds"] = round(time.time() - t_start, 2)
    all_results["summary"]["data_source"] = "AICity22 Track1 MTMC Tracking"
    all_results["summary"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 保存结果 JSON
    results_path = output_dir / "cityflow_results.json"
    # 保存时不包含 numpy array (特征向量已单独保存为 npz)
    serializable_results = _make_serializable(all_results)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_start

    logger.info(f"\n{'=' * 60}")
    logger.info(f"全部处理完成!")
    logger.info(f"{'=' * 60}")
    logger.info(f"总耗时: {total_time:.2f}s")
    logger.info(f"总帧数: {all_results['summary']['total_frames']}")
    logger.info(f"总检测数: {all_results['summary']['total_detections']}")
    logger.info(f"总 track 数: {all_results['summary']['total_tracks']}")
    logger.info(f"总车辆数: {all_results['summary']['vehicle_count']}")
    logger.info(f"总摄像头数: {all_results['summary']['camera_count']}")
    logger.info(f"结果已保存: {results_path}")
    logger.info(f"{'=' * 60}")


def _make_serializable(obj: Any) -> Any:
    """将对象转为 JSON 可序列化格式 (移除 numpy array 等大对象)"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 跳过大型向量字段 (已单独保存为 npz)
            if k in ("reid_vector", "clip_vector"):
                continue
            new_dict[k] = _make_serializable(v)
        return new_dict
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


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
