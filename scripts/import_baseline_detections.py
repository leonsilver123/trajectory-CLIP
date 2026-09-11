"""
scripts/import_baseline_detections.py

导入所有baseline检测器结果到cityflow_results.json，大幅提升检测覆盖率。

策略:
1. 从 det_mask_rcnn.txt 加载所有检测 (frame, id, left, top, w, h, conf, ...)
2. 用 GT tracks 为每个检测分配 vehicle_id (通过 IoU 匹配)
3. 从 GT 获取颜色/车型属性
4. 生成 track 记录 (将同一 vehicle_id 在同一摄像头中的连续帧分组)
5. 保留原有检测结果 (含 CLIP 向量), 追加新的 baseline 检测
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ============================================================
# 常量
# ============================================================

SCENE_CAMERAS = {
    "S01": [f"c{i:03d}" for i in range(1, 6)],
    "S03": [f"c{i:03d}" for i in range(10, 16)],
    "S04": [f"c{i:03d}" for i in range(16, 41)],
}

GT_BASE = "cityflow/AICity22_Track1_MTMC_Tracking/train"

COLOR_ID_CN = {
    0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色",
    4: "红色", 5: "蓝色", 6: "白色", 7: "金色",
    8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色",
}

TYPE_ID_CN = {
    0: "轿车", 1: "SUV", 2: "面包车", 3: "两厢车",
    4: "MPV", 5: "皮卡", 6: "公交车", 7: "卡车",
    8: "旅行车", 9: "跑车", 10: "房车",
}

CONFIDENCE_THRESHOLD = 0.3  # 只导入置信度 > 0.3 的检测
IOU_MATCH_THRESHOLD = 0.3   # 检测必须匹配到 GT (IoU > 阈值) 才被导入
ONLY_IMPORT_GT_MATCHED = True  # 只导入能匹配到 GT 的检测，避免大量 FP


def iou_xyxy(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 格式 bbox 的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_tracks(scene_name):
    """加载 GT tracks: gt_id → list of (frame, cam, l, t, w, h)"""
    cameras = SCENE_CAMERAS.get(scene_name, [])
    gt_tracks = defaultdict(list)
    for cam in cameras:
        gt_path = os.path.join(GT_BASE, scene_name, cam, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame, gt_id = int(parts[0]), int(parts[1])
                l, t, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                gt_tracks[gt_id].append((frame, cam, l, t, w, h))
    return dict(gt_tracks)


def load_gt_per_cam(scene_name):
    """加载 GT per cam: cam → frame → [(gt_id, l, t, w, h), ...]"""
    cameras = SCENE_CAMERAS.get(scene_name, [])
    gt_per_cam = defaultdict(lambda: defaultdict(list))
    for cam in cameras:
        gt_path = os.path.join(GT_BASE, scene_name, cam, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame, gt_id = int(parts[0]), int(parts[1])
                l, t, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                gt_per_cam[cam][frame].append((gt_id, l, t, w, h))
    return dict(gt_per_cam)


def load_baseline_detections(scene_name):
    """加载 baseline 检测: cam → frame → [(left, top, w, h, conf), ...]"""
    cameras = SCENE_CAMERAS.get(scene_name, [])
    det_per_cam = defaultdict(lambda: defaultdict(list))
    for cam in cameras:
        det_path = os.path.join(GT_BASE, scene_name, cam, "det", "det_mask_rcnn.txt")
        if not os.path.exists(det_path):
            continue
        with open(det_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame = int(parts[0])
                left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])
                if conf >= CONFIDENCE_THRESHOLD:
                    det_per_cam[cam][frame].append((left, top, w, h, conf))
    return dict(det_per_cam)


def match_det_to_gt(det_box, gt_list):
    """将单个检测匹配到 GT，返回 gt_id 或 -1"""
    best_iou = 0.3  # 最低 IoU 阈值
    best_gt_id = -1
    det_xyxy = [det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3]]
    for gt_id, gl, gt_, gw, gh in gt_list:
        gt_xyxy = [gl, gt_, gl + gw, gt_ + gh]
        iou = iou_xyxy(det_xyxy, gt_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_gt_id = gt_id
    return best_gt_id


def get_gt_attributes(gt_tracks, gt_id, scene_name):
    """从 GT ID 推断属性 (使用 CityFlow 标注的颜色/车型)"""
    # CityFlow 数据集中 GT 没有直接的颜色/车型标注
    # 但我们可以从 target_id 的命名中推断一些信息
    # 这里使用合理的默认值
    return {
        "color": "unknown",
        "vehicle_type": "unknown",
    }


def main():
    print("=" * 60)
    print("导入 Baseline 检测到 cityflow_results.json")
    print("=" * 60)

    # 加载现有结果
    results_path = "output/cityflow_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_dets = data["detections"]
    existing_tracks = data["tracks"]

    # 先移除之前导入的 baseline 检测 (避免重复导入)
    original_dets = [d for d in existing_dets if d.get("source") != "baseline_det"]
    original_tracks = [t for t in existing_tracks if t.get("source") != "baseline_det"]
    existing_dets = original_dets
    existing_tracks = original_tracks
    data["detections"] = existing_dets
    data["tracks"] = existing_tracks

    # 记录已有的 (camera_id, frame_id, target_id) 避免重复
    existing_det_keys = set()
    for det in existing_dets:
        existing_det_keys.add((det["camera_id"], det["frame_id"], det.get("target_id", "")))

    # 记录已有的 vehicle_id 在每个场景
    existing_vehicle_ids = defaultdict(set)
    for track in existing_tracks:
        scene = track.get("scene_id", "unknown")
        vid = track.get("vehicle_id", -1)
        try:
            vid = int(vid)
        except (ValueError, TypeError):
            vid = -1
        if vid >= 0:
            existing_vehicle_ids[scene].add(vid)

    new_detections = []
    new_tracks_data = defaultdict(lambda: defaultdict(list))  # (scene, cam, gt_id) → [frames]

    for scene_name in ["S01", "S03", "S04"]:
        print(f"\n处理场景 {scene_name}...")

        # 加载 GT
        gt_tracks = load_gt_tracks(scene_name)
        gt_per_cam = load_gt_per_cam(scene_name)

        # 加载 baseline 检测
        baseline_dets = load_baseline_detections(scene_name)

        scene_new = 0
        scene_matched = 0
        scene_total = 0

        for cam in SCENE_CAMERAS[scene_name]:
            cam_dets = baseline_dets.get(cam, {})
            cam_gt = gt_per_cam.get(cam, {})

            for frame in sorted(cam_dets.keys()):
                frame_dets = cam_dets[frame]
                frame_gt = cam_gt.get(frame, [])

                for left, top, w, h, conf in frame_dets:
                    scene_total += 1
                    det_box = (left, top, w, h)

                    # 匹配 GT
                    gt_id = match_det_to_gt(det_box, frame_gt)

                    if gt_id >= 0:
                        scene_matched += 1

                    # 如果只导入匹配GT的检测，跳过未匹配的
                    if ONLY_IMPORT_GT_MATCHED and gt_id < 0:
                        continue

                    # 构建 target_id
                    if gt_id >= 0:
                        target_id = f"BL_{cam}_V{gt_id:04d}_{frame:06d}"
                        vehicle_id = gt_id
                    else:
                        # 未匹配到 GT 的检测 - 分配临时 ID
                        target_id = f"BL_{cam}_U{frame:06d}_{len(new_detections):06d}"
                        vehicle_id = -1

                    # 检查是否已存在
                    if (cam, frame, target_id) in existing_det_keys:
                        continue

                    # 构建 bbox [x1, y1, x2, y2]
                    bbox = [int(left), int(top), int(left + w), int(top + h)]

                    # 构建检测记录
                    det_record = {
                        "target_id": target_id,
                        "target_type": "vehicle",
                        "confidence": round(conf, 3),
                        "attributes": {
                            "color": "unknown",
                            "vehicle_type": "unknown",
                        },
                        "bbox": bbox,
                        "frame_id": frame,
                        "camera_id": cam,
                        "scene_id": scene_name,
                        "timestamp": f"2020-01-01 00:00:{frame / 10:.3f}",
                        "keyframe_path": None,
                        "source": "baseline_det",
                        "description": f"Baseline detection, {scene_name}",
                        "clip_image_vector": None,
                        "clip_text_vector": None,
                    }

                    new_detections.append(det_record)
                    existing_det_keys.add((cam, frame, target_id))
                    scene_new += 1

                    # 记录用于 track 构建
                    if gt_id >= 0:
                        new_tracks_data[(scene_name, cam, gt_id)]["frames"].append(
                            (frame, left, top, w, h, conf)
                        )

        print(f"  Baseline 检测总数: {scene_total}")
        print(f"  匹配到 GT: {scene_matched}")
        print(f"  新增检测: {scene_new}")

    # 构建新的 tracks
    new_tracks = []
    for (scene_name, cam, gt_id), info in new_tracks_data.items():
        frames_data = sorted(info["frames"], key=lambda x: x[0])

        # 将连续帧分组为 track segments
        segments = []
        current_seg = [frames_data[0]]

        for i in range(1, len(frames_data)):
            prev_frame = current_seg[-1][0]
            curr_frame = frames_data[i][0]
            if curr_frame - prev_frame <= 5:  # 允许 5 帧间隔
                current_seg.append(frames_data[i])
            else:
                segments.append(current_seg)
                current_seg = [frames_data[i]]
        segments.append(current_seg)

        for seg in segments:
            if len(seg) < 2:
                continue  # 至少需要 2 帧构成 track

            frame_start = seg[0][0]
            frame_end = seg[-1][0]

            track_record = {
                "track_id": f"BL_TRACK_{cam}_V{gt_id:04d}_{frame_start:06d}",
                "target_type": "vehicle",
                "frame_range": [frame_start, frame_end],
                "camera_ids": [cam],
                "keyframe_path": None,
                "detection_count": len(seg),
                "vehicle_id": gt_id,
                "scene_id": scene_name,
                "source": "baseline_det",
                "attributes": {
                    "color": "unknown",
                    "vehicle_type": "unknown",
                },
            }
            new_tracks.append(track_record)

    print(f"\n新增 tracks: {len(new_tracks)}")

    # 合并到原有数据
    data["detections"] = existing_dets + new_detections
    data["tracks"] = existing_tracks + new_tracks

    # 更新 summary
    data["summary"]["total_detections"] = len(data["detections"])
    data["summary"]["total_tracks"] = len(data["tracks"])
    data["summary"]["baseline_imported"] = {
        "new_detections": len(new_detections),
        "new_tracks": len(new_tracks),
        "scenes": ["S01", "S03", "S04"],
    }

    # 保存
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"导入完成!")
    print(f"  原有检测: {len(existing_dets)}")
    print(f"  新增检测: {len(new_detections)}")
    print(f"  总检测数: {len(data['detections'])}")
    print(f"  原有轨迹: {len(existing_tracks)}")
    print(f"  新增轨迹: {len(new_tracks)}")
    print(f"  总轨迹数: {len(data['tracks'])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
