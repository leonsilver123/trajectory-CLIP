"""
scripts.quick_demo - 快速演示脚本

从 AICity22 Track1 MTMC 数据集中取视频帧，用 YOLO 检测所有车辆和行人，
在原图上画检测框，保存到 output/demo_detections/，裁剪目标到 output/crops/，
并生成前端需要的 results.json。

运行方式:
    python -m scripts.quick_demo
    或
    python scripts/quick_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import ImageFont, ImageDraw, Image as PILImage
import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 切换到项目目录（YOLO 模型路径是相对路径）
os.chdir(_PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 配置
# ============================================================

# COCO 类别 ID -> 系统类别映射（与 detector.py 保持一致）
CLASS_MAPPING: Dict[int, str] = {
    0: "pedestrian",          # person
    1: "non_motor_vehicle",   # bicycle
    2: "vehicle",             # car
    3: "non_motor_vehicle",   # motorcycle
    5: "vehicle",             # bus
    7: "vehicle",             # truck
}

# 类别中文标签
TYPE_LABELS = {
    "vehicle": "车辆",
    "pedestrian": "行人",
    "non_motor_vehicle": "非机动车",
}

# 检测框颜色（BGR 格式，OpenCV 使用）
# vehicle=绿色, pedestrian=蓝色, non_motor=黄色
TYPE_COLORS = {
    "vehicle": (0, 255, 0),           # 绿色
    "pedestrian": (255, 0, 0),        # 蓝色
    "non_motor_vehicle": (0, 255, 255),  # 黄色
}

# 输入输出路径
_DATASET_DIR = _PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking"
_DATASET_IMG_DIR = _PROJECT_ROOT / "data" / "VisDrone2019-DET" / "images"  # 回退
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_DETECTIONS_DIR = _OUTPUT_DIR / "demo_detections"
_CROPS_DIR = _OUTPUT_DIR / "crops"
_RESULTS_JSON = _OUTPUT_DIR / "results.json"

# 检测参数
CONFIDENCE_THRESHOLD = 0.35
NUM_IMAGES = 10
MODEL_NAME = "yolov8x.pt"


def find_dataset_images() -> List[Path]:
    """从 AICity22 数据集中找到可用的图片，回退到 VisDrone"""
    # 尝试从 AICity22 视频文件中提取帧
    if _DATASET_DIR.exists():
        # 从第一个可用的视频中提取帧
        for split in ["train", "validation"]:
            split_dir = _DATASET_DIR / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                for cam_dir in sorted(scene_dir.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    video_path = cam_dir / "vdo.avi"
                    if video_path.exists():
                        # 从视频中提取帧作为图片
                        return _extract_frames_from_video(video_path, NUM_IMAGES)

    # 回退到 VisDrone 图片
    if _DATASET_IMG_DIR.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted(
            p for p in _DATASET_IMG_DIR.iterdir()
            if p.suffix.lower() in exts
        )
        return images

    print(f"[ERROR] 数据集目录不存在: {_DATASET_DIR} 且 {_DATASET_IMG_DIR}")
    return []


def _extract_frames_from_video(video_path: Path, num_frames: int) -> List[Path]:
    """从视频中提取帧并保存为临时图片"""
    frames_dir = _OUTPUT_DIR / "temp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # 均匀采样
    sample_indices = np.linspace(0, min(total_frames - 1, 500), min(num_frames, total_frames), dtype=int)
    extracted = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        fname = frames_dir / f"frame_{int(idx):06d}.jpg"
        cv2.imwrite(str(fname), frame)
        extracted.append(fname)
    cap.release()
    return extracted


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """在原图上绘制检测框和标签（支持中文）"""
    annotated = frame.copy()

    # 加载中文字体（多重回退）
    pil_font = None
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
               r"C:\Windows\Fonts\msyhbd.ttc"]:
        try:
            pil_font = ImageFont.truetype(fp, 18)
            break
        except Exception:
            continue
    if pil_font is None:
        pil_font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        target_type = det["target_type"]
        confidence = det["confidence"]
        label = TYPE_LABELS.get(target_type, target_type)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = TYPE_COLORS.get(target_type, (200, 200, 200))

        # 画检测框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 用 PIL 绘制中文标签
        text = f"{label} {confidence:.0%}"
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # 标签背景
        bbox_text = draw.textbbox((0, 0), text, font=pil_font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 4, y1], fill=color)
        # 标签文字
        draw.text((x1 + 2, y1 - th - 6), text, fill=(255, 255, 255), font=pil_font)
        annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return annotated


def crop_target(frame: np.ndarray, bbox: List[float], padding: float = 0.1) -> np.ndarray:
    """从帧中裁剪目标区域"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0] - (bbox[2] - bbox[0]) * padding))
    y1 = max(0, int(bbox[1] - (bbox[3] - bbox[1]) * padding))
    x2 = min(w, int(bbox[2] + (bbox[2] - bbox[0]) * padding))
    y2 = min(h, int(bbox[3] + (bbox[3] - bbox[1]) * padding))
    return frame[y1:y2, x1:x2].copy()


def _build_attributes(target_type: str, confidence: float) -> Dict[str, Any]:
    """根据目标类型构建属性"""
    attrs: Dict[str, Any] = {}
    if target_type == "vehicle":
        attrs["color"] = "unknown"
        attrs["vehicle_type"] = "car"
    elif target_type == "pedestrian":
        attrs["gender"] = "unknown"
        attrs["age_group"] = "adult"
    elif target_type == "non_motor_vehicle":
        attrs["vehicle_type"] = "bicycle"
    attrs["detection_confidence"] = f"{confidence:.0%}"
    return attrs


def _check_cuda() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_demo() -> None:
    """执行快速演示"""
    print("=" * 60)
    print("  交通风险感知系统 - 快速演示")
    print("  数据集: AICity22 Track1 MTMC Tracking")
    print("=" * 60)
    print()

    # 1. 查找数据集图片
    print("[1/5] 查找数据集图片...")
    all_images = find_dataset_images()
    if not all_images:
        print("  [ERROR] 未找到数据集图片，请确认 cityflow/AICity22_Track1_MTMC_Tracking/ 目录")
        return

    selected_images = all_images[:NUM_IMAGES]
    print(f"  找到 {len(all_images)} 张图片，选取 {len(selected_images)} 张")
    print()

    # 2. 加载 YOLO 模型
    print("[2/5] 加载 YOLO 模型...")
    model_path = _PROJECT_ROOT / MODEL_NAME
    if not model_path.exists():
        print(f"  [ERROR] 模型文件不存在: {model_path}")
        return

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        device = "cuda" if _check_cuda() else "cpu"
        print(f"  模型加载完成，使用设备: {device}")
    except Exception as e:
        print(f"  [ERROR] 模型加载失败: {e}")
        return
    print()

    # 3. 创建输出目录
    _DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[3/5] 输出目录:")
    print(f"  标注图片: {_DETECTIONS_DIR}")
    print(f"  裁剪目标: {_CROPS_DIR}")
    print(f"  结果 JSON: {_RESULTS_JSON}")
    print()

    # 4. 逐张检测
    print("[4/5] 开始检测...")
    print("-" * 60)

    all_detections: List[Dict[str, Any]] = []
    type_totals: Counter = Counter()
    det_id_counter = 0

    for idx, img_path in enumerate(selected_images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [{idx+1}/{len(selected_images)}] 跳过: {img_path.name}")
            continue

        h, w = frame.shape[:2]
        rel_path = str(img_path.relative_to(_PROJECT_ROOT)).replace("\\", "/")

        # YOLO 推理
        yolo_results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
            imgsz=640,
            device=device,
            verbose=False,
        )

        # 解析检测结果
        frame_detections: List[Dict[str, Any]] = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                target_type = CLASS_MAPPING.get(cls_id)
                if target_type is None:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())
                det_id_counter += 1
                target_id = f"DET_{det_id_counter:04d}"

                # 裁剪目标
                crop = crop_target(frame, xyxy.tolist())
                crop_filename = f"{target_id}.jpg"
                crop_path = _CROPS_DIR / crop_filename
                if crop.size > 0:
                    cv2.imwrite(str(crop_path), crop)

                frame_detections.append({
                    "target_id": target_id,
                    "target_type": target_type,
                    "confidence": round(conf, 4),
                    "attributes": _build_attributes(target_type, conf),
                    "bbox": [round(float(v), 1) for v in xyxy],
                    "frame_id": idx,
                    "image_path": rel_path,
                    "keyframe_path": f"output/crops/{crop_filename}",
                    "detection_image_path": "",  # 稍后填充
                })

                type_totals[target_type] += 1

        # 在原图上画检测框
        annotated = draw_detections(frame, frame_detections)
        det_filename = f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(_DETECTIONS_DIR / det_filename), annotated)

        # 回填 detection_image_path
        for det in frame_detections:
            det["detection_image_path"] = f"output/demo_detections/{det_filename}"

        all_detections.extend(frame_detections)

        # 打印单张结果
        frame_type_str = ", ".join(
            f"{TYPE_LABELS.get(t, t)}:{c}"
            for t, c in sorted(
                {d["target_type"]: sum(1 for dd in frame_detections if dd["target_type"] == d["target_type"])
                 for d in frame_detections}.items()
            )
        ) if frame_detections else "无"
        print(f"  [{idx+1}/{len(selected_images)}] {img_path.name}: "
              f"{len(frame_detections)} 个目标 ({frame_type_str}) -> {det_filename}")

    # 5. 生成 results.json
    print()
    print("[5/5] 生成 results.json...")

    # 简化的 tracks：每个检测视为一个独立 tracklet（无跟踪器）
    tracks = []
    for det in all_detections:
        tracks.append({
            "track_id": f"TRACK_{det['target_id'].split('_')[1]}",
            "target_type": det["target_type"],
            "frame_range": [det["frame_id"], det["frame_id"]],
            "keyframe_path": det["keyframe_path"],
            "detection_count": 1,
        })

    total_dets = len(all_detections)
    results_json = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "dataset": "AICity22-Track1-MTMC",
        "detections": all_detections,
        "tracks": tracks,
        "summary": {
            "total_frames": len(selected_images),
            "total_detections": total_dets,
            "total_tracks": len(tracks),
            "vehicle_count": type_totals.get("vehicle", 0),
            "pedestrian_count": type_totals.get("pedestrian", 0),
            "non_motor_vehicle_count": type_totals.get("non_motor_vehicle", 0),
        },
    }

    with open(_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print()
    print("=" * 60)
    print("  检测结果摘要")
    print("=" * 60)
    print(f"  数据集: AICity22 Track1 MTMC Tracking")
    print(f"  处理图片数: {len(selected_images)}")
    print(f"  检测目标总数: {total_dets}")
    print(f"    - 车辆 (vehicle): {type_totals.get('vehicle', 0)}")
    print(f"    - 行人 (pedestrian): {type_totals.get('pedestrian', 0)}")
    print(f"    - 非机动车 (non_motor_vehicle): {type_totals.get('non_motor_vehicle', 0)}")
    print(f"  标注图片: {_DETECTIONS_DIR}")
    print(f"  裁剪目标: {_CROPS_DIR} ({det_id_counter} 个)")
    print(f"  结果 JSON: {_RESULTS_JSON}")
    print("=" * 60)
    print()
    print("[OK] 演示完成！可以启动前端查看真实检测结果：")
    print(f"  streamlit run {_PROJECT_ROOT / 'frontend' / 'app.py'}")
    print()


if __name__ == "__main__":
    run_demo()
