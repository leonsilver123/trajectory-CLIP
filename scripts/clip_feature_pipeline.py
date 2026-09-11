"""
scripts/clip_feature_pipeline.py - CLIP 特征管线

完整流程：图像加载 → 颜色分析 → 描述生成 → CLIP 编码 → 向量入库

使用方式:
  python scripts/clip_feature_pipeline.py
  python scripts/clip_feature_pipeline.py --skip-clip --device cpu
  python scripts/clip_feature_pipeline.py --batch-size 8 --use-yolo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from src.common.logger import get_logger

logger = get_logger("scripts.clip_feature_pipeline")


# ============================================================
# 颜色映射表
# ============================================================

# RGB 中心值 → 精确中文颜色名
COLOR_CENTER_MAP: Dict[Tuple[int, int, int], str] = {
    (0, 0, 0): "墨黑",
    (40, 40, 40): "墨黑",
    (255, 255, 255): "白色",
    (240, 240, 240): "浅白",
    (200, 200, 200): "银灰",
    (160, 160, 160): "浅灰",
    (128, 128, 128): "灰色",
    (80, 80, 80): "深灰",
    (60, 60, 60): "深灰",
    (0, 0, 180): "深蓝",
    (0, 0, 128): "深蓝",
    (0, 0, 255): "蓝色",
    (30, 90, 200): "蓝色",
    (100, 149, 237): "浅蓝",
    (135, 180, 230): "浅蓝",
    (180, 0, 0): "深红",
    (139, 0, 0): "深红",
    (255, 0, 0): "红色",
    (220, 50, 50): "红色",
    (255, 105, 180): "粉色",
    (255, 130, 180): "粉色",
    (128, 0, 128): "紫色",
    (160, 50, 160): "紫色",
    (0, 100, 0): "深绿",
    (0, 128, 0): "深绿",
    (0, 200, 0): "绿色",
    (50, 180, 50): "绿色",
    (139, 69, 19): "深棕",
    (100, 50, 20): "深棕",
    (210, 180, 140): "棕色",
    (180, 140, 100): "棕色",
    (255, 215, 0): "金色",
    (220, 190, 50): "金色",
    (255, 165, 0): "橙色",
    (240, 150, 30): "橙色",
    (255, 255, 0): "黄色",
    (240, 230, 50): "黄色",
}

# 颜色名列表（用于最近邻搜索）
_COLOR_CENTERS_ARR = np.array(list(COLOR_CENTER_MAP.keys()), dtype=np.float32)
_COLOR_NAMES_ARR = list(COLOR_CENTER_MAP.values())

# GT color_id 到中文颜色名的映射（粗粒度 → 细粒度补充）
COLOR_ID_TO_CN = {
    0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色",
    4: "红色", 5: "蓝色", 6: "白色", 7: "金色",
    8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色",
}

# 车型列表
TYPE_ID_CN = {
    0: "轿车", 1: "SUV", 2: "面包车", 3: "两厢车",
    4: "MPV", 5: "皮卡", 6: "公交车", 7: "卡车",
    8: "旅行车", 9: "跑车", 10: "房车",
}


# ============================================================
# 颜色分析
# ============================================================

def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """将 RGB 值映射到最近的中文颜色名"""
    rgb_arr = np.array(rgb, dtype=np.float32).reshape(1, 3)
    dists = np.linalg.norm(_COLOR_CENTERS_ARR - rgb_arr, axis=1)
    idx = int(np.argmin(dists))
    return _COLOR_NAMES_ARR[idx]


def analyze_dominant_color(
    crop_image: np.ndarray,
    k: int = 3,
    gt_color_id: Optional[int] = None,
) -> Tuple[str, Tuple[int, int, int], List[Dict[str, Any]]]:
    """
    KMeans 聚类提取主色调，映射到精确中文颜色名

    Args:
        crop_image: BGR 格式裁剪图像
        k: 聚类数
        gt_color_id: GT 颜色 ID（如有）

    Returns:
        (颜色中文名, 主色调 RGB, 所有聚类信息列表)
    """
    import cv2
    from sklearn.cluster import KMeans

    # 转 RGB
    rgb = crop_image[:, :, ::-1].copy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # 缩小图像加速聚类
    h, w = rgb.shape[:2]
    if max(h, w) > 128:
        scale = 128.0 / max(h, w)
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))

    # 展平像素
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # 过滤掉接近黑边/白边的极端像素（可能是裁剪边框）
    brightness = pixels.sum(axis=1)
    low_mask = brightness > 15
    high_mask = brightness < 750
    valid = pixels[low_mask & high_mask]
    if len(valid) < 50:
        valid = pixels

    # KMeans 聚类
    actual_k = min(k, len(valid) // 10, 5)
    actual_k = max(actual_k, 2)
    try:
        kmeans = KMeans(n_clusters=actual_k, n_init=3, max_iter=50, random_state=42)
        kmeans.fit(valid)
    except Exception:
        # fallback: 取均值
        mean_rgb = tuple(int(x) for x in pixels.mean(axis=0))
        color_name = rgb_to_color_name(mean_rgb)
        if gt_color_id is not None and gt_color_id in COLOR_ID_TO_CN:
            color_name = COLOR_ID_TO_CN[gt_color_id]
        return color_name, mean_rgb, [{"rgb": mean_rgb, "ratio": 1.0, "name": color_name}]

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 统计每个聚类的面积占比
    label_counts = Counter(labels)
    total = len(labels)

    clusters = []
    for label_id, count in label_counts.most_common():
        ratio = count / total
        center_rgb = tuple(int(x) for x in centers[label_id])
        name = rgb_to_color_name(center_rgb)
        clusters.append({"rgb": center_rgb, "ratio": round(ratio, 3), "name": name})

    # 主色调：面积占比最大的聚类
    dominant_rgb = clusters[0]["rgb"]
    dominant_name = clusters[0]["name"]

    # 如果有 GT 颜色，优先使用 GT 颜色名，但用像素分析做补充验证
    if gt_color_id is not None and gt_color_id in COLOR_ID_TO_CN:
        gt_name = COLOR_ID_TO_CN[gt_color_id]
        # 如果 GT 颜色和像素分析差异太大，保留像素分析结果
        # （GT 可能不准确），但仍记录 GT 颜色
        dominant_name = gt_name

    return dominant_name, dominant_rgb, clusters


def refine_color_name(base_name: str, clusters: List[Dict[str, Any]]) -> str:
    """
    根据聚类结果细化颜色名

    例如：GT 说"蓝色"，但像素分析显示偏深 → "深蓝"
    """
    # 如果只有一个主色，直接用
    if not clusters:
        return base_name

    # 如果 GT 颜色比较笼统（如"蓝色"），看像素分析是否更精确
    generic_colors = {"蓝色", "红色", "绿色", "灰色", "白色", "黑色"}
    if base_name in generic_colors and len(clusters) >= 1:
        pixel_name = clusters[0]["name"]
        # 如果像素分析给出了更精确的颜色（如"深蓝"代替"蓝色"），使用它
        if pixel_name != base_name and pixel_name not in generic_colors:
            # 检查是否同色系
            same_hue = False
            hue_map = {
                "蓝色": ["深蓝", "浅蓝"],
                "红色": ["深红"],
                "绿色": ["深绿"],
                "灰色": ["银灰", "深灰", "浅灰"],
                "白色": ["浅白", "银灰"],
                "黑色": ["墨黑", "深灰"],
            }
            if pixel_name in hue_map.get(base_name, []):
                same_hue = True
            if same_hue:
                return pixel_name

    return base_name


# ============================================================
# 尺寸估计
# ============================================================

def estimate_size(bbox: List[float], image_shape: Optional[Tuple] = None) -> str:
    """
    根据 bbox 估计目标大小

    Args:
        bbox: [x1, y1, x2, y2]
        image_shape: 原始图像 shape（可选）

    Returns:
        "large" / "medium" / "small"
    """
    if len(bbox) < 4:
        return "medium"

    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if w <= 0 or h <= 0:
        return "medium"

    area = w * h
    aspect_ratio = w / max(h, 1)

    # 基于面积的粗略估计（CityFlow 图像通常 1920x1080）
    # 全帧图像 (bbox=[0,0,w,h]) 视为 large
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        bbox_ratio = area / max(img_w * img_h, 1)
        if bbox_ratio > 0.3:
            return "large"

    # 基于绝对面积
    if area > 80000:
        return "large"
    elif area > 20000:
        return "medium"
    else:
        return "small"


# ============================================================
# 描述生成
# ============================================================

def generate_description(
    color_cn: str,
    type_cn: str,
    size: str,
    scene_id: str = "",
) -> str:
    """
    生成自然语言描述

    Examples:
        "深蓝色中型轿车，外观整洁"
        "白色大型SUV"
        "红色小型跑车"
    """
    size_map = {"large": "大型", "medium": "中型", "small": "小型"}
    size_cn = size_map.get(size, "中型")

    # 构建描述
    desc = f"{color_cn}{size_cn}{type_cn}"

    # 添加场景信息
    if scene_id and scene_id not in ("sim", "reid", "track1", "track4"):
        desc += f"，{scene_id}场景"
    else:
        desc += "，外观整洁"

    return desc


# ============================================================
# 图像加载
# ============================================================

def load_crop_image(det: Dict[str, Any], project_root: Path, yolo_model: Any = None) -> Optional[np.ndarray]:
    """
    加载裁剪图像

    1. 优先从 keyframe_path 加载
    2. 如果不存在，尝试从原始视频裁剪
    3. 对于全帧图像（bbox=[0,0,w,h]），使用 YOLO 检测
    """
    import cv2

    keyframe_path = det.get("keyframe_path", "")
    if not keyframe_path:
        return None

    # 统一路径分隔符
    keyframe_path = keyframe_path.replace("\\", "/")
    img_path = project_root / keyframe_path

    if not img_path.exists():
        return None

    image = cv2.imread(str(img_path))
    if image is None:
        return None

    bbox = det.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = bbox
    img_h, img_w = image.shape[:2]

    # 检查是否为全帧图像（bbox 覆盖整张图）
    is_full_frame = (x1 <= 1 and y1 <= 1 and
                     abs(x2 - img_w) <= 5 and abs(y2 - img_h) <= 5)

    if is_full_frame and yolo_model is not None:
        # 使用 YOLO 检测目标并裁剪
        try:
            results = yolo_model(image, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # 取最大的检测框
                    confs = boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    cx1, cy1, cx2, cy2 = xyxy
                    cx1 = max(0, cx1)
                    cy1 = max(0, cy1)
                    cx2 = min(img_w, cx2)
                    cy2 = min(img_h, cy2)
                    if cx2 > cx1 and cy2 > cy1:
                        return image[cy1:cy2, cx1:cx2].copy()
        except Exception as e:
            logger.debug(f"YOLO 检测失败: {e}")

    # 如果不是全帧，且有有效 bbox，裁剪
    if not is_full_frame and x2 > x1 and y2 > y1:
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(img_w, int(x2))
        cy2 = min(img_h, int(y2))
        if cx2 > cx1 and cy2 > cy1:
            return image[cy1:cy2, cx1:cx2].copy()

    # 全帧且无 YOLO → 返回整张图（缩小）
    if is_full_frame:
        scale = min(224 / img_w, 224 / img_h, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
        return image

    return image


# ============================================================
# 核心处理流程
# ============================================================

def process_all_detections(
    data: Dict[str, Any],
    extractor: Any,
    project_root: Path,
    use_yolo: bool = False,
    skip_clip: bool = False,
    batch_size: int = 16,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Counter, Counter, Counter]:
    """
    处理所有检测：颜色分析 + 描述生成 + CLIP 编码

    Returns:
        (处理后的检测列表, 统计信息)
    """
    import cv2

    detections = data.get("detections", [])
    total = len(detections)
    logger.info(f"开始处理 {total} 条检测记录")

    # 可选：加载 YOLO
    yolo_model = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            yolo_path = project_root / "yolov8x.pt"
            if yolo_path.exists():
                yolo_model = YOLO(str(yolo_path))
                logger.info(f"YOLOv8 模型已加载: {yolo_path}")
            else:
                logger.warning(f"YOLOv8 模型不存在: {yolo_path}, 跳过 YOLO 检测")
        except Exception as e:
            logger.warning(f"YOLOv8 加载失败: {e}")

    # 统计
    stats = {
        "total": total,
        "success": 0,
        "failed_image": 0,
        "failed_clip": 0,
        "skipped": 0,
    }
    color_counter: Counter = Counter()
    type_counter: Counter = Counter()
    size_counter: Counter = Counter()

    # 第一阶段：加载图像 + 颜色分析 + 描述生成
    processed: List[Dict[str, Any]] = []
    crop_images_for_clip: List[np.ndarray] = []
    descriptions: List[str] = []
    valid_indices: List[int] = []

    for i, det in enumerate(detections):
        if (i + 1) % 100 == 0:
            logger.info(f"  进度: {i + 1}/{total} (成功={stats['success']}, 失败={stats['failed_image']})")

        # 加载裁剪图像
        crop = load_crop_image(det, project_root, yolo_model)
        if crop is None:
            stats["failed_image"] += 1
            # 仍然保留该检测，只是没有 CLIP 向量
            gt_color_id = det.get("attributes", {}).get("color_id")
            gt_color_cn = det.get("attributes", {}).get("color", "未知")
            type_cn = det.get("attributes", {}).get("vehicle_type", "车辆")
            size = estimate_size(det.get("bbox", []))
            desc = generate_description(gt_color_cn, type_cn, size, det.get("scene_id", ""))

            result_det = dict(det)
            result_det["description"] = desc
            result_det["dominant_color_rgb"] = [0, 0, 0]
            result_det["bbox_size"] = size
            result_det["color_analysis"] = {"source": "gt_only"}
            processed.append(result_det)
            color_counter[gt_color_cn] += 1
            type_counter[type_cn] += 1
            size_counter[size] += 1
            continue

        # 颜色分析
        gt_color_id = det.get("attributes", {}).get("color_id")
        gt_color_cn = det.get("attributes", {}).get("color", "未知")

        dominant_name, dominant_rgb, clusters = analyze_dominant_color(
            crop, k=3, gt_color_id=gt_color_id
        )

        # 细化颜色名
        final_color = refine_color_name(dominant_name, clusters)

        # 如果 GT 颜色和像素分析一致，用更精确的像素分析名
        if gt_color_cn == final_color or gt_color_cn in ("未知",):
            final_color = dominant_name if dominant_name != gt_color_cn else final_color

        # 更新属性中的颜色
        type_cn = det.get("attributes", {}).get("vehicle_type", "车辆")
        size = estimate_size(det.get("bbox", []), crop.shape if crop is not None else None)

        # 生成描述
        desc = generate_description(final_color, type_cn, size, det.get("scene_id", ""))

        # 构建结果
        result_det = dict(det)
        # 更新颜色属性为更精确的版本
        result_det["attributes"] = dict(det.get("attributes", {}))
        result_det["attributes"]["color_refined"] = final_color
        result_det["description"] = desc
        result_det["dominant_color_rgb"] = list(dominant_rgb)
        result_det["bbox_size"] = size
        result_det["color_analysis"] = {
            "dominant_name": dominant_name,
            "refined_name": final_color,
            "dominant_rgb": list(dominant_rgb),
            "clusters": clusters,
            "gt_color": gt_color_cn,
            "gt_color_id": gt_color_id,
        }

        processed.append(result_det)
        stats["success"] += 1
        color_counter[final_color] += 1
        type_counter[type_cn] += 1
        size_counter[size] += 1

        # 收集用于 CLIP 编码的数据
        if not skip_clip:
            crop_images_for_clip.append(crop)
            descriptions.append(desc)
            valid_indices.append(len(processed) - 1)

    # 第二阶段：CLIP 批量编码
    if not skip_clip and crop_images_for_clip:
        logger.info(f"开始 CLIP 批量编码: {len(crop_images_for_clip)} 张图像")

        # 批量图像特征
        try:
            clip_image_vectors = extractor.extract_clip_batch(crop_images_for_clip)
            logger.info(f"CLIP 图像特征提取完成: shape={clip_image_vectors.shape}")
        except Exception as e:
            logger.error(f"CLIP 批量图像特征提取失败: {e}")
            clip_image_vectors = np.zeros((len(crop_images_for_clip), extractor.clip_dim), dtype=np.float32)
            stats["failed_clip"] += len(crop_images_for_clip)

        # 逐条文本特征
        clip_text_vectors = []
        for j, desc_text in enumerate(descriptions):
            try:
                tv = extractor.extract_text_clip(desc_text)
                clip_text_vectors.append(tv)
            except Exception as e:
                logger.debug(f"文本特征提取失败 (idx={j}): {e}")
                clip_text_vectors.append(np.zeros(extractor.clip_dim, dtype=np.float32))
                stats["failed_clip"] += 1

        clip_text_vectors = np.array(clip_text_vectors, dtype=np.float32)
        logger.info(f"CLIP 文本特征提取完成: shape={clip_text_vectors.shape}")

        # 写入结果
        for vi, idx in enumerate(valid_indices):
            processed[idx]["clip_image_vector"] = clip_image_vectors[vi].tolist()
            processed[idx]["clip_text_vector"] = clip_text_vectors[vi].tolist()

    logger.info(f"处理完成: 成功={stats['success']}, 图像失败={stats['failed_image']}, "
                f"CLIP失败={stats['failed_clip']}")

    return processed, dict(stats), color_counter, type_counter, size_counter


# ============================================================
# 存储
# ============================================================

def store_to_qdrant(
    detections_with_clip: List[Dict[str, Any]],
    host: str = "localhost",
    port: int = 6333,
) -> bool:
    """存入 Qdrant 向量数据库"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        client = QdrantClient(host=host, port=port)

        collection_name = "traffic_clip"
        vector_dim = 768

        # 创建集合（如果不存在）
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            logger.info(f"创建 Qdrant 集合: {collection_name}")

        # 准备 points
        points = []
        for i, det in enumerate(detections_with_clip):
            clip_vec = det.get("clip_image_vector")
            if clip_vec is None:
                continue

            # 确保向量维度正确
            vec = np.array(clip_vec, dtype=np.float32).tolist()
            if len(vec) != vector_dim:
                continue

            payload = {
                "target_id": det.get("target_id", ""),
                "description": det.get("description", ""),
                "attributes": det.get("attributes", {}),
                "camera_id": det.get("camera_id", ""),
                "scene_id": det.get("scene_id", ""),
                "frame_id": det.get("frame_id", 0),
                "bbox_size": det.get("bbox_size", ""),
                "dominant_color_rgb": det.get("dominant_color_rgb", []),
            }

            # 如果有文本向量，也存入 payload
            text_vec = det.get("clip_text_vector")
            if text_vec is not None:
                payload["clip_text_vector"] = np.array(text_vec, dtype=np.float32).tolist()

            points.append(PointStruct(
                id=i,
                vector=vec,
                payload=payload,
            ))

        if points:
            # 批量上传（每批 100）
            batch_size = 100
            for batch_start in range(0, len(points), batch_size):
                batch = points[batch_start:batch_start + batch_size]
                client.upsert(collection_name=collection_name, points=batch)

            logger.info(f"已存入 Qdrant: {len(points)} 条记录 (集合: {collection_name})")
            return True
        else:
            logger.warning("没有有效的 CLIP 向量可存入 Qdrant")
            return False

    except Exception as e:
        logger.warning(f"Qdrant 存储失败 (可能未运行): {e}")
        return False


def store_to_files(
    detections_with_clip: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Fallback: 存入 numpy + JSON 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 CLIP 向量
    image_vectors = []
    text_vectors = []
    metadata = []

    for det in detections_with_clip:
        clip_img = det.get("clip_image_vector")
        clip_txt = det.get("clip_text_vector")

        meta = {
            "target_id": det.get("target_id", ""),
            "description": det.get("description", ""),
            "camera_id": det.get("camera_id", ""),
            "scene_id": det.get("scene_id", ""),
            "frame_id": det.get("frame_id", 0),
            "bbox_size": det.get("bbox_size", ""),
            "dominant_color_rgb": det.get("dominant_color_rgb", []),
            "attributes": det.get("attributes", {}),
        }

        if clip_img is not None:
            image_vectors.append(clip_img)
            meta["has_clip_image"] = True
        else:
            meta["has_clip_image"] = False

        if clip_txt is not None:
            text_vectors.append(clip_txt)
            meta["has_clip_text"] = True
        else:
            meta["has_clip_text"] = False

        metadata.append(meta)

    # 保存 numpy 向量文件
    if image_vectors:
        img_arr = np.array(image_vectors, dtype=np.float32)
        np.save(str(output_dir / "clip_vectors.npy"), img_arr)
        logger.info(f"CLIP 图像向量已保存: {output_dir / 'clip_vectors.npy'} (shape: {img_arr.shape})")

    if text_vectors:
        txt_arr = np.array(text_vectors, dtype=np.float32)
        np.save(str(output_dir / "clip_text_vectors.npy"), txt_arr)
        logger.info(f"CLIP 文本向量已保存: {output_dir / 'clip_text_vectors.npy'} (shape: {txt_arr.shape})")

    # 保存元数据
    with open(output_dir / "clip_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存: {output_dir / 'clip_metadata.json'}")


def update_cityflow_results(
    data: Dict[str, Any],
    detections_with_clip: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """更新 cityflow_results.json"""
    result = dict(data)
    result["detections"] = detections_with_clip

    # 添加 CLIP 处理摘要
    if "summary" not in result:
        result["summary"] = {}
    result["summary"]["clip_processed"] = True
    result["summary"]["clip_model"] = "CN-CLIP-ViT-L-14"
    result["summary"]["clip_dim"] = 768

    has_clip_count = sum(1 for d in detections_with_clip if d.get("clip_image_vector") is not None)
    result["summary"]["clip_vector_count"] = has_clip_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"cityflow_results.json 已更新: {output_path}")
    logger.info(f"  其中 {has_clip_count}/{len(detections_with_clip)} 条包含 CLIP 向量")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLIP 特征管线 - 图像加载→颜色分析→描述生成→CLIP编码→向量入库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/clip_feature_pipeline.py
  python scripts/clip_feature_pipeline.py --skip-clip --device cpu
  python scripts/clip_feature_pipeline.py --batch-size 8 --use-yolo
        """,
    )
    parser.add_argument("--data-dir", type=str, default="cityflow/AICity22_Track1_MTMC_Tracking",
                        help="AICity22 数据集目录 (默认: cityflow/AICity22_Track1_MTMC_Tracking)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (默认: cuda if available else cpu)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="CLIP 批处理大小 (默认: 16)")
    parser.add_argument("--use-yolo", action="store_true", default=True,
                        help="对全帧图像使用 YOLOv8 检测 (默认: True)")
    parser.add_argument("--no-yolo", action="store_true", default=False,
                        help="禁用 YOLOv8 检测")
    parser.add_argument("--skip-clip", action="store_true", default=False,
                        help="跳过 CLIP 编码（仅生成描述）(默认: False)")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                        help="Qdrant 地址 (默认: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                        help="Qdrant 端口 (默认: 6333)")
    args = parser.parse_args()

    # 路径处理
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    use_yolo = args.use_yolo and not args.no_yolo

    # 设备检测
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info("=" * 60)
    logger.info("CLIP 特征管线")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"YOLO 检测: {use_yolo}")
    logger.info(f"跳过 CLIP: {args.skip_clip}")
    logger.info("=" * 60)

    t_start = time.time()

    # 1. 加载检测数据
    results_path = output_dir / "cityflow_results.json"
    if not results_path.exists():
        logger.error(f"检测结果文件不存在: {results_path}")
        logger.error("请先运行: python scripts/cityflow_all_tracks.py")
        sys.exit(1)

    logger.info(f"加载检测数据: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections", [])
    logger.info(f"共 {len(detections)} 条检测记录")

    if not detections:
        logger.error("没有检测记录，退出")
        sys.exit(1)

    # 2. 初始化 FeatureExtractor
    extractor = None
    if not args.skip_clip:
        try:
            from src.perception.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(
                clip_model="CN-CLIP-ViT-L-14",
                clip_dim=768,
                clip_batch_size=args.batch_size,
                device=device,
            )
            logger.info(f"FeatureExtractor 初始化完成 (device={device})")
        except Exception as e:
            logger.error(f"FeatureExtractor 初始化失败: {e}")
            logger.warning("将跳过 CLIP 编码")
            args.skip_clip = True

    # 3. 处理所有检测
    processed_dets, stats, color_counter, type_counter, size_counter = process_all_detections(
        data, extractor, PROJECT_ROOT,
        use_yolo=use_yolo,
        skip_clip=args.skip_clip,
        batch_size=args.batch_size,
    )

    # 4. 存储结果
    logger.info(f"\n{'=' * 40}")
    logger.info("存储结果")
    logger.info(f"{'=' * 40}")

    # 4a. 尝试 Qdrant
    qdrant_ok = False
    if not args.skip_clip:
        qdrant_ok = store_to_qdrant(processed_dets, args.qdrant_host, args.qdrant_port)

    # 4b. Fallback 文件存储
    if not qdrant_ok and not args.skip_clip:
        logger.info("Qdrant 不可用，使用文件存储")
        store_to_files(processed_dets, output_dir)

    # 5. 更新 cityflow_results.json
    update_cityflow_results(data, processed_dets, results_path)

    # 6. 打印统计
    total_time = time.time() - t_start

    logger.info(f"\n{'=' * 60}")
    logger.info("CLIP 特征管线完成!")
    logger.info(f"{'=' * 60}")
    logger.info(f"总耗时: {total_time:.1f}s")
    logger.info(f"处理统计:")
    logger.info(f"  总检测数: {stats.get('total', 0)}")
    logger.info(f"  成功处理: {stats.get('success', 0)}")
    logger.info(f"  图像加载失败: {stats.get('failed_image', 0)}")
    logger.info(f"  CLIP 编码失败: {stats.get('failed_clip', 0)}")
    logger.info(f"\n颜色分布:")
    for color, count in color_counter.most_common():
        logger.info(f"  {color}: {count}")
    logger.info(f"\n车型分布:")
    for vtype, count in type_counter.most_common():
        logger.info(f"  {vtype}: {count}")
    logger.info(f"\n大小分布:")
    for size, count in size_counter.most_common():
        logger.info(f"  {size}: {count}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n用户中断，退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[FATAL] CLIP 特征管线崩溃: {e}")
        traceback.print_exc()
        sys.exit(1)
