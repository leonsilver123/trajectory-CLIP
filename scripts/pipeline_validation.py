"""
scripts/pipeline_validation.py - 全流程管线验证脚本

串联完整算法管线并用真实数据集跑通:
  视频帧/图片序列读取 → 目标检测 (YOLO) → 单摄多目标跟踪 (ByteTrack)
  → 属性识别 → 质量评分 → ReID 特征提取 → CLIP 特征提取
  → Tracklet 生成 → 轨迹关联 → 输出处理结果
"""

import sys
import os
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 解决 OpenMP 重复加载冲突（PyTorch + NumPy/OpenCV）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = r"H:\trajectory CLIP"
sys.path.insert(0, PROJECT_ROOT)

# 切换到项目目录（YOLO 模型路径是相对路径）
os.chdir(PROJECT_ROOT)


# 类别中文标签
TYPE_LABELS = {
    "vehicle": "车辆",
    "pedestrian": "行人",
    "non_motor_vehicle": "非机动车",
}


def _load_chinese_font(size=16):
    """加载中文字体（多重回退）"""
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
               r"C:\Windows\Fonts\msyhbd.ttc"]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_chinese_text(frame, text, position, color, font):
    """在 OpenCV 帧上用 PIL 绘制中文文字"""
    x, y = position
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    # 标签背景
    draw.rectangle([x, y - th - 8, x + tw + 4, y], fill=color)
    # 标签文字
    draw.text((x + 2, y - th - 6), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
# 工具函数
# ============================================================

def crop_target(frame: np.ndarray, bbox, padding: float = 0.1) -> np.ndarray:
    """从帧中裁剪目标区域，带 padding"""
    h, w = frame.shape[:2]
    bw, bh = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
    x1 = max(0, int(bbox.x1 - bw * padding))
    y1 = max(0, int(bbox.y1 - bh * padding))
    x2 = min(w, int(bbox.x2 + bw * padding))
    y2 = min(h, int(bbox.y2 + bh * padding))
    crop = frame[y1:y2, x1:x2].copy()
    return crop


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 主流程
# ============================================================

def main():
    report_lines = []

    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)

    def section(title: str):
        log(f"\n[Step {title}]")

    # 输出目录
    output_dir = Path(PROJECT_ROOT) / "output" / "pipeline_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("全流程管线验证报告")
    log("=" * 60)
    log(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"数据目录: {PROJECT_ROOT}/data")
    log(f"输出目录: {output_dir}")

    # ----------------------------------------------------------
    # Step 1: 数据加载
    # ----------------------------------------------------------
    section("1: 数据加载")
    t0 = time.time()

    # 优先使用 AICity22 Track1 MTMC 数据
    cityflow_dir = Path(PROJECT_ROOT) / "cityflow" / "AICity22_Track1_MTMC_Tracking"
    # 尝试找到第一个有视频的场景/摄像头
    aic22_img_files = []
    if cityflow_dir.exists():
        for split in ["train", "validation"]:
            split_dir = cityflow_dir / split
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
                        aic22_img_files.append(video_path)
                        if len(aic22_img_files) >= 5:
                            break
                if len(aic22_img_files) >= 5:
                    break
            if len(aic22_img_files) >= 5:
                break

    # 回退到 VisDrone2019-DET
    det_img_dir = Path(PROJECT_ROOT) / "data" / "VisDrone2019-DET" / "images"
    mot_img_dir = Path(PROJECT_ROOT) / "data" / "VisDrone-MOT" / "images"

    import cv2

    frames = []
    dataset_name = ""
    img_files = []  # 原始图片路径列表

    if aic22_img_files:
        dataset_name = "AICity22-Track1-MTMC"
        for video_path in aic22_img_files:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            while frame_count < 20:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                img_files.append(video_path)  # 用视频路径作为图片路径
                frame_count += 1
            cap.release()
            if len(frames) >= 100:
                break
    elif det_img_dir.exists():
        img_files = sorted(det_img_dir.glob("*.jpg"))[:100]
        dataset_name = "VisDrone2019-DET"
        for f in img_files:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
    elif mot_img_dir.exists():
        img_files = sorted(mot_img_dir.glob("*.jpg"))[:100]
        dataset_name = "VisDrone-MOT"
        for f in img_files:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
    else:
        log("  [ERROR] 未找到数据目录!")
        return

    load_time = time.time() - t0
    log(f"  数据集: {dataset_name}")
    log(f"  图片数量: {len(frames)}")
    log(f"  加载耗时: {load_time:.2f}s")

    if len(frames) < 5:
        log("  [ERROR] 图片数量太少，无法进行跟踪验证!")
        return

    # ----------------------------------------------------------
    # Step 2: 目标检测
    # ----------------------------------------------------------
    section("2: 目标检测")
    t0 = time.time()

    from src.perception.detector import VehicleDetector

    detector = VehicleDetector(
        model_name="yolov8x",
        confidence_threshold=0.3,  # 降低阈值以获得更多检测
        nms_threshold=0.45,
        device="cuda",
    )

    all_detections = []  # List[List[Tuple[str, BoundingBox]]]
    total_det_count = 0
    class_counter = Counter()
    det_errors = 0

    for i, frame in enumerate(frames):
        try:
            dets = detector.detect(frame)
            all_detections.append(dets)
            total_det_count += len(dets)
            for cls, _ in dets:
                class_counter[cls] += 1
        except Exception as e:
            all_detections.append([])
            det_errors += 1
            if det_errors <= 3:
                log(f"  [WARN] 帧 {i} 检测失败: {e}")

    det_time = time.time() - t0
    avg_per_frame = total_det_count / max(len(frames), 1)

    log(f"  总检测框数: {total_det_count}")
    log(f"  平均每帧检测数: {avg_per_frame:.1f}")
    log(f"  类别分布: {dict(class_counter)}")
    log(f"  检测总耗时: {det_time:.2f}s")
    log(f"  平均每帧耗时: {det_time / len(frames) * 1000:.1f}ms")
    if det_errors > 0:
        log(f"  检测错误帧数: {det_errors}")

    # ----------------------------------------------------------
    # Step 3: 单摄跟踪
    # ----------------------------------------------------------
    section("3: 单摄跟踪")
    t0 = time.time()

    from src.perception.tracker import SingleCameraTracker

    tracker = SingleCameraTracker(max_age=15, min_hits=3, iou_threshold=0.2)

    # 存储每帧的跟踪结果
    frame_tracks = []  # List[List[Tuple[int, str, BoundingBox]]]
    track_id_set = set()
    track_frame_count = defaultdict(int)  # track_id -> 出现帧数

    for i, dets in enumerate(all_detections):
        tracks = tracker.update(dets, frame_id=i)
        frame_tracks.append(tracks)
        for tid, ttype, tbbox in tracks:
            track_id_set.add(tid)
            track_frame_count[tid] += 1

    track_time = time.time() - t0
    n_tracks = len(track_id_set)
    track_lengths = list(track_frame_count.values())
    avg_track_len = np.mean(track_lengths) if track_lengths else 0
    max_track_len = max(track_lengths) if track_lengths else 0

    log(f"  生成 track 数: {n_tracks}")
    log(f"  平均 track 长度: {avg_track_len:.1f} 帧")
    log(f"  最长 track: {max_track_len} 帧")
    log(f"  跟踪总耗时: {track_time:.2f}s")

    # ----------------------------------------------------------
    # Step 4: 特征提取
    # ----------------------------------------------------------
    section("4: 特征提取")
    t0 = time.time()

    from src.perception.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(
        reid_dim=512,
        clip_dim=768,
        device="cuda",
    )

    # 为每个 track 收集特征
    track_reid_features = defaultdict(list)  # track_id -> [vec, ...]
    track_clip_features = defaultdict(list)
    feature_extract_count = 0
    feature_errors = 0

    for i, (frame, tracks) in enumerate(zip(frames, frame_tracks)):
        for tid, ttype, tbbox in tracks:
            crop = crop_target(frame, tbbox)
            if crop.size == 0:
                continue

            try:
                reid_vec = extractor.extract_reid(crop)
                if reid_vec is not None and reid_vec.shape[0] == 512:
                    track_reid_features[tid].append(reid_vec)
                    feature_extract_count += 1
            except Exception as e:
                feature_errors += 1

            try:
                clip_vec = extractor.extract_clip(crop)
                if clip_vec is not None and clip_vec.shape[0] == 768:
                    track_clip_features[tid].append(clip_vec)
            except Exception as e:
                feature_errors += 1

    feat_time = time.time() - t0

    # 计算同 track 内相似度和不同 track 间相似度
    intra_sims = []  # 同 track 内
    inter_sims = []  # 不同 track 间

    # ReID 同 track 内相似度
    for tid, vecs in track_reid_features.items():
        if len(vecs) >= 2:
            for j in range(len(vecs)):
                for k in range(j + 1, len(vecs)):
                    intra_sims.append(cosine_sim(vecs[j], vecs[k]))

    # ReID 不同 track 间相似度
    track_ids_with_reid = [tid for tid, vecs in track_reid_features.items() if len(vecs) > 0]
    if len(track_ids_with_reid) >= 2:
        # 取每个 track 的平均向量
        track_avg_reid = {}
        for tid in track_ids_with_reid:
            vecs = track_reid_features[tid]
            track_avg_reid[tid] = np.mean(np.stack(vecs), axis=0)

        tid_list = list(track_avg_reid.keys())
        for j in range(len(tid_list)):
            for k in range(j + 1, len(tid_list)):
                inter_sims.append(cosine_sim(track_avg_reid[tid_list[j]], track_avg_reid[tid_list[k]]))

    avg_intra_sim = np.mean(intra_sims) if intra_sims else 0.0
    avg_inter_sim = np.mean(inter_sims) if inter_sims else 0.0

    # 验证向量维度
    sample_reid_dim = 0
    sample_clip_dim = 0
    for vecs in track_reid_features.values():
        if vecs:
            sample_reid_dim = vecs[0].shape[0]
            break
    for vecs in track_clip_features.values():
        if vecs:
            sample_clip_dim = vecs[0].shape[0]
            break

    log(f"  ReID 向量维度: {sample_reid_dim}")
    log(f"  CLIP 向量维度: {sample_clip_dim}")
    log(f"  特征提取次数: {feature_extract_count}")
    log(f"  同 track 内平均余弦相似度: {avg_intra_sim:.4f} (应 > 0.5)")
    log(f"  不同 track 间平均余弦相似度: {avg_inter_sim:.4f} (应 < 0.5)")
    log(f"  特征提取总耗时: {feat_time:.2f}s")
    if feature_errors > 0:
        log(f"  特征提取错误次数: {feature_errors}")

    # ----------------------------------------------------------
    # Step 5: 质量评分和属性识别
    # ----------------------------------------------------------
    section("5: 质量评分与属性识别")
    t0 = time.time()

    from src.perception.quality import QualityScorer
    from src.perception.attribute import AttributeRecognizer

    quality_scorer = QualityScorer(min_score=0.3)
    attr_recognizer = AttributeRecognizer(device="cuda")

    quality_scores = []
    attr_results = []
    high_quality_count = 0
    quality_errors = 0

    for i, (frame, tracks) in enumerate(zip(frames, frame_tracks)):
        for tid, ttype, tbbox in tracks:
            try:
                qscore = quality_scorer.score(frame, tbbox, detection_confidence=tbbox.confidence)
                quality_scores.append(qscore)
                if quality_scorer.is_qualified(qscore):
                    high_quality_count += 1
            except Exception as e:
                quality_errors += 1

            # 属性识别（只对部分目标做，避免太慢）
            if len(attr_results) < 200:
                crop = crop_target(frame, tbbox)
                if crop.size > 0:
                    try:
                        attrs = attr_recognizer.recognize(crop, ttype)
                        attr_results.append((tid, ttype, attrs))
                    except Exception:
                        pass

    qual_time = time.time() - t0
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    high_quality_ratio = high_quality_count / max(len(quality_scores), 1) * 100

    log(f"  平均质量分: {avg_quality:.2f}")
    log(f"  高质量目标占比: {high_quality_ratio:.1f}%")
    log(f"  质量评分+属性识别耗时: {qual_time:.2f}s")
    if quality_errors > 0:
        log(f"  质量评分错误: {quality_errors}")

    # 属性识别统计
    attr_type_counter = Counter()
    for _, ttype, attrs in attr_results:
        if attrs:
            attr_type_counter[ttype] += 1
    log(f"  属性识别结果数: {len(attr_results)}")
    if attr_type_counter:
        log(f"  属性识别类别: {dict(attr_type_counter)}")
        # 展示几个样例
        shown = 0
        for tid, ttype, attrs in attr_results[:5]:
            log(f"    样例 track={tid}, type={ttype}: {attrs}")
            shown += 1

    # ----------------------------------------------------------
    # Step 6: Tracklet 生成
    # ----------------------------------------------------------
    section("6: Tracklet 生成")
    t0 = time.time()

    from src.tracking.tracklet import TrackletGenerator

    tracklet_gen = TrackletGenerator(camera_id="c001", min_instances=3, keyframe_interval=5)

    # 模拟时间戳（假设 25fps）
    base_time = datetime(2024, 1, 15, 8, 30, 0)

    all_tracklets = []
    for i, tracks in enumerate(frame_tracks):
        ts = datetime.fromtimestamp(base_time.timestamp() + i / 25.0)
        tracklet_gen.add_frame(tracks, timestamp=ts, frame_id=i, frame=frames[i])
        completed = tracklet_gen.get_completed_tracklets()
        all_tracklets.extend(completed)

    # flush 剩余的
    remaining = tracklet_gen.flush()
    all_tracklets.extend(remaining)

    tracklet_time = time.time() - t0

    log(f"  生成 Tracklet 数: {len(all_tracklets)}")
    if all_tracklets:
        inst_counts = [t.instance_count for t in all_tracklets]
        log(f"  平均实例数: {np.mean(inst_counts):.1f}")
        log(f"  最长 Tracklet: {max(inst_counts)} 实例")

        # 类型分布
        trk_type_counter = Counter(t.target_type for t in all_tracklets)
        log(f"  Tracklet 类型分布: {dict(trk_type_counter)}")

        # 检查有 ReID/CLIP 向量的 tracklet（回填前可能为 0）
        has_reid = sum(1 for t in all_tracklets if t.avg_reid_vector is not None)
        has_clip = sum(1 for t in all_tracklets if t.avg_clip_vector is not None)
        if has_reid > 0 or has_clip > 0:
            log(f"  含 ReID 向量的 Tracklet: {has_reid}")
            log(f"  含 CLIP 向量的 Tracklet: {has_clip}")
    else:
        log(f"  [WARN] 未生成任何 Tracklet!")

    log(f"  Tracklet 生成耗时: {tracklet_time:.2f}s")

    # ----------------------------------------------------------
    # Step 7: 为 Tracklet 填充特征向量
    # ----------------------------------------------------------
    # TrackletGenerator 创建 TargetInstance 时没有特征向量
    # 我们需要将提取的特征回填到 tracklet instances 中
    # 然后重新计算 avg_reid/avg_clip

    # 建立 track_id -> 特征列表的映射，用于回填 tracklet
    # 通过匹配 tracklet instances 的 frame_id 序列与原始 track 的 frame_id 序列
    track_frame_ids = defaultdict(set)  # track_id -> set of frame_ids
    for i, tracks in enumerate(frame_tracks):
        for tid, ttype, tbbox in tracks:
            track_frame_ids[tid].add(i)

    filled_reid = 0
    filled_clip = 0
    for trk in all_tracklets:
        trk_fids = {inst.frame_id for inst in trk.instances}
        # 找到 frame_id 重叠最多的原始 track_id
        best_tid = None
        best_overlap = 0
        for tid, fids in track_frame_ids.items():
            overlap = len(trk_fids & fids)
            if overlap > best_overlap:
                best_overlap = overlap
                best_tid = tid
        if best_tid is None:
            continue
        # 用该 track_id 的平均特征填充 tracklet
        if best_tid in track_reid_features and track_reid_features[best_tid]:
            vecs = track_reid_features[best_tid]
            avg_reid = np.mean(np.stack(vecs), axis=0)
            norm = np.linalg.norm(avg_reid)
            if norm > 0:
                trk.avg_reid_vector = (avg_reid / norm).astype(np.float32)
                filled_reid += 1
        if best_tid in track_clip_features and track_clip_features[best_tid]:
            vecs = track_clip_features[best_tid]
            avg_clip = np.mean(np.stack(vecs), axis=0)
            norm = np.linalg.norm(avg_clip)
            if norm > 0:
                trk.avg_clip_vector = (avg_clip / norm).astype(np.float32)
                filled_clip += 1

    log(f"  回填 ReID 向量的 Tracklet: {filled_reid}")
    log(f"  回填 CLIP 向量的 Tracklet: {filled_clip}")

    # ----------------------------------------------------------
    # Step 8: 输出可视化
    # ----------------------------------------------------------
    section("7: 输出可视化")
    t0 = time.time()

    vis_count = 0
    # 颜色映射
    COLOR_MAP = {
        "vehicle": (0, 255, 0),       # 绿色
        "pedestrian": (255, 0, 0),     # 蓝色
        "non_motor_vehicle": (0, 255, 255),  # 黄色
    }

    # 每隔 N 帧保存一张可视化图片
    vis_interval = max(1, len(frames) // 20)  # 保存约 20 张
    cn_font = _load_chinese_font(16)

    for i in range(0, len(frames), vis_interval):
        frame = frames[i].copy()
        tracks = frame_tracks[i]

        for tid, ttype, tbbox in tracks:
            color = COLOR_MAP.get(ttype, (200, 200, 200))
            x1, y1 = int(tbbox.x1), int(tbbox.y1)
            x2, y2 = int(tbbox.x2), int(tbbox.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cn_label = TYPE_LABELS.get(ttype, ttype)
            label = f"ID:{tid} {cn_label}"
            frame = _draw_chinese_text(frame, label, (x1, y1), color, cn_font)

        out_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        vis_count += 1

    vis_time = time.time() - t0
    log(f"  保存可视化图片: {vis_count} 张")
    log(f"  输出目录: {output_dir}")
    log(f"  可视化耗时: {vis_time:.2f}s")

    # ----------------------------------------------------------
    # Step 8b: 生成前端需要的 results.json + crops
    # ----------------------------------------------------------
    section("8: 生成前端输出 (results.json + crops)")
    import json

    crops_dir = Path(PROJECT_ROOT) / "output" / "crops"
    demo_det_dir = Path(PROJECT_ROOT) / "output" / "demo_detections"
    crops_dir.mkdir(parents=True, exist_ok=True)
    demo_det_dir.mkdir(parents=True, exist_ok=True)

    # 生成 crops：为每个 track 裁剪一个关键帧
    crop_count = 0
    track_keyframe_map = {}  # track_id -> crop_path
    for i, tracks in enumerate(frame_tracks):
        for tid, ttype, tbbox in tracks:
            if tid not in track_keyframe_map:
                crop = crop_target(frames[i], tbbox)
                if crop.size > 0:
                    crop_path = crops_dir / f"TRACK_{tid:04d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    track_keyframe_map[tid] = str(crop_path.relative_to(Path(PROJECT_ROOT)))
                    crop_count += 1

    # 收集检测结果
    det_records = []
    det_id_counter = 0
    # 为每个 track 生成检测记录
    track_type_counter = Counter()
    for i, tracks in enumerate(frame_tracks):
        for tid, ttype, tbbox in tracks:
            det_id_counter += 1
            det_record = {
                "target_id": f"DET_{det_id_counter:04d}",
                "target_type": ttype,
                "confidence": round(tbbox.confidence, 4),
                "attributes": {},
                "bbox": [round(tbbox.x1, 1), round(tbbox.y1, 1), round(tbbox.x2, 1), round(tbbox.y2, 1)],
                "frame_id": i,
                "image_path": str(img_files[i].relative_to(Path(PROJECT_ROOT))) if i < len(img_files) else "",
                "keyframe_path": track_keyframe_map.get(tid, ""),
                "detection_image_path": "",
            }
            # 关联属性
            for attr_tid, attr_ttype, attr_attrs in attr_results:
                if attr_tid == tid and attr_attrs:
                    det_record["attributes"] = attr_attrs
                    break
            det_records.append(det_record)

    # 为 detection_image_path 关联可视化图片
    vis_frame_indices = list(range(0, len(frames), vis_interval))
    for rec in det_records:
        fid = rec["frame_id"]
        # 找最近的可视化帧
        closest_vis = min(vis_frame_indices, key=lambda vi: abs(vi - fid))
        rec["detection_image_path"] = str((output_dir / f"frame_{closest_vis:04d}.jpg").relative_to(Path(PROJECT_ROOT)))

    # 生成 tracks 记录
    track_records = []
    for tid in sorted(track_id_set):
        fids = sorted(i for i, tracks in enumerate(frame_tracks) for t, _, _ in tracks if t == tid)
        if not fids:
            continue
        # 确定 track 类型
        trk_type = None
        for tracks in frame_tracks:
            for t, tt, _ in tracks:
                if t == tid:
                    trk_type = tt
                    break
            if trk_type:
                break
        track_records.append({
            "track_id": f"TRACK_{tid:04d}",
            "target_type": trk_type or "unknown",
            "frame_range": [fids[0], fids[-1]],
            "keyframe_path": track_keyframe_map.get(tid, ""),
            "detection_count": track_frame_count[tid],
        })
        if trk_type:
            track_type_counter[trk_type] += 1

    # 汇总
    vehicle_count = class_counter.get("vehicle", 0)
    pedestrian_count = class_counter.get("pedestrian", 0)
    non_motor_count = class_counter.get("non_motor_vehicle", 0)

    results_json = {
        "detections": det_records,
        "tracks": track_records,
        "summary": {
            "total_frames": len(frames),
            "total_detections": total_det_count,
            "total_tracks": n_tracks,
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "non_motor_vehicle_count": non_motor_count,
        },
    }

    results_json_path = Path(PROJECT_ROOT) / "output" / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    log(f"  生成 crops: {crop_count} 张")
    log(f"  生成 detections 记录: {len(det_records)}")
    log(f"  生成 tracks 记录: {len(track_records)}")
    log(f"  results.json 已保存: {results_json_path}")
    log(f"  类别统计: vehicle={vehicle_count}, pedestrian={pedestrian_count}, non_motor={non_motor_count}")

    # 尝试生成视频
    try:
        if vis_count > 1:
            vis_frames = []
            cn_font_vid = _load_chinese_font(14)
            for i in range(0, len(frames), vis_interval):
                frame = frames[i].copy()
                tracks = frame_tracks[i]
                for tid, ttype, tbbox in tracks:
                    color = COLOR_MAP.get(ttype, (200, 200, 200))
                    x1, y1 = int(tbbox.x1), int(tbbox.y1)
                    x2, y2 = int(tbbox.x2), int(tbbox.y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cn_label = TYPE_LABELS.get(ttype, ttype)
                    label = f"ID:{tid} {cn_label}"
                    frame = _draw_chinese_text(frame, label, (x1, y1), color, cn_font_vid)
                vis_frames.append(frame)

            if vis_frames:
                h, w = vis_frames[0].shape[:2]
                video_path = output_dir / "pipeline_tracking.avi"
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
                for vf in vis_frames:
                    writer.write(vf)
                writer.release()
                log(f"  视频已保存: {video_path}")
    except Exception as e:
        log(f"  [WARN] 视频生成失败: {e}")

    # ----------------------------------------------------------
    # 最终报告
    # ----------------------------------------------------------
    log("")
    log("=" * 60)
    log("管线验证总结")
    log("=" * 60)

    checks = []

    # 检查 1: 至少处理了 5 帧
    ok = len(frames) >= 5
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 处理帧数: {len(frames)} (>= 5)")

    # 检查 2: 有检测结果
    ok = total_det_count > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 检测目标数: {total_det_count} (> 0)")

    # 检查 3: 有跟踪结果
    ok = n_tracks > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 跟踪 track 数: {n_tracks} (> 0)")

    # 检查 4: 有特征提取
    ok = feature_extract_count > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 特征提取次数: {feature_extract_count} (> 0)")

    # 检查 5: 同 track 内相似度
    ok = avg_intra_sim > 0.5
    log(f"  [{'OK' if ok else 'WARN'}] 同 track ReID 相似度: {avg_intra_sim:.4f} (> 0.5)")

    # 检查 6: 至少 3 个 Tracklet
    ok = len(all_tracklets) >= 3
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] Tracklet 数: {len(all_tracklets)} (>= 3)")

    # 检查 7: 有可视化输出
    ok = vis_count > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 可视化图片: {vis_count} 张")

    # 检查 8: Tracklet 含 ReID 向量
    final_reid_count = sum(1 for t in all_tracklets if t.avg_reid_vector is not None)
    final_clip_count = sum(1 for t in all_tracklets if t.avg_clip_vector is not None)
    ok = final_reid_count > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 含 ReID 向量的 Tracklet: {final_reid_count}")
    log(f"  [INFO] 含 CLIP 向量的 Tracklet: {final_clip_count}")

    # 检查 9: 检测到车辆
    ok = vehicle_count > 0
    checks.append(ok)
    log(f"  [{'OK' if ok else 'FAIL'}] 检测到车辆数: {vehicle_count} (> 0)")
    log(f"  [INFO] 检测类别分布: vehicle={vehicle_count}, pedestrian={pedestrian_count}, non_motor={non_motor_count}")

    all_pass = all(checks)
    log("")
    if all_pass:
        log("管线验证状态: ✓ 全流程跑通")
    else:
        fail_count = sum(1 for c in checks if not c)
        log(f"管线验证状态: ✗ {fail_count} 项检查未通过")

    log("=" * 60)

    # 保存报告
    report_path = Path(PROJECT_ROOT) / "output" / "pipeline_validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] 管线运行崩溃: {e}")
        traceback.print_exc()
        sys.exit(1)
