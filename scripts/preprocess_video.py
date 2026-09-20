"""
scripts/preprocess_video.py - 视频预处理（检测 → 单摄跟踪 → 属性/质量）

用法:
    python scripts/preprocess_video.py --video cityflow/AICity22_Track1_MTMC_Tracking/test/S06/c041/vdo.avi \
                                       --camera c041 --max-frames 200

    # 跳过属性识别（省时；只要轨迹时用）
    python scripts/preprocess_video.py --video <path> --camera c041 --no-attributes

## 这个脚本做什么

把一段视频跑成结构化的目标实例与单摄轨迹（Tracklet），并落盘为 JSON：

    逐帧 → YOLOv8x 检测 → ByteTrack 关联 → 每个跟踪目标：质量评分 + 属性识别
         → TargetInstance 列表 + Tracklet 列表

它**直接复用 `src/perception/` 的实现**（检测器/跟踪器/质量评分/属性识别），
不重新实现任何算法。与 `scripts/cityflow_preprocess.py` 的区别是：
后者针对 AICity22 数据集做了批量化与特定目录约定，本脚本是**单视频通用入口**，
也是 `setup.py` 注册的 `traffic-preprocess` 命令。

## 诚实边界

- **不产出车辆身份（vehicle_id）**。跟踪器给的只是单摄内的 track 编号，
  不是跨镜的车辆身份。因此本脚本的 ID 刻意**不含 `V####` 段** ——
  项目里 `V####` 是真实身份的格式，`src.common.ids.extract_vehicle_id`
  会把 `c041_V0000_...` 解析成 `V0000` 并让下游走"强身份"路径，
  那是把跟踪编号伪装成了真值。
- **不产车牌**：`src/perception/plate_ocr.py` 的内置降级路径在没有
  EasyOCR/PaddleOCR 时只会返回 `(None, conf)`，即拿不到真实车牌字符串。
  与其输出一个"看起来有车牌"的空值，这里不产出车牌字段。
- 输出里不写任何模型没给出的字段；缺失即缺失（统一为 `null`）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logger import get_logger

logger = get_logger("scripts.preprocess_video")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频预处理：检测 + 单摄跟踪 + 属性/质量")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--camera", type=str, required=True, help="摄像头 ID（如 c041）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径（默认 output/preprocess_<camera>.json）")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧（0=全部）")
    parser.add_argument("--no-attributes", action="store_true",
                        help="跳过属性识别（更快，但输出无颜色/车型字段）")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="低于该质量分的实例被丢弃（0=不过滤）")
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    from src.common.config import get_config

    cfg = get_config(args.config)

    try:
        import cv2
    except ImportError:
        logger.error("需要 opencv-python 才能读视频：pip install opencv-python")
        sys.exit(2)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频不存在: {video_path}")
        sys.exit(2)

    # ── 1. 打开视频 ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        sys.exit(2)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    limit = args.max_frames if args.max_frames > 0 else total_frames
    logger.info(
        f"视频: {video_path.name} | 总帧={total_frames} | fps={fps:.2f} | "
        f"本次处理={limit} 帧 | 摄像头={args.camera}"
    )

    # ── 2. 初始化感知组件（全部复用 src/perception）──
    from src.perception.attribute import AttributeRecognizer
    from src.perception.detector import VehicleDetector
    from src.perception.quality import QualityScorer
    from src.perception.tracker import SingleCameraTracker

    device = cfg.get("system.device", "cpu")
    detector = VehicleDetector(
        model_name=cfg.get("detection.model", "yolov8x"),
        confidence_threshold=cfg.get("detection.confidence_threshold", 0.5),
        nms_threshold=cfg.get("detection.nms_threshold", 0.45),
        device=device,
    )
    tracker = SingleCameraTracker(
        max_age=cfg.get("tracking.max_age", 30),
        min_hits=cfg.get("tracking.min_hits", 3),
        iou_threshold=cfg.get("tracking.iou_threshold", 0.3),
    )
    quality_scorer = QualityScorer()

    recognizer = None
    if not args.no_attributes:
        try:
            recognizer = AttributeRecognizer(device=device)
        except Exception as e:
            logger.warning(f"属性识别器初始化失败，本次跳过属性识别: {e}")

    # ── 3. 逐帧处理 ──
    instances: list = []
    instance_seq = 0
    frame_id = 0
    t_start = time.time()

    while frame_id < limit:
        ok, frame = cap.read()
        if not ok:
            logger.info(f"视频读取结束于第 {frame_id} 帧")
            break
        frame_id += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_id)

        for track_id, target_type, bbox in tracks:
            quality = quality_scorer.score(
                frame, bbox, detection_confidence=bbox.confidence
            )
            if quality < args.min_quality:
                continue

            attributes: dict = {}
            if recognizer is not None and target_type == "vehicle":
                crop = _crop(frame, bbox)
                if crop is not None:
                    try:
                        attributes = recognizer.recognize(crop, target_type) or {}
                    except Exception as e:
                        logger.debug(f"属性识别失败 frame={frame_id} track={track_id}: {e}")

            instance_seq += 1
            instances.append({
                # 注意 ID 里**刻意不含 `V####` 段**：项目里 V#### 是"真实车辆身份"
                # 的格式，`src.common.ids.extract_vehicle_id` 会把它解析出来，
                # 下游 TrajectoryBuilder 据此走"强身份"路径。而本脚本只有跟踪器
                # 给出的 track 编号，**没有真实车辆身份** —— 若把 track 编号塞进
                # V#### 槽位（如 c041_V0000_...），就会被误当成真值。
                "instance_id": f"PRE_{args.camera}_f{frame_id:06d}_i{instance_seq:06d}",
                "camera_id": args.camera,
                "frame_id": frame_id,
                "timestamp": _frame_timestamp(frame_id, fps),
                "track_id": f"PRE_TRACK_{args.camera}_{track_id:04d}",
                # 本脚本不产出车辆身份；跨镜身份必须由后续环节确定
                "vehicle_id": None,
                "target_type": target_type,
                "bbox": {
                    "x1": float(bbox.x1), "y1": float(bbox.y1),
                    "x2": float(bbox.x2), "y2": float(bbox.y2),
                },
                "confidence": float(bbox.confidence),
                "quality_score": float(quality),
                "attributes": attributes or None,   # 未识别到就是 None，不填 {}
                # 车牌不在本脚本产出范围（见模块 docstring 的诚实边界）
                "plate_number": None,
            })

        if frame_id % 50 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"进度 {frame_id}/{limit} 帧 | 实例 {len(instances)} | "
                f"耗时 {elapsed:.1f}s ({frame_id / max(elapsed, 1e-6):.1f} fps)"
            )

    cap.release()
    elapsed = time.time() - t_start

    # ── 4. 汇总为 Tracklet ──
    tracklets = _build_tracklets(instances)

    summary = {
        "camera_id": args.camera,
        "video": str(video_path),
        "fps": fps,
        "frames_processed": frame_id,
        "instances": len(instances),
        "tracklets": len(tracklets),
        "elapsed_seconds": round(elapsed, 2),
        "attributes_enabled": recognizer is not None,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = Path(args.output) if args.output else (
        Path("output") / f"preprocess_{args.camera}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"summary": summary, "tracklets": tracklets, "instances": instances},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        f"完成: {frame_id} 帧 / {len(instances)} 实例 / {len(tracklets)} Tracklet "
        f"| 耗时 {elapsed:.1f}s | 输出 {out_path}"
    )


def _crop(frame, bbox):
    """按检测框裁剪（越界自动收敛到图像范围）"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1)); y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2)); y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _frame_timestamp(frame_id: int, fps: float) -> str:
    """帧号 → 相对时间字符串（视频起点为 00:00:00）"""
    if not fps or fps <= 0:
        return f"frame_{frame_id}"
    seconds = frame_id / fps
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def _build_tracklets(instances: list) -> list:
    """把逐帧实例按 track_id 聚成 Tracklet（不做跨镜推断，只做单摄聚合）"""
    grouped: dict = {}
    for inst in instances:
        grouped.setdefault(inst["track_id"], []).append(inst)

    tracklets = []
    for track_id, items in grouped.items():
        items.sort(key=lambda x: x["frame_id"])
        first, last = items[0], items[-1]
        tracklets.append({
            "track_id": track_id,
            "camera_id": first["camera_id"],
            "target_type": first["target_type"],
            "start_frame": first["frame_id"],
            "end_frame": last["frame_id"],
            "start_time": first["timestamp"],
            "end_time": last["timestamp"],
            "instance_count": len(items),
            "avg_quality": round(
                sum(i["quality_score"] for i in items) / len(items), 4
            ),
            "instance_ids": [i["instance_id"] for i in items],
        })
    tracklets.sort(key=lambda t: t["start_frame"])
    return tracklets


if __name__ == "__main__":
    main()
