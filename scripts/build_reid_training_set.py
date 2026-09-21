"""
scripts/build_reid_training_set.py - 构建 ReID 训练/验证集（PLAN5-C1）

用法:
    python scripts/build_reid_training_set.py
    python scripts/build_reid_training_set.py --val-cameras-per-id 2 --min-images 4

产物:
    output/reid_train/{split}.json    每行 {crop_path, vehicle_id, camera_id}
    output/reid_train/summary.json    统计与切分口径

## 切分口径：按**摄像头**切，不按图片随机切

ReID 的验证集必须是"**同一批身份、不同摄像头视角**"，否则验证毫无意义：
- 若按图片随机切，同一辆车的相邻帧会同时落进 train 和 val ——
  模型只需记住"这一帧的邻居"，val 指标虚高，而真实的跨镜能力根本没被考到。
- 因此这里对**每个身份**保留若干摄像头进 val，其余进 train。
  身份在两侧都出现（闭集协议），但视角不重叠。

## 单位是"轨迹片段"而非"图片"

同一辆车在同一摄像头内的连续帧高度相似。若把 30 帧都当成 30 个样本，
训练会被这几帧主导。这里对每个 (身份, 摄像头) 组合按时间**抽稀**到至多
`--max-per-camera` 张，保证样本多样性。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger

logger = get_logger("scripts.build_reid_training_set")

DEFAULT_OUTDIR = ROOT / "output" / "reid_train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 ReID 训练/验证集")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="输出目录")
    p.add_argument("--val-cameras-per-id", type=int, default=1,
                   help="每个身份留几个摄像头进验证集")
    p.add_argument("--max-per-camera", type=int, default=8,
                   help="同一(身份,摄像头)最多抽稀到多少张")
    p.add_argument("--min-images", type=int, default=6,
                   help="身份至少要有多少张图才纳入（太少的丢掉）")
    p.add_argument("--min-cameras", type=int, default=2,
                   help="身份至少要出现在几个摄像头（跨镜才有意义）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_crop(raw, root: Path = ROOT) -> Path | None:
    if not raw:
        return None
    p = Path(str(raw).replace("\\", "/").lstrip("./"))
    if not p.is_absolute():
        p = root / p
    return p if p.exists() else None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    from src.common.ids import extract_vehicle_id
    from src.storage.datastore import load_results

    data = load_results()
    if not data:
        sys.exit("[fatal] 数据不可用")
    detections = data.get("detections", [])
    logger.info("检测总数: %d", len(detections))

    # ── 按 (身份, 摄像头) 归组 ──
    by_identity: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped_no_crop = 0
    for det in detections:
        vid = extract_vehicle_id(det.get("target_id") or "")
        if not vid:
            continue
        crop = _resolve_crop(det.get("crop_path"))
        if crop is None:
            skipped_no_crop += 1
            continue
        cam = det.get("camera_id") or ""
        by_identity[vid][cam].append({
            "crop_path": str(crop.relative_to(ROOT)).replace("\\", "/"),
            "vehicle_id": vid,
            "camera_id": cam,
            "frame_id": det.get("frame_id", 0),
        })
    logger.info("身份数（去重后）: %d | 因 crop 缺失跳过 %d 条",
                len(by_identity), skipped_no_crop)

    # ── 过滤 + 抽稀 + 按摄像头切分 ──
    train, val = [], []
    kept_ids, dropped = [], defaultdict(int)

    for vid, cams in sorted(by_identity.items()):
        n_imgs = sum(len(v) for v in cams.values())
        if n_imgs < args.min_images:
            dropped["图片太少"] += 1
            continue
        if len(cams) < args.min_cameras:
            dropped["摄像头太少"] += 1
            continue

        cam_ids = sorted(cams.keys())
        random.shuffle(cam_ids)
        n_val = min(args.val_cameras_per_id, max(1, len(cam_ids) - 1))
        val_cams = set(cam_ids[:n_val])
        # 确保验证集侧也有足够样本，否则该身份的 val 形同虚设
        if sum(len(cams[c]) for c in val_cams) < 2:
            dropped["验证侧样本不足"] += 1
            continue

        kept_ids.append(vid)
        for cam in cam_ids:
            items = sorted(cams[cam], key=lambda x: x["frame_id"])
            if len(items) > args.max_per_camera:
                step = len(items) / args.max_per_camera
                items = [items[int(i * step)] for i in range(args.max_per_camera)]
            (val if cam in val_cams else train).append(items)

    train_flat = [x for grp in train for x in grp]
    val_flat = [x for grp in val for x in grp]

    # ── 一致性自检 ──
    train_ids = {x["vehicle_id"] for x in train_flat}
    val_ids = {x["vehicle_id"] for x in val_flat}

    only_val = val_ids - train_ids
    only_train = train_ids - val_ids
    if only_val or only_train:
        logger.warning(
            "身份未在两侧同时出现：仅 val %d 个、仅 train %d 个。"
            "闭集 ReID 评测要求两侧身份一致，评估脚本会取交集处理。",
            len(only_val), len(only_train))

    # **真正的泄漏检查**：同一身份的同一个摄像头，不能同时出现在 train 和 val。
    #
    # 注意不要用"摄像头全局是否重叠"来判断 —— 46 个摄像头、215 个身份，
    # 每个摄像头必然装着很多辆车，所以全局看摄像头**一定**是重叠的，
    # 那是正常的（身份 A 的 c001 在 train、身份 B 的 c001 在 val，互不泄漏）。
    # 要查的是**逐身份**的摄像头是否互斥。
    id_cams: dict[str, dict[str, set]] = defaultdict(lambda: {"train": set(), "val": set()})
    for x in train_flat:
        id_cams[x["vehicle_id"]]["train"].add(x["camera_id"])
    for x in val_flat:
        id_cams[x["vehicle_id"]]["val"].add(x["camera_id"])

    leaks = {
        vid: sorted(c["train"] & c["val"])
        for vid, c in id_cams.items()
        if c["train"] & c["val"]
    }
    if leaks:
        logger.error(
            "发现 %d 个身份的同摄像头视角跨集泄漏（例如 %s）",
            len(leaks), list(leaks.items())[:2],
        )
    else:
        logger.info("逐身份摄像头互斥检查通过：%d 个身份均无同摄像头跨集泄漏",
                    len(id_cams))

    train_cams = {x["camera_id"] for x in train_flat}
    val_cams_all = {x["camera_id"] for x in val_flat}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train.json").write_text(
        json.dumps(train_flat, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "val.json").write_text(
        json.dumps(val_flat, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "detections_total": len(detections),
        "identities_total": len(by_identity),
        "identities_kept": len(kept_ids),
        "dropped": dict(dropped),
        "train_images": len(train_flat),
        "val_images": len(val_flat),
        "train_identities": len(train_ids),
        "val_identities": len(val_ids),
        "identities_in_both": len(train_ids & val_ids),
        "train_cameras": len(train_cams),
        "val_cameras": len(val_cams_all),
        # 逐身份摄像头互斥：**这条才是泄漏指标**
        "per_identity_camera_leaks": len(leaks),
        # 全局摄像头重叠是正常的（一个摄像头装多辆车），仅供了解规模
        "camera_seen_in_both_splits": sorted(train_cams & val_cams_all),
        "split_policy": "按摄像头切分（同一身份保留若干摄像头进 val，逐身份互斥）",
        "sampling": {
            "val_cameras_per_id": args.val_cameras_per_id,
            "max_per_camera": args.max_per_camera,
            "min_images": args.min_images,
            "min_cameras": args.min_cameras,
            "seed": args.seed,
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("train: %d 张 / %d 身份 | val: %d 张 / %d 身份 | 两侧共有身份 %d",
                len(train_flat), len(train_ids),
                len(val_flat), len(val_ids), len(train_ids & val_ids))
    logger.info("产物: %s", outdir)


if __name__ == "__main__":
    main()
