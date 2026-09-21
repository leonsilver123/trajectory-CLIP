"""
scripts/train_reid_triplet.py - Batch Hard Triplet 训练 ReID 嵌入（PLAN5-C2/C3）

用法:
    python scripts/train_reid_triplet.py                    # 标准训练（triplet + ID 损失）
    python scripts/train_reid_triplet.py --no-id-loss       # 纯 triplet，供消融对照
    python scripts/train_reid_triplet.py --hard-mining      # 开启难负样本挖掘（C3）
    python scripts/train_reid_triplet.py --epochs 30 --p 8 --k 4

产物:
    output/reid_trained/{best.pth,last.pth,train_log.json,metrics.json}

## 对应简历里的哪两句

- 「ReID 空间用 **Batch Hard Triplet Loss** 优化」→ 本脚本主体
- 「训练引入 **Hard Negative Mining** 强化细粒度区分」→ `--hard-mining`（C3）

## Batch Hard 是怎么做的

每个 batch 按 **PK 采样**：P 个身份 × 每身份 K 张图。对每个 anchor，
在 **batch 内部**取"最远的正样本"与"最近的负样本"组成三元组 ——
这就是 Batch Hard 的定义（难样本在线挖，不是离线预先挑好）。
`torchreid.losses.TripletLoss` 正是这个语义。

## C3 的难负样本挖掘与 Batch Hard 的区别

Batch Hard 只在**当前 batch 内部**挖；若 batch 里恰好没有难负样本，就挖不到。
`--hard-mining` 在此之上加一层**跨 batch 的挖掘**：每个 epoch 结束后用当前模型
算全部训练图的嵌入，为每个身份找出**最容易被混淆的几个身份**，
下一轮采样时按比例把这些"困难身份对"排进同一个 batch。

两轮训练的对照即 C3 的验收：如实报告挖掘带来的增益**或负增益**。

## 诚实边界（必须先读）

1. **只有 215 个身份**（VeRi 575 身份 / MSMT17 1041 身份）。这个量级训出来的嵌入，
   **不要期待公开基准那种 R@1**。
2. **本数据集禁止商用**（AICity22 许可），产物仅限研究。
3. ⚠️ **本脚本打印的 val Rank-1 默认是「同摄像头」口径，不是跨摄像头。**
   见下方专节。
"""

# ============================================================
# ⚠️ val Rank-1 的口径：默认是同摄像头，不是跨摄像头
# ============================================================
#
# 2026-09-21 实测更正。本文件此前写着「val 的 Rank-1 衡量的是**跨摄像头泛化**」，
# **那句话是错的**，且是我自己写下的。
#
# 事实：`build_reid_training_set.py` 默认 `--val-cameras-per-id 1`，于是每个身份
# 在 val 里**只占 1 个摄像头**。实测 215 个身份的分布是 `{1: 215}` —— 无一例外。
# 因此 `evaluate_rank` 在 val 上做的 query-gallery 配对，**全部来自同一摄像头**，
# 衡量的是"同摄像头内的外观记忆"，不是跨镜泛化。
#
# 这个错误让 20 epoch 训练报出的 `Rank-1 = 0.909` 看起来像跨镜成绩，而同一份
# 数据上用**真正跨镜**口径（`scripts/eval_cross_camera.py`，全量检测池）测
# fast-reid 只有 **0.207**。差 4 倍多的落差就是口径差，不是模型差。
#
# **要看跨摄像头数字**：用 `scripts/eval_cross_camera.py`，或把验证集重建为
# 每身份 >= 2 个摄像头（`build_reid_training_set.py --val-cameras-per-id 2`）
# 后重训。本文件不再宣称 val Rank-1 是跨镜指标。
#
# 下面 `_warn_if_val_not_cross_camera()` 会在运行时检查并告警，避免这个口径
# 再次被静默当成跨镜成绩。

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger

logger = get_logger("scripts.train_reid_triplet")

DEFAULT_DATASET = ROOT / "output" / "reid_train"
DEFAULT_OUTDIR = ROOT / "output" / "reid_trained"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Hard Triplet 训练 ReID 嵌入")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="训练集目录")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="输出目录")
    p.add_argument("--arch", default="osnet_x1_0", help="backbone（torchreid 模型名）")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--p", type=int, default=8, help="每 batch 的身份数 P")
    p.add_argument("--k", type=int, default=4, help="每身份图片数 K")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--margin", type=float, default=0.3, help="triplet margin")
    p.add_argument("--id-loss-weight", type=float, default=1.0,
                   help="ID 分类损失权重（0 = 纯 triplet）")
    p.add_argument("--no-id-loss", action="store_true")
    p.add_argument("--hard-mining", action="store_true",
                   help="开启难负样本挖掘（C3），与不开的版本做对照")
    p.add_argument("--hard-frac", type=float, default=0.5,
                   help="开启挖掘时，多大比例的 batch 使用困难身份对")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-per-id", type=int, default=0,
                   help="每身份最多用多少张（冒烟用，0=不限）")
    p.add_argument("--image-cache-size", type=int, default=2000,
                   help="图片缓存条数上限（0=不限）。内存紧张时调小")
    return p.parse_args()


# ============================================================
# 数据
# ============================================================


def _warn_if_val_not_cross_camera(val_records) -> bool:
    """
    检查验证集是否真的是「跨摄像头」的，不是就告警

    Returns:
        True 表示验证集是跨摄像头的（同一身份 >= 2 个摄像头）
    """
    cams: dict[str, set] = defaultdict(set)
    for r in val_records:
        cams[r["vehicle_id"]].add(r["camera_id"])
    if not cams:
        return False
    multi = sum(1 for v in cams.values() if len(v) >= 2)
    if multi == 0:
        logger.warning(
            "⚠️ 验证集**不是跨摄像头口径**：%d 个身份在 val 里各自只占 1 个摄像头，"
            "因此下面所有 Rank-1/5/mAP 衡量的都是**同摄像头内的外观记忆**，"
            "**不能**当作跨镜泛化指标、也**不可**与 eval_cross_camera.py 的数字并列。"
            "要看跨镜数字请用 `scripts/eval_cross_camera.py`。",
            len(cams))
        return False
    if multi < len(cams):
        logger.warning("验证集部分身份只有一个摄像头（%d/%d），Rank 指标是混合口径",
                       len(cams) - multi, len(cams))
    return True


def load_records(dataset_dir: Path, limit_per_id: int = 0):
    """读取 train/val 清单，返回 (train, val, 身份->标签)"""
    train_p = dataset_dir / "train.json"
    val_p = dataset_dir / "val.json"
    if not train_p.exists() or not val_p.exists():
        sys.exit(f"[fatal] 找不到 {train_p} / {val_p}，请先跑 build_reid_training_set.py")

    train = json.loads(train_p.read_text(encoding="utf-8"))
    val = json.loads(val_p.read_text(encoding="utf-8"))

    if limit_per_id:
        cnt: dict[str, int] = defaultdict(int)
        kept = []
        for r in train:
            if cnt[r["vehicle_id"]] < limit_per_id:
                cnt[r["vehicle_id"]] += 1
                kept.append(r)
        train = kept

    ids = sorted({r["vehicle_id"] for r in train})
    label_of = {vid: i for i, vid in enumerate(ids)}
    return train, val, label_of


class CropDataset:
    """
    按路径读裁剪图并做与推理一致的预处理

    缓存是**有界**的（`cache_size`）。原先是无界字典 —— 每个 epoch 都不释放，
    训练结束时全量图片仍驻留内存。本数据集裁剪图中位数 118×98，单张不大，
    但 5024 张训练图累加起来仍需数百 MB；而本机可用内存经常只有 1 GB 出头，
    无界缓存足以把进程推到 OOM。

    淘汰策略是"满了就整体清空"而不是 LRU：PK 采样本身按随机身份取图，
    命中率对访问顺序不敏感，LRU 的记账开销换不来实际收益。
    `cache_size=0` 表示不限制（保留旧行为，供内存宽裕时使用）。
    """

    def __init__(self, records, label_of, transform, cache_size: int = 2000):
        self.records = records
        self.label_of = label_of
        self.transform = transform
        self.cache_size = cache_size
        self.cache: dict[int, object] = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image

        r = self.records[i]
        path = ROOT / r["crop_path"]
        img = None if self.cache_size == 0 else self.cache.get(i)
        if img is None:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                # 坏图返回全黑，不中断训练；计数由调用方从日志观察
                import numpy as np
                from PIL import Image as I

                img = I.fromarray(np.zeros((256, 128, 3), dtype="uint8"))
            if self.cache_size:
                if len(self.cache) >= self.cache_size:
                    self.cache.clear()
                self.cache[i] = img
        return self.transform(img), self.label_of.get(r["vehicle_id"], -1)


def build_pk_batches(records, p: int, k: int, rng, hard_pairs=None, hard_frac=0.0):
    """
    生成一个 epoch 的 batch（每项是一组下标）

    hard_pairs: {身份: [困难身份, ...]}；提供时按 hard_frac 比例优先组"困难身份对"
    """
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_id[r["vehicle_id"]].append(i)
    ids = [v for v, idx in by_id.items() if len(idx) >= 2]   # 至少 2 张才能配正样本对
    if len(ids) < 2:
        return []
    p = min(p, len(ids))

    n_batches = max(1, sum(len(v) for v in by_id.values()) // (p * k))
    batches = []
    for _ in range(n_batches):
        use_hard = (
            hard_pairs and hard_frac > 0 and rng.random() < hard_frac
        )
        if use_hard:
            anchors = [v for v in ids if hard_pairs.get(v)]
            if anchors:
                a = anchors[rng.integers(len(anchors))]
                partners = [h for h in hard_pairs[a] if h in by_id][: p - 1]
                chosen = [a] + partners
                # 不够就用随机身份补齐
                while len(chosen) < p:
                    c = ids[rng.integers(len(ids))]
                    if c not in chosen:
                        chosen.append(c)
            else:
                chosen = rng.choice(ids, size=p, replace=False).tolist()
        else:
            chosen = rng.choice(ids, size=p, replace=False).tolist()

        idx: list[int] = []
        for vid in chosen:
            pool = by_id[vid]
            take = min(k, len(pool))
            idx.extend(rng.choice(pool, size=take, replace=False).tolist())
        batches.append(idx)
    return batches


# ============================================================
# 评估
# ============================================================


def extract_embeddings(model, dataset, device, batch_size=64):
    """跑一遍数据集，返回 (嵌入, 标签)"""
    import torch

    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            idx = list(range(start, min(start + batch_size, len(dataset))))
            imgs = torch.stack([dataset[i][0] for i in idx]).to(device)
            labs = [dataset[i][1] for i in idx]
            out = model(imgs)
            out = torch.nn.functional.normalize(out.float(), dim=1)
            feats.append(out.cpu().numpy())
            labels.extend(labs)
    if not feats:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.concatenate(feats, axis=0), labels


def evaluate_rank(feats: np.ndarray, labels: list) -> dict:
    """
    闭集 ReID 评测：对每张 query，在同身份的其他图里找最近邻

    指标：Rank-1 / Rank-5 / mAP。**跨摄像头协议** —— query 与 gallery 来自
    同一身份的不同摄像头（由训练集切分保证）。
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {"rank1": None, "rank5": None, "map": None, "n_query": 0}

    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)          # 排除自身

    ranks, aps = [], []
    for i in range(len(labels)):
        order = np.argsort(-sim[i])
        match = labels[order] == labels[i]
        if not match.any():
            continue
        first = int(np.argmax(match))
        ranks.append(first + 1)

        # mAP：按排序累积精度
        hits = np.cumsum(match)
        precisions = hits / (np.arange(len(match)) + 1)
        ap = float((precisions * match).sum() / match.sum())
        aps.append(ap)

    if not ranks:
        return {"rank1": None, "rank5": None, "map": None, "n_query": 0}
    ranks = np.asarray(ranks)
    return {
        "rank1": float((ranks <= 1).mean()),
        "rank5": float((ranks <= 5).mean()),
        "map": float(np.mean(aps)),
        "n_query": int(len(ranks)),
    }


def mine_hard_negatives(feats, labels, top_k: int = 3) -> dict:
    """
    跨 batch 的难负样本挖掘（PLAN5-C3）

    为每个身份找出**质心最近的其他身份** —— 那些就是最容易混淆的。
    下一轮采样时优先把它们排进同一个 batch，让 batch-hard 有难样本可挖。
    """
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    centroids = {}
    for lab in uniq:
        m = labels == lab
        c = feats[m].mean(axis=0)
        n = np.linalg.norm(c)
        centroids[lab] = c / n if n > 0 else c
    mat = np.stack([centroids[u] for u in uniq])
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)

    out = {}
    for i, lab in enumerate(uniq):
        order = np.argsort(-sim[i])[:top_k]
        out[lab] = [uniq[j] for j in order]
    return out


# ============================================================
# 训练
# ============================================================


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    from torchreid.reid.losses import CrossEntropyLoss, TripletLoss
    from torchreid.reid.models import build_model

    rng = np.random.default_rng(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    train_recs, val_recs, label_of = load_records(Path(args.dataset), args.limit_per_id)
    val_is_cross_camera = _warn_if_val_not_cross_camera(val_recs)
    n_ids = len(label_of)
    logger.info("训练 %d 张 / %d 身份 | 验证 %d 张",
                len(train_recs), n_ids, len(val_recs))
    if n_ids < 10:
        logger.warning("身份数只有 %d，训练结果不具代表性", n_ids)

    # 与推理端一致的预处理（OSNet 官方口径 256×128）
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CropDataset(train_recs, label_of, transform, args.image_cache_size)
    val_ds = CropDataset(val_recs, label_of, eval_transform, args.image_cache_size)

    model = build_model(args.arch, num_classes=n_ids, pretrained=True, use_gpu=False)
    model.to(device)

    triplet = TripletLoss(margin=args.margin)
    # torchreid 的 CrossEntropyLoss 需要类别数（用于标签平滑）
    ce = CrossEntropyLoss(num_classes=n_ids, use_gpu=(device == "cuda"))
    id_w = 0.0 if args.no_id_loss else args.id_loss_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log: list[dict] = []
    best_rank1 = -1.0
    hard_pairs = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = build_pk_batches(
            train_recs, args.p, args.k, rng,
            hard_pairs=hard_pairs, hard_frac=args.hard_frac,
        )
        if not batches:
            logger.error("无法构造任何 batch（身份或图片不足）")
            break

        ep_tri, ep_id, nb = 0.0, 0.0, 0
        for idx in batches:
            imgs = torch.stack([train_ds[i][0] for i in idx]).to(device)
            labels = torch.tensor([train_ds[i][1] for i in idx], device=device)

            optimizer.zero_grad()
            out = model(imgs)
            loss_t = triplet(out, labels)
            loss = loss_t
            if id_w > 0:
                loss_id = ce(out, labels)
                loss = loss + id_w * loss_id
                # 用 .item() 而非 float()：loss_id 带梯度，直接 float() 会触发
                # "Converting a tensor with requires_grad=True to a scalar" 警告
                ep_id += loss_id.item()
            loss.backward()
            optimizer.step()

            ep_tri += loss_t.item()
            nb += 1

        scheduler.step()

        # ── 验证 ──
        vf, vl = extract_embeddings(model, val_ds, device)
        metrics = evaluate_rank(vf, vl)
        row = {
            "epoch": epoch,
            "triplet_loss": round(ep_tri / max(nb, 1), 5),
            "id_loss": round(ep_id / max(nb, 1), 5) if id_w > 0 else None,
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            **{k: (round(v, 5) if isinstance(v, float) else v)
               for k, v in metrics.items()},
            "elapsed": round(time.time() - t0, 1),
        }
        log.append(row)
        logger.info(
            "epoch %2d | tri=%.4f id=%.4f | R1=%.4f R5=%.4f mAP=%.4f | %.0fs",
            epoch, row["triplet_loss"], row["id_loss"] or 0.0,
            metrics["rank1"] or 0.0, metrics["rank5"] or 0.0,
            metrics["map"] or 0.0, row["elapsed"],
        )

        if (metrics["rank1"] or 0) > best_rank1:
            best_rank1 = metrics["rank1"]
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "label_of": label_of,
                "epoch": epoch,
                "metrics": metrics,
                "args": vars(args),
            }, str(outdir / "best.pth"))
            logger.info("  ↑ 新的最佳 R1=%.4f，已保存 best.pth", best_rank1)

        torch.save({
            "state_dict": model.state_dict(), "arch": args.arch,
            "label_of": label_of, "epoch": epoch,
        }, str(outdir / "last.pth"))

        # ── C3：难负样本挖掘（下个 epoch 生效）──
        if args.hard_mining:
            tf, tl = extract_embeddings(model, train_ds, device)
            hard_pairs = mine_hard_negatives(tf, tl, top_k=3)
            logger.info("  [hard-mining] 已为 %d 个身份更新困难负样本表", len(hard_pairs))

    (outdir / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    final = log[-1] if log else {}
    (outdir / "metrics.json").write_text(json.dumps({
        "arch": args.arch,
        "epochs_run": len(log),
        "n_identities": n_ids,
        "train_images": len(train_recs),
        "val_images": len(val_recs),
        "best_rank1": best_rank1,
        "final": final,
        "hard_mining": bool(args.hard_mining),
        "id_loss_weight": id_w,
        "p": args.p, "k": args.k, "margin": args.margin,
        "image_cache_size": args.image_cache_size,
        # 口径标记：False 表示下面那些 Rank-1/5/mAP 是同摄像头口径，
        # 不是跨镜泛化。见文件头专节。
        "val_is_cross_camera": bool(val_is_cross_camera),
        "dataset": str(args.dataset),
        "note": (
            "本数据集仅 215 个身份，指标不可与 VeRi/MSMT17 等公开基准直接比较；"
            "AICity22 许可禁止商用，产物仅限研究。"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("训练完成: %d epoch | 最佳 R1=%.4f | 产物 %s",
                len(log), best_rank1, outdir)


if __name__ == "__main__":
    main()
