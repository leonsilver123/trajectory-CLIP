"""
scripts/train_edge_scorer.py - 用 BCE + Ranking Loss 训练跨镜边评分器（PLAN5-C4）

用法:
    python scripts/train_edge_scorer.py
    python scripts/train_edge_scorer.py --epochs 200 --neg-ratio 3

产物:
    output/edge_scorer/{model.pt,metrics.json,feature_stats.json}

## 对应简历里的哪一句

「边评分用 **BCE 加 Ranking Loss** 联合监督」——本脚本就是它的实现。

## 与现有 `CrossCameraScorer` 的关系（重要）

现有的 `src/stitching/scoring.py` 是一个**手写加权和**：六维分数按
`configs/default.yaml` 的权重相加。那些权重是**网格搜索调出来的**（PLAN2 的 P-A），
不是学出来的。

本脚本**不改动那条在线路径**，而是并排做一件事：把六维分项当作**特征**，
用带标签的边训练一个判别模型 —— 即"学权重"而非"调权重"。
两者的 IDF1 对照即本任务的验收。

## 标签怎么来

数据集里每条检测带真实 `vehicle_id`，所以边的正负是**确定的**：
- 两个 tracklet 的 `vehicle_id` 相同 → 正边（同一辆车）
- 不同 → 负边（不同车）

这不是启发式标注，是数据集给定的事实。

## 🔴 结果不可当作"学到了正确权重"的证据（2026-09-21 实测后的结论）

跑完全量（218 身份）后得到 val AUC **0.855**，手写加权和同批边是 **0.716**，
表面看"学习型完胜"。**但这个结论是错的**，原因是**负样本的采样分布不对**：

| 维度 | 正边均值 | 负边均值 | 真相 |
|---|---|---|---|
| `spatial_score` | **0.5000** | 0.2006 | **正边 100% 取值恰为 0.5（中性）** |
| `temporal_score` | 0.2876 | 0.2084 | **正边 70% 取值恰为 0** |

`spatial` 在这个数据上不是"空间可达性"，而是**「是否同场景」的指示器** ——
同一场景的摄像头 GPS 坐标相同（全数据集 46 个摄像头只有 4 组坐标），
距离算出来是 0，按"未知距离"给了中性 0.5。于是：

- 正边（同一辆车）**必然同场景** ⇒ 100% 落在中性组
- 负边是**随机跨车配对**，包含跨场景对 ⇒ 63.3% 落在非中性组

**AUC 0.796 全部来自这一条**。"剔除 `spatial==0.5` 的子集"里有 723 条边、
**0 条正边** —— 判别度完全来自"正边必同场景"这个采样假象。

**为什么这毁掉了整个对照**：生产路径的 `CandidateEdgeGenerator` 有空间搜索半径，
**跨场景对根本不会成为候选边**。也就是说这个判别器在生产分布上**根本不存在**，
学到的权重（`spatial` 拿到 +4.98 的压倒性权重）**不可迁移**。

**因此**：本脚本产出的 AUC/权重，**不能**用来支持"学权重优于调权重"，
也**不能**用来质疑手写权重。它只说明"损失实现正确、能训练、在**这个**分布上有区分度"。

**要得到真结论，负样本必须从生产候选边的分布里取** —— 即先跑
`CandidateEdgeGenerator` 拿到真实的候选对，再在其中标正负。
本脚本目前是直接随机配对，这是它最大的方法缺陷，见 `known_limitation` 字段。

## 两个损失各管什么

- **BCE**：逐边判断"这条边是不是真的"。给出校准过的概率。
- **Ranking**（pairwise margin）：正边的分必须**高于**负边。BCE 只管单点，
  可能出现"所有边都预测 0.5"这种没区分度但 loss 不高的情况；ranking 直接惩罚
  正负边的**相对次序**错误。

两者相加：`loss = BCE + λ · Ranking`。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger

logger = get_logger("scripts.train_edge_scorer")

DEFAULT_OUTDIR = ROOT / "output" / "edge_scorer"
FEATURE_NAMES = [
    "appearance_score", "attribute_score", "plate_score",
    "temporal_score", "spatial_score", "direction_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BCE + Ranking 训练跨镜边评分器")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--ranking-weight", type=float, default=1.0, help="ranking 损失权重 λ")
    p.add_argument("--margin", type=float, default=0.2, help="ranking 的间隔")
    p.add_argument("--neg-ratio", type=float, default=3.0, help="负边:正边 采样比")
    p.add_argument("--max-identities", type=int, default=60, help="参与构建的身份数上限")
    p.add_argument("--rebuild-edges", action="store_true",
                   help="忽略构边缓存，强制重新构边（构边在 218 身份上约 10 分钟）")
    p.add_argument("--negatives-from-candidates", action="store_true",
                   help="负样本取自生产候选边（正确做法）；默认是随机跨车配对（有采样假象）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# 从 datastore 构建 tracklet 与带标签的边
# ============================================================


def build_tracklets(max_identities: int, seed: int):
    """
    把检测按 (vehicle_id, camera_id) 聚成 tracklet

    这是**评测用的构建方式**，与在线路径的 Tracklet 构造不同 ——
    在线是用跟踪算法分段，这里直接用真实 vehicle_id 分组，
    为的是拿到"哪些边是真连接"的确定标签。
    """
    from src.common.ids import extract_vehicle_id
    from src.storage.datastore import load_results

    data = load_results()
    if not data:
        sys.exit("[fatal] 数据不可用")

    groups: dict[tuple, list] = defaultdict(list)
    rows_of: dict[tuple, list] = defaultdict(list)
    for row, det in enumerate(data.get("detections", [])):
        vid = extract_vehicle_id(det.get("target_id") or "")
        cam = det.get("camera_id")
        if vid and cam:
            groups[(vid, cam)].append(det)
            rows_of[(vid, cam)].append(row)

    # 只保留出现在 ≥2 个摄像头的身份（跨镜边才有意义）
    by_id: dict[str, list] = defaultdict(list)
    for (vid, cam), dets in groups.items():
        by_id[vid].append((cam, dets))

    ids = [v for v, cams in by_id.items() if len(cams) >= 2]
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:max_identities]

    logger.info("参与构建的身份: %d（原本 %d 个跨多摄像头身份）", len(ids), len([1 for v, c in by_id.items() if len(c) >= 2]))
    return {v: by_id[v] for v in ids}, {k: v for k, v in rows_of.items() if k[0] in set(ids)}


def load_tracklet_reid(rows_of, dim_hint: int = 2048):
    """
    按 (身份, 摄像头) 求检测级 ReID 向量的均值，得到"轨迹级"外观向量

    ## 为什么必须有这一步（2026-09-21 实测踩到）

    `make_tracklet` 初版把 `avg_reid_vector` / `avg_clip_vector` 都留成 None，
    于是 `CandidateEdgeGenerator._compute_appearance_similarity` 恒返回中性 0.5，
    而规则 8 的阈值来自配置的 `min_appearance_score`（0.95）——
    **0.5 < 0.95，所有候选边被无条件拒掉**。实测：792,368 对 → 粗筛通过 138,000
    → **有效候选边 0 条**。

    这是构造缺陷，**不是"生产门控把边全滤掉了"** —— 生产路径的 tracklet 是带
    ReID 向量的（轨迹级旁挂表），门控有真实证据可判。不把这个补上，
    `--negatives-from-candidates` 这条正确路线根本跑不起来。

    用检测级 fast-reid 向量（68,349 × 2048，行号与 detections 对齐）按组求均值，
    等效于生产从轨迹级旁挂表取向量的做法。用 mmap 逐块求均值，避免把
    560MB 的矩阵整个读进内存（本机可用内存常年只有 1–3 GB）。
    """
    npy = ROOT / "output" / "reid" / "reid_vectors.npy"
    if not npy.exists():
        logger.warning("找不到 %s —— 轨迹级外观向量将缺失，候选边门控会拒掉全部边", npy)
        return {}
    arr = np.load(str(npy), mmap_mode="r")
    out = {}
    n_ok = 0
    for key, rows in rows_of.items():
        vecs = []
        for r in rows:
            if r >= arr.shape[0]:
                continue
            v = np.asarray(arr[r], dtype=np.float32)
            if not np.isnan(v).any():
                vecs.append(v)
        if vecs:
            m = np.mean(vecs, axis=0)
            nrm = float(np.linalg.norm(m))
            out[key] = m / nrm if nrm > 0 else m
            n_ok += 1
    logger.info("轨迹级 ReID 向量: %d/%d 个 (身份,摄像头) 组有有效向量（dim=%d）",
                n_ok, len(rows_of), arr.shape[1])
    return out


def make_tracklet(cam: str, dets: list, vid: str, reid_vec=None):
    """
    把一组检测包成 scoring 需要的 Tracklet 对象

    注意 `Tracklet.start_time/end_time` 的类型是 **datetime**，不是字符串 ——
    `scoring._score_temporal` 会对它们做减法（`time_diff_seconds`），
    传字符串会在运行时抛 `unsupported operand type(s) for -: 'str' and 'str'`。
    这里复用 builder 同款的 `_parse_timestamp` 解析。
    """
    from src.common.data_models import Tracklet
    from src.trajectory.builder import _parse_timestamp

    dets = sorted(dets, key=lambda d: d.get("timestamp", ""))
    attrs = (dets[0].get("attributes") or {}) if dets else {}
    first = _parse_timestamp(dets[0].get("timestamp")) if dets else None
    last = _parse_timestamp(dets[-1].get("timestamp")) if dets else None
    if first is None or last is None:
        # 时间解析不出来时用同一时刻占位：时间维度会退化为中性，
        # 但**不会**让整条边构造失败（缺失该维本就有中性分处理）
        from datetime import datetime
        first = last = datetime(2020, 1, 1)

    return Tracklet(
        tracklet_id=f"{vid}_{cam}",
        camera_id=cam,
        target_type=dets[0].get("target_type", "vehicle") if dets else "vehicle",
        start_time=first,
        end_time=last,
        instances=[],
        direction="",
        plate_number=None,
        attributes=attrs,
        # 必须带上外观向量，否则规则 8 的外观门控会把所有候选边拒掉（见
        # `load_tracklet_reid` 的说明）。ReID 优先于 CLIP，与生产同序。
        avg_reid_vector=reid_vec,
        avg_clip_vector=None,
        keyframe_paths=[],
    )


def build_edges_from_candidates(tracklet_index, seed: int):
    """
    用**生产候选边**构造训练/验证边（正确做法，见 docstring 的 known_limitation）

    与 `build_edges` 的区别只在**负样本怎么来**：

    - `build_edges`：随机跨车配对 → 包含跨场景对 → `spatial_score` 退化成
      「是否同场景」指示器（正边 100% 中性），学到的权重不可迁移到线上。
    - 本函数：先跑生产同款 `CandidateEdgeGenerator`（含拓扑/时间/距离/方向/
      类别粗筛 + 外观精筛），**只在这些真实候选对里标正负**。这样负样本分布
      与线上一致，`spatial_score` 才真的是"空间可达性"而不是同场景指示器。

    标签仍是数据集给定的事实：`vehicle_id` 相同 = 正边、不同 = 负边。
    """
    from src.common.config import get_config
    from src.data_governance.camera_manager import CameraManager
    from src.data_governance.road_topology import RoadTopology
    from src.stitching.candidate_edge import CandidateEdgeGenerator
    from src.stitching.scoring import CrossCameraScorer

    meta = ROOT / "configs" / "cityflow_camera_metadata.yaml"
    cm = CameraManager(str(meta))
    rt = RoadTopology(str(meta))

    # 阈值走生产同款配置，不在这里硬编码
    cfg = get_config()
    gen = CandidateEdgeGenerator(
        camera_manager=cm,
        road_topology=rt,
        max_time_gap_seconds=float(cfg.get("stitching.max_time_gap_seconds", 600.0)),
        min_appearance_score=float(cfg.get("stitching.min_appearance_score", 0.95)),
        spatial_search_radius_km=float(cfg.get("stitching.spatial_search_radius_km", 10.0)),
        scorer=CrossCameraScorer(cm, rt),
    )

    all_tracklets = [tl for _, cams in tracklet_index.items() for _, tl in cams]
    vid_of = {tl.tracklet_id: vid for vid, cams in tracklet_index.items() for _, tl in cams}
    logger.info("跑生产候选边生成器，输入 %d 个 tracklet", len(all_tracklets))
    edges = gen.generate_candidates(all_tracklets)
    logger.info("生产候选边: %d 条", len(edges))

    rows = []
    for e in edges:
        src_id = getattr(e, "source_tracklet_id", None) or getattr(e, "source_id", None)
        dst_id = getattr(e, "target_tracklet_id", None) or getattr(e, "target_id", None)
        if src_id is None or dst_id is None:
            continue
        va, vb = vid_of.get(src_id), vid_of.get(dst_id)
        if va is None or vb is None:
            continue
        feats = [float(getattr(e, n, 0.0) or 0.0) for n in FEATURE_NAMES]
        rows.append({"features": feats, "label": 1 if va == vb else 0,
                     "src": src_id, "dst": dst_id})

    n_pos = sum(r["label"] for r in rows)
    logger.info("其中正边 %d 条 / 负边 %d 条", n_pos, len(rows) - n_pos)
    if n_pos == 0 or n_pos == len(rows):
        logger.error("候选边里只有单一类别，无法训练 —— 说明生产门控把某一类全滤掉了")

    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.8)
    return rows[:cut], rows[cut:]


def build_edges(tracklet_index, neg_ratio: float, seed: int):
    """
    生成带标签的候选边

    Returns:
        rows: [{"features": [...], "label": 0/1, "src": ..., "dst": ...}]
    """
    from src.data_governance.camera_manager import CameraManager
    from src.data_governance.road_topology import RoadTopology
    from src.stitching.scoring import CrossCameraScorer

    meta = ROOT / "configs" / "cityflow_camera_metadata.yaml"
    scorer = CrossCameraScorer(CameraManager(str(meta)), RoadTopology(str(meta)))

    # (身份, 摄像头) -> Tracklet
    index = [(vid, cam, tl) for vid, cams in tracklet_index.items() for cam, tl in cams]

    pos, neg = [], []
    for i, (vid_a, cam_a, tl_a) in enumerate(index):
        for vid_b, cam_b, tl_b in index[i + 1:]:
            if cam_a == cam_b:
                continue                       # 同摄像头的边不是跨镜边
            edge = scorer.score(tl_a, tl_b)
            feats = [float(getattr(edge, n)) for n in FEATURE_NAMES]
            row = {"features": feats, "src": tl_a.tracklet_id, "dst": tl_b.tracklet_id}
            if vid_a == vid_b:
                pos.append({**row, "label": 1})
            else:
                neg.append({**row, "label": 0})

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_neg = min(len(neg), int(len(pos) * neg_ratio))
    logger.info("正边 %d 条 | 负边 %d 条（采样后 %d 条）", len(pos), len(neg), n_neg)

    rows = pos + neg[:n_neg]
    rng.shuffle(rows)

    # 按边分组切分（同一对 tracklet 只能落一侧，避免泄漏）
    cut = int(len(rows) * 0.8)
    return rows[:cut], rows[cut:]


# ============================================================
# 模型与损失
# ============================================================


def train(rows_train, rows_val, args):
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)

    def to_tensors(rows):
        X = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        return X, y

    Xtr, ytr = to_tensors(rows_train)
    Xva, yva = to_tensors(rows_val)
    if len(Xtr) == 0 or len(Xva) == 0:
        sys.exit("[fatal] 训练或验证边为空")

    # 线性打分器：输出 logit。**刻意用线性** ——
    # 目的是"学到六维的权重"，与手写加权和可比；上 MLP 会让对照失去意义。
    model = nn.Linear(Xtr.shape[1], 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr).squeeze(-1)
        loss_bce = bce(logits, ytr)

        # Ranking：随机配对正负边，惩罚"正边分不高于负边分"
        pos_idx = (ytr > 0.5).nonzero(as_tuple=True)[0]
        neg_idx = (ytr <= 0.5).nonzero(as_tuple=True)[0]
        if len(pos_idx) and len(neg_idx):
            perm = torch.randperm(len(neg_idx))[: len(pos_idx)]
            pair_neg = neg_idx[perm]
            diff = logits[pos_idx] - logits[pair_neg]
            loss_rank = torch.clamp(args.margin - diff, min=0).mean()
        else:
            loss_rank = torch.tensor(0.0)

        loss = loss_bce + args.ranking_weight * loss_rank
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                auc = _auc(model(Xva).squeeze(-1).numpy(), yva.numpy())
                acc = float(((model(Xva).squeeze(-1) > 0).float() == yva).float().mean())
            # 用 .item() 取值：loss_bce 带梯度，直接 float() 会触发
            # "Calling float on a tensor that requires grad is deprecated" 警告
            bce_v, rank_v = loss_bce.item(), float(loss_rank)
            log.append({"epoch": epoch, "bce": round(bce_v, 5),
                        "ranking": round(rank_v, 5),
                        "val_auc": round(auc, 5), "val_acc": round(acc, 5)})
            logger.info("epoch %3d | bce=%.4f rank=%.4f | val AUC=%.4f acc=%.4f",
                        epoch, bce_v, rank_v, auc, acc)

    return model, log


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC（用秩和公式，不依赖 sklearn）"""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 对每个正样本，统计有多少负样本分数更低
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> None:
    args = parse_args()
    import torch

    index, rows_of = build_tracklets(args.max_identities, args.seed)
    reid_of = load_tracklet_reid(rows_of)
    # 把 (cam, dets) 变成 (cam, Tracklet)，并挂上轨迹级 ReID 向量
    tl_index = {
        vid: [(cam, make_tracklet(cam, dets, vid, reid_of.get((vid, cam))))
              for cam, dets in cams]
        for vid, cams in index.items()
    }

    # ── 构边缓存 ──
    #
    # 构边是 O(tracklet²) 次 `scorer.score()`，在 218 个身份（约 650 个 tracklet）
    # 上是约 **10 分钟**，而训练本身只要几秒。调参时反复重付这 10 分钟毫无意义，
    # 因此把边缓存到磁盘，键含所有影响构边的参数。
    # 换参数（identities / neg_ratio / seed）会自动 miss，不会拿错缓存。
    outdir_early = Path(args.outdir)
    outdir_early.mkdir(parents=True, exist_ok=True)
    mode = "cand" if args.negatives_from_candidates else "rand"
    cache_p = outdir_early / (
        f"edges_{mode}_n{args.max_identities}_r{args.neg_ratio}_s{args.seed}.json")

    if cache_p.exists() and not args.rebuild_edges:
        blob = json.loads(cache_p.read_text(encoding="utf-8"))
        rows_train, rows_val = blob["train"], blob["val"]
        logger.info("复用构边缓存 %s（train %d / val %d）",
                    cache_p.name, len(rows_train), len(rows_val))
    else:
        if args.negatives_from_candidates:
            rows_train, rows_val = build_edges_from_candidates(tl_index, args.seed)
        else:
            rows_train, rows_val = build_edges(tl_index, args.neg_ratio, args.seed)
        if not rows_train:
            # **不要把空结果写进缓存** —— 2026-09-21 踩到：初版无条件写缓存，
            # 一次因缺少外观向量而"0 条边"的失败运行把空结果落盘，
            # 之后每次重跑都命中这个空缓存、每次都以同样的方式失败，
            # 看起来像"构边就是产不出边"，掩盖了真正的原因。
            logger.error("构边产出为空，**不写缓存**（避免把失败结果固化下来）")
        else:
            cache_p.write_text(json.dumps(
                {"train": rows_train, "val": rows_val,
                 "max_identities": args.max_identities, "neg_ratio": args.neg_ratio,
                 "seed": args.seed, "negatives_from_candidates": args.negatives_from_candidates,
                 "feature_names": FEATURE_NAMES},
                ensure_ascii=False), encoding="utf-8")
            logger.info("构边缓存已写出: %s", cache_p.name)

    if not rows_train:
        sys.exit("[fatal] 没有构造出任何边")

    # ── 护栏：正样本太少时 AUC 没有意义 ──
    #
    # 2026-09-21 踩到。用生产候选边做负采样时（--negatives-from-candidates），
    # 实测只产出 534 条边、其中正边 13 条（验证集里约 3 条），训练出
    # **val AUC = 1.0000**。这个 1.0 不是"完美"，是**退化的**：3 个正样本
    # 恰好排在 104 个负样本前面而已，换一批样本就会剧烈波动。
    #
    # 不加护栏的话，日志里那行 `val AUC=1.0000` 会被当成"学习型打分器完胜
    # 手写权重"的证据，而事实恰恰相反 —— 它说明候选边分布极度不平衡。
    n_val_pos = sum(1 for r in rows_val if r["label"] == 1)
    n_val_neg = len(rows_val) - n_val_pos
    n_tr_pos = sum(1 for r in rows_train if r["label"] == 1)
    degenerate = n_val_pos < 20 or n_val_neg < 20
    if degenerate:
        logger.error(
            "⚠️ 验证边严重不平衡：正 %d / 负 %d（训练 正 %d / 负 %d）—— "
            "**此条件下的 AUC / acc 不可解释**，AUC=1.0 只说明正样本恰好排前面，"
            "不代表模型可用。读数前请先看这里。",
            n_val_pos, n_val_neg, n_tr_pos, len(rows_train) - n_tr_pos)

    model, log = train(rows_train, rows_val, args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "args": vars(args)}, str(outdir / "model.pt"))

    w = model.weight.detach().squeeze(0).numpy()
    b = float(model.bias.detach())
    learned = {n: round(float(v), 5) for n, v in zip(FEATURE_NAMES, w)}

    # ── 特征统计：识别"常数维度" ──
    #
    # 这一步不是装饰。冒烟测试里 `appearance_score` / `plate_score` /
    # `direction_score` **三个维度的学到的系数完全相同（-0.50091）** ——
    # 这不是巧合，而是它们在**所有边上取值恒定**、方差为 0 的表现：
    #   - plate：本数据集 68,349 条检测中带车牌的为 0 条 → 恒为中性
    #   - appearance：tracklet 没有 CLIP/ReID 向量 → 恒为中性
    #   - direction：上面构造 Tracklet 时 direction="" → 恒为无证据
    # 常数维度的"权重"没有意义（只是替 bias 分担了一点偏移），
    # 但**光看数字看不出来**，所以这里显式标出来。
    #
    # 统计从 `rows_val` 现算，不要伸手去拿 `train()` 的局部张量 ——
    # 2026-09-21：初版写成 `Xva.numpy()`，`Xva` 是 `train()` 的局部变量，
    # 在 `main()` 里根本不存在，于是**跑完十分钟构边、训完 300 epoch 之后**
    # 才在最后一步抛 `NameError` 把整轮结果丢掉。
    Xva_np = np.asarray([r["features"] for r in rows_val], dtype=np.float32)
    stats = {}
    for i, n in enumerate(FEATURE_NAMES):
        col = Xva_np[:, i]
        std = float(col.std())
        stats[n] = {
            "mean": round(float(col.mean()), 5),
            "std": round(std, 6),
            "min": round(float(col.min()), 5),
            "max": round(float(col.max()), 5),
            "learned_weight": learned[n],
            # 方差为 0 ⇒ 该维度不携带任何区分信息，权重不可解释
            "constant_on_val": bool(std < 1e-9),
        }
    const_dims = [n for n, s in stats.items() if s["constant_on_val"]]
    if const_dims:
        logger.warning(
            "以下维度在验证边上取值恒定（无区分力），其学到的权重**不可解释**：%s",
            const_dims)
    (outdir / "feature_stats.json").write_text(
        json.dumps({"val_set": stats, "constant_dimensions": const_dims},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 与在线手写权重的对照（只列能对上的维度）
    from src.stitching.scoring import DEFAULT_VEHICLE_WEIGHTS
    manual = {
        "appearance_score": DEFAULT_VEHICLE_WEIGHTS.get("reid"),
        "attribute_score": DEFAULT_VEHICLE_WEIGHTS.get("attribute"),
        "plate_score": DEFAULT_VEHICLE_WEIGHTS.get("plate"),
        "temporal_score": DEFAULT_VEHICLE_WEIGHTS.get("temporal"),
        "spatial_score": DEFAULT_VEHICLE_WEIGHTS.get("topology"),
    }

    metrics = {
        "n_train_edges": len(rows_train),
        "n_val_edges": len(rows_val),
        "learned_weights": learned,
        "bias": b,
        "manual_weights_for_reference": manual,
        "final": log[-1] if log else None,
        "constant_dimensions": const_dims,
        "class_balance": {
            "train_positive": n_tr_pos,
            "train_negative": len(rows_train) - n_tr_pos,
            "val_positive": n_val_pos,
            "val_negative": n_val_neg,
            "degenerate": bool(degenerate),
        },
        "known_limitation": (
            "**负样本不是生产分布**：负边由随机跨车配对生成，包含跨场景对；"
            "而生产路径 CandidateEdgeGenerator 有空间搜索半径，跨场景对不会成为候选。"
            "后果是 spatial_score 在本实验里退化成「是否同场景」的指示器"
            "（正边 100% 取值中性 0.5，负边仅 36.7%），学到的权重不可迁移到线上。"
            "本文件的 AUC/权重**不能**用来支持或否定手写权重。"
            "要做真对照，需从生产候选边里取负样本。详见脚本 docstring。"
        ),
        "note": (
            "learned_weights 是**线性打分器学出来的**六维系数，可与 "
            "configs/default.yaml 的手写权重对照。注意两者量纲不同："
            "手写权重是加权和的系数（和≈1），这里是 logit 的系数（无归一化约束），"
            "**只能比相对大小与符号，不能直接比绝对值**。"
            "另：`constant_dimensions` 列出的维度在数据上取值恒定（方差为 0），"
            "它们的权重不含信息，读数时应忽略 —— 见 feature_stats.json。"
        ),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("学到的六维系数: %s", learned)
    logger.info("（对照）手写权重: %s", manual)
    logger.info("产物: %s", outdir)


if __name__ == "__main__":
    main()
