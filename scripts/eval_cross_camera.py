"""
scripts.eval_cross_camera - 跨镜车辆再识别基准（二期 T7）

用「弱身份」方式评测跨镜匹配：给定某摄像头里的一条检测（query），在
其它摄像头里按特征余弦相似度检索，看 top-K 命中的是否同一辆车
（真值来自 cityflow_results.json 的 vehicle_id，与 GT track_id 同源：
V0034 ↔ GT 34）。

本脚本不依赖 vehicle_id 参与匹配（只用它当评测真值），因此能真实反映
「没有车牌/真值时的弱身份能力」——这是系统在真实场景下的价值所在。

用法:
    python scripts/eval_cross_camera.py --features output/clip_vectors.faiss --sample 5000
    python scripts/eval_cross_camera.py --features output/reid/reid_vectors.faiss --sample 5000

指标: Rank-1 / Rank-5 / mAP（标准跨镜 ReID 指标）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _vehicle_id(target_id: str) -> str:
    for p in (target_id or "").split("_"):
        if p.startswith("V") and p[1:].isdigit():
            return p
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="FAISS 索引路径或 .npy 特征矩阵")
    ap.add_argument("--sample", type=int, default=5000, help="query 采样数（全量较慢）")
    ap.add_argument("--k", type=int, default=50, help="检索 top-K")
    args = ap.parse_args()

    data = json.load(open(_PROJECT_ROOT / "output" / "cityflow_results.json", encoding="utf-8"))
    dets = data["detections"]
    n = len(dets)
    cam = np.array([d.get("camera_id", "") for d in dets])
    vid = np.array([_vehicle_id(d.get("target_id", "")) for d in dets])

    # 加载特征矩阵
    fp = Path(args.features)
    import faiss
    if fp.suffix == ".faiss":
        index = faiss.read_index(str(fp))
        dim = index.d
        X = np.zeros((index.ntotal, dim), dtype=np.float32)
        for i in range(index.ntotal):
            index.reconstruct(i, X[i])
        if index.ntotal != n:
            print(f"[warn] 索引行数 {index.ntotal} != 检测数 {n}，按 min 截断")
            n = min(index.ntotal, n)
            X, cam, vid = X[:n], cam[:n], vid[:n]
    else:
        X = np.load(str(fp))
        n = min(len(X), n)
        X, cam, vid = X[:n], cam[:n], vid[:n]
        dim = X.shape[1]

    # 归一化（FAISS IndexFlatIP 用内积，等价余弦）
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    idx = faiss.IndexFlatIP(dim)
    idx.add(X)

    # 采样 query
    rng = np.random.default_rng(0)
    q_ids = rng.choice(n, size=min(args.sample, n), replace=False)

    # 先取 top-M，再剔除同摄像头与自身（同摄像头里同车检测极多，不剔除会虚高）
    M = 200
    D, I = idx.search(X[q_ids], M)
    aps = []
    r1 = r5 = 0
    for qi, q in enumerate(q_ids):
        qvid = vid[q]
        qcam = cam[q]
        hits = []
        for gi in I[qi]:
            if gi == q or cam[gi] == qcam:  # 排除自身 + 同摄像头
                continue
            hits.append(1 if vid[gi] == qvid else 0)
            if len(hits) >= args.k:
                break
        if not hits:
            continue
        if hits[0] == 1:
            r1 += 1
        if sum(hits[:5]) > 0:
            r5 += 1
        # AP
        precs = [sum(hits[:i + 1]) / (i + 1) for i in range(len(hits)) if hits[i] == 1]
        aps.append(sum(precs) / len(precs) if precs else 0.0)

    nq = len(aps)
    print(f"query 数 = {nq}")
    print(f"Rank-1 = {r1/nq:.4f}  ({r1}/{nq})")
    print(f"Rank-5 = {r5/nq:.4f}  ({r5}/{nq})")
    print(f"mAP    = {np.mean(aps):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
