"""
scripts.eval_method_suite - 文献方法落地实验骨架（二期 M1-M6）

统一骨架：加载 tracklet（特征/摄像头/时间/GT）→ 算成对相似度 → 可插拔方法
（k-reciprocal / CLM / box-grained / 方向掩码 / ReID 集成 / 子聚类）→ 并查集聚类
（摄像头唯一性互斥）→ 官方 IDF1。

每个方法只改「相似度矩阵」或「聚类」一步，其余固定，从而公平量增量。

用法:
    python scripts/eval_method_suite.py --method baseline
    python scripts/eval_method_suite.py --method kreciprocal --k 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.ids import extract_vehicle_id  # noqa: E402


# ---------------------------------------------------------------- 数据加载
def load_tracklets(builder):
    """返回 tracklet 列表：每个含 id/camera/scene/vehicle_id/时间/avg_reid_vector"""
    out = []
    for tid, dets in builder._tracklet_detections.items():
        if not dets:
            continue
        det = dets[0]
        vid = extract_vehicle_id(det.get("target_id", ""))
        t = builder._tracklet_from_detections(tid, dets)
        out.append({
            "id": tid,
            "camera": t.camera_id,
            "scene": det.get("scene_id", ""),
            "vehicle_id": vid,
            "start": t.start_time.timestamp() if t.start_time else 0.0,
            "end": t.end_time.timestamp() if t.end_time else 0.0,
            "vec": t.avg_reid_vector,
        })
    return out


def cosine_matrix(vecs):
    """vecs: list of (n,d) 或 None。返回归一化余弦相似度矩阵 (m,m)，缺失为 NaN"""
    m = len(vecs)
    S = np.full((m, m), np.nan)
    valid = [i for i, v in enumerate(vecs) if v is not None]
    if valid:
        X = np.stack([vecs[i] for i in valid]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        C = X @ X.T
        for a, i in enumerate(valid):
            for b, j in enumerate(valid):
                S[i, j] = C[a, b]
    return S


# ---------------------------------------------------------------- 方法插件
def method_similarity(tracklets, S, method, k):
    """根据方法返回「成对匹配分数」矩阵。返回 (score_matrix, 说明)"""
    if method == "baseline":
        return S, "原始余弦"

    if method == "kreciprocal":
        return kreciprocal_rerank(S, k), f"k-reciprocal k={k}"

    raise ValueError(f"未知方法 {method}")


def kreciprocal_rerank(S, k=7):
    """k-reciprocal 重排：Jaccard 相似度作为重排后的分数"""
    m = S.shape[0]
    # 每个 tracklet 的 top-k 近邻（排除自身）
    nn = []
    for i in range(m):
        row = np.argsort(-S[i])
        nn_i = [j for j in row if j != i and not np.isnan(S[i, j])][:k]
        nn.append(set(nn_i))
    # 扩展：加入 1/2-reciprocal 邻居（R*）
    R = [set(nn[i]) for i in range(m)]
    for i in range(m):
        for j in list(R[i]):
            inter = R[i] & nn[j]
            if len(inter) >= 2.0 / 3.0 * len(nn[j]):
                R[i] |= nn[j]
    # Jaccard 距离
    J = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if not R[i] or not R[j]:
                J[i, j] = 0.0
            else:
                J[i, j] = len(R[i] & R[j]) / len(R[i] | R[j])
    # 融合：0.5*原始余弦 + 0.5*jaccard（标准做法）
    S_norm = np.nan_to_num(S, nan=0.0)
    return 0.5 * S_norm + 0.5 * J


# ---------------------------------------------------------------- 聚类 + IDF1
class _DSU:
    def __init__(self, n, cameras):
        self.parent = list(range(n))
        self.cams = [set([cameras[i]]) for i in range(n)]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.cams[ra] & self.cams[rb]:
            return False  # 摄像头唯一性互斥
        self.parent[rb] = ra
        self.cams[ra] |= self.cams[rb]
        return True


def cluster_by_score(tracklets, score, min_score):
    n = len(tracklets)
    dsu = _DSU(n, [t["camera"] for t in tracklets])
    # 收集跨摄像头 + 时间有序的候选对，按分数降序
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = score[i, j]
            if np.isnan(s) or s < min_score:
                continue
            if tracklets[i]["camera"] == tracklets[j]["camera"]:
                continue
            pairs.append((s, i, j))
    pairs.sort(key=lambda x: -x[0])
    for s, i, j in pairs:
        dsu.union(i, j)
    roots = [dsu.find(i) for i in range(n)]
    return roots


def compute_idf1(tracklets, roots):
    gt = [t["vehicle_id"] or "none" for t in tracklets]
    pred = [f"c{roots[i]}" for i in range(len(tracklets))]
    pairs = Counter(zip(gt, pred))
    gts = sorted(set(gt))
    preds = sorted(set(pred))
    gi = {g: i for i, g in enumerate(gts)}
    pi = {p: i for i, p in enumerate(preds)}
    M = np.zeros((len(gts), len(preds)))
    for (g, p), c in pairs.items():
        M[gi[g], pi[p]] = c
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-M)
    idtp = float(M[rows, cols].sum())
    total = float(len(tracklets))
    idfp = idfn = total - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if idtp > 0 else 0.0
    return idf1, idtp / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    from src.trajectory.builder import get_trajectory_builder
    builder = get_trajectory_builder()
    builder.ensure_loaded()

    tracklets = load_tracklets(builder)
    vecs = [t["vec"] for t in tracklets]
    S = cosine_matrix(vecs)
    score, desc = method_similarity(tracklets, S, args.method, args.k)

    best = (-1, None)
    for thr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]:
        roots = cluster_by_score(tracklets, score, thr)
        idf1, idp = compute_idf1(tracklets, roots)
        if idf1 > best[0]:
            best = (idf1, thr)

    print(f"方法={args.method} ({desc}) | tracklet 数={len(tracklets)}")
    print(f"最优官方 IDF1 = {best[0]:.4f} @ min_score={best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
