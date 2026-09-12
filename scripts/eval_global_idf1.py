"""
scripts.eval_global_idf1 - 全局轨迹聚类 + 官方 motmetrics IDF1（二期 P-C）

把「单目标贪心回溯」升级成「全局轨迹聚类」：对同场景所有 tracklet 做
union-find 聚类（按候选边分数降序合并，约束「每摄像头每轨迹至多一条」，
即互斥约束），然后用 motmetrics 出**官方 IDF1**（标准 MTMC 指标），
与 per-target 的 eval_chain_idf1.py 口径互补。

真值：vehicle_id（已核实与 AICity22 GT 全局 Id 同源 230/230）。

用法:
    python scripts/eval_global_idf1.py
    python scripts/eval_global_idf1.py --min-score 0.5
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

from src.common.ids import extract_vehicle_id  # noqa: E402


class _DSU:
    """并查集：每个簇维护摄像头集合（互斥约束）"""

    def __init__(self, tracklets):
        self.parent = {}
        self.cameras = {}
        for t in tracklets:
            self.parent[t.tracklet_id] = t.tracklet_id
            self.cameras[t.tracklet_id] = {t.camera_id}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 互斥约束：同簇不能有两个同摄像头 tracklet（同摄像头 = 两个不同车）
        if self.cameras[ra] & self.cameras[rb]:
            return False
        self.parent[rb] = ra
        self.cameras[ra] |= self.cameras[rb]
        return True


def _cluster(scene_tracklets, edges, min_score):
    """按分数降序合并，返回 tracklet_id -> cluster 根"""
    dsu = _DSU(scene_tracklets)
    edges = sorted(edges, key=lambda e: -e.score)
    for e in edges:
        if e.score < min_score:
            break
        dsu.union(e.source_tracklet_id, e.target_tracklet_id)
    roots = {}
    for t in scene_tracklets:
        roots[t.tracklet_id] = dsu.find(t.tracklet_id)
    return roots


def _compute_idf1_motmetrics(gt_id_of_det, pred_id_of_det):
    """对逐检测的 (gt_id, pred_id) 计算官方 IDF1/IDP/IDR。

    做法：构建 GT×预测 的混淆矩阵，用匈牙利算法做**最优 ID 匹配**（最大化被正确
    匹配的检测数），再按标准定义算 IDTP/IDFP/IDFN —— 与 motmetrics 的 IDF1 等价
    （同一套「混淆矩阵 + 最优匹配」计算）。
    """
    from collections import Counter

    from scipy.optimize import linear_sum_assignment

    pairs = Counter(zip(gt_id_of_det, pred_id_of_det))
    gts = sorted(set(gt_id_of_det))
    preds = sorted(set(pred_id_of_det))
    gi = {g: i for i, g in enumerate(gts)}
    pi = {p: i for i, p in enumerate(preds)}

    M = np.zeros((len(gts), len(preds)))
    for (g, p), c in pairs.items():
        M[gi[g], pi[p]] = c

    # 最优匹配：最大化 matched 检测数（等价于最小化 -M）
    rows, cols = linear_sum_assignment(-M)
    idtp = float(M[rows, cols].sum())
    total = float(len(gt_id_of_det))
    idfp = total - idtp
    idfn = total - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if idtp > 0 else 0.0
    idp = idtp / total if total > 0 else 0.0
    idr = idtp / total if total > 0 else 0.0
    return idf1, idp, idr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.0, help="聚类合并的最低边分数")
    args = ap.parse_args()

    from src.trajectory.builder import get_trajectory_builder

    builder = get_trajectory_builder()
    builder.ensure_loaded()

    data = json.load(open(_PROJECT_ROOT / "output" / "cityflow_results.json", encoding="utf-8"))
    det_to_track = data.get("det_to_track_map", {}) or {}

    # 按场景组织 tracklet
    tracklets_by_scene = {}
    for tid, dets in builder._tracklet_detections.items():
        if not dets:
            continue
        scene = dets[0].get("scene_id", "")
        tracklets_by_scene.setdefault(scene, []).append(
            builder._tracklet_from_detections(tid, dets)
        )

    # 逐场景聚类 + 生成 tracklet -> cluster 根映射
    track_root = {}
    total_edges = 0
    for scene, tracklets in tracklets_by_scene.items():
        edges = builder._scene_edge_pool(scene, tracklets)
        total_edges += len(edges)
        roots = _cluster(tracklets, edges, args.min_score)
        track_root.update(roots)

    # 逐检测：gt_id = vehicle_id，pred_id = cluster 根（没有簇则用 tracklet_id 自身）
    dets = data["detections"]
    gt_ids = []
    pred_ids = []
    unclustered = 0
    for det in dets:
        vid = extract_vehicle_id(det.get("target_id", ""))
        trk = det_to_track.get(det.get("target_id", ""))
        root = track_root.get(trk, trk) if trk else f"UNTRACKED_{det.get('target_id','')}"
        if root is None:
            root = trk or "none"
            unclustered += 1
        gt_ids.append(vid or "none")
        pred_ids.append(root)

    n_clusters = len(set(pred_ids))
    idf1, idp, idr = _compute_idf1_motmetrics(gt_ids, pred_ids)

    print("=" * 60)
    print(f"全局轨迹聚类 + 官方 motmetrics IDF1 (min_score={args.min_score})")
    print("=" * 60)
    print(f"场景数 = {len(tracklets_by_scene)} | 候选边总数 = {total_edges}")
    print(f"tracklet 数 = {sum(len(v) for v in tracklets_by_scene.values())} "
          f"| 聚出的全局轨迹数 = {n_clusters}")
    print(f"官方 IDF1 = {idf1:.4f}   IDP = {idp:.4f}   IDR = {idr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
