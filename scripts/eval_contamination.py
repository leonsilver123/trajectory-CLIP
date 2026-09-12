"""
scripts.eval_contamination - 数据污染量化（用户提出的诊断）

检测污染车（车内跨镜 tracklet 外观余弦 < 阈值的，说明 GT vehicle_id 把不同车标成同一辆），
对比「全量 IDF1」与「干净（剔除污染车）IDF1」，确定数据干净时可达的上限。

用法:
    python scripts/eval_contamination.py --cos-threshold 0.15
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.ids import extract_vehicle_id  # noqa: E402


def _cos(a, b):
    return float((a / (np.linalg.norm(a) + 1e-8)) @ (b / (np.linalg.norm(b) + 1e-8)))


class _DSU:
    def __init__(self, tracklets):
        self.p = {t["id"]: t["id"] for t in tracklets}
        self.cams = {t["id"]: {t["camera"]} for t in tracklets}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.cams[ra] & self.cams[rb]:
            return False
        self.p[rb] = ra
        self.cams[ra] |= self.cams[rb]
        return True


def _idf1(gt_ids, pred_ids):
    pairs = Counter(zip(gt_ids, pred_ids))
    gts = sorted(set(gt_ids))
    preds = sorted(set(pred_ids))
    gi = {g: i for i, g in enumerate(gts)}
    pi = {p: i for i, p in enumerate(preds)}
    M = np.zeros((len(gts), len(preds)))
    for (g, p), c in pairs.items():
        M[gi[g], pi[p]] = c
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-M)
    idtp = float(M[rows, cols].sum())
    total = float(len(gt_ids))
    return 2 * idtp / (2 * idtp + (total - idtp) * 2) if idtp > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos-threshold", type=float, default=0.15)
    args = ap.parse_args()

    from src.trajectory.builder import get_trajectory_builder
    builder = get_trajectory_builder()
    builder.ensure_loaded()

    # 加载 tracklet
    tracklets = []
    for tid, dets in builder._tracklet_detections.items():
        if not dets:
            continue
        vid = extract_vehicle_id(dets[0].get("target_id", ""))
        t = builder._tracklet_from_detections(tid, dets)
        tracklets.append({
            "id": tid, "camera": t.camera_id, "scene": dets[0].get("scene_id", ""),
            "vehicle_id": vid, "vec": t.avg_reid_vector,
        })

    # 检测污染车：车内跨镜最小余弦 < 阈值
    by_veh = defaultdict(list)
    for t in tracklets:
        if t["vec"] is not None:
            by_veh[t["vehicle_id"]].append(t)

    contaminated = set()
    for vid, ts in by_veh.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                if ts[i]["camera"] == ts[j]["camera"]:
                    continue
                if _cos(ts[i]["vec"], ts[j]["vec"]) < args.cos_threshold:
                    contaminated.add(vid)

    # 聚类（union-find on candidate edges, 按场景）
    track_root = {}
    for scene, ts in defaultdict(list, {t["scene"]: t for t in tracklets}).items():
        pass

    # 用 builder 的候选边做聚类
    track_by_id = {t["id"]: t for t in tracklets}
    dsu = _DSU(tracklets)
    for scene, ts in __import__("collections").defaultdict(list).items():
        pass
    # 按场景聚类
    from collections import defaultdict as dd
    by_scene = dd(list)
    for t in tracklets:
        by_scene[t["scene"]].append(t)
    for scene, ts in by_scene.items():
        edges = builder._scene_edge_pool(scene, [builder._tracklet_from_detections(t["id"], builder._tracklet_detections[t["id"]]) for t in ts])
        edges = sorted(edges, key=lambda e: -e.score)
        for e in edges:
            dsu.union(e.source_tracklet_id, e.target_tracklet_id)

    roots = [dsu.find(t["id"]) for t in tracklets]

    def report(mask, label):
        gt = [t["vehicle_id"] or "none" for i, t in enumerate(tracklets) if mask(t)]
        pred = [f"c{roots[i]}" for i, t in enumerate(tracklets) if mask(t)]
        print(f"  {label}: IDF1 = {_idf1(gt, pred):.4f}  (tracklet 数={len(gt)})")

    print(f"污染车数 = {len(contaminated)} / {len(by_veh)} ({len(contaminated)/len(by_veh)*100:.1f}%)")
    report(lambda t: True, "全量")
    report(lambda t: t["vehicle_id"] not in contaminated, "干净(剔除污染车)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
