# encoding: utf-8
"""
scripts.eval_attribute_consistency - 属性标注跨镜一致率评测（二期 T1 验收）。

指标定义（严格对齐 ASSESSMENT.md §3.1 的既有口径，便于与基线直接对比）:
    - 摄像头**内**一致率：每辆车在每个摄像头内取"主导标签"，该摄像头内主导标签的占比
      的均值（基线 78.0%）
    - 摄像头**间**一致率：一辆车若出现在多个摄像头，其各摄像头的"主导标签"**是否全同**；
      对全部这样的车取比例（基线 **8.3%**，共 218 辆车）
    - 随机基线：假设标签按**边缘分布独立随机**抽取，同样算法下的期望跨镜一致率
      （基线 31.9%）
    - 覆盖车数：出现在 >=2 个摄像头的车辆数

为什么这个指标重要：跨镜一致率**低于随机基线**意味着属性不是"噪声大"，
而是**与摄像头系统性相关**——拿它做跨镜过滤会反向有害（惩罚正确轨迹）。

用法:
    python scripts/eval_attribute_consistency.py --source old     # 现有 JSON 标注
    python scripts/eval_attribute_consistency.py --source pulc    # T1 新模型输出
    python scripts/eval_attribute_consistency.py --source both
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_PULC_PROBS = _PROJECT_ROOT / "output" / "attributes" / "pulc_probs.npy"

COLORS = ["yellow", "orange", "green", "gray", "red", "blue", "white",
          "golden", "brown", "black"]
TYPES = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
         "truck", "estate"]
COLOR_THRESHOLD = 0.5
TYPE_THRESHOLD = 0.5


def vehicle_of(target_id: str) -> str:
    for p in (target_id or "").split("_"):
        if p.startswith("V") and p[1:].isdigit():
            return p
    return ""


def random_baseline(labels_by_cam: dict, marginal: Counter) -> tuple:
    """
    随机基线：各摄像头的主导标签按边缘分布独立抽取时，跨镜"全同"的期望比例。

    返回两个口径：
      - actual: 按每辆车**实际的摄像头数 k** 算 sum_l p_l^k，这是严格正确的定义
      - pair  : 一律按 k=2 算 sum_l p_l^2

    ⚠️ 为什么要给两个：ASSESSMENT.md §3.1 记的 31.9% 实测对应的是 **pair** 口径
    （本函数在旧标注上复现出 32.1%）。但本数据集的车辆**大多出现在 3~5 个摄像头**
    （79 辆在 5 个、47 辆在 4 个，最多 24 个），k>2 时 sum p^k 会显著小于 sum p^2，
    所以 pair 口径**高估**了随机基线，也就**夸大**了"低于随机"的幅度。
    两个都报出来，避免用一个偏保守的基线替结论撑腰。
    """
    keys = list(marginal)
    probs = np.array([marginal[k] for k in keys], dtype=np.float64)
    probs /= probs.sum()
    actual = pair = 0.0
    for _, cams in labels_by_cam.items():
        k = len(cams)
        actual += float((probs ** k).sum())
        pair += float((probs ** 2).sum())
    m = max(len(labels_by_cam), 1)
    return actual / m, pair / m


def evaluate(name: str, labels: list, cams: list, vids: list) -> dict:
    """labels/cams/vids 等长：每条检测的标签、摄像头、车辆号。"""
    # 车 -> 摄像头 -> 该摄像头内的标签计数
    per = defaultdict(lambda: defaultdict(Counter))
    for lab, cam, v in zip(labels, cams, vids):
        if v:
            per[v][cam][lab] += 1

    # 内一致率（每车每摄像头的主导占比，再对所有(车,摄像头)取均值）
    within = []
    labels_by_cam = {}
    marginal = Counter()
    for v, cams_d in per.items():
        if len(cams_d) < 2:
            continue
        dom = {}
        for cam, cnt in cams_d.items():
            top, topn = cnt.most_common(1)[0]
            dom[cam] = top
            within.append(topn / sum(cnt.values()))
            marginal[top] += 1
        labels_by_cam[v] = dom

    cross = sum(1 for dom in labels_by_cam.values() if len(set(dom.values())) == 1)
    n = len(labels_by_cam)
    r_actual, r_pair = random_baseline(labels_by_cam, marginal) if n else (float("nan"),) * 2
    return {
        "name": name,
        "within": float(np.mean(within)) if within else float("nan"),
        "cross": cross / n if n else float("nan"),
        "cross_hits": cross,
        "n_vehicles": n,
        "random": r_actual,
        "random_pair": r_pair,
        "distinct_labels": len(marginal),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="both", choices=("old", "pulc", "both"))
    args = ap.parse_args()

    data = json.load(open(_RESULTS, encoding="utf-8"))
    dets = data["detections"]
    cams = [d.get("camera_id", "") for d in dets]
    vids = [vehicle_of(d.get("target_id", "")) for d in dets]

    results = []

    if args.source in ("old", "both"):
        old = []
        for d in dets:
            a = d.get("attributes") or {}
            old.append(a.get("vehicle_type") or "unknown")
        results.append(evaluate("现有 JSON 标注 (vehicle_type)", old, cams, vids))

    if args.source in ("pulc", "both"):
        if not _PULC_PROBS.is_file():
            sys.exit(f"[fatal] 缺 {_PULC_PROBS}（先跑 scripts/extract_attributes.py）")
        P = np.load(_PULC_PROBS)
        if len(P) != len(dets):
            sys.exit(f"[fatal] 行数不符: probs {len(P)} vs dets {len(dets)}")
        ti = P[:, 10:].argmax(axis=1)
        tc = P[np.arange(len(P)), 10 + ti]
        new = [TYPES[i] if tc[k] >= TYPE_THRESHOLD else "unknown" for k, i in enumerate(ti)]
        results.append(evaluate("T1 PULC 模型 (type)", new, cams, vids))

        ci = P[:, :10].argmax(axis=1)
        cc = P[np.arange(len(P)), ci]
        new_c = [COLORS[i] if cc[k] >= COLOR_THRESHOLD else "unknown" for k, i in enumerate(ci)]
        results.append(evaluate("T1 PULC 模型 (color)", new_c, cams, vids))

    print(f"{'来源':<30} {'内一致':>7} {'跨镜一致':>9} {'随机(k)':>8} {'随机(k=2)':>10} {'车辆':>5} {'标签':>5}")
    print("-" * 82)
    for r in results:
        print(f"{r['name']:<30} {r['within']:>7.1%} {r['cross']:>9.1%} "
              f"{r['random']:>8.1%} {r['random_pair']:>10.1%} {r['n_vehicles']:>5} {r['distinct_labels']:>5}")
    print("\n参考基线（ASSESSMENT.md §3.1）: 内一致 78.0% / 跨镜一致 8.3% / 随机 31.9%(=k=2口径) / 218 辆")
    print("验收目标（PLAN2）            : 跨镜一致率 显著高于 8.3%，目标 > 60%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
