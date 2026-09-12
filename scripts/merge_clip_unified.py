# encoding: utf-8
"""
scripts.merge_clip_unified - 把 T3 重编码的 512 维 CLIP 向量合并回检测记录（二期 T4-b）。

背景（T3 的收尾）:
    `scripts/unify_clip_949.py` 已经把 949 条内联了 **768 维**（CN-CLIP ViT-L-14）
    `clip_image_vector` 的检测，用 ViT-B-16 重新编码成 **512 维**并落盘为
    `output/clip_949_unified.npy` + `clip_949_unified_rows.npy`，但**当时刻意没有接线**
    （其 docstring 末句：「接线是后续波次的事」）。本脚本就是那条接线。

    为什么必须接：`src/trajectory/builder.py` 会把检测的 `clip_image_vector` 求均值
    成 `Tracklet.avg_clip_vector`，而 `src/stitching/scoring.py:530` 会把它喂给
    `cosine_similarity` —— 该函数内部是 `np.dot(vec1, vec2)`，**两个向量维度不一致会直接
    抛 ValueError**（768 vs 512）。也就是说，768/512 混在库里是一个**潜在崩溃**，
    不只是"特征空间不统一"这种措辞问题。

本脚本做什么:
    按 `clip_949_unified_rows.npy` 给出的下标，把对应检测的 `clip_image_vector`
    替换成 512 维版本。**只改这一列**，其它字段一律不动。
    `clip_text_vector` 保持原样（768 维）——文本向量本轮没有重编码，
    截断 768→512 是错的，故不动它（datastore 把两者存成不同的向量列，维度可不同）。

安全性:
    - 幂等：已经是 512 维的行跳过，重复跑结果一致。
    - 原子写：先写 .tmp 再 replace，中途失败不会留下半个 JSON。
    - 写前校验：行数、下标范围、目标行确实是带向量的那批。

用法:
    python scripts/merge_clip_unified.py --check    # 只报告，不改文件
    python scripts/merge_clip_unified.py            # 执行合并
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_UNIFIED = _PROJECT_ROOT / "output" / "clip_949_unified.npy"
_ROWS = _PROJECT_ROOT / "output" / "clip_949_unified_rows.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告现状，不改文件")
    args = ap.parse_args()

    for p in (_UNIFIED, _ROWS):
        if not p.is_file():
            sys.exit(f"[fatal] 缺少 {p}（先跑 scripts/unify_clip_949.py）")

    V = np.load(_UNIFIED)
    R = np.load(_ROWS)
    print(f"[info] 统一向量 : {V.shape}  dtype={V.dtype}")
    print(f"[info] 行下标   : {R.shape}  范围 [{R.min()}, {R.max()}]")
    if len(V) != len(R):
        sys.exit(f"[fatal] 向量 {len(V)} 行与下标 {len(R)} 个不匹配")

    data = json.load(open(_RESULTS, encoding="utf-8"))
    dets = data["detections"]
    print(f"[info] 检测总数 : {len(dets)}")
    if R.max() >= len(dets):
        sys.exit(f"[fatal] 下标越界: {R.max()} >= {len(dets)}")

    # 现状统计
    dims_before: dict[int, int] = {}
    for d in dets:
        v = d.get("clip_image_vector")
        if v:
            dims_before[len(v)] = dims_before.get(len(v), 0) + 1
    print(f"[info] 合并前维度分布 : {dict(sorted(dims_before.items()))}")

    changed = already = bad = 0
    for k, row in enumerate(R):
        d = dets[int(row)]
        cur = d.get("clip_image_vector")
        if not cur:
            bad += 1
            continue
        if len(cur) == V.shape[1]:
            already += 1
            continue
        if len(cur) != 768:
            bad += 1
            continue
        d["clip_image_vector"] = V[k].astype(np.float32).tolist()
        changed += 1

    print(f"\n===== 合并 =====")
    print(f"替换为 512 维 : {changed}")
    print(f"已是 512 维   : {already}  （幂等跳过）")
    print(f"异常/跳过     : {bad}  （下标处无向量，或原维度既非 768 也非 512）")

    # 预期分布
    dims_after: dict[int, int] = {}
    for d in dets:
        v = d.get("clip_image_vector")
        if v:
            dims_after[len(v)] = dims_after.get(len(v), 0) + 1
    print(f"合并后维度分布 : {dict(sorted(dims_after.items()))}")

    if args.check:
        print("\n[check] 只报告，未写出文件")
        return 0

    if changed == 0:
        print("\n无需改动，未写出文件")
        return 0

    # 原子写：先落 .tmp 再 replace
    tmp = _RESULTS.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, _RESULTS)
    print(f"\n[done] 已原地更新 {_RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
