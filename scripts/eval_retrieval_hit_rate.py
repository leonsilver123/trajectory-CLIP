"""
scripts.eval_retrieval_hit_rate - 文本检索命中率评测（二期 T7-a）

## 评测对象

**部署中的检索路径**（`api/routes/search.py`，与前端「检索」页同一个函数）：
    `_extract_query_features` → `_attribute_filter`（属性粗筛）
    → `_encode_text` + FAISS 内积（CLIP 精排）→ 按轨迹聚合去重 → top-K

## 真值

`vehicle_id`（＝ AICity22 GT 的全局 Id，已在 `eval_chain_idf1.py` 核实
230/230 全部同源）。检索**不使用** vehicle_id 参与排序，只当评测真值。

## 口径

对每辆出现在 ≥2 个摄像头的车 v：

1. 取它时间上最早的一条检测作为**锚点**（模拟"用户最早看到它的那一眼"）；
2. 用锚点所在**单摄轨迹内的多数属性**拼出自然语言描述（用户描述的是他看见的东西）：
   例如 `白色` + `轿车` → `"白色轿车"`；
3. 走真实检索路径拿 top-K 候选；
4. **命中** = 候选中存在 vehicle_id == v 的项。

    检索命中率@K = 命中的查询数 / 有效查询数

同时报告 **属性粗筛保留率**（粗筛后是否还剩该车的任何检测）。
两者分开，是为了区分失败来自「粗筛就把目标滤掉了」还是「精排没把它排进 top-K」。

## [!] 如实标注的口径限制

- **描述来自模型自己的属性预测**（AICity22 的 GT 只有 bbox 和 ID，没有颜色/车型真值）。
  因此描述与 `_attribute_filter` 的过滤键同源，粗筛**大概率**会保留目标——
  本指标主要测的是 **CLIP 精排的排序能力**，不是端到端"用户任意措辞"的鲁棒性。
  `--no-clip` 可关掉精排做对照，看精排到底有没有增益。
- 候选按**轨迹**去重，因此 top-K 是 K 条轨迹，不是 K 张图。

用法:
    python scripts/eval_retrieval_hit_rate.py
    python scripts/eval_retrieval_hit_rate.py --k 20 --no-clip
    python scripts/eval_retrieval_hit_rate.py --json output/_retrieval_hit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.ids import extract_vehicle_id  # noqa: E402

# 视为「无信息量」的属性值，不参与拼描述
_UNKNOWN = {"", "unknown", "未知", "None", "none", "其他", "other"}


def _majority(values: list[str]) -> str:
    vals = [v for v in values if v and v not in _UNKNOWN]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="top-K 候选（部署默认 20）")
    ap.add_argument("--no-clip", action="store_true", help="关闭 CLIP 精排（对照实验）")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 辆（调试用）")
    ap.add_argument("--json", type=str, default="", help="结果写 JSON")
    args = ap.parse_args()

    import api.routes.search as S

    if args.no_clip:
        S._encode_text = lambda _text: None  # 强制走纯属性分数路径
        print("[ablation] CLIP 精排已关闭，退化为属性分数排序")

    data = S._load_results()
    if data is None:
        print("[error] 检索数据不可用")
        return 2
    detections = data.get("detections", [])
    d2t = data.get("det_to_track_map", {}) or {}

    # 与 FAISS 索引对齐性核查：索引行号即 detections 下标
    index = S._get_clip_index()
    if index is not None and index.ntotal != len(detections):
        print(f"[warn] FAISS 行数 {index.ntotal} != 检测数 {len(detections)}，向量精排会错位")

    # ---- 构建评测目标 ----
    by_vehicle: dict[str, list] = {}
    for det in detections:
        vid = extract_vehicle_id(det.get("target_id", ""))
        if vid:
            by_vehicle.setdefault(vid, []).append(det)

    track_to_dets: dict[str, list] = {}
    for det in detections:
        trk = d2t.get(det.get("target_id", ""))
        if trk:
            track_to_dets.setdefault(trk, []).append(det)

    targets = []
    for vid, dets in by_vehicle.items():
        cams = {d.get("camera_id", "") for d in dets}
        cams.discard("")
        if len(cams) >= 2:
            targets.append((vid, dets))
    targets.sort(key=lambda x: x[0])
    if args.limit:
        targets = targets[: args.limit]

    print(f"评测车辆数（>=2 摄像头）= {len(targets)} ; top-K = {args.k}")

    hits = 0
    coarse_kept = 0
    valid = 0
    no_desc = 0
    rr_sum = 0.0
    candidates_returned: list[int] = []
    per_query = []

    for vid, dets in targets:
        anchor = min(dets, key=lambda d: d.get("timestamp", ""))
        trk = d2t.get(anchor.get("target_id", ""))
        anchor_dets = track_to_dets.get(trk, [anchor])

        color = _majority([(d.get("attributes") or {}).get("color", "") for d in anchor_dets])
        vtype = _majority([(d.get("attributes") or {}).get("vehicle_type", "") for d in anchor_dets])
        query_text = f"{color}{vtype}".strip()
        if not query_text:
            no_desc += 1
            continue
        valid += 1

        features = S._extract_query_features(query_text)

        # 属性粗筛是否还留着目标
        kept = S._attribute_filter(detections, features)
        kept_target = any(extract_vehicle_id(d.get("target_id", "")) == vid for d in kept)
        if kept_target:
            coarse_kept += 1

        cands = S._build_candidates_with_clip(data, query_text, features, top_k=args.k)
        candidates_returned.append(len(cands))

        rank = 0
        for i, c in enumerate(cands, 1):
            if extract_vehicle_id(c.get("instance_id", "")) == vid:
                rank = i
                break
        if rank:
            hits += 1
            rr_sum += 1.0 / rank

        per_query.append({
            "vehicle_id": vid,
            "query_text": query_text,
            "anchor_camera": anchor.get("camera_id", ""),
            "n_candidates": len(cands),
            "hit_rank": rank,
            "coarse_kept_target": kept_target,
        })

    if valid == 0:
        print("[error] 没有有效查询")
        return 2

    print("\n" + "=" * 72)
    print(f"文本检索命中率（{'关闭 CLIP 精排' if args.no_clip else '部署路径：属性粗筛 + CLIP 精排'}）")
    print("=" * 72)
    print(f"有效查询数（能拼出描述）    = {valid}（另有 {no_desc} 辆属性全为 unknown，无描述，剔除）")
    print(f"检索命中率@{args.k}             = {hits/valid:.4f}   ({hits}/{valid})")
    print(f"MRR                       = {rr_sum/valid:.4f}")
    print(f"属性粗筛保留率              = {coarse_kept/valid:.4f}   ({coarse_kept}/{valid})")
    print(f"平均返回候选数              = {sum(candidates_returned)/len(candidates_returned):.2f}")
    print("-" * 72)
    print("解读：粗筛保留率高而命中率低 => 瓶颈在精排排序，不在粗筛。")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = _PROJECT_ROOT / out
        out.write_text(json.dumps({
            "metric": "retrieval_hit_rate",
            "clip_rerank": not args.no_clip,
            "top_k": args.k,
            "summary": {
                "valid_queries": valid,
                "skipped_no_description": no_desc,
                f"hit_rate@{args.k}": round(hits / valid, 6),
                "mrr": round(rr_sum / valid, 6),
                "coarse_filter_retention": round(coarse_kept / valid, 6),
                "mean_candidates_returned": round(sum(candidates_returned) / len(candidates_returned), 4),
            },
            "per_query": per_query,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐查询结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
