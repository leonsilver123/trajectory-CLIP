"""
scripts.eval_chain_idf1 - 跨镜观测链 IDF1 评测（二期 T7-b）

## 评测对象

系统的**弱身份路径**（`mode="stitch"`）：用户确认一个锚点目标后，系统在同场景
的单摄轨迹池上做六维评分拼接，产出一条跨镜观测链。

## 为什么必须显式指定 mode="stitch"

`src/trajectory/builder.py:787`：

    use_strong = bool(vehicle_id) and mode != "stitch"

本数据集的**每一条检测都带真实 vehicle_id**（GT track_id 同源，V0034 ↔ GT 34），
所以 `mode="auto"` **永远**走强身份路径——那条路径直接把同一 vehicle_id 的检测
聚合成一个目标，拿它评 IDF1 会**恒等于 1.0**，没有任何信息量。
只有强制 `mode="stitch"` 才绕过真值，测的是真正的跨镜匹配能力。

## 口径（用户 2026-09-12 确认）

对每辆 GT 车 v（出现在 ≥2 个摄像头）：

    pred(v) = 观测链覆盖到的全部检测
    gt(v)   = vehicle_id == v 的全部检测
    IDTP = |pred ∩ gt|   IDFP = |pred \\ gt|   IDFN = |gt \\ pred|

全局 IDF1 = 2·ΣIDTP / (2·ΣIDTP + ΣIDFP + ΣIDFN)（motmetrics 定义，micro 累加）。

测的是「链路扩展器该收的收没收、不该收的收没收」；
**不测**全库无锚点聚类——系统本来就没有这个能力。

## ⚠️ 会抬高结果的已知因素（如实标注，见报告）

**轨迹池是「按摄像头 × 车辆」预先切好的 GT 纯净单元**：经实测，926 条
`det_to_track_map` 轨迹**全部**只含单一 vehicle_id（926/926 纯净，0 混杂），
形如 `CF3_TRACK_c001_V0034`。也就是说**单摄 MOT 的分割被当作真值送进来了**，
摄像头内的 ID switch 不在评测范围内。真实 MTMC 场景没有这个先验。
→ 因此本指标是「跨镜**关联**」能力，**不是**端到端 MTMC 能力，不可与 AICity22 榜单直接比。

（原以为「候选池按场景圈定 ⇒ 跨场景车辆有天花板」也是一项限制，实测**不成立**：
218 辆多摄像头车辆**全部**单场景（0/218 跨场景），AICity22 的 MTMC 本就在场景内定义。
该因素已删除。）

用法:
    python scripts/eval_chain_idf1.py
    python scripts/eval_chain_idf1.py --limit 20 --json output/_idf1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.ids import extract_vehicle_id  # noqa: E402

_GT_DIR = _PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking" / "eval"


def _load_gt_ids(fname: str) -> set[str]:
    """读 AICity22 GT（CameraId Id FrameId X Y W H Xworld Yworld），取全局 Id 集合。

    用于核实「我们检测里的 vehicle_id 确实来自 GT」——即评测真值同源。
    """
    path = _GT_DIR / fname
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                ids.add(f"V{int(parts[1]):04d}")
    return ids


def _track_to_dets(data: dict) -> dict[str, list[str]]:
    """track_id → 该轨迹下的 target_id 列表（口径与 builder 一致：按 det_to_track_map 的值域）"""
    d2t = data.get("det_to_track_map", {}) or {}
    out: dict[str, list[str]] = defaultdict(list)
    for det in data.get("detections", []):
        tid = det.get("target_id", "")
        trk = d2t.get(tid)
        if tid and trk:
            out[trk].append(tid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 辆（调试用，0=全量）")
    ap.add_argument("--json", type=str, default="", help="把逐车结果写成 JSON")
    args = ap.parse_args()

    from src.trajectory.builder import get_trajectory_builder

    builder = get_trajectory_builder()
    builder.ensure_loaded()
    if not builder._loaded:
        print("[error] 真实数据不可用，无法评测")
        return 2

    data = json.load(open(_PROJECT_ROOT / "output" / "cityflow_results.json", encoding="utf-8"))
    track_to_dets = _track_to_dets(data)

    # ---- 真值来源核查：我们的 vehicle_id 是否真的来自 AICity22 GT ----
    gt_train = _load_gt_ids("ground_truth_train.txt")
    gt_val = _load_gt_ids("ground_truth_validation.txt")
    gt_all = gt_train | gt_val
    our_ids = set(builder._vehicle_detections)
    if gt_all:
        covered = len(our_ids & gt_all)
        print(f"[GT 同源核查] AICity22 GT 全局 Id：train={len(gt_train)} validation={len(gt_val)} "
              f"并集={len(gt_all)}")
        print(f"              我方 vehicle_id 数 = {len(our_ids)}；能在 GT 中找到的 = {covered} "
              f"({covered/len(our_ids)*100:.1f}%)；不在 GT 中的 = {len(our_ids - gt_all)}")
    else:
        print(f"[GT 同源核查] 未找到 GT 文件目录 {_GT_DIR}，跳过")

    # ---- 目标集合：出现在 >=2 个摄像头的车 ----
    all_vehicles = list(builder._vehicle_detections.items())
    vehicles = []
    for vid, dets in all_vehicles:
        cams = {d.get("camera_id", "") for d in dets}
        cams.discard("")
        if len(cams) >= 2:
            vehicles.append((vid, dets, cams))
    vehicles.sort(key=lambda x: x[0])
    single_cam = len(all_vehicles) - len(vehicles)
    if args.limit:
        vehicles = vehicles[: args.limit]

    print(f"评测车辆数（>=2 摄像头）= {len(vehicles)}"
          f"{'（--limit 截断）' if args.limit else ''}")

    idtp = idfp = idfn = 0
    per_vehicle = []
    failures = []
    cam_exact = cam_recall_sum = cam_prec_sum = 0.0
    n_cam_vehicles = 0
    cross_scene_vehicles = 0

    for idx, (vid, dets, gt_cams) in enumerate(vehicles, 1):
        # 锚点：该车在时间上最早的一条检测（模拟"用户最早确认到它"）
        anchor_det = min(dets, key=lambda d: d.get("timestamp", ""))
        anchor_target_id = anchor_det.get("target_id", "")
        scene = anchor_det.get("scene_id", "")

        try:
            result = builder.build(anchor_target_id, mode="stitch")
        except Exception as exc:  # 锚点无轨迹 / 数据缺失等
            failures.append({"vehicle_id": vid, "anchor": anchor_target_id,
                             "error": f"{type(exc).__name__}: {exc}"})
            idfn += len(dets)
            continue

        pred_tracklets = [n.get("tracklet_id") for n in result.get("observation_nodes", [])]
        pred_target_ids: set[str] = set()
        for trk in pred_tracklets:
            pred_target_ids.update(track_to_dets.get(trk, []))

        gt_target_ids = {d.get("target_id", "") for d in dets}

        tp = len(pred_target_ids & gt_target_ids)
        fp = len(pred_target_ids - gt_target_ids)
        fn = len(gt_target_ids - pred_target_ids)
        idtp += tp
        idfp += fp
        idfn += fn

        # 摄像头层面的还原度
        pred_cams = {c for c in result.get("camera_sequence", []) if c}
        inter = len(pred_cams & gt_cams)
        cam_recall_sum += inter / len(gt_cams) if gt_cams else 0.0
        cam_prec_sum += inter / len(pred_cams) if pred_cams else 0.0
        cam_exact += 1 if pred_cams == gt_cams else 0
        n_cam_vehicles += 1

        # 跨场景车辆：结构上天花板受限
        gt_scenes = {d.get("scene_id", "") for d in dets}
        is_cross_scene = len(gt_scenes) > 1
        if is_cross_scene:
            cross_scene_vehicles += 1

        per_vehicle.append({
            "vehicle_id": vid,
            "anchor_target_id": anchor_target_id,
            "anchor_camera": anchor_det.get("camera_id", ""),
            "anchor_scene": scene,
            "gt_detections": len(gt_target_ids),
            "pred_detections": len(pred_target_ids),
            "idtp": tp, "idfp": fp, "idfn": fn,
            "gt_cameras": sorted(gt_cams),
            "pred_cameras": sorted(pred_cams),
            "camera_recall": round(inter / len(gt_cams), 4) if gt_cams else 0.0,
            "camera_precision": round(inter / len(pred_cams), 4) if pred_cams else 0.0,
            "cross_scene": is_cross_scene,
            "chain_confidence": result.get("overall_confidence"),
        })

    # ---- 汇总 ----
    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom else 0.0
    idp = (idtp / (idtp + idfp)) if (idtp + idfp) else 0.0
    idr = (idtp / (idtp + idfn)) if (idtp + idfn) else 0.0

    print("\n" + "=" * 72)
    print("跨镜观测链 IDF1（mode=stitch，弱身份路径，无真值参与匹配）")
    print("=" * 72)
    print(f"评测车辆数        = {n_cam_vehicles}（>=2 摄像头；另有 {single_cam} 辆仅出现在 1 个摄像头，不参与）")
    print(f"IDTP = {idtp}   IDFP = {idfp}   IDFN = {idfn}")
    print(f"IDF1 = {idf1:.4f}")
    print(f"IDP  = {idp:.4f}")
    print(f"IDR  = {idr:.4f}")

    # macro 口径：逐车 IDF1 后取平均。micro 会被高检测量的大车主导
    # （例如 V0260 一辆就贡献 1944 IDFN，占全局 IDFN 的 4.7%），两个都要报。
    macro_vals = [
        (2 * r["idtp"] / (2 * r["idtp"] + r["idfp"] + r["idfn"]))
        for r in per_vehicle if (2 * r["idtp"] + r["idfp"] + r["idfn"]) > 0
    ]
    macro_idf1 = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    print(f"IDF1 = {macro_idf1:.4f}   (macro：逐车 IDF1 平均，{len(macro_vals)} 辆)")
    print("-" * 72)
    print("摄像头层面还原度（跨镜匹配的直接体现）")
    print(f"  摄像头集合完全一致（exact match） = {cam_exact}/{n_cam_vehicles} "
          f"= {cam_exact/n_cam_vehicles:.4f}" if n_cam_vehicles else "  (无)")
    print(f"  平均摄像头召回 (Camera Recall)   = {cam_recall_sum/n_cam_vehicles:.4f}" if n_cam_vehicles else "")
    print(f"  平均摄像头精度 (Camera Precision)= {cam_prec_sum/n_cam_vehicles:.4f}" if n_cam_vehicles else "")
    print("-" * 72)
    print(f"跨场景出现的车辆 = {cross_scene_vehicles}/{n_cam_vehicles}"
          f"（实测为 0 ⇒ 候选池按场景圈定**不构成**限制，AICity22 的 MTMC 本身在场景内定义）")
    print(f"构建失败（无轨迹）的车辆数 = {len(failures)}")

    # ---- 链扩展诊断：链长 vs GT 摄像头数 ----
    if per_vehicle:
        zero_expand = [r for r in per_vehicle if len(r["pred_cameras"]) <= 1]
        print("-" * 72)
        print("链扩展诊断")
        print(f"  完全没扩展（链只含锚点摄像头）的车辆 = {len(zero_expand)}/{len(per_vehicle)} "
              f"= {len(zero_expand)/len(per_vehicle):.4f}")
        gt_cam_mean = sum(len(r["gt_cameras"]) for r in per_vehicle) / len(per_vehicle)
        pred_cam_mean = sum(len(r["pred_cameras"]) for r in per_vehicle) / len(per_vehicle)
        print(f"  平均 GT 摄像头数 = {gt_cam_mean:.2f} ；平均预测摄像头数 = {pred_cam_mean:.2f}"
              f"（缺口 {gt_cam_mean - pred_cam_mean:+.2f}）")
        no_false_pos = [r for r in per_vehicle if r["idfp"] == 0]
        print(f"  零误收（IDFP==0）的车辆 = {len(no_false_pos)}/{len(per_vehicle)}"
              f" —— 链路保守，宁可少收不多收")

    if per_vehicle:
        ranked = sorted(per_vehicle, key=lambda x: (x["camera_recall"], x["idtp"]))
        print("\n最差的 10 辆（按摄像头召回）:")
        print(f"  {'车辆':<7} {'GT检测':<7} {'预测检测':<9} {'IDTP':<6} {'IDFP':<6} {'IDFN':<6} {'GT摄像头数':<10} {'命中':<5} {'跨场景'}")
        for r in ranked[:10]:
            print(f"  {r['vehicle_id']:<7} {r['gt_detections']:<7} {r['pred_detections']:<9} "
                  f"{r['idtp']:<6} {r['idfp']:<6} {r['idfn']:<6} {len(r['gt_cameras']):<10} "
                  f"{len(set(r['pred_cameras']) & set(r['gt_cameras'])):<5} {r['cross_scene']}")
        print("\n最好的 5 辆（按摄像头召回）:")
        for r in ranked[-5:][::-1]:
            print(f"  {r['vehicle_id']:<7} {r['gt_detections']:<7} {r['pred_detections']:<9} "
                  f"{r['idtp']:<6} {r['idfp']:<6} {r['idfn']:<6} {len(r['gt_cameras']):<10} "
                  f"{len(set(r['pred_cameras']) & set(r['gt_cameras'])):<5} {r['cross_scene']}")

    if failures:
        print(f"\n失败样例（前 5）:")
        for f in failures[:5]:
            print(f"  {f['vehicle_id']:<7} anchor={f['anchor']}  {f['error']}")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = _PROJECT_ROOT / out
        payload = {
            "metric": "IDF1",
            "mode": "stitch",
            "note": "弱身份路径；候选池为按摄像头×车辆预切的 GT 纯净轨迹",
            "summary": {
                "vehicles_evaluated": n_cam_vehicles,
                "idtp": idtp, "idfp": idfp, "idfn": idfn,
                "idf1": round(idf1, 6), "idp": round(idp, 6), "idr": round(idr, 6),
                "idf1_macro": round(macro_idf1, 6),
                "camera_exact_match": cam_exact,
                "camera_recall_mean": round(cam_recall_sum / n_cam_vehicles, 6) if n_cam_vehicles else None,
                "camera_precision_mean": round(cam_prec_sum / n_cam_vehicles, 6) if n_cam_vehicles else None,
                "cross_scene_vehicles": cross_scene_vehicles,
                "build_failures": len(failures),
            },
            "per_vehicle": per_vehicle,
            "failures": failures,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐车结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
