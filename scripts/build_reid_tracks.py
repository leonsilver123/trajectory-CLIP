# encoding: utf-8
"""
scripts.build_reid_tracks - 把检测级 ReID 向量聚合成**轨迹级**向量（二期 T4）。

为什么需要这一步：
    T2（`scripts/extract_reid.py`）产出的是**检测级**向量，行号对齐
    `cityflow_results.json` 的 `detections`。但下游消费 ReID 的地方要的是
    **轨迹级**（Tracklet 级）向量：

        src/stitching/scoring.py:495,528      -> source/target.avg_reid_vector
        src/stitching/candidate_edge.py:601   -> 同上
        src/trajectory/builder.py:1181        -> 构造 Tracklet 时写死 avg_reid_vector=None

    所以只跑完 T2、不跑本脚本，`avg_reid_vector` 仍然恒为 None，
    scoring 的 reid 分支依旧不会触发——T2 的效果等于没接上。

本脚本做什么：
    1. 读 output/reid/reid_vectors.npy（(N,2048)，第 i 行 <-> detections[i]）
    2. 按 det_to_track_map 把检测归到轨迹，对每条轨迹的检测向量求**平均后 L2 重新归一化**
       （ReID 的标准做法：先在单位球面上平均，再归一化，避免长轨迹因模长占优）
    3. 落盘**旁挂文件**（不改动 107MB 的 cityflow_results.json，避免体积膨胀）：

        output/reid/track_reid_vectors.npy   float32, (T, 2048)，已 L2 归一化
        output/reid/track_ids.json           list[str]，长度 T，与上面的行一一对应

关键防御：
    - T2 里读取/前向失败的检测写的是**全零向量**（见 extract_reid.py 的 failed_rows.txt）。
      全零向量若参与平均会稀释掉真实信号，若整条轨迹都是失败行则平均结果模长为 0——
      这种轨迹必须判为「无 ReID 证据」而不是塞一个零向量给评分器
      （零向量的余弦相似度恒为 0，会被误当成"有证据且完全不像"）。

用法:
    python scripts/build_reid_tracks.py
    python scripts/build_reid_tracks.py --check    # 只做自检，不写文件
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_DET_VEC = _PROJECT_ROOT / "output" / "reid" / "reid_vectors.npy"
_OUT_VEC = _PROJECT_ROOT / "output" / "reid" / "track_reid_vectors.npy"
_OUT_IDS = _PROJECT_ROOT / "output" / "reid" / "track_ids.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检，不写出文件")
    args = ap.parse_args()

    if not _DET_VEC.is_file():
        sys.exit(f"[fatal] 检测级向量不存在: {_DET_VEC}（先跑 scripts/extract_reid.py）")

    data = json.load(open(_RESULTS, encoding="utf-8"))
    dets = data["detections"]
    tracks = data["tracks"]
    det_to_track = data["det_to_track_map"]

    X = np.load(_DET_VEC)
    print(f"[info] 检测级向量 : {X.shape}  dtype={X.dtype}")
    if len(X) != len(dets):
        sys.exit(f"[fatal] 行数不匹配: 向量 {len(X)} vs 检测 {len(dets)}")
    print(f"[info] 检测数     : {len(dets)}   轨迹数: {len(tracks)}")

    # 行范数用于识别「失败行」（T2 写的是全零）
    norms = np.linalg.norm(X, axis=1)
    zero_rows = int((norms == 0).sum())
    print(f"[info] 全零行数   : {zero_rows}  （T2 里读取/前向失败的检测）")

    # 轨迹 -> 检测行号
    rows_by_track: dict[str, list[int]] = defaultdict(list)
    for i, det in enumerate(dets):
        tid = det_to_track.get(det.get("target_id", ""))
        if tid:
            rows_by_track[tid].append(i)

    track_ids: list[str] = []
    vecs: list[np.ndarray] = []
    all_zero: list[str] = []      # 检测全为失败行 -> 无证据

    # 迭代**检测推出来的**轨迹集合（= det_to_track_map 的值域），而不是 tracks[] 数组。
    # 实测两处对不上，这里是刻意的选择：
    #   - det_to_track_map 指向 926 条轨迹，但 tracks[] 里只有其中 370 条存在；
    #     另外 556 条在 tracks[] 里查无此 id（track_id 前缀不同：CF3_* vs BL_*/CF2S_*），
    #     若按 tracks[] 迭代会把这 556 条轨迹的 ReID 证据全部丢掉。
    #   - tracks[] 另有 1700 条（同样是 BL_*/CF2S_* 等别的来源）压根没有任何检测。
    # builder 生成 tracklet 时用的正是 det_to_track_map（见 builder.py 的
    # _tracklet_detections），所以这里以它为准，保证「有向量的轨迹」与
    # 「会被构造出来的 tracklet」是同一集合。
    for tid, rows in rows_by_track.items():
        sub = X[rows]
        sub_norms = np.linalg.norm(sub, axis=1)
        # 只保留有效（非全零）行
        valid = sub[sub_norms > 0]
        if len(valid) == 0:
            all_zero.append(tid)
            continue
        # 先单位化再平均（ReID 标准聚合），最后再归一化
        unit = valid / np.linalg.norm(valid, axis=1, keepdims=True)
        mean = unit.mean(axis=0)
        n = np.linalg.norm(mean)
        if n == 0:
            all_zero.append(tid)
            continue
        track_ids.append(tid)
        vecs.append((mean / n).astype(np.float32))

    Y = np.stack(vecs) if vecs else np.zeros((0, X.shape[1]), dtype=np.float32)
    print(f"\n===== 结果 =====")
    print(f"有检测的轨迹(基准) : {len(rows_by_track)}  （det_to_track_map 的值域）")
    print(f"tracks[] 条目数    : {len(tracks)}  （含无检测的其它来源，见上方注释）")
    print(f"有 ReID 证据       : {len(track_ids)}  ({len(track_ids)/max(len(rows_by_track),1):.1%})")
    print(f"检测全为失败行     : {len(all_zero)}")
    print(f"输出矩阵           : {Y.shape}")

    if len(Y):
        yn = np.linalg.norm(Y, axis=1)
        print(f"输出行范数         : min={yn.min():.4f} max={yn.max():.4f} （应为 1.0）")
        uniq = len({v.tobytes() for v in Y})
        print(f"去重后的不同向量数 : {uniq}  （远小于行数则说明聚合退化）")

    if args.check:
        print("\n[check] 只自检，未写出文件")
        return 0

    _OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
    np.save(_OUT_VEC, Y)
    with open(_OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False)
    print(f"\n[done] wrote {_OUT_VEC}  shape={Y.shape}")
    print(f"[done] wrote {_OUT_IDS}  n={len(track_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
