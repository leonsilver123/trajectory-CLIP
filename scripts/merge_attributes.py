# encoding: utf-8
"""
scripts.merge_attributes - 把 T1 的属性预测合并进检测记录（二期 T1 收尾）。

为什么必须一次性把 7 个字段全改掉（不是只改 vehicle_type）:
    现有 attributes 里颜色/车型是**多份冗余且互相矛盾**的（即 E1）。实测样例：

        {"color": "白色", "颜色": "白色", "vehicle_type": "SUV", "车型": "SUV",
         "vehicle_type_en": "van", "type_id": 2}
                         ^^^ 车型相关四个字段里，前两个说 SUV，后两个说 van/面包车

    而下游取值有**优先级链**，且中文键排在前面——
      src/trajectory/builder.py:1605-1619
        type_cn = attrs.get("车型") or _TYPE_CN_MAP[type_id] or _EN_TYPE_MAP[vehicle_type] or vehicle_type
        color_cn= attrs.get("颜色") or _COLOR_CN_MAP[color_id] or _EN_COLOR_MAP[color]  or color
    所以只改 `vehicle_type` 会被 `车型` **完全覆盖**，等于白改。

    另外 `type_id` 只在 949/68349 条上存在（其余为 None），所以不能依赖它兜底。

字段口径（本脚本写入后 7 个字段两两一致）:
    color / 颜色     : 中文（attribute_filter 用 COLOR_ALIAS_MAP 比中文）
    color_id        : PULC 颜色下标 0-9（builder 的 _COLOR_CN_MAP 直接用下标）
    vehicle_type / 车型 : 中文（builder 优先读这个）
    vehicle_type_en : 英文原标签（展示用）
    type_id         : PULC 车型下标 0-8（builder 的 _TYPE_CN_MAP 直接用下标）

    下标可直接复用，是因为 **PULC 的标签顺序与 builder 的两张表逐位对齐**（实测核对）：
      PULC colors = yellow,orange,green,gray,red,blue,white,golden,brown,black
      _COLOR_CN_MAP = 黄色,橙色,绿色,灰色,红色,蓝色,白色,金色,棕色,黑色,(紫,粉)
      PULC types  = sedan,suv,van,hatchback,mpv,pickup,bus,truck,estate
      _TYPE_CN_MAP  = 轿车,SUV,面包车,两厢车,MPV,皮卡,公交车,卡车,旅行车,(跑车,房车)
    → 无需任何有损映射。置信度低于阈值的判为 "unknown"/未知，不硬猜。

可选 --track-vote（E2）: 按 track 做多数投票后再写回该 track 的所有检测，
    压制单帧误判（摄像头内一致率已 78%，投票后应更高）。投票是按**轨迹**做的，
    而轨迹是单摄的，因此不会把不同摄像头的信息混在一起。

用法:
    python scripts/merge_attributes.py --check
    python scripts/merge_attributes.py --track-vote
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_PROBS = _PROJECT_ROOT / "output" / "attributes" / "pulc_probs.npy"

# 与 PULC 输出顺序、与 builder 下标表逐位对齐（见脚本 docstring）
COLORS_EN = ["yellow", "orange", "green", "gray", "red", "blue", "white",
             "golden", "brown", "black"]
TYPES_EN = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"]
COLORS_CN = ["黄色", "橙色", "绿色", "灰色", "红色", "蓝色", "白色",
             "金色", "棕色", "黑色"]
TYPES_CN = ["轿车", "SUV", "面包车", "两厢车", "MPV", "皮卡", "公交车", "卡车", "旅行车"]
UNKNOWN_CN = "未知"
COLOR_THRESHOLD = 0.5
TYPE_THRESHOLD = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只统计，不写文件")
    ap.add_argument("--track-vote", action="store_true",
                    help="先按轨迹多数投票再写回（E2）")
    args = ap.parse_args()

    if not _PROBS.is_file():
        sys.exit(f"[fatal] 缺 {_PROBS}（先跑 scripts/extract_attributes.py）")

    data = json.load(open(_RESULTS, encoding="utf-8"))
    dets = data["detections"]
    P = np.load(_PROBS)
    if len(P) != len(dets):
        sys.exit(f"[fatal] 行数不符: probs {len(P)} vs dets {len(dets)}")

    ci = P[:, :10].argmax(axis=1)
    cconf = P[np.arange(len(P)), ci]
    ti = P[:, 10:].argmax(axis=1)
    tconf = P[np.arange(len(P)), 10 + ti]
    c_known = cconf >= COLOR_THRESHOLD
    t_known = tconf >= TYPE_THRESHOLD

    print(f"[info] 颜色 known : {c_known.mean():.1%}   车型 known : {t_known.mean():.1%}")

    if args.track_vote:
        d2t = data["det_to_track_map"]
        by_track = defaultdict(list)
        for i, d in enumerate(dets):
            tid = d2t.get(d.get("target_id", ""))
            if tid:
                by_track[tid].append(i)
        voted = 0
        for tid, rows in by_track.items():
            # 只在"已知"的检测里投票；全未知则该轨迹保持未知
            crows = [r for r in rows if c_known[r]]
            trows = [r for r in rows if t_known[r]]
            if crows:
                best = Counter(int(ci[r]) for r in crows).most_common(1)[0][0]
                for r in crows:
                    ci[r] = best
                voted += 1
            if trows:
                best = Counter(int(ti[r]) for r in trows).most_common(1)[0][0]
                for r in trows:
                    ti[r] = best
        print(f"[info] 轨迹多数投票 : 覆盖 {voted} 条轨迹")

    # 写入
    changed = 0
    for i, d in enumerate(dets):
        attrs = d.setdefault("attributes", {})
        if c_known[i]:
            c_cn, c_en, c_id = COLORS_CN[ci[i]], COLORS_EN[ci[i]], int(ci[i])
        else:
            c_cn, c_en, c_id = UNKNOWN_CN, "unknown", -1
        if t_known[i]:
            t_cn, t_en, t_id = TYPES_CN[ti[i]], TYPES_EN[ti[i]], int(ti[i])
        else:
            t_cn, t_en, t_id = UNKNOWN_CN, "unknown", -1

        before = (attrs.get("color"), attrs.get("颜色"), attrs.get("color_id"),
                  attrs.get("vehicle_type"), attrs.get("车型"),
                  attrs.get("vehicle_type_en"), attrs.get("type_id"))
        attrs["color"] = c_cn
        attrs["颜色"] = c_cn
        attrs["color_id"] = c_id
        attrs["vehicle_type"] = t_cn
        attrs["车型"] = t_cn
        attrs["vehicle_type_en"] = t_en
        attrs["type_id"] = t_id
        if before != (c_cn, c_cn, c_id, t_cn, t_cn, t_en, t_id):
            changed += 1

    print(f"\n===== 合并 =====")
    print(f"字段发生变化 : {changed} / {len(dets)}  ({changed/len(dets):.1%})")
    cdist = Counter(COLORS_CN[ci[i]] if c_known[i] else UNKNOWN_CN for i in range(len(dets)))
    tdist = Counter(TYPES_CN[ti[i]] if t_known[i] else UNKNOWN_CN for i in range(len(dets)))
    print(f"颜色分布 : {cdist.most_common()}")
    print(f"车型分布 : {tdist.most_common()}")

    if args.check:
        print("\n[check] 只统计，未写出文件")
        return 0

    tmp = _RESULTS.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, _RESULTS)
    print(f"\n[done] 已原地更新 {_RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
