# encoding: utf-8
"""
scripts.extract_attributes - 用 PULC 车辆属性模型重跑全量裁剪图（二期 T1）。

背景（为什么必须重跑）:
    现有 `attributes` 里的颜色/车型是**系统性相机偏置**的——同一辆车换一个摄像头，
    属性就变（跨镜一致率实测 8.3%，比随机基线 31.9% 还低）。这不是"噪声大"，
    而是标注与摄像头相关，直接毁掉跨镜属性过滤。故用有预训练权重的模型重跑。

模型（实测取证，勿凭记忆）:
    PULC vehicle_attribute（PP-LCNet_x1_0，Apache-2.0），推理模型 7.5MB：
      https://paddleclas.bj.bcebos.com/models/PULC/inference/vehicle_attribute_infer.tar
    解压到 models/pulc/vehicle_attribute_infer/。

    输入 : 3x192x256（H=192, W=256，横版），Scale 1/255，
           Mean [0.485,0.456,0.406]，Std [0.229,0.224,0.225]（ImageNet），RGB。
    输出 : **19 维、且已在图内做过 sigmoid**（实测输出名 sigmoid_2.tmp_0，
           取值范围 0.0000~0.9996）。
           ⚠️ PaddleClas 官方的 VehicleAttribute 后处理里又调了一次 F.sigmoid——
           在本推理模型上是**重复激活**。本脚本按实测直接使用原值，不再套 sigmoid。
           前 10 维 = 颜色，后 9 维 = 车型；各自 argmax，置信度低于阈值则判 unknown。

本脚本**只产出旁挂文件，不改动 cityflow_results.json**（红线 4/5：属性字段是对外契约，
改它会影响检索结果）。先用旁挂结果做离线的跨镜一致率验证，验证通过再由
`merge_attributes.py` 显式合并。

用法:
    python scripts/extract_attributes.py --limit 200 --device cpu   # 冒烟
    python scripts/extract_attributes.py --device gpu --resume      # 全量
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 模型输出的英文标签（顺序即输出维度顺序，取自 PaddleClas
# ppcls/data/postprocess/attr_rec.py 的 VehicleAttribute）
COLORS = ["yellow", "orange", "green", "gray", "red", "blue", "white",
          "golden", "brown", "black"]
TYPES = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
         "truck", "estate"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE_HW = (192, 256)          # 网络的 (H, W)
COLOR_THRESHOLD = 0.5
TYPE_THRESHOLD = 0.5

DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "pulc" / "vehicle_attribute_infer"
DEFAULT_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
DEFAULT_OUTDIR = _PROJECT_ROOT / "output" / "attributes"


def parse_args():
    p = argparse.ArgumentParser(description="PULC vehicle attribute extraction (T1)")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    p.add_argument("--resume", action="store_true", help="从 progress.json 断点续跑")
    return p.parse_args()


def build_predictor(model_dir: Path, device: str):
    from paddle.inference import Config, create_predictor

    model_file = model_dir / "inference.pdmodel"
    params_file = model_dir / "inference.pdiparams"
    if not model_file.is_file():
        sys.exit(f"[fatal] 模型不存在: {model_file}（见本文件 docstring 的下载地址）")

    cfg = Config(str(model_file), str(params_file))
    if device == "gpu":
        # Paddle 3.x 的 API 是 enable_use_gpu（旧版的 enable_gpu 已移除）
        cfg.enable_use_gpu(1000, 0)
    else:
        cfg.disable_gpu()
    # 实测：开启 MKLDNN 会报 "OneDnnContext does not have the input Filter"
    # （CPU 后端 bug），直接关掉，PP-LCNet 本身很小，性能影响可接受。
    cfg.disable_mkldnn()
    predictor = create_predictor(cfg)
    return predictor


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> float32 CHW（192x256），照 PaddleClas 的 Resize+Normalize。"""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (SIZE_HW[1], SIZE_HW[0]), interpolation=cv2.INTER_LINEAR)
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    dets = data["detections"]
    total = len(dets)
    n = min(args.limit, total) if args.limit else total
    print(f"[info] detections : {total} (processing {n})")
    print(f"[info] model dir  : {args.model_dir}")
    print(f"[info] device     : {args.device}")

    predictor = build_predictor(Path(args.model_dir), args.device)
    in_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    out_handle = predictor.get_output_handle(predictor.get_output_names()[0])

    probs_path = outdir / "pulc_probs.npy"
    progress_path = outdir / "progress.json"

    start = 0
    failed: list[list] = []
    if args.resume and probs_path.is_file() and progress_path.is_file():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        probs = np.load(probs_path, mmap_mode="r+")
        if probs.shape != (n, 19):
            sys.exit(f"[fatal] --resume 的矩阵 {probs.shape} 与本次 ({n}, 19) 不符")
        start = int(prev.get("last_index", -1)) + 1
        failed = [list(x) for x in prev.get("failed", [])]
        print(f"[resume] 从第 {start} 行继续")
    else:
        # 磁盘映射：本机内存紧张，避免整块常驻（T2 的教训）
        probs = np.lib.format.open_memmap(
            probs_path, mode="w+", dtype=np.float32, shape=(n, 19)
        )
        probs[:] = 0

    def save_progress(last_index: int) -> None:
        progress_path.write_text(
            json.dumps({"last_index": last_index, "failed": failed}, ensure_ascii=False),
            encoding="utf-8",
        )

    batch_imgs: list[np.ndarray] = []
    batch_rows: list[int] = []

    def flush():
        if not batch_imgs:
            return
        in_handle.copy_from_cpu(np.stack(batch_imgs, axis=0))
        predictor.run()
        out = out_handle.copy_to_cpu()
        for r, v in zip(batch_rows, out):
            probs[r] = v
        batch_imgs.clear()
        batch_rows.clear()

    t0 = time.time()
    for i in range(start, n):
        raw = dets[i].get("crop_path")
        img = cv2.imread(str(raw).replace("\\", "/")) if raw else None
        if img is None:
            failed.append([i, str(raw), "read_failed"])
            continue
        try:
            batch_imgs.append(preprocess(img))
            batch_rows.append(i)
        except Exception as e:
            failed.append([i, str(raw), f"preprocess_error:{type(e).__name__}"])
            continue
        if len(batch_imgs) >= args.batch_size:
            flush()
        if (i + 1) % 10000 == 0:
            el = time.time() - t0
            done = i + 1 - start
            print(f"  [progress] {i + 1}/{n}  elapsed {el:.1f}s  "
                  f"({done / max(el, 1e-9):.1f} img/s)  failed={len(failed)}", flush=True)
            probs.flush()
            save_progress(i)
    flush()
    probs.flush()
    save_progress(n - 1)
    el = time.time() - t0
    print(f"[done] {probs_path}  shape={probs.shape}  {el:.1f}s")

    fail_log = outdir / "failed_rows.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        for row, path, reason in failed:
            f.write(f"{row}\t{reason}\t{path}\n")

    # ---- 派生：颜色/车型 argmax + 阈值 ----
    P = np.asarray(probs, dtype=np.float32)
    color_idx = P[:, :10].argmax(axis=1)
    type_idx = P[:, 10:].argmax(axis=1)
    color_conf = P[np.arange(len(P)), color_idx]
    type_conf = P[np.arange(len(P)), 10 + type_idx]
    color_known = color_conf >= COLOR_THRESHOLD
    type_known = type_conf >= TYPE_THRESHOLD

    meta = {
        "colors": COLORS,
        "types": TYPES,
        "color_threshold": COLOR_THRESHOLD,
        "type_threshold": TYPE_THRESHOLD,
        "note": "输出已在图内 sigmoid；未再套 PaddleClas 的二次 sigmoid",
    }
    with open(outdir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(f"rows           : {n}")
    print(f"failed rows    : {len(failed)}  (log: {fail_log})")
    print(f"颜色置信度     : mean={color_conf.mean():.4f}  >=0.5 占比={color_known.mean():.1%}")
    print(f"车型置信度     : mean={type_conf.mean():.4f}  >=0.5 占比={type_known.mean():.1%}")
    print(f"颜色 unknown   : {(~color_known).sum()}  ({(~color_known).mean():.1%})")
    print(f"车型 unknown   : {(~type_known).sum()}  ({(~type_known).mean():.1%})")

    from collections import Counter
    print("\n颜色分布 top5  :", Counter(COLORS[i] for i in color_idx[color_known]).most_common(5))
    print("车型分布 top5  :", Counter(TYPES[i] for i in type_idx[type_known]).most_common(5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
