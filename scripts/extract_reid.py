# encoding: utf-8
"""
车辆 ReID 特征批量提取脚本（二期 T2）。

用途：
    读取 output/cityflow_results.json 的全部 detections（保持原始顺序），
    用每条 detection 的 crop_path 找到裁剪图，批量前向 fast-reid 模型
    （VeRi SBS R50-ibn，**2048 维**，L2 归一化），落盘为：

        output/reid/reid_vectors.npy     float32, (N, 2048)，第 i 行 <-> detections[i]
        output/reid/reid_vectors.faiss   FAISS IndexFlatIP，向量顺序同上
        output/reid/failed_rows.txt      读取/前向失败的 detection 行号（该行填空零向量）

设计要点：
    - 第 i 个向量严格对应第 i 条 detection（下游 stitching 依赖该顺序契约）。
    - 依赖 fast-reid 源码树（third_party/fast-reid），不走 pip 安装。
    - 前处理完全照抄 fast-reid/demo/predictor.py 的 FeatureExtractionDemo.run_on_image：
        BGR -> RGB、cv2.resize 到 tuple(INPUT.SIZE_TEST[::-1])（即 (W,H) 给 cv2，
        SIZE_TEST 自身是 [H,W]）、float32、HWC -> CHW。
      fast-reid 不做 ImageNet 均值方差归一化（实测默认 transformation 不含 Normalize），
      故此处也不归一化。

    - **实测纠正（勿凭直觉）**：
      ① 本权重的嵌入维度是 **2048**，不是 512。取证：`models/veri_sbs_R50-ibn.pth`
         里 `heads.bottleneck.0.weight` = (2048,)、`heads.classifier.weight` = (575, 2048)，
         且 `MODEL.HEADS.EMBEDDING_DIM` 未配置 → 取 `BACKBONE.FEAT_DIM`(=2048)。
         下游若按 512 维断言会直接错。
      ② 模型输出**并非** L2 归一化：实测探针向量模长 15.05（`run_on_image` 的
         docstring 写 "normalized feature"，与实际不符）。因此下面 `flush()` 里的
         防御性归一化是**必需**的，不是可选兜底——去掉它会让 IndexFlatIP 变成
         比"向量模长"而不是比余弦。

用法：
    python scripts/extract_reid.py                       # 全量 68349 张
    python scripts/extract_reid.py --limit 200           # 快速冒烟测试
    python scripts/extract_reid.py --batch-size 128 --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 项目根目录（本脚本位于 <root>/scripts/）
ROOT = Path(__file__).resolve().parent.parent
FASTREID_ROOT = ROOT / "third_party" / "fast-reid"

# fast-reid 无 setup.py，直接把它挂到 sys.path 上导入
sys.path.insert(0, str(FASTREID_ROOT))

DEFAULT_CFG = FASTREID_ROOT / "configs" / "VeRi" / "sbs_R50-ibn.yml"
DEFAULT_WEIGHTS = ROOT / "models" / "veri_sbs_R50-ibn.pth"
DEFAULT_RESULTS = ROOT / "output" / "cityflow_results.json"
DEFAULT_OUTDIR = ROOT / "output" / "reid"


def parse_args():
    p = argparse.ArgumentParser(description="Extract vehicle ReID features with fast-reid")
    p.add_argument("--config-file", default=str(DEFAULT_CFG), help="fast-reid config yml")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="fast-reid checkpoint (.pth)")
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="cityflow_results.json")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="output directory")
    p.add_argument("--batch-size", type=int, default=128, help="inference batch size")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--limit", type=int, default=0, help="only process first N detections (smoke test)")
    p.add_argument("--resume", action="store_true",
                   help="从 progress.json 记录的断点续跑（崩溃后不必从头再来）")
    p.add_argument("--save-npy", action="store_true", default=True, help="write reid_vectors.npy")
    p.add_argument("--save-faiss", action="store_true", default=True, help="write reid_vectors.faiss")
    return p.parse_args()


def build_predictor(cfg_file, weights, device):
    """构造 fast-reid 的 DefaultPredictor（模型加载 + 前处理由它负责）。"""
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_file))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg, DefaultPredictor(cfg)


def resolve_crop_path(raw, root=ROOT):
    """把 detection 里的 crop_path 解析成真实文件路径。

    JSON 里存的是 Windows 相对路径（如 'output\\aicity22_crops\\c001\\x.jpg'），
    相对项目根。同时兼容正斜杠与绝对路径。
    """
    if not raw:
        return None
    norm = str(raw).replace("\\", "/")
    p = Path(norm)
    if not p.is_absolute():
        p = root / norm
    return p


def preprocess(img_bgr, size_test):
    """照抄 FeatureExtractionDemo.run_on_image 的前处理，返回 float32 CHW。"""
    rgb = img_bgr[:, :, ::-1]
    # SIZE_TEST 是 [H, W]，cv2.resize 要 (W, H)，故反转
    resized = cv2.resize(rgb, tuple(size_test[::-1]), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(resized.astype("float32").transpose(2, 0, 1))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not FASTREID_ROOT.is_dir():
        sys.exit(f"[fatal] fast-reid not found at {FASTREID_ROOT}; "
                 f"run: git clone --depth 1 https://github.com/JDAI-CV/fast-reid.git third_party/fast-reid")
    if not Path(args.weights).is_file():
        sys.exit(f"[fatal] weights not found: {args.weights}")

    print(f"[info] project root     : {ROOT}")
    print(f"[info] fast-reid root   : {FASTREID_ROOT}")
    print(f"[info] config           : {args.config_file}")
    print(f"[info] weights          : {args.weights}")
    print(f"[info] device           : {args.device}")

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    detections = data["detections"]
    total = len(detections)
    n = min(args.limit, total) if args.limit else total
    print(f"[info] detections total : {total} (processing {n})")

    cfg, predictor = build_predictor(args.config_file, args.weights, args.device)
    size_test = list(cfg.INPUT.SIZE_TEST)
    print(f"[info] INPUT.SIZE_TEST  : {size_test}")

    # 探测特征维度（跑一张，顺带验证权重能加载 + torch 兼容性）
    probe_path = resolve_crop_path(detections[0].get("crop_path"))
    probe_img = cv2.imread(str(probe_path))
    if probe_img is None:
        sys.exit(f"[fatal] cannot read probe crop: {probe_path}")
    t0 = time.time()
    with torch.no_grad():
        probe = predictor(torch.as_tensor(preprocess(probe_img, size_test))[None])
    feat_dim = int(probe.shape[1])
    print(f"[info] feature dim      : {feat_dim}  (probe latency {time.time() - t0:.2f}s)")
    print(f"[info] probe raw norm   : {float(probe.norm(dim=1).mean()):.4f} (expect ~1.0 if L2-normalized)")

    # ---- 输出矩阵：磁盘映射，不整块吃内存 ----
    # 实测教训：全量矩阵 (68349, 2048) float32 常驻内存 = 560MB，叠加 torch/CUDA
    # 的宿主侧占用后，在本机（16GB 物理内存）上跑会在 3.5 万张附近 OOM 崩掉
    # （报 "Unable to allocate 96.0 MiB"，即连一个 batch 的缓冲都开不出来）。
    # 改用 np.memmap 直接写盘：进程内只有当前 batch 的缓冲，且崩溃后已写的数据还在。
    npy_path = outdir / "reid_vectors.npy"
    progress_path = outdir / "progress.json"

    start_index = 0
    failed = []           # (row_index, crop_path, reason)
    if args.resume and npy_path.is_file() and progress_path.is_file():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        vectors = np.load(npy_path, mmap_mode="r+")
        if vectors.shape != (n, feat_dim):
            sys.exit(f"[fatal] --resume 的已有矩阵 shape={vectors.shape} 与本次 ({n}, {feat_dim}) 不符，"
                     f"请删除 {npy_path} 后重跑")
        start_index = int(prev.get("last_index", -1)) + 1
        failed = [tuple(x) for x in prev.get("failed", [])]
        print(f"[resume] 从第 {start_index} 行继续（已有 {len(failed)} 条失败记录）")
    else:
        vectors = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float32, shape=(n, feat_dim)
        )
        vectors[:] = 0     # 失败行保持全零（与旧行为一致）

    batch_imgs, batch_rows = [], []

    def flush():
        """对一个 batch 做前向并写回 vectors。"""
        if not batch_imgs:
            return
        tensor = torch.as_tensor(np.stack(batch_imgs, axis=0))
        with torch.no_grad():
            feats = predictor(tensor).cpu().numpy().astype(np.float32)
        # 防御性归一化：权重输出已是 L2 归一化，这里再兜一次底
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        for r, v in zip(batch_rows, feats):
            vectors[r] = v
        batch_imgs.clear()
        batch_rows.clear()

    def save_progress(last_index: int) -> None:
        """记录进度，便于崩溃后续跑（与 memmap 里已落盘的数据配套）。"""
        progress_path.write_text(
            json.dumps({"last_index": last_index, "failed": failed}, ensure_ascii=False),
            encoding="utf-8",
        )

    t_start = time.time()
    for i in range(start_index, n):
        crop = resolve_crop_path(detections[i].get("crop_path"))
        img = cv2.imread(str(crop)) if crop is not None else None
        if img is None:
            failed.append((i, str(crop), "read_failed"))
            continue
        try:
            batch_imgs.append(preprocess(img, size_test))
            batch_rows.append(i)
        except Exception as e:  # 前处理异常（尺寸为 0 等）
            failed.append((i, str(crop), f"preprocess_error:{type(e).__name__}"))
            continue
        if len(batch_imgs) >= args.batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            done = i + 1 - start_index
            print(f"  [progress] {i + 1}/{n}  elapsed {elapsed:.1f}s  "
                  f"({done / max(elapsed, 1e-9):.1f} img/s)  failed={len(failed)}", flush=True)
            vectors.flush()          # 把 memmap 的脏页刷到磁盘，崩溃也不丢
            save_progress(i)
    flush()
    vectors.flush()
    save_progress(n - 1)
    infer_sec = time.time() - t_start

    # 失败行写日志（该行保持全零向量）
    fail_log = outdir / "failed_rows.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        for row, path, reason in failed:
            f.write(f"{row}\t{reason}\t{path}\n")

    # ---- 落盘 ----
    # vectors 已是磁盘上的 memmap，数据早在运行中逐批写好了，这里无需再 np.save
    print(f"[done] {npy_path}  shape={vectors.shape}  dtype={vectors.dtype}（memmap，已逐批落盘）")

    if args.save_faiss:
        import faiss
        index = faiss.IndexFlatIP(feat_dim)          # 向量已 L2 归一化 -> 内积 == 余弦相似度
        # 分块 add：避免一次性把 560MB 矩阵读进内存（本机内存紧张，见上方注释）
        chunk = 8192
        for s in range(0, n, chunk):
            index.add(np.asarray(vectors[s:s + chunk], dtype=np.float32))
        faiss_path = outdir / "reid_vectors.faiss"
        faiss.write_index(index, str(faiss_path))
        print(f"[done] wrote {faiss_path}  ntotal={index.ntotal}  d={index.d}")

    # 零范数行统计（分块，避免一次性物化大数组）
    zeros = 0
    for s in range(0, n, 8192):
        blk = np.asarray(vectors[s:s + 8192], dtype=np.float32)
        zeros += int((np.linalg.norm(blk, axis=1) == 0).sum())
    print("\n===== SUMMARY =====")
    print(f"rows                 : {n}")
    print(f"feature dim          : {feat_dim}")
    print(f"failed rows          : {len(failed)} (zero vectors); log: {fail_log}")
    print(f"zero-norm rows       : {zeros}")
    print(f"total infer time     : {infer_sec:.1f}s")
    if n:
        print(f"throughput           : {n / max(infer_sec, 1e-9):.1f} img/s")
        print(f"per-image            : {1000.0 * infer_sec / n:.2f} ms")


if __name__ == "__main__":
    main()
