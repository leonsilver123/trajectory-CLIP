"""
scripts/extract_reid_osnet.py - 用 OSNet 提取车辆 ReID 特征

用法:
    python scripts/extract_reid_osnet.py --outdir output/reid_osnet
    python scripts/extract_reid_osnet.py --limit 500          # 冒烟

产物（与 fast-reid 版格式一致，便于直接对比）:
    <outdir>/reid_vectors.npy     float32, (N, 512)，第 i 行 <-> detections[i]
    <outdir>/reid_vectors.faiss   FAISS IndexFlatIP（向量已 L2 归一化）
    <outdir>/progress.json        断点续跑信息

## 为什么单独一个脚本，而不是给 extract_reid.py 加 --backbone

`extract_reid.py` 走 fast-reid 的源码树（`third_party/fast-reid`）与 yml 配置，
OSNet 走 torchreid 的 Python API，两者的模型构造与预处理完全不同。
硬塞进一个脚本会让两条路互相牵制；分开写、**保持产物格式一致**，
同样能完成"两种 backbone 对比"的目标（见 `scripts/eval_cross_camera.py`）。

## 重要：OSNet 是**行人**再识别模型

OSNet 的公开权重是在行人数据集（MSMT17 / Market1501 / ImageNet）上训练的，
而本项目要识别的是**车辆**。这与 fast-reid 的 VeRi（车辆）权重相比是**跨域**使用，
预期效果会差 —— 必须实测后如实报告，不要假设"换了 OSNet 就更好"。
（同类问题本项目已踩过一次：VeRi 权重在自己的数据集上 R@1 97%，到 AICity22 只剩两成。）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger

logger = get_logger("scripts.extract_reid_osnet")

DEFAULT_MODEL = ROOT / "models" / "osnet_x1_0_imagenet.pth"
DEFAULT_OUTDIR = ROOT / "output" / "reid_osnet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract vehicle ReID features with OSNet")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="OSNet 权重路径")
    p.add_argument("--arch", default="osnet_x1_0", help="torchreid 模型名")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="输出目录")
    p.add_argument("--batch-size", type=int, default=64, help="推理批大小")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="每多少条落盘一次（中断后已算的部分不丢；0=只在结束时写）")
    p.add_argument("--no-resume", action="store_true",
                   help="忽略已有检查点，从第 0 行重算（默认会续跑）")
    return p.parse_args()


def resolve_crop_path(raw, root: Path = ROOT) -> Path | None:
    """
    把 detection 里的 crop_path 解析成真实文件路径

    JSON 里存的是 Windows 相对路径（如 'output\\aicity22_crops\\c001\\x.jpg'），
    需要做分隔符归一化并相对项目根解析。
    """
    if not raw:
        return None
    rel = str(raw).replace("\\", "/").lstrip("./")
    p = Path(rel)
    if not p.is_absolute():
        p = root / p
    return p if p.exists() else None


def build_osnet(arch: str, weights: str, device: str):
    """构造 OSNet 并加载权重，返回处于 eval 模式的模型"""
    import torch
    from torchreid.reid.models import build_model

    if not Path(weights).exists():
        sys.exit(
            f"[fatal] 找不到 OSNet 权重: {weights}\n"
            f"  下载方式: python -m gdown 1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY "
            f"-O {weights}"
        )

    model = build_model(arch, num_classes=1000, pretrained=False, use_gpu=False)
    state = torch.load(weights, map_location="cpu")
    # 官方 zoo 的 checkpoint 有的直接是 state_dict，有的包在 'state_dict' 键里
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("权重缺失键 %d 个（通常是分类头，不影响特征提取）: %s",
                       len(missing), missing[:3])
    model.to(device).eval()
    logger.info("OSNet 已加载: arch=%s device=%s", arch, device)
    return model


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # ── 数据 ──
    from src.storage.datastore import load_results

    data = load_results()
    if not data:
        sys.exit("[fatal] 数据不可用（output/datastore 与 JSON 均缺失）")
    detections = data.get("detections", [])
    if args.limit:
        detections = detections[: args.limit]
    logger.info("待处理检测数: %d", len(detections))

    model = build_osnet(args.arch, args.model, device)

    # OSNet 的标准输入：256×128（行人比例）。车辆裁剪图更接近方形，
    # 这里仍按官方预处理走 —— 改比例会偏离预训练分布，属另一个变量。
    preprocess = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dim: int | None = None
    vectors: list = []
    failed: list = []
    t0 = time.time()

    batch_imgs, batch_rows = [], []

    # ── 续跑：读回上次的检查点，跳过已算过的行 ──
    #
    # 2026-09-21 补。此前 `save_checkpoint` 只**写**不**读** —— 中断后已算的
    # 向量虽在磁盘上，重跑却仍从第 0 行开始，等于白算。
    # 全量实测 7.65 张/秒 ⇒ 68,349 张约 2.5 小时，中途被系统回收的概率不低，
    # 能续跑是这类长任务的基本要求。
    done_rows: set[int] = set()
    npy_path = outdir / "reid_vectors.npy"
    done_path = outdir / "done_rows.npy"
    if not args.no_resume and npy_path.exists() and done_path.exists():
        try:
            # ⚠️ 这里**不能**用 mmap_mode="r"：映射会一直持有文件句柄，
            # 而 save_checkpoint 稍后要写回同一个 .npy —— Windows 下覆盖一个
            # 仍被映射的文件会抛 `OSError: [Errno 22] Invalid argument`
            # （Linux 允许，所以这个坑只在 Windows 上炸）。
            prev = np.load(str(npy_path))
            rows = np.load(str(done_path))
            if prev.shape[0] == len(detections):
                dim = int(prev.shape[1])
                for r in rows:
                    r = int(r)
                    vec = np.asarray(prev[r], dtype=np.float32)
                    if not np.isnan(vec).any():
                        vectors.append((r, vec))
                        done_rows.add(r)
                logger.info("续跑：从检查点读回 %d 条已算向量，将跳过这些行",
                            len(done_rows))
            else:
                logger.warning(
                    "检查点形状不匹配（%s vs 当前 %d 条检测），忽略，从头开始",
                    prev.shape, len(detections))
            del prev, rows
        except Exception as e:
            logger.warning("检查点读取失败（%s），从头开始", e)

    def save_checkpoint(processed: int, final: bool = False) -> None:
        """
        周期性落盘（**不要只在最后写**）

        2026-09-21 的教训：本脚本原先只在跑完时写文件，结果一次因系统内存不足
        被中断的运行，把约 22,000 张（半小时算力）的结果全部丢掉了。
        现在每 `--checkpoint-every` 条写一次，中断后已算的部分仍在磁盘上。
        """
        if not vectors:
            return
        d = dim or 512
        mat = np.full((len(detections), d), np.nan, dtype=np.float32)
        for row, vec in vectors:
            mat[row] = vec
        np.save(str(outdir / "reid_vectors.npy"), mat)
        np.save(str(outdir / "done_rows.npy"),
                np.array([r for r, _ in vectors], dtype=np.int64))
        (outdir / "progress.json").write_text(json.dumps({
            "total": len(detections),
            "extracted": len(vectors),
            "failed": len(failed),
            "last_processed_index": processed,
            "dim": d,
            "arch": args.arch,
            "weights": str(args.model),
            "complete": final,
            "elapsed_seconds": round(time.time() - t0, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if failed:
            (outdir / "failed_rows.txt").write_text(
                "\n".join(f"{i}\t{p}\t{r}" for i, p, r in failed), encoding="utf-8")

    def flush() -> None:
        """跑一个 batch 并把 512 维特征写回 vectors"""
        nonlocal batch_imgs, batch_rows, dim
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            # OSNet 的 forward 返回分类 logits（512 维），
            # 这正是 global_avgpool 之后的 embedding —— 公开权重的标准用法。
            feats = model(tensor)
        feats = torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()
        dim = feats.shape[1]
        for row, vec in zip(batch_rows, feats):
            vectors.append((row, vec.astype(np.float32)))
        batch_imgs, batch_rows = [], []

    for idx, det in enumerate(detections):
        if idx in done_rows:
            continue                       # 续跑：该行上次已算完
        path = resolve_crop_path(det.get("crop_path"))
        if path is None:
            failed.append((idx, det.get("crop_path"), "crop 不存在"))
            continue
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_rows.append(idx)
        except Exception as e:
            failed.append((idx, str(path), f"{type(e).__name__}: {e}"))
            continue

        if len(batch_imgs) >= args.batch_size:
            flush()
        if (idx + 1) % 2000 == 0:
            el = time.time() - t0
            logger.info("进度 %d/%d | 失败 %d | %.1f 张/秒",
                        idx + 1, len(detections), len(failed), (idx + 1) / max(el, 1e-6))
        if args.checkpoint_every and (idx + 1) % args.checkpoint_every == 0:
            flush()
            save_checkpoint(idx + 1)
            logger.info("  [checkpoint] 已落盘 %d 条", len(vectors))
    flush()

    if not vectors:
        sys.exit("[fatal] 没有成功提取任何向量")

    dim = dim or 512
    save_checkpoint(len(detections), final=True)

    npy_path = outdir / "reid_vectors.npy"
    mat = np.load(str(npy_path), mmap_mode="r")
    logger.info("已写出 %s  shape=%s", npy_path, mat.shape)

    # FAISS（跳过 NaN 行）
    valid = ~np.isnan(np.asarray(mat)).any(axis=1)
    try:
        import faiss

        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(mat)[valid])
        faiss.write_index(index, str(outdir / "reid_vectors.faiss"))
        logger.info("已写出 FAISS 索引: ntotal=%d dim=%d", index.ntotal, index.d)
    except Exception as e:
        logger.warning("FAISS 索引未写出: %s", e)

    logger.info("完成: 成功 %d / 失败 %d / 耗时 %.1fs",
                int(valid.sum()), len(failed), time.time() - t0)


if __name__ == "__main__":
    main()
