"""
scripts/eval_reid_trained_cross_camera.py - 对训练好的 ReID 模型做**真·跨摄像头**评测

用法:
    python scripts/eval_reid_trained_cross_camera.py --model output/reid_trained_c2/best.pth
    python scripts/eval_reid_trained_cross_camera.py --model ... --batch-size 32

## 为什么需要这个脚本（而不是直接用 train_reid_triplet.py 报的 Rank-1）

`train_reid_triplet.py` 每个 epoch 打印的 val Rank-1 **是同摄像头口径**：
`build_reid_training_set.py` 默认 `--val-cameras-per-id 1`，每个身份在 val 里
只占 1 个摄像头，于是 val 内部的 query-gallery 配对全部来自同一摄像头。

实测证明：训练出的模型在该口径下报 **Rank-1 = 0.909**；而同一份数据用**真正
跨镜**口径（`scripts/eval_cross_camera.py`，全量检测池）测 fast-reid 只有
**0.207**。4 倍多的落差是口径差，不是模型差。

本脚本把口径摆正：

    query   = 该身份在 **val 摄像头** 里的图
    gallery = 同一身份在 **train 摄像头**（与 val 摄像头不同）里的图

两侧摄像头严格不同，所以测的是真的跨镜匹配。身份是闭集（215 个身份两侧都出现）。

## 诚实边界

- 本数据集只有 215 个身份、裁剪图中位数 118×98 像素，跨镜 ReID 本身很弱
  （fast-reid 的 d-prime 只有 0.78）。**不要拿这个数字与 VeRi/MSMT17 比。**
- 本脚本测的是**训练产物**，训练集是按摄像头切分的，query 摄像头从未参与训练，
  因此不存在"背下这一帧"的捷径。
- AICity22 许可禁止商用，产物仅限研究。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger

logger = get_logger("scripts.eval_reid_trained_cross_camera")

DEFAULT_DATASET = ROOT / "output" / "reid_train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练好的 ReID 模型的跨摄像头评测")
    p.add_argument("--model", required=True, help="best.pth / last.pth 路径")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default="", help="结果写 JSON")
    p.add_argument("--max-per-camera", type=int, default=8,
                   help="每 (身份,摄像头) 最多取多少张，控制评测规模")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from torchvision import transforms

    from torchreid.reid.models import build_model

    ckpt = torch.load(args.model, map_location="cpu")
    arch = ckpt.get("arch", "osnet_x1_0")
    label_of = ckpt.get("label_of", {})
    n_ids = max(len(label_of), 2)

    device = args.device if (args.device == "cuda" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_model(arch, num_classes=n_ids, pretrained=False, use_gpu=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    logger.info("已加载 %s（arch=%s，训练时 epoch=%s）",
                args.model, arch, ckpt.get("epoch"))

    train = json.loads((Path(args.dataset) / "train.json").read_text(encoding="utf-8"))
    val = json.loads((Path(args.dataset) / "val.json").read_text(encoding="utf-8"))

    # 身份 -> 摄像头 -> 记录
    def group(recs):
        g: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in recs:
            g[r["vehicle_id"]][r["camera_id"]].append(r)
        return g

    g_train, g_val = group(train), group(val)

    def thin(items):
        if len(items) <= args.max_per_camera:
            return items
        step = len(items) / args.max_per_camera
        return [items[int(i * step)] for i in range(args.max_per_camera)]

    queries, gallery, q_ids = [], [], []
    n_cross = 0
    for vid, vcams in g_val.items():
        tcams = g_train.get(vid)
        if not tcams:
            continue
        # gallery = 该身份在**与 val 摄像头不同**的训练摄像头里的图
        gal = []
        for cam, items in tcams.items():
            if cam in vcams:
                continue
            gal.extend(thin(sorted(items, key=lambda x: x.get("frame_id", 0))))
        if not gal:
            continue
        qs = []
        for cam, items in vcams.items():
            qs.extend(thin(sorted(items, key=lambda x: x.get("frame_id", 0))))
        if not qs:
            continue
        n_cross += 1
        queries.extend(qs)
        q_ids.extend([vid] * len(qs))
        gallery.extend(gal)

    if not queries or not gallery:
        logger.error("构造不出跨摄像头 query/gallery（val 与 train 的摄像头没有区分开）")
        return 1

    g_ids = [r["vehicle_id"] for r in gallery]
    logger.info("跨镜评测: query %d 张（%d 个身份）| gallery %d 张",
                len(queries), n_cross, len(gallery))

    tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def embed(records):
        from PIL import Image

        feats = []
        with torch.no_grad():
            for start in range(0, len(records), args.batch_size):
                chunk = records[start:start + args.batch_size]
                imgs = []
                for r in chunk:
                    p = ROOT / r["crop_path"]
                    try:
                        im = Image.open(p).convert("RGB")
                    except Exception:
                        im = Image.fromarray(np.zeros((256, 128, 3), dtype="uint8"))
                    imgs.append(tf(im))
                out = model(torch.stack(imgs).to(device))
                out = torch.nn.functional.normalize(out.float(), dim=1)
                feats.append(out.cpu().numpy())
                if start and start % (args.batch_size * 20) == 0:
                    logger.info("  嵌入进度 %d/%d", start, len(records))
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), np.float32)

    logger.info("提取 query 嵌入…")
    Q = embed(queries)
    logger.info("提取 gallery 嵌入…")
    G = embed(gallery)

    g_ids_arr = np.asarray(g_ids)
    q_ids_arr = np.asarray(q_ids)

    # 逐个 query 排序，统计 Rank-1/5 与 mAP（同身份即命中）
    r1 = r5 = 0
    aps = []
    for i in range(len(Q)):
        sim = G @ Q[i]
        order = np.argsort(-sim)
        match = g_ids_arr[order] == q_ids_arr[i]
        if not match.any():
            continue
        first = int(np.argmax(match))
        if first == 0:
            r1 += 1
        if first < 5:
            r5 += 1
        hits = np.cumsum(match)
        precs = hits / (np.arange(len(match)) + 1)
        aps.append(float((precs * match).sum() / match.sum()))

    nq = len(aps)
    out = {
        "model": args.model,
        "protocol": "cross_camera (query 摄像头 ∉ gallery 摄像头)",
        "n_query": nq,
        "n_gallery": len(gallery),
        "n_identities": n_cross,
        "rank1": round(r1 / nq, 4) if nq else None,
        "rank5": round(r5 / nq, 4) if nq else None,
        "map": round(float(np.mean(aps)), 4) if aps else None,
    }
    logger.info("跨镜 Rank-1 = %.4f | Rank-5 = %.4f | mAP = %.4f | query %d",
                out["rank1"] or 0, out["rank5"] or 0, out["map"] or 0, nq)

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("结果已写出: %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
