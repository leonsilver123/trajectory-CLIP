"""
scripts.unify_clip_949 - 统一 CLIP 特征空间（二期 T3）

背景：output/cityflow_results.json 里有 949 条检测内联了 768 维的
clip_image_vector（CN-CLIP ViT-L-14 产物），而检索路径用的是 512 维
ViT-B-16 的 FAISS 索引（clip_vectors.faiss，覆盖全部 68349 条）。
两套向量维度不同、不可混用——拼接的外观分对这 949 条用的是 768 维，
其余用的是 512 维索引，造成特征空间分裂。

本脚本把这 949 张裁剪图用 ViT-B-16 重新编码成 512 维，落盘为
  output/clip_949_unified.npy      (949 x 512, float32, 已 L2 归一化)
  output/clip_949_unified_rows.npy (949, int64, 对应 detections 里的原始下标)
不覆盖任何现有文件；接线（让 builder 改用这份）是后续波次的事。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_OUT_VEC = _PROJECT_ROOT / "output" / "clip_949_unified.npy"
_OUT_ROWS = _PROJECT_ROOT / "output" / "clip_949_unified_rows.npy"


def main() -> int:
    data = json.load(open(_RESULTS, encoding="utf-8"))
    dets = data["detections"]

    # 找出内联 768 维向量的那批（下标 + 裁剪图路径）
    rows = [i for i, x in enumerate(dets) if x.get("clip_image_vector") is not None]
    print(f"内联 768 维检测数: {len(rows)}")

    from cn_clip.clip import load_from_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 Chinese-CLIP ViT-B-16, device={device}")
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root=str(_PROJECT_ROOT / "models"))
    model.eval()

    vecs = np.zeros((len(rows), 512), dtype=np.float32)
    failed: list[int] = []
    with torch.no_grad():
        for j, i in enumerate(rows):
            crop = dets[i].get("crop_path", "")
            p = _PROJECT_ROOT / crop if not Path(crop).is_absolute() else Path(crop)
            try:
                img = Image.open(p).convert("RGB")
                inp = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                vecs[j] = feat.cpu().numpy().astype(np.float32)[0]
            except Exception as e:  # 读图失败则留全零并记录
                failed.append((i, str(e)))
            if (j + 1) % 200 == 0:
                print(f"  进度 {j + 1}/{len(rows)}")

    np.save(_OUT_VEC, vecs)
    np.save(_OUT_ROWS, np.array(rows, dtype=np.int64))
    print(f"完成: {_OUT_VEC.name} shape={vecs.shape}")
    print(f"失败 {len(failed)} 条" + (f": {failed[:5]}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
