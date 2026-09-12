"""
scripts.ensemble_reid - ReID 多模型集成（二期 M5）

把多个检测级 ReID 向量（VeRi / VehicleID / VERI-Wild，均 R50-ibn、2048 维、L2 归一化）
做**归一化平均**集成，落盘为 output/reid/reid_vectors.npy（覆盖），随后由
build_reid_tracks.py 重算轨迹级向量。

步骤：
    1. python scripts/extract_reid.py --weights models/vehicleid_bot_R50-ibn.pth \
           --config-file .../VehicleID/bagtricks_R50-ibn.yml --outdir output/reid/vehicleid
    2. python scripts/extract_reid.py --weights models/veriwild_bot_R50-ibn.pth \
           --config-file .../VERIWild/bagtricks_R50-ibn.yml --outdir output/reid/veriwild
    3. python scripts/ensemble_reid.py          # 平均三个检测级向量
    4. python scripts/build_reid_tracks.py      # 重算轨迹级
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REID = _PROJECT_ROOT / "output" / "reid"

# (路径, 是否必须存在)
SOURCES = [
    (REID / "reid_vectors.npy", True),                 # VeRi（原始）
    (REID / "vehicleid" / "reid_vectors.npy", True),   # VehicleID
    (REID / "veriwild" / "reid_vectors.npy", False),   # VERI-Wild（可选）
]


def _load(p: Path) -> np.ndarray:
    v = np.load(str(p)).astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    # 归一化；零向量（提取失败行）保持零，参与平均前会单独处理
    return v / (n + 1e-8)


def main() -> int:
    vecs = []
    names = []
    for path, required in SOURCES:
        if not path.exists():
            if required:
                print(f"[error] 缺少 {path}")
                return 1
            print(f"[skip] {path} 不存在，跳过")
            continue
        vecs.append(_load(path))
        names.append(path.parent.name if path.parent.name != "reid" else "veri")
        print(f"  载入 {path} 形状 {vecs[-1].shape}")

    if not vecs:
        return 1

    # 检测级维度对齐（提取失败行为全零，各模型失败行可能不同，统一用全零处理）
    N = vecs[0].shape[0]
    for i, v in enumerate(vecs):
        assert v.shape[0] == N, f"{names[i]} 行数 {v.shape[0]} != {N}"

    # 归一化平均：对每个检测，平均各模型的归一化向量，再重新归一化。
    # 全零向量（某模型提取失败）在平均前被其它模型补足；若所有模型都失败则保持零。
    summed = np.zeros_like(vecs[0])
    for v in vecs:
        summed += v
    ensemble = summed / len(vecs)
    norms = np.linalg.norm(ensemble, axis=1, keepdims=True)
    ensemble = ensemble / (norms + 1e-8)
    # 全零行还原为零（下游把零向量判为"无证据"）
    ensemble[norms.reshape(-1) < 1e-6] = 0.0

    # 备份原始 veri，再覆盖
    bak = REID / "reid_vectors.veri_backup.npy"
    if not bak.exists():
        shutil.copy(REID / "reid_vectors.npy", bak)
        print(f"  已备份原始 VeRi 向量到 {bak.name}")

    np.save(str(REID / "reid_vectors.npy"), ensemble)
    print(f"  集成向量已写入 reid_vectors.npy 形状 {ensemble.shape}（{len(names)} 模型：{', '.join(names)}）")
    print("  下一步：python scripts/build_reid_tracks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
