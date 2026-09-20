"""
scripts/build_index.py - 构建/校验 FAISS 向量索引

用法:
    # 从向量矩阵构建索引
    python scripts/build_index.py --vectors output/clip_949_unified.npy --out output/clip_949.faiss

    # 校验已有索引的维度与规模（对照在线检索的期望值）
    python scripts/build_index.py --verify output/clip_vectors.faiss

    # 校验并断言必须为 512 维（与在线检索的 ViT-B-16 特征空间一致）
    python scripts/build_index.py --verify output/clip_vectors.faiss --expect-dim 512

## 与原 TODO 的差异（重要）

原实现的 TODO 写的是"从 TrackManager 加载目标实例 → 导入 Qdrant"。
但实测：**Qdrant 在项目里没有任何消费者**（`docker-compose.yml` 起了容器，
`configs/default.yaml` 配了地址，`src/retrieval/vector_recall.py` 有实现，
但在线检索走的是 FAISS + `output/clip_vectors.faiss`）。
照那个 TODO 实现会造出一个**没人用**的脚本。

因此这里实现的是**真正被用到的那个索引**（FAISS）的构建与校验工具，
并且把"特征空间必须一致"这条不变量做成可执行的断言（`--expect-dim`）——
因为项目真的因 768/512 混库出过事故（见 scripts/unify_clip_949.py）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.common.logger import get_logger

logger = get_logger("scripts.build_index")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建/校验 FAISS 向量索引")
    parser.add_argument("--vectors", type=str, default=None,
                        help="输入向量矩阵 .npy（形状 N×D，应已 L2 归一化）")
    parser.add_argument("--out", type=str, default=None,
                        help="输出索引文件路径（.faiss）")
    parser.add_argument("--verify", type=str, default=None,
                        help="改为校验模式：检查指定索引文件的维度与规模")
    parser.add_argument("--expect-dim", type=int, default=None,
                        help="期望维度；不匹配则非零退出（用于 CI/回归）")
    parser.add_argument("--expect-ntotal", type=int, default=None,
                        help="期望向量条数；不匹配则非零退出")
    return parser.parse_args()


def build_index(vectors_path: str, out_path: str) -> int:
    """从向量矩阵构建 IndexFlatIP 索引"""
    import faiss

    src = Path(vectors_path)
    if not src.exists():
        logger.error(f"向量文件不存在: {src}")
        return 2

    vectors = np.load(str(src)).astype(np.float32)
    if vectors.ndim != 2:
        logger.error(f"向量矩阵应为二维 (N, D)，实际 {vectors.shape}")
        return 2

    n, dim = vectors.shape
    logger.info(f"加载向量: {src} -> shape=({n}, {dim})")

    # 内积索引要求向量已归一化（否则内积不等于余弦）
    norms = np.linalg.norm(vectors, axis=1)
    if n and not np.allclose(norms, 1.0, atol=1e-3):
        logger.warning(
            "向量未归一化（模长中位数 %.4f）—— IndexFlatIP 的内积将不等于余弦相似度，"
            "正在就地 L2 归一化", float(np.median(norms))
        )
        faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    logger.info(f"索引已写入: {out} (ntotal={index.ntotal}, dim={index.d})")
    return 0


def verify_index(path: str, expect_dim, expect_ntotal) -> int:
    """校验索引文件的维度与规模"""
    import faiss

    p = Path(path)
    if not p.exists():
        logger.error(f"索引文件不存在: {p}")
        return 2

    index = faiss.read_index(str(p))
    logger.info(f"索引 {p}: ntotal={index.ntotal}, dim={index.d}")

    ok = True
    if expect_dim is not None and index.d != expect_dim:
        logger.error(
            f"维度不匹配: 索引为 {index.d} 维，期望 {expect_dim} 维。"
            f"这通常意味着索引与在线检索用的 CLIP 模型不是同一套特征空间"
            f"（项目曾因 768/512 混库出过事故，见 scripts/unify_clip_949.py）"
        )
        ok = False
    if expect_ntotal is not None and index.ntotal != expect_ntotal:
        logger.error(f"条数不匹配: 索引为 {index.ntotal} 条，期望 {expect_ntotal} 条")
        ok = False

    if ok:
        logger.info("索引校验通过")
        return 0
    return 1


def main() -> None:
    """主函数"""
    args = parse_args()

    if args.verify:
        sys.exit(verify_index(args.verify, args.expect_dim, args.expect_ntotal))

    if not args.vectors or not args.out:
        logger.error("构建模式需要同时提供 --vectors 与 --out（或改用 --verify）")
        sys.exit(2)

    sys.exit(build_index(args.vectors, args.out))


if __name__ == "__main__":
    main()
