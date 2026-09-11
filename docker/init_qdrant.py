"""
docker/init_qdrant.py - Qdrant 向量数据库初始化脚本

创建向量集合、配置索引参数。
在 Docker 容器启动后运行一次即可完成初始化。

用法:
    python docker/init_qdrant.py
    或
    docker exec traffic-backtrack-backend python docker/init_qdrant.py
"""

from __future__ import annotations

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("init_qdrant")


def wait_for_qdrant(host: str, port: int, max_retries: int = 30, delay: int = 3):
    """等待 Qdrant 服务就绪"""
    import requests

    for i in range(max_retries):
        try:
            resp = requests.get(f"http://{host}:{port}/healthz", timeout=5)
            if resp.status_code == 200:
                logger.info("Qdrant 服务已就绪")
                return True
        except Exception:
            pass
        logger.info(f"等待 Qdrant 启动... ({i + 1}/{max_retries})")
        time.sleep(delay)

    logger.error("Qdrant 服务启动超时")
    return False


def init_collections(host: str, port: int):
    """创建向量集合"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        OptimizersConfigDiff,
    )

    client = QdrantClient(host=host, port=port)

    # ----------------------------------------------------------
    # 集合 1: ReID 特征向量 (512 维)
    # ----------------------------------------------------------
    reid_collection = "traffic_reid"
    try:
        client.get_collection(reid_collection)
        logger.info(f"集合 '{reid_collection}' 已存在，跳过创建")
    except Exception:
        logger.info(f"创建集合 '{reid_collection}' (dim=512, Cosine)")
        client.create_collection(
            collection_name=reid_collection,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )
        logger.info(f"集合 '{reid_collection}' 创建成功")

    # ----------------------------------------------------------
    # 集合 2: CLIP 特征向量 (768 维)
    # ----------------------------------------------------------
    clip_collection = "traffic_clip"
    try:
        client.get_collection(clip_collection)
        logger.info(f"集合 '{clip_collection}' 已存在，跳过创建")
    except Exception:
        logger.info(f"创建集合 '{clip_collection}' (dim=768, Cosine)")
        client.create_collection(
            collection_name=clip_collection,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )
        logger.info(f"集合 '{clip_collection}' 创建成功")

    # ----------------------------------------------------------
    # 集合 3: 综合目标向量 (用于检索，512 维 ReID)
    # ----------------------------------------------------------
    targets_collection = "traffic_targets"
    try:
        client.get_collection(targets_collection)
        logger.info(f"集合 '{targets_collection}' 已存在，跳过创建")
    except Exception:
        logger.info(f"创建集合 '{targets_collection}' (dim=512, Cosine)")
        client.create_collection(
            collection_name=targets_collection,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )
        logger.info(f"集合 '{targets_collection}' 创建成功")

    # ----------------------------------------------------------
    # 打印集合信息
    # ----------------------------------------------------------
    collections = client.get_collections()
    logger.info("当前 Qdrant 集合列表:")
    for c in collections.collections:
        info = client.get_collection(c.name)
        logger.info(
            f"  - {c.name}: "
            f"points={info.points_count}, "
            f"vectors={info.vectors_count}, "
            f"status={info.status}"
        )


def main():
    """主入口"""
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    logger.info(f"连接 Qdrant: {host}:{port}")

    if not wait_for_qdrant(host, port):
        sys.exit(1)

    init_collections(host, port)

    logger.info("Qdrant 初始化完成!")


if __name__ == "__main__":
    main()
