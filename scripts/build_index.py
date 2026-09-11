"""
scripts/build_index.py - 构建向量索引脚本

将目标实例的特征向量导入 Qdrant 向量数据库:
- CLIP 图像向量 (用于图文检索)
- ReID 向量 (用于跨镜匹配)

使用方式:
    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("scripts.build_index")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--rebuild", action="store_true", help="是否重建索引")
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    config = get_config(args.config)

    logger.info("开始构建向量索引...")

    # TODO: 实现向量索引构建
    # 1. 从 TrackManager 加载所有目标实例
    # 2. 初始化 Qdrant 客户端
    # 3. 创建/重建集合
    # 4. 批量导入 CLIP 向量和 ReID 向量
    # 5. 验证索引完整性

    logger.info("向量索引构建完成")


if __name__ == "__main__":
    main()
