"""
scripts/demo_inference.py - 推理演示脚本

演示完整的检索和回溯流程:
1. 文本查询 → 候选图片
2. 用户确认 → 轨迹回溯
3. 输出观测链和候选路径

使用方式:
    python scripts/demo_inference.py --query "蓝色背包的男人"
    python scripts/demo_inference.py --query "苏E12345"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("scripts.demo_inference")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="推理演示脚本")
    parser.add_argument("--query", type=str, required=True, help="查询文本")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回候选数量")
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    config = get_config(args.config)

    logger.info(f"演示推理: query='{args.query}', top_k={args.top_k}")

    # TODO: 实现演示推理流程
    # 1. 解析查询
    # 2. 属性过滤
    # 3. 向量召回
    # 4. 重排
    # 5. 展示候选结果
    # 6. 模拟用户确认(选择第一个)
    # 7. 轨迹回溯
    # 8. 输出结果

    logger.info("演示推理完成")


if __name__ == "__main__":
    main()
