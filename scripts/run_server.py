"""
scripts/run_server.py - 启动 API 服务

使用方式:
    python scripts/run_server.py
    python scripts/run_server.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动 API 服务")
    parser.add_argument("--host", type=str, default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--reload", action="store_true", help="开发模式(热重载)")
    return parser.parse_args()


def main() -> None:
    """主函数"""
    import uvicorn

    args = parse_args()

    # 加载配置
    from src.common.config import get_config
    config = get_config(args.config)

    host = args.host or config.get("api.host", "0.0.0.0")
    port = args.port or config.get("api.port", 8000)
    workers = args.workers or config.get("api.workers", 2)

    print(f"启动 API 服务: {host}:{port}, workers={workers}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers if not args.reload else 1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
