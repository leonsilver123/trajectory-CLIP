"""
api.main - FastAPI 主入口

交通风险感知子系统 REST API 服务。
提供检索、确认、回溯和仪表盘接口。

启动方式:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    或
    python scripts/run_server.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("交通风险感知子系统 API 服务启动中...")
    config = get_config()
    logger.info(f"系统版本: {config.get('system.version')}")
    yield
    # 关闭时清理
    logger.info("API 服务正在关闭...")


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例

    Returns:
        FastAPI 应用
    """
    config = get_config()

    app = FastAPI(
        title="交通风险感知子系统 API",
        description="文本驱动的交通目标检索与跨镜时空回溯",
        version=config.get("system.version", "1.0.0"),
        lifespan=lifespan,
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from api.routes.search import router as search_router
    from api.routes.confirm import router as confirm_router
    from api.routes.backtrack import router as backtrack_router
    from api.routes.dashboard import router as dashboard_router

    app.include_router(search_router, prefix="/api/v1/search", tags=["检索"])
    app.include_router(confirm_router, prefix="/api/v1/confirm", tags=["确认"])
    app.include_router(backtrack_router, prefix="/api/v1/backtrack", tags=["回溯"])
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["仪表盘"])

    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        return {"status": "ok", "version": config.get("system.version", "1.0.0")}

    # 静态文件服务：将 output/ 目录映射到 /static 路径
    # 前端可通过 /static/crops/xxx.jpg 访问真实检测图片
    output_dir = Path(__file__).resolve().parent.parent / "output"
    if output_dir.exists():
        app.mount("/static", StaticFiles(directory=str(output_dir)), name="static")
        logger.info(f"静态文件服务已挂载: /static -> {output_dir}")

    return app


# 全局应用实例(供 uvicorn 使用)
app = create_app()
