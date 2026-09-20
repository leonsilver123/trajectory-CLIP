"""
api.main - FastAPI 主入口

交通风险感知子系统 REST API 服务。
提供检索、确认、回溯和仪表盘接口。

启动方式:
    uvicorn api.main:app --host 127.0.0.1 --port 8000
    或
    python scripts/run_server.py

默认只监听 127.0.0.1（本服务无鉴权，避免局域网暴露）。需要跨机访问时
显式传 --host 0.0.0.0，并自行评估风险。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("api.main")


# 前端构建产物目录（webapp/dist，由 `npm run build` 生成）
_WEBAPP_DIST = Path(__file__).resolve().parent.parent / "webapp" / "dist"

# 允许经 /static 对外提供的 output/ 子目录（**只放图片**）。
# 新增目录必须显式登记 —— 默认不可达，避免数据文件被整包下载。
# 依据：datastore 中 68,349 条检测的 crop_path/image_path/keyframe_path
# 分别指向 aicity22_crops / aicity22_frames / cityflow_crops。
_STATIC_IMAGE_DIRS = (
    "aicity22_crops",     # 检测裁剪图（crop_path，68,349 张）
    "aicity22_frames",    # 原始抽帧图（image_path）
    "cityflow_crops",     # 关键帧图（keyframe_path，949 张）
    "crops",              # 早期裁剪图目录
    "demo_detections",    # quick_demo 产物
    "pipeline_test",      # 管线验证产物
)


def _mount_frontend(app: FastAPI) -> None:
    """
    把前端构建产物 webapp/dist 挂到根路径，实现「一条命令同时服务 API 与 UI」。

    兼容性：dist/ 不存在（尚未 npm run build）时直接跳过，API 行为与从前完全一致，
    前端仍可 `npm run dev` 单独起（vite proxy 转发 /api 到本服务）。

    挂载方式：
      1. /assets 用 StaticFiles —— Vite 产物目录，JS/CSS 走标准静态文件服务
      2. 其余路径用兜底路由 —— 命中 dist 下的真实文件就返回该文件，
         否则一律回 index.html，交给前端 BrowserRouter 处理深链接。
         这样直接刷新 /backtrack、/trajectory、/dashboard 不会 404。

    本函数必须在所有 API 路由注册**之后**调用：Starlette 按注册顺序匹配，
    兜底路由放最后才不会截胡 /api/v1/*、/health、/docs。
    """
    index_file = _WEBAPP_DIST / "index.html"
    if not index_file.exists():
        logger.info(
            f"未找到前端构建产物 {_WEBAPP_DIST}，本次仅提供 API；"
            f"如需一并托管 UI，请先在 webapp/ 下执行 npm run build"
        )
        return

    assets_dir = _WEBAPP_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="webapp-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """SPA 兜底：静态文件优先，其余交给前端路由"""
        # /api 与 /static 下的未知路径保持 404 语义，不被前端路由吞掉
        if full_path.startswith(("api/", "static/")):
            raise HTTPException(status_code=404, detail="Not Found")

        if full_path:
            candidate = (_WEBAPP_DIST / full_path).resolve()
            # 防目录穿越：解析后的路径必须仍在 dist 内
            if candidate.is_relative_to(_WEBAPP_DIST) and candidate.is_file():
                return FileResponse(candidate)

        return FileResponse(index_file)

    logger.info(f"前端已挂载: / -> {_WEBAPP_DIST}（SPA 兜底已启用）")


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
    from api.routes.report import router as report_router

    app.include_router(search_router, prefix="/api/v1/search", tags=["检索"])
    app.include_router(confirm_router, prefix="/api/v1/confirm", tags=["确认"])
    app.include_router(backtrack_router, prefix="/api/v1/backtrack", tags=["回溯"])
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["仪表盘"])
    app.include_router(report_router, prefix="/api/v1/report", tags=["研判报告"])

    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        return {"status": "ok", "version": config.get("system.version", "1.0.0")}

    # 静态图片服务：**只挂白名单里的图片目录**，而不是整个 output/。
    #
    # 早先这里把整个 output/ 挂到 /static，于是 output/cityflow_results.json（75MB 全量数据）、
    # output/datastore/*.parquet、meta.sqlite、各类调试 .log 全部可经 HTTP 直接下载。
    # 前端真正需要的只是检测裁剪图，所以改成逐目录白名单 —— 新增目录必须显式登记，
    # 默认不可达。
    #
    # 白名单是**从数据反推**出来的（见 PLAN3-E1）：datastore 里 68,349 条检测的
    #   crop_path -> aicity22_crops/     image_path -> aicity22_frames/
    #   keyframe_path -> cityflow_crops/
    output_dir = Path(__file__).resolve().parent.parent / "output"
    if output_dir.exists():
        mounted = []
        for name in _STATIC_IMAGE_DIRS:
            sub = output_dir / name
            if sub.is_dir():
                app.mount(
                    f"/static/{name}",
                    StaticFiles(directory=str(sub)),
                    name=f"static-{name}",
                )
                mounted.append(name)
        logger.info(f"静态图片服务已挂载: /static/{{{','.join(mounted)}}} -> {output_dir}")

    # 前端托管放在最后：API 路由优先，未命中的路径才落到 SPA 兜底
    _mount_frontend(app)

    return app


# 全局应用实例(供 uvicorn 使用)
app = create_app()
