@echo off
chcp 65001 >nul
echo ============================================================
echo   交通风险感知子系统 - Docker 一键启动
echo ============================================================
echo.

cd /d "%~dp0\.."

echo [1/3] 检查 Docker 环境...
docker info >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)
echo       Docker 环境正常

echo [2/3] 构建并启动服务...
docker-compose up -d --build
if errorlevel 1 (
    echo [错误] 服务启动失败，请检查日志
    docker-compose logs --tail=50
    pause
    exit /b 1
)

echo.
echo [3/3] 等待服务就绪...
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo   服务已启动:
echo.
echo     前端首页:    http://localhost:8501
echo     后端 API:    http://localhost:8000
echo     API 文档:    http://localhost:8000/docs
echo     Qdrant:      http://localhost:6333/dashboard
echo ============================================================
echo.
echo 提示: 首次启动后请运行 Qdrant 初始化脚本:
echo   docker exec traffic-backtrack-backend python docker/init_qdrant.py
echo.
echo 查看日志: scripts\docker_logs.bat
echo 停止服务: scripts\docker_stop.bat
echo.
pause
