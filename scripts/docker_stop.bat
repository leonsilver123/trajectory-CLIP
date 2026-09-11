@echo off
chcp 65001 >nul
echo ============================================================
echo   交通风险感知子系统 - Docker 一键停止
echo ============================================================
echo.

cd /d "%~dp0\.."

echo 正在停止所有服务...
docker-compose down

echo.
echo 所有服务已停止。
echo.
echo 提示: 数据文件保留在本项目目录下，不会被删除。
echo 如需清除所有数据卷，请运行:
echo   docker-compose down -v
echo.
pause
