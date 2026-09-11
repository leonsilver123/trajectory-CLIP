@echo off
chcp 65001 >nul
echo ============================================================
echo   交通风险感知子系统 - Docker 日志查看
echo ============================================================
echo.

cd /d "%~dp0\.."

if "%1"=="" (
    echo 查看所有服务日志:
    echo.
    docker-compose logs -f --tail=100
) else (
    echo 查看 %1 服务日志:
    echo.
    docker-compose logs -f --tail=100 %1
)
