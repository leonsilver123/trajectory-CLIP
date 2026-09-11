@echo off
REM ============================================================
REM GPU 环境诊断工具 (Windows 一键运行)
REM 双击此文件即可运行 GPU 环境检测
REM ============================================================

echo.
echo  交通风险感知子系统 - GPU 环境诊断
echo.

REM 切换到项目根目录
cd /d "%~dp0\.."

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] 未找到 Python，请先安装 Python 3.10+
    echo     下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 运行诊断脚本
python scripts/check_gpu_env.py --verbose

pause
