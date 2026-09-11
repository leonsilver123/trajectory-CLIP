@echo off
cd /d "%~dp0.."
echo ========================================
echo  交通风险感知子系统 - 前端服务
echo ========================================
echo.
echo  [移动端访问] 请确保手机与本电脑在同一网络
echo.
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
