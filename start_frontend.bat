@echo off
chcp 65001 >nul
title 跌倒/在床行为判别系统 - 前端服务 (端口8524)

echo ========================================
echo   跌倒/在床行为精准判别系统 - 前端
echo   端口: 8524
echo ========================================
echo.

cd /d "%~dp0frontend"

echo 启动前端静态文件服务器...
echo 访问地址: http://localhost:8524
echo.
python server.py

pause
