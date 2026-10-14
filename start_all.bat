@echo off
chcp 65001 >nul
title 跌倒/在床行为判别系统 - 一键启动

echo ========================================
echo   跌倒/在床行为精准判别系统
echo   多Agent协同与CV智能感知
echo ========================================
echo.
echo 正在启动后端服务 (端口8564)...
start "后端服务-8564" cmd /k "%~dp0start_backend.bat"

echo 等待后端初始化...
timeout /t 5 /nobreak >nul

echo 正在启动前端服务 (端口8524)...
start "前端服务-8524" cmd /k "%~dp0start_frontend.bat"

echo.
echo ========================================
echo   系统启动完成!
echo   前端地址: http://localhost:8524
echo   后端API:  http://localhost:8564/docs
echo ========================================
echo.

REM 等待3秒后自动打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost:8524

echo.
echo 按任意键关闭此窗口（服务将继续运行）...
pause >nul
