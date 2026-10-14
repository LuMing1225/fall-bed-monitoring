@echo off
chcp 65001 >nul
title 跌倒/在床行为判别系统 - 后端服务 (端口8564)

echo ========================================
echo   跌倒/在床行为精准判别系统 - 后端
echo   端口: 8564
echo   Python环境: yolo_313
echo ========================================
echo.

cd /d "%~dp0backend"

REM 使用 yolo_313 conda 环境
call C:\Users\luming\anaconda3\Scripts\activate.bat yolo_313

echo 启动 FastAPI 后端服务...
echo.
python app.py

pause
