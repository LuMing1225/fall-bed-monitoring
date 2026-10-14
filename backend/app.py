"""
FastAPI 后端服务
基于多模态分割与姿态估计的跌倒/在床行为精准判别系统

端口: 8564
功能:
- REST API: 视频上传、单帧检测、系统状态、模型管理
- WebSocket: 实时视频流检测结果推送
- 静态文件: 前端页面服务
"""
import cv2
import numpy as np
import logging
import time
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BACKEND_HOST, BACKEND_PORT, FRONTEND_PORT, LOG_DIR, BehaviorState
from pipeline.video_processor import VideoProcessor

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backend.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ============ FastAPI 应用 ============
app = FastAPI(
    title="跌倒/在床行为精准判别系统",
    description="基于多模态分割与姿态估计的跌倒/在床行为精准判别系统 API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局视频处理器
processor = VideoProcessor()

# 上传目录
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ============ 启动事件 ============
@app.on_event("startup")
async def startup_event():
    logger.info("后端服务启动中...")
    # 预加载模型（后台线程）
    import threading
    def load_models_async():
        try:
            processor.load_models()
            logger.info("模型预加载完成")
        except Exception as e:
            logger.error(f"模型预加载失败: {e}")
    threading.Thread(target=load_models_async, daemon=True).start()


# ============ REST API ============

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "models_loaded": processor._models_loaded,
        "timestamp": time.time(),
    }


@app.get("/api/status")
async def system_status():
    """系统状态"""
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}GB"
    except:
        pass

    return {
        "models_loaded": processor._models_loaded,
        "is_running": processor.is_running,
        "current_source": processor.current_source,
        "gpu": gpu_info,
        "bed_segmentation": {
            "model": "MobileSAM",
            "device": "cuda" if processor.bed_segmenter.device == "cuda" else "cpu",
            "bed_mask_cached": processor.bed_segmenter.bed_mask is not None,
        },
        "pose_detection": {
            "model": "YOLOv8n-Pose",
            "device": "cuda" if processor.pose_detector.device == "cuda" else "cpu",
            "tracked_persons": len(processor.pose_detector.track_history),
        },
        "llm_agent": {
            "enabled": processor.pose_corrector.enabled,
            "ollama_available": processor.pose_corrector._ollama_available,
            "model": processor.pose_corrector.model,
        },
    }


@app.post("/api/detect/frame")
def detect_frame(file: UploadFile = File(...), return_image: bool = False):
    """
    单帧图像检测
    上传一张图片，返回检测结果。
    return_image=True 时附带标注图（base64，较大）。
    """
    if not processor._models_loaded:
        processor.load_models()

    # 读取并解码图片
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    # 处理
    annotated, result = processor.process_frame(frame, 0)

    response = {"result": result.to_dict()}

    # 可选返回标注图（低质量压缩以减小体积）
    if return_image:
        import base64
        _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"

    return response


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...), max_frames: Optional[int] = 100):
    """
    视频文件检测
    上传视频，逐帧处理后返回汇总结果
    """
    if not processor._models_loaded:
        processor.load_models()

    # 保存上传文件
    suffix = Path(file.filename).suffix or ".mp4"
    temp_path = UPLOAD_DIR / f"temp_{int(time.time())}{suffix}"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        # 处理视频
        all_results = []
        fall_events = []
        frame_count = 0

        for annotated, result in processor.process_video_file(
            str(temp_path), max_frames=max_frames
        ):
            all_results.append(result.to_dict())
            frame_count += 1
            if result.fall_detected:
                fall_events.append({
                    "frame_idx": result.frame_idx,
                    "timestamp": result.timestamp,
                    "persons": [p["classification"]["label"] for p in result.to_dict()["persons"]],
                })

        # 统计
        state_counts = {"on_bed": 0, "fallen": 0, "active": 0, "unknown": 0}
        for r in all_results:
            for p in r["persons"]:
                state = p["classification"]["state"]
                state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_frames": frame_count,
            "fall_detected": len(fall_events) > 0,
            "fall_events": fall_events[:20],  # 最多返回20个跌倒事件
            "state_statistics": state_counts,
            "frames": all_results[::max(1, len(all_results) // 50)],  # 降采样返回
        }
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/bed/reset")
async def reset_bed_mask():
    """重置床体掩膜（下一帧重新分割）"""
    processor.bed_segmenter.bed_mask = None
    processor.bed_segmenter._last_mask_frame = 0
    return {"status": "ok", "message": "床体掩膜已重置，下一帧将重新分割"}


@app.post("/api/bed/set")
async def set_bed_mask(file: UploadFile = File(...)):
    """
    手动设置床体掩膜
    上传一张床体区域的二值掩码图片
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=400, detail="无法解析掩膜图片")
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    processor.bed_segmenter._set_bed_mask(mask)
    return {"status": "ok", "message": "床体掩膜已手动设置"}


@app.get("/api/report/{track_id}")
async def get_report(track_id: int):
    """获取指定跟踪ID的行为分析报告"""
    report = processor.pose_corrector.generate_report(track_id)
    return {"track_id": track_id, "report": report}


# ============ WebSocket: 实时视频流检测 ============

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket 实时检测
    客户端发送视频帧（JPEG/base64），服务端返回检测结果
    """
    await websocket.accept()
    logger.info("WebSocket连接建立")

    if not processor._models_loaded:
        processor.load_models()

    frame_idx = 0
    try:
        while True:
            data = await websocket.receive_text()

            # 解析消息
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "无效的JSON格式"})
                continue

            msg_type = msg.get("type", "frame")

            if msg_type == "frame":
                # 接收图像帧
                img_data = msg.get("image", "")
                if img_data.startswith("data:image"):
                    img_data = img_data.split(",")[1]

                import base64
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "无法解码图像"})
                    continue

                # 处理帧
                annotated, result = processor.process_frame(frame, frame_idx)
                frame_idx += 1

                # 返回结果（包含标注图）
                _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')

                response = {
                    "type": "result",
                    "frame_idx": result.frame_idx,
                    "result": result.to_dict(),
                    "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
                }
                await websocket.send_json(response)

            elif msg_type == "reset":
                processor.reset()
                frame_idx = 0
                await websocket.send_json({"type": "status", "message": "系统已重置"})

            elif msg_type == "reset_bed":
                processor.bed_segmenter.bed_mask = None
                await websocket.send_json({"type": "status", "message": "床体掩膜已重置"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# ============ 前端静态文件服务 ============
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """服务前端首页"""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "前端页面未找到"}, status_code=404)


# ============ 主入口 ============
if __name__ == "__main__":
    logger.info(f"启动后端服务: {BACKEND_HOST}:{BACKEND_PORT}")
    uvicorn.run(
        "app:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=False,
        log_level="info",
    )
