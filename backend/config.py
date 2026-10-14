"""
系统全局配置
基于多模态分割与姿态估计的跌倒/在床行为精准判别系统
"""
import os
from pathlib import Path

# ============ 路径配置 ============
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============ 模型路径 ============
MOBILE_SAM_PATH = str(MODEL_DIR / "mobile_sam.pt")
SAM_B_PATH = str(MODEL_DIR / "sam_b.pt")
YOLOV8_POSE_PATH = str(MODEL_DIR / "yolov8n-pose.pt")
YOLO11_POSE_PATH = str(MODEL_DIR / "yolo11n-pose.pt")

# ============ 后端服务配置 ============
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8564

# ============ 前端配置 ============
FRONTEND_PORT = 8524

# ============ 床体分割配置 ============
SEGMENTATION = {
    "model_type": "sam_b",              # sam_b (vit_b, 稳定可用) / mobile_sam (需额外安装mobile-sam包)
    "device": "cuda",                   # cuda / cpu
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "min_mask_area": 500,               # 最小掩膜面积（像素）
    "bed_class_hint": "bed",            # 床体类别提示词
    "static_mask_refresh_interval": 300, # 静态掩膜刷新间隔（帧）
}

# ============ 姿态检测配置 ============
POSE_DETECTION = {
    "model": "yolov8n-pose",           # yolov8n-pose / yolo11n-pose
    "device": "cuda",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4,
    "keypoint_conf_threshold": 0.3,    # 关键点置信度阈值
    "max_num_people": 5,
}

# ============ COCO 17关键点定义 ============
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点索引映射
KP_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# 重点关注关键点（脊椎、臀部、膝盖、脚踝）
KEY_KEYPOINTS = [
    KP_INDEX["left_shoulder"], KP_INDEX["right_shoulder"],  # 肩部（脊椎上端）
    KP_INDEX["left_hip"], KP_INDEX["right_hip"],              # 臀部
    KP_INDEX["left_knee"], KP_INDEX["right_knee"],            # 膝盖
    KP_INDEX["left_ankle"], KP_INDEX["right_ankle"],          # 脚踝
]

# 骨骼连接对（用于绘制）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿部
]

# ============ 几何判别配置 ============
CLASSIFICATION = {
    "iou_threshold_on_bed": 0.5,       # 在床IoU阈值 τ
    "iou_threshold_fallen": 0.15,       # 跌倒IoU上限
    "body_axis_angle_threshold": 30,    # 人体轴线与地面平行角度阈值（度）
    "hip_height_ratio_threshold": 0.3,  # 臀部高度占画面比例阈值（区分床上/地面）
    "keypoint_in_bed_ratio": 0.6,       # 关键点落入床体Mask比例阈值
    "temporal_smoothing_frames": 5,     # 时序平滑帧数
    "fall_confidence_frames": 3,        # 连续N帧判定为跌倒才报警
}

# ============ LLM Agent 配置 ============
LLM_AGENT = {
    "enabled": True,
    "provider": "ollama",
    "model": "qwen2.5:7b",              # Ollama本地模型（模拟Llama V2 Pose）
    "base_url": "http://localhost:11434",
    "temporal_window": 10,               # 时序分析窗口（帧）
    "drift_correction_enabled": True,    # 关键点漂移纠错
    "max_retries": 2,
    "timeout": 10,                       # LLM调用超时（秒），演示时避免卡顿
}

# ============ 视频处理配置 ============
VIDEO_PROCESSING = {
    "target_fps": 15,                    # 处理帧率
    "max_frame_size": (1280, 720),      # 最大帧尺寸
    "jpeg_quality": 85,                  # WebSocket推送JPEG质量
    "buffer_size": 30,                   # 帧缓冲区大小
}

# ============ 状态枚举 ============
class BehaviorState:
    ACTIVE = "active"                     # 活动/站立
    ON_BED = "on_bed"                     # 在床
    FALLEN = "fallen"                     # 跌倒
    UNKNOWN = "unknown"                   # 未知

STATE_LABELS = {
    BehaviorState.ACTIVE: "活动中",
    BehaviorState.ON_BED: "在床休息",
    BehaviorState.FALLEN: "跌倒报警",
    BehaviorState.UNKNOWN: "待检测",
}

STATE_COLORS = {
    BehaviorState.ACTIVE: (0, 255, 0),      # 绿色
    BehaviorState.ON_BED: (255, 200, 0),    # 橙色
    BehaviorState.FALLEN: (0, 0, 255),      # 红色
    BehaviorState.UNKNOWN: (128, 128, 128), # 灰色
}
