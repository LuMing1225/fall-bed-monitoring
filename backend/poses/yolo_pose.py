"""
YOLOv8-Pose 人体姿态检测模块
提取人体17个COCO关键点，重点关注脊椎、臀部、膝盖、脚踝
支持时序漂移纠错与多目标跟踪
"""
import cv2
import numpy as np
import torch
import logging
from typing import Optional, List, Dict, Tuple
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    YOLOV8_POSE_PATH, YOLO11_POSE_PATH, POSE_DETECTION,
    COCO_KEYPOINTS, KP_INDEX, KEY_KEYPOINTS, SKELETON_CONNECTIONS,
    CLASSIFICATION
)

logger = logging.getLogger(__name__)


class PersonPose:
    """单个人体姿态数据结构"""

    def __init__(self, keypoints: np.ndarray, bbox: np.ndarray,
                 confidence: float, track_id: int = -1):
        """
        Args:
            keypoints: [17, 3] array (x, y, confidence)
            bbox: [x1, y1, x2, y2]
            confidence: detection confidence
            track_id: tracking ID
        """
        self.keypoints = keypoints  # [17, 3]
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.track_id = track_id
        self.frame_idx = 0

    @property
    def center(self) -> Tuple[float, float]:
        """人体中心（肩部中点与髋部中点的平均）"""
        left_shoulder = self.keypoints[KP_INDEX["left_shoulder"]]
        right_shoulder = self.keypoints[KP_INDEX["right_shoulder"]]
        left_hip = self.keypoints[KP_INDEX["left_hip"]]
        right_hip = self.keypoints[KP_INDEX["right_hip"]]

        shoulder_center = np.mean([left_shoulder[:2], right_shoulder[:2]], axis=0)
        hip_center = np.mean([left_hip[:2], right_hip[:2]], axis=0)
        center = np.mean([shoulder_center, hip_center], axis=0)
        return float(center[0]), float(center[1])

    @property
    def hip_center(self) -> Tuple[float, float]:
        """臀部中心"""
        left_hip = self.keypoints[KP_INDEX["left_hip"]]
        right_hip = self.keypoints[KP_INDEX["right_hip"]]
        hip = np.mean([left_hip[:2], right_hip[:2]], axis=0)
        return float(hip[0]), float(hip[1])

    @property
    def body_axis_angle(self) -> float:
        """
        人体轴线与水平方向的夹角（度）
        轴线 = 肩部中心到髋部中心的连线
        0度 = 完全水平（躺卧），90度 = 完全垂直（站立）
        """
        left_shoulder = self.keypoints[KP_INDEX["left_shoulder"]]
        right_shoulder = self.keypoints[KP_INDEX["right_shoulder"]]
        left_hip = self.keypoints[KP_INDEX["left_hip"]]
        right_hip = self.keypoints[KP_INDEX["right_hip"]]

        # 检查关键点置信度
        conf_threshold = 0.3
        valid = all(kp[2] > conf_threshold for kp in
                    [left_shoulder, right_shoulder, left_hip, right_hip])
        if not valid:
            return -1  # 无法计算

        shoulder_center = np.mean([left_shoulder[:2], right_shoulder[:2]], axis=0)
        hip_center = np.mean([left_hip[:2], right_hip[:2]], axis=0)

        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return -1

        # 与水平方向的夹角
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        return float(angle)

    @property
    def is_horizontal(self) -> bool:
        """人体是否接近水平（躺卧姿态）"""
        angle = self.body_axis_angle
        if angle < 0:
            return False
        return angle < CLASSIFICATION["body_axis_angle_threshold"]

    def get_valid_keypoints(self, conf_threshold: float = 0.3) -> np.ndarray:
        """获取有效关键点（置信度高于阈值）"""
        mask = self.keypoints[:, 2] > conf_threshold
        return self.keypoints[mask]

    def get_key_keypoints(self) -> np.ndarray:
        """获取重点关键点（肩/髋/膝/踝）"""
        return self.keypoints[KEY_KEYPOINTS]

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "track_id": self.track_id,
            "confidence": float(self.confidence),
            "bbox": self.bbox.tolist(),
            "keypoints": self.keypoints.tolist(),
            "center": self.center,
            "hip_center": self.hip_center,
            "body_axis_angle": self.body_axis_angle,
            "is_horizontal": self.is_horizontal,
        }


class PoseDetector:
    """
    YOLOv8-Pose 姿态检测器
    - 支持 yolov8n-pose / yolo11n-pose
    - 多目标检测与跟踪
    - 时序关键点平滑（卡尔曼滤波/滑动平均）
    - 关键点漂移纠错
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**POSE_DETECTION, **(config or {})}
        self.device = self.config["device"] if torch.cuda.is_available() else "cpu"
        self.model = None
        self._model_loaded = False

        # 时序跟踪
        self.track_history: Dict[int, deque] = {}
        self.smoothing_window = CLASSIFICATION["temporal_smoothing_frames"]
        self.next_track_id = 0
        self.frame_count = 0

    def load_model(self):
        """加载 YOLO-Pose 模型"""
        if self._model_loaded:
            return

        from ultralytics import YOLO

        model_name = self.config["model"]
        model_path = YOLOV8_POSE_PATH if "yolov8" in model_name else YOLO11_POSE_PATH

        logger.info(f"加载姿态模型: {model_name} ({model_path}) on {self.device}")
        try:
            self.model = YOLO(model_path)
            self._model_loaded = True
            logger.info("姿态模型加载成功")
        except Exception as e:
            logger.error(f"姿态模型加载失败: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[PersonPose]:
        """
        检测帧中所有人的姿态
        Returns:
            List[PersonPose]: 检测到的人体姿态列表
        """
        if self.model is None:
            self.load_model()

        self.frame_count += 1
        results = self.model(
            frame,
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
            device=self.device,
            verbose=False,
        )

        poses = []
        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints_data = result.keypoints.data.cpu().numpy()  # [N, 17, 3]
                boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                confidences = result.boxes.conf.cpu().numpy()  # [N]

                for i in range(len(keypoints_data)):
                    kps = keypoints_data[i]  # [17, 3]
                    bbox = boxes[i]
                    conf = float(confidences[i])

                    # 简单跟踪：基于IoU匹配
                    track_id = self._match_track(bbox, kps)

                    pose = PersonPose(kps, bbox, conf, track_id)
                    pose.frame_idx = self.frame_count

                    # 时序平滑
                    pose = self._temporal_smooth(pose)

                    poses.append(pose)

        # 清理过期跟踪
        self._cleanup_tracks()

        return poses

    def _match_track(self, bbox: np.ndarray, keypoints: np.ndarray) -> int:
        """基于IoU和中心距离匹配跟踪ID"""
        best_id = -1
        best_iou = 0.3  # 最小IoU阈值

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        for track_id, history in self.track_history.items():
            if len(history) == 0:
                continue
            last_pose = history[-1]
            last_bbox = last_pose.bbox

            # 计算IoU
            x1 = max(bbox[0], last_bbox[0])
            y1 = max(bbox[1], last_bbox[1])
            x2 = min(bbox[2], last_bbox[2])
            y2 = min(bbox[3], last_bbox[3])

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area2 = (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1])
                iou = intersection / (area1 + area2 - intersection + 1e-6)
            else:
                iou = 0

            # 中心距离惩罚
            last_center = np.array(last_pose.center)
            dist = np.linalg.norm(center - last_center)
            bbox_diag = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
            dist_score = max(0, 1 - dist / (bbox_diag * 1.5))

            combined = iou * 0.7 + dist_score * 0.3
            if combined > best_iou:
                best_iou = combined
                best_id = track_id

        if best_id == -1:
            best_id = self.next_track_id
            self.next_track_id += 1
            self.track_history[best_id] = deque(maxlen=self.smoothing_window * 3)

        return best_id

    def _temporal_smooth(self, pose: PersonPose) -> PersonPose:
        """时序关键点平滑（滑动平均）"""
        track_id = pose.track_id
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.smoothing_window * 3)

        history = self.track_history[track_id]
        history.append(pose)

        if len(history) >= 2:
            # 对关键点进行加权滑动平均（越近权重越大）
            n = min(len(history), self.smoothing_window)
            weights = np.linspace(0.5, 1.0, n)
            weights = weights / weights.sum()

            smoothed_kps = np.zeros_like(pose.keypoints)
            for i in range(n):
                hist_pose = history[-(i + 1)]
                smoothed_kps[:, :2] += hist_pose.keypoints[:, :2] * weights[-(i + 1)]
                smoothed_kps[:, 2] = np.maximum(smoothed_kps[:, 2], hist_pose.keypoints[:, 2])

            pose.keypoints = smoothed_kps

        return pose

    def _cleanup_tracks(self):
        """清理超过30帧未更新的跟踪"""
        expired = []
        for track_id, history in self.track_history.items():
            if len(history) > 0:
                last_frame = history[-1].frame_idx
                if self.frame_count - last_frame > 30:
                    expired.append(track_id)
        for tid in expired:
            del self.track_history[tid]

    def correct_keypoint_drift(self, pose: PersonPose) -> PersonPose:
        """
        关键点漂移纠错
        检测异常跳变的关键点，用历史值替代
        """
        track_id = pose.track_id
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return pose

        history = list(self.track_history[track_id])
        prev_pose = history[-2] if len(history) >= 2 else None
        if prev_pose is None:
            return pose

        corrected = pose.keypoints.copy()
        bbox_diag = np.sqrt(
            (pose.bbox[2] - pose.bbox[0]) ** 2 + (pose.bbox[3] - pose.bbox[1]) ** 2
        )
        max_jump = bbox_diag * 0.3  # 单帧最大位移不超过bbox对角线的30%

        for kp_idx in range(17):
            curr = pose.keypoints[kp_idx, :2]
            prev = prev_pose.keypoints[kp_idx, :2]
            curr_conf = pose.keypoints[kp_idx, 2]
            prev_conf = prev_pose.keypoints[kp_idx, 2]

            if curr_conf > 0.3 and prev_conf > 0.3:
                jump = np.linalg.norm(curr - prev)
                if jump > max_jump:
                    # 异常跳变，用历史值替代
                    corrected[kp_idx, :2] = prev
                    corrected[kp_idx, 2] = prev_conf * 0.8

        pose.keypoints = corrected
        return pose

    def visualize(self, frame: np.ndarray, poses: List[PersonPose],
                  draw_skeleton: bool = True, draw_keypoints: bool = True,
                  draw_bbox: bool = True) -> np.ndarray:
        """在帧上可视化姿态"""
        vis = frame.copy()

        for pose in poses:
            color = (0, 255, 255)  # 青色

            # 绘制边界框
            if draw_bbox:
                x1, y1, x2, y2 = map(int, pose.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                label = f"ID:{pose.track_id} {pose.confidence:.2f}"
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 绘制骨骼连接
            if draw_skeleton:
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    start_kp = pose.keypoints[start_idx]
                    end_kp = pose.keypoints[end_idx]
                    if start_kp[2] > 0.3 and end_kp[2] > 0.3:
                        pt1 = (int(start_kp[0]), int(start_kp[1]))
                        pt2 = (int(end_kp[0]), int(end_kp[1]))
                        cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

            # 绘制关键点
            if draw_keypoints:
                for i, kp in enumerate(pose.keypoints):
                    if kp[2] > 0.3:
                        pt = (int(kp[0]), int(kp[1]))
                        # 重点关键点用红色，其他用黄色
                        if i in KEY_KEYPOINTS:
                            cv2.circle(vis, pt, 4, (0, 0, 255), -1)
                        else:
                            cv2.circle(vis, pt, 3, (0, 255, 255), -1)

        return vis

    def reset(self):
        """重置检测器状态"""
        self.track_history.clear()
        self.next_track_id = 0
        self.frame_count = 0
