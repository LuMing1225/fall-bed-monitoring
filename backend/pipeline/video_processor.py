"""
视频处理流水线
整合床体分割、姿态检测、几何判别、LLM纠错四大模块
支持视频文件处理和实时摄像头流
通过 WebSocket 向前端推送实时检测结果
"""
import cv2
import numpy as np
import logging
import time
import base64
import json
from typing import Optional, List, Dict, Tuple, Generator
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VIDEO_PROCESSING, BehaviorState, STATE_LABELS
from segments.bed_segment import BedSegmenter
from poses.yolo_pose import PoseDetector
from classifier.rule_engine import BehaviorClassifier
from llm_agent.pose_corrector import PoseCorrector

logger = logging.getLogger(__name__)


class FrameResult:
    """单帧处理结果"""

    def __init__(self, frame_idx: int, timestamp: float):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.poses = []
        self.results = []
        self.bed_mask = None
        self.floor_edge_y = None
        self.processing_time = 0.0
        self.fps = 0.0
        self.fall_detected = False
        self.on_bed_count = 0
        self.active_count = 0

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "processing_time_ms": round(self.processing_time * 1000, 1),
            "fps": round(self.fps, 1),
            "fall_detected": self.fall_detected,
            "on_bed_count": self.on_bed_count,
            "active_count": self.active_count,
            "person_count": len(self.poses),
            "persons": [
                {
                    **pose.to_dict(),
                    "classification": result.to_dict(),
                }
                for pose, result in zip(self.poses, self.results)
            ],
        }


class VideoProcessor:
    """
    视频处理流水线
    整合四大模块，提供统一的帧处理接口
    """

    def __init__(self):
        self.bed_segmenter = BedSegmenter()
        self.pose_detector = PoseDetector()
        self.classifier = BehaviorClassifier()
        self.pose_corrector = PoseCorrector()

        self._models_loaded = False
        self._frame_times = []  # 用于FPS计算
        self.is_running = False
        self.current_source = None

    def load_models(self):
        """加载所有模型（懒加载）"""
        if self._models_loaded:
            return
        logger.info("开始加载所有模型...")
        t0 = time.time()
        self.bed_segmenter.load_model()
        self.pose_detector.load_model()
        self.pose_corrector.check_ollama()
        self._models_loaded = True
        logger.info(f"所有模型加载完成，耗时 {time.time() - t0:.1f}s")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple[np.ndarray, FrameResult]:
        """
        处理单帧

        Returns:
            (annotated_frame, frame_result)
        """
        if not self._models_loaded:
            self.load_models()

        t0 = time.time()
        result = FrameResult(frame_idx, frame_idx / 30.0)

        # 1. 床体分割
        bed_mask = self.bed_segmenter.get_bed_mask(frame)
        result.bed_mask = bed_mask
        result.floor_edge_y = self.bed_segmenter.floor_edge_y

        # 2. 姿态检测
        poses = self.pose_detector.detect(frame)
        result.poses = poses

        # 3. 几何判别
        classifications = self.classifier.classify_all(poses, bed_mask, self.bed_segmenter.floor_edge_y)

        # 4. LLM纠错
        final_results = []
        for pose, cls_result in zip(poses, classifications):
            corrected_state, corrected_conf, reason = self.pose_corrector.correct_classification(
                pose, cls_result.state, cls_result.details
            )
            cls_result.state = corrected_state
            cls_result.confidence = corrected_conf
            cls_result.label = STATE_LABELS.get(corrected_state, corrected_state)
            cls_result.details["llm_correction"] = reason
            final_results.append(cls_result)

        result.results = final_results

        # 统计
        for r in final_results:
            if r.state == BehaviorState.FALLEN:
                result.fall_detected = True
            elif r.state == BehaviorState.ON_BED:
                result.on_bed_count += 1
            elif r.state == BehaviorState.ACTIVE:
                result.active_count += 1

        # 5. 可视化
        annotated = frame.copy()
        annotated = self.bed_segmenter.visualize(annotated)
        annotated = self.pose_detector.visualize(annotated, poses)
        annotated = self.classifier.visualize(annotated, poses, final_results)

        # 处理时间和FPS
        result.processing_time = time.time() - t0
        self._frame_times.append(time.time())
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            result.fps = 1.0 / max(np.mean(np.diff(self._frame_times)), 1e-6)

        return annotated, result

    def process_video_file(self, video_path: str,
                            output_path: Optional[str] = None,
                            max_frames: Optional[int] = None,
                            display: bool = False) -> Generator[Tuple[np.ndarray, FrameResult], None, None]:
        """
        处理视频文件，逐帧yield结果

        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径（可选）
            max_frames: 最大处理帧数
            display: 是否显示窗口
        """
        self.load_models()
        self.is_running = True
        self.current_source = video_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                annotated, result = self.process_frame(frame, frame_idx)
                result.timestamp = frame_idx / fps

                if writer:
                    writer.write(annotated)

                yield annotated, result

                frame_idx += 1

                if display:
                    cv2.imshow("Fall Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False

        logger.info(f"视频处理完成: {frame_idx}帧")

    def process_webcam(self, camera_id: int = 0,
                        display: bool = False) -> Generator[Tuple[np.ndarray, FrameResult], None, None]:
        """处理摄像头实时流"""
        self.load_models()
        self.is_running = True
        self.current_source = f"webcam_{camera_id}"

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")

        frame_idx = 0
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, result = self.process_frame(frame, frame_idx)
                yield annotated, result
                frame_idx += 1

                if display:
                    cv2.imshow("Fall Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False

    def stop(self):
        """停止处理"""
        self.is_running = False

    def reset(self):
        """重置所有模块状态"""
        self.bed_segmenter.reset()
        self.pose_detector.reset()
        self.classifier.reset()
        self.pose_corrector.reset()
        self._frame_times.clear()

    @staticmethod
    def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
        """将帧编码为JPEG字节"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return encoded.tobytes()

    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
        """将帧编码为base64字符串"""
        jpeg_bytes = VideoProcessor.encode_frame_to_jpeg(frame, quality)
        return base64.b64encode(jpeg_bytes).decode('utf-8')
