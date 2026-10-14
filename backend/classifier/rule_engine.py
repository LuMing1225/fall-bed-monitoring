"""
几何判别规则引擎
结合人体骨骼中心与床体Mask的拓扑相交比例(IoU)、质心相对床沿的几何距离
实现"在床"与"跌倒在地"的精确分类

状态决策方程:
    Status = Active/On_Bed,  if Kn ∩ Mask_bed / Kn >= τ and Y_hip < Y_floor_edge
           = Fallen,           otherwise

其中:
    Kn = 躯干关键点集合
    τ = 面积重合阈值 (0.5)
    Y_hip = 臀部关键点y坐标
    Y_floor_edge = 地面边缘y坐标
"""
import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLASSIFICATION, BehaviorState, STATE_LABELS, STATE_COLORS,
    COCO_KEYPOINTS, KP_INDEX, KEY_KEYPOINTS
)
from poses.yolo_pose import PersonPose

logger = logging.getLogger(__name__)


class ClassificationResult:
    """分类结果数据结构"""

    def __init__(self, state: str, confidence: float, details: Dict):
        self.state = state
        self.confidence = confidence
        self.details = details
        self.label = STATE_LABELS.get(state, state)
        self.color = STATE_COLORS.get(state, (128, 128, 128))

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "label": self.label,
            "confidence": float(self.confidence),
            "details": self.details,
        }


class BehaviorClassifier:
    """
    行为分类器
    - IoU计算：人体区域与床体Mask的相交比例
    - 几何距离：质心相对床沿/地面边缘的距离
    - 姿态角度：人体轴线与地面的夹角
    - 时序平滑：连续帧确认机制
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**CLASSIFICATION, **(config or {})}
        # 每个人的状态历史（用于时序确认）
        self.state_history: Dict[int, deque] = {}
        self.fall_confidence_frames = self.config["fall_confidence_frames"]
        self.smoothing_frames = self.config["temporal_smoothing_frames"]

    def classify(self, pose: PersonPose, bed_mask: np.ndarray,
                 floor_edge_y: Optional[int] = None) -> ClassificationResult:
        """
        对单个人体进行行为分类

        Args:
            pose: 人体姿态
            bed_mask: 床体掩膜 [H, W]
            floor_edge_y: 地面边缘y坐标

        Returns:
            ClassificationResult: 分类结果
        """
        h, w = bed_mask.shape[:2]

        if floor_edge_y is None:
            floor_edge_y = h  # 默认画面底部

        # ============ 1. 计算IoU ============
        iou, person_mask = self._compute_iou(pose, bed_mask)

        # ============ 2. 计算关键点落入床体比例 ============
        kp_in_bed_ratio, kp_in_bed_count, kp_total = self._compute_keypoint_in_bed_ratio(
            pose, bed_mask
        )

        # ============ 3. 计算几何距离 ============
        hip_x, hip_y = pose.hip_center
        center_x, center_y = pose.center

        # 臀部相对地面边缘的高度（正值=在地面边缘上方，负值=在地面边缘下方/地面）
        hip_relative_height = floor_edge_y - hip_y
        hip_height_ratio = hip_relative_height / h if h > 0 else 0

        # 质心到床体掩膜的距离
        dist_to_bed = self._compute_distance_to_mask(center_x, center_y, bed_mask)

        # ============ 4. 姿态角度 ============
        body_angle = pose.body_axis_angle
        is_horizontal = pose.is_horizontal

        # ============ 5. 决策逻辑 ============
        state, confidence = self._decision_logic(
            iou=iou,
            kp_in_bed_ratio=kp_in_bed_ratio,
            hip_height_ratio=hip_height_ratio,
            hip_y=hip_y,
            floor_edge_y=floor_edge_y,
            is_horizontal=is_horizontal,
            body_angle=body_angle,
            dist_to_bed=dist_to_bed,
            h=h,
        )

        # ============ 6. 时序平滑与确认 ============
        state, confidence = self._temporal_confirm(pose.track_id, state, confidence)

        details = {
            "iou": float(iou),
            "keypoint_in_bed_ratio": float(kp_in_bed_ratio),
            "keypoint_in_bed_count": int(kp_in_bed_count),
            "keypoint_total": int(kp_total),
            "hip_y": float(hip_y),
            "hip_relative_height": float(hip_relative_height),
            "hip_height_ratio": float(hip_height_ratio),
            "floor_edge_y": int(floor_edge_y),
            "body_axis_angle": float(body_angle) if body_angle >= 0 else -1,
            "is_horizontal": bool(is_horizontal),
            "distance_to_bed": float(dist_to_bed),
            "person_center": [float(center_x), float(center_y)],
            "hip_center": [float(hip_x), float(hip_y)],
        }

        return ClassificationResult(state, confidence, details)

    def _compute_iou(self, pose: PersonPose, bed_mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算人体区域与床体Mask的IoU
        人体区域 = 关键点凸包 + 边界框
        """
        h, w = bed_mask.shape[:2]

        # 构建人体掩膜：使用关键点凸包
        valid_kps = pose.get_valid_keypoints(0.3)
        if len(valid_kps) < 3:
            # 关键点不足，使用边界框
            x1, y1, x2, y2 = map(int, np.clip(pose.bbox, 0, [w - 1, h - 1, w - 1, h - 1]))
            person_mask = np.zeros((h, w), dtype=np.uint8)
            person_mask[y1:y2, x1:x2] = 1
        else:
            # 关键点凸包
            points = valid_kps[:, :2].astype(np.int32)
            hull = cv2.convexHull(points)
            person_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(person_mask, hull, 1)

            # 膨胀一点以覆盖身体宽度
            kernel = np.ones((15, 15), np.uint8)
            person_mask = cv2.dilate(person_mask, kernel, iterations=1)

        # 计算IoU
        intersection = np.logical_and(person_mask > 0, bed_mask > 0).sum()
        union = np.logical_or(person_mask > 0, bed_mask > 0).sum()

        if union == 0:
            return 0.0, person_mask

        iou = intersection / union
        return float(iou), person_mask

    def _compute_keypoint_in_bed_ratio(self, pose: PersonPose,
                                         bed_mask: np.ndarray) -> Tuple[float, int, int]:
        """计算关键点落入床体Mask的比例"""
        h, w = bed_mask.shape[:2]
        valid_kps = pose.get_valid_keypoints(0.3)

        if len(valid_kps) == 0:
            return 0.0, 0, 0

        in_bed_count = 0
        for kp in valid_kps:
            x, y = int(np.clip(kp[0], 0, w - 1)), int(np.clip(kp[1], 0, h - 1))
            if bed_mask[y, x] > 0:
                in_bed_count += 1

        ratio = in_bed_count / len(valid_kps)
        return float(ratio), in_bed_count, len(valid_kps)

    def _compute_distance_to_mask(self, x: float, y: float,
                                    bed_mask: np.ndarray) -> float:
        """计算点到床体掩膜的最近距离（像素）"""
        h, w = bed_mask.shape[:2]
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

        if bed_mask[yi, xi] > 0:
            return 0.0

        # 距离变换
        dist_transform = cv2.distanceTransform(
            (bed_mask == 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        return float(dist_transform[yi, xi])

    def _decision_logic(self, iou: float, kp_in_bed_ratio: float,
                         hip_height_ratio: float, hip_y: float,
                         floor_edge_y: float, is_horizontal: bool,
                         body_angle: float, dist_to_bed: float,
                         h: int) -> Tuple[str, float]:
        """
        核心决策逻辑

        状态决策方程:
            Status = On_Bed,   if IoU >= τ AND keypoint_in_bed_ratio >= 0.6 AND Y_hip < Y_floor_edge
                   = Fallen,   if is_horizontal AND IoU < 0.15 AND keypoint_in_bed_ratio < 0.3
                   = Active,   otherwise
        """
        tau = self.config["iou_threshold_on_bed"]  # τ = 0.5
        kp_ratio_thresh = self.config["keypoint_in_bed_ratio"]  # 0.6
        fallen_iou_thresh = self.config["iou_threshold_fallen"]  # 0.15

        # ===== 在床判定 =====
        # 条件1: IoU >= τ (人体与床体重叠度高)
        # 条件2: 关键点落入床体比例 >= 0.6
        # 条件3: 臀部在地面边缘上方 (Y_hip < Y_floor_edge)
        on_bed_conditions = {
            "iou >= tau": iou >= tau,
            "kp_in_bed_ratio >= 0.6": kp_in_bed_ratio >= kp_ratio_thresh,
            "hip_above_floor": hip_y < floor_edge_y,
        }
        on_bed_score = sum(on_bed_conditions.values()) / len(on_bed_conditions)

        if on_bed_score >= 0.66:  # 至少满足2/3条件
            confidence = 0.5 + 0.5 * min(iou / tau, 1.0)
            return BehaviorState.ON_BED, min(confidence, 0.98)

        # ===== 跌倒判定 =====
        # 条件1: 人体接近水平 (躺卧)
        # 条件2: IoU低 (不在床上)
        # 条件3: 关键点落入床体比例低
        # 条件4: 臀部接近或低于地面边缘
        fallen_conditions = {
            "is_horizontal": is_horizontal,
            "iou < 0.15": iou < fallen_iou_thresh,
            "kp_in_bed_ratio < 0.3": kp_in_bed_ratio < 0.3,
            "hip_near_or_below_floor": hip_y >= floor_edge_y * 0.85,
        }
        fallen_score = sum(fallen_conditions.values()) / len(fallen_conditions)

        if fallen_score >= 0.75:  # 至少满足3/4条件
            confidence = 0.5 + 0.5 * fallen_score
            return BehaviorState.FALLEN, min(confidence, 0.99)

        # ===== 活动/未知 =====
        if not is_horizontal and body_angle > 45:
            # 明显站立/坐姿
            return BehaviorState.ACTIVE, 0.7 + 0.2 * min(body_angle / 90, 1.0)

        # 过渡状态，置信度较低
        return BehaviorState.UNKNOWN, 0.4

    def _temporal_confirm(self, track_id: int, state: str,
                            confidence: float) -> Tuple[str, float]:
        """
        时序确认机制
        - 跌倒需要连续N帧确认才报警
        - 状态切换有迟滞，避免抖动
        """
        if track_id not in self.state_history:
            self.state_history[track_id] = deque(maxlen=self.smoothing_frames * 2)

        history = self.state_history[track_id]
        history.append(state)

        # 跌倒需要连续确认
        if state == BehaviorState.FALLEN:
            recent_fallen = sum(1 for s in list(history)[-self.fall_confidence_frames:]
                                 if s == BehaviorState.FALLEN)
            if recent_fallen < self.fall_confidence_frames:
                # 未达到连续帧数，降级为UNKNOWN或保持前一状态
                if len(history) >= 2 and history[-2] != BehaviorState.FALLEN:
                    return history[-2], confidence * 0.5
                return BehaviorState.UNKNOWN, confidence * 0.3

        # 状态迟滞：如果最近多数帧是同一状态，保持该状态
        if len(history) >= 3:
            recent = list(history)[-3:]
            from collections import Counter
            most_common = Counter(recent).most_common(1)[0]
            if most_common[1] >= 2 and most_common[0] != state:
                # 多数票与当前不同，且当前不是跌倒（跌倒需要严格确认）
                if state != BehaviorState.FALLEN:
                    state = most_common[0]
                    confidence = confidence * 0.8

        return state, confidence

    def classify_all(self, poses: List[PersonPose], bed_mask: np.ndarray,
                     floor_edge_y: Optional[int] = None) -> List[ClassificationResult]:
        """对所有人进行分类"""
        results = []
        for pose in poses:
            result = self.classify(pose, bed_mask, floor_edge_y)
            results.append(result)
        return results

    def visualize(self, frame: np.ndarray, poses: List[PersonPose],
                  results: List[ClassificationResult]) -> np.ndarray:
        """在帧上可视化分类结果"""
        vis = frame.copy()

        for pose, result in zip(poses, results):
            x1, y1, x2, y2 = map(int, pose.bbox)
            color_bgr = (int(result.color[2]), int(result.color[1]), int(result.color[0]))

            # 绘制状态边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)

            # 状态标签
            label = f"{result.label} ({result.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 跌倒报警特效
            if result.state == BehaviorState.FALLEN:
                # 红色闪烁边框
                cv2.rectangle(vis, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 3)
                # 报警文字
                cv2.putText(vis, "!!! FALL ALARM !!!", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis

    def reset(self):
        """重置分类器状态"""
        self.state_history.clear()
