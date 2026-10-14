"""
床体语义分割模块
使用 MobileSAM 进行精准床体分割，生成静态 Bed Mask 并缓存
支持多相机视角下的透视变换校正
"""
import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple, List
from segment_anything import sam_model_registry, SamPredictor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MOBILE_SAM_PATH, SAM_B_PATH, SEGMENTATION,
    BehaviorState, STATE_COLORS
)

logger = logging.getLogger(__name__)


class BedSegmenter:
    """
    床体语义分割器
    - 初始化时对首帧进行床体分割，生成静态掩膜
    - 支持手动框选 / 自动检测床体区域
    - 支持透视变换校正（俯视/侧视视角）
    - 掩膜周期性刷新机制
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**SEGMENTATION, **(config or {})}
        self.device = self.config["device"] if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.bed_mask: Optional[np.ndarray] = None
        self.bed_contour: Optional[np.ndarray] = None
        self.bed_bbox: Optional[Tuple[int, int, int, int]] = None
        self.floor_edge_y: Optional[int] = None  # 地面边缘y坐标
        self.homography_matrix: Optional[np.ndarray] = None  # 透视变换矩阵
        self.frame_count = 0
        self._last_mask_frame = 0
        self._model_loaded = False

    def load_model(self):
        """加载 SAM 模型（支持 sam_b / mobile_sam，自动降级）"""
        if self._model_loaded:
            return

        model_type = self.config["model_type"]

        # 尝试加载指定模型
        try:
            if model_type == "mobile_sam":
                model_path = MOBILE_SAM_PATH
                model_key = "vit_t"
                # 尝试使用 mobile_sam 包
                try:
                    from mobile_sam import sam_model_registry as ms_registry
                    sam = ms_registry[model_key](checkpoint=model_path)
                except ImportError:
                    # mobile_sam 包未安装，降级为 sam_b
                    logger.warning("mobile_sam 包未安装，降级使用 sam_b (vit_b)")
                    model_type = "sam_b"
                    model_path = SAM_B_PATH
                    model_key = "vit_b"
                    sam = sam_model_registry[model_key](checkpoint=model_path)
            else:
                model_path = SAM_B_PATH
                model_key = "vit_b"
                sam = sam_model_registry[model_key](checkpoint=model_path)

            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self._model_loaded = True
            logger.info(f"分割模型加载成功: {model_type} ({model_key}) on {self.device}")
        except Exception as e:
            logger.error(f"分割模型加载失败: {e}")
            raise

    def segment_bed_auto(self, frame: np.ndarray) -> np.ndarray:
        """
        自动床体分割
        策略：使用 SAM 的 everything 模式，结合面积/位置先验筛选床体区域
        床体通常位于画面中下部，面积较大，呈矩形
        """
        if self.predictor is None:
            self.load_model()

        h, w = frame.shape[:2]
        self.predictor.set_image(frame)

        # 生成候选框：画面中下部的大区域
        # 床体通常占据画面宽度的40%-90%，高度的30%-60%，位于中下部
        candidate_boxes = [
            # 左半床
            np.array([int(w * 0.05), int(h * 0.3), int(w * 0.55), int(h * 0.95)]),
            # 右半床
            np.array([int(w * 0.45), int(h * 0.3), int(w * 0.95), int(h * 0.95)]),
            # 整床
            np.array([int(w * 0.05), int(h * 0.25), int(w * 0.95), int(h * 0.95)]),
            # 中央区域
            np.array([int(w * 0.15), int(h * 0.35), int(w * 0.85), int(h * 0.9)]),
        ]

        best_mask = None
        best_score = -1

        for box in candidate_boxes:
            try:
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=True,
                )
                # 选择置信度最高且面积合理的掩膜
                for mask, score in zip(masks, scores):
                    area = np.sum(mask)
                    total_area = h * w
                    area_ratio = area / total_area

                    # 床体面积通常占画面的15%-60%
                    if 0.05 < area_ratio < 0.7:
                        # 评分：置信度 + 面积合理性 + 位置合理性（偏下）
                        mask_y_center = np.mean(np.where(mask)[0]) / h
                        position_score = 1.0 - abs(mask_y_center - 0.6)  # 中心偏下最佳
                        combined_score = score * 0.6 + position_score * 0.4

                        if combined_score > best_score:
                            best_score = combined_score
                            best_mask = mask.astype(np.uint8)
            except Exception as e:
                logger.warning(f"候选框分割失败: {e}")
                continue

        if best_mask is None:
            # Fallback: 使用画面中下部作为默认床体区域
            logger.warning("自动分割失败，使用默认床体区域")
            best_mask = np.zeros((h, w), dtype=np.uint8)
            best_mask[int(h * 0.35):int(h * 0.95), int(w * 0.1):int(w * 0.9)] = 1

        self._set_bed_mask(best_mask)
        return best_mask

    def segment_bed_with_prompt(self, frame: np.ndarray,
                                  point_coords: Optional[np.ndarray] = None,
                                  point_labels: Optional[np.ndarray] = None,
                                  box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        带提示的床体分割（点提示/框提示）
        """
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(frame)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        self._set_bed_mask(mask)
        return mask

    def _set_bed_mask(self, mask: np.ndarray):
        """设置床体掩膜并计算轮廓/边界框/地面边缘"""
        self.bed_mask = mask
        self._last_mask_frame = self.frame_count

        # 计算轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            self.bed_contour = largest
            x, y, w, h = cv2.boundingRect(largest)
            self.bed_bbox = (x, y, w, h)
            # 地面边缘 = 床体底部边界
            self.floor_edge_y = y + h
        else:
            self.bed_contour = None
            self.bed_bbox = None
            self.floor_edge_y = mask.shape[0]

    def compute_homography(self, frame: np.ndarray,
                            bed_corners: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        计算透视变换矩阵，校正相机视角畸变
        将床体从任意视角变换为俯视图
        """
        h, w = frame.shape[:2]

        if bed_corners is None and self.bed_contour is not None:
            # 从轮廓近似四边形
            epsilon = 0.02 * cv2.arcLength(self.bed_contour, True)
            approx = cv2.approxPolyDP(self.bed_contour, epsilon, True)
            if len(approx) >= 4:
                bed_corners = [tuple(pt[0]) for pt in approx[:4]]
            else:
                x, y, bw, bh = self.bed_bbox
                bed_corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
        elif bed_corners is None:
            x, y = int(w * 0.1), int(h * 0.35)
            bw, bh = int(w * 0.8), int(h * 0.55)
            bed_corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]

        # 目标：标准矩形（俯视图）
        dst_w, dst_h = 600, 400
        dst_points = np.array([
            [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]
        ], dtype=np.float32)

        src_points = np.array(bed_corners, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        return self.homography_matrix

    def apply_homography(self, frame: np.ndarray) -> np.ndarray:
        """应用透视变换，生成俯视图"""
        if self.homography_matrix is None:
            self.compute_homography(frame)
        return cv2.warpPerspective(frame, self.homography_matrix, (600, 400))

    def get_bed_mask(self, frame: np.ndarray, force_refresh: bool = False) -> np.ndarray:
        """
        获取床体掩膜（带缓存机制）
        每隔 N 帧或强制刷新时重新分割
        """
        self.frame_count += 1

        if (self.bed_mask is None or force_refresh or
                self.frame_count - self._last_mask_frame > self.config["static_mask_refresh_interval"]):
            self.segment_bed_auto(frame)

        return self.bed_mask

    def visualize(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """在帧上可视化床体掩膜"""
        vis = frame.copy()
        if self.bed_mask is not None:
            # 半透明绿色掩膜
            colored_mask = np.zeros_like(vis)
            colored_mask[self.bed_mask > 0] = [0, 255, 100]
            vis = cv2.addWeighted(vis, 1 - alpha, colored_mask, alpha, 0)

            # 绘制轮廓
            if self.bed_contour is not None:
                cv2.drawContours(vis, [self.bed_contour], -1, (0, 200, 0), 2)

            # 绘制边界框和地面边缘线
            if self.bed_bbox:
                x, y, w, h = self.bed_bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.line(vis, (0, self.floor_edge_y), (vis.shape[1], self.floor_edge_y),
                         (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, "Floor Edge", (10, self.floor_edge_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return vis

    def reset(self):
        """重置分割器状态"""
        self.bed_mask = None
        self.bed_contour = None
        self.bed_bbox = None
        self.floor_edge_y = None
        self.homography_matrix = None
        self.frame_count = 0
        self._last_mask_frame = 0
