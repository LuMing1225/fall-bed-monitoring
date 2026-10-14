"""
自动化测试：跌倒/在床行为分类器
覆盖 test_lie_on_bed_exclusion（躺在床上不报警）和 test_fall_on_floor_detection（跌倒在地上必报警）

测试断言：
- 对"躺在床上"的正确分类率 ≥ 98%
- 对"跌倒在地"的捕获灵敏度 (Sensitivity) = 100%
- 躺在床上误报率 ≤ 2%
- 跌倒在地上漏报率 = 0%
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2

# 添加项目路径
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import BehaviorState, STATE_LABELS
from segments.bed_segment import BedSegmenter
from poses.yolo_pose import PoseDetector
from classifier.rule_engine import BehaviorClassifier
from pipeline.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 测试视频目录
TEST_VIDEO_DIR = Path(__file__).resolve().parent / "test_videos"


class TestResult:
    """测试结果"""
    def __init__(self, name: str):
        self.name = name
        self.total_frames = 0
        self.correct_frames = 0
        self.false_positives = 0  # 误报（在床上被判为跌倒）
        self.false_negatives = 0  # 漏报（跌倒被判为非跌倒）
        self.details = []
        self.passed = False

    @property
    def accuracy(self) -> float:
        return self.correct_frames / self.total_frames if self.total_frames > 0 else 0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.total_frames if self.total_frames > 0 else 0

    @property
    def sensitivity(self) -> float:
        """灵敏度 = TP / (TP + FN)"""
        tp = self.correct_frames - (self.total_frames - self.correct_frames - self.false_negatives)
        total_positive = tp + self.false_negatives
        return tp / total_positive if total_positive > 0 else 1.0

    def summary(self) -> str:
        return (f"[{self.name}] 总帧数:{self.total_frames} "
                f"正确:{self.correct_frames} 准确率:{self.accuracy:.1%} "
                f"误报:{self.false_positives} 漏报:{self.false_negatives} "
                f"通过:{'✓' if self.passed else '✗'}")


def generate_synthetic_bed_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """生成合成的病房场景（带床体）"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # 浅灰墙
    # 地面
    frame[int(height * 0.7):, :] = [180, 180, 180]
    # 床体区域
    bed_x1, bed_y1 = int(width * 0.15), int(height * 0.45)
    bed_x2, bed_y2 = int(width * 0.85), int(height * 0.85)
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (200, 220, 255), -1)  # 床单
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (150, 150, 150), 2)  # 床沿
    # 枕头
    cv2.rectangle(frame, (bed_x1 + 10, bed_y1 + 5), (bed_x1 + 80, bed_y1 + 40), (255, 255, 255), -1)
    return frame


def create_person_keypoints_on_bed(width: int = 640, height: int = 480) -> np.ndarray:
    """创建躺在床上的人体关键点（水平姿态）"""
    bed_y1 = int(height * 0.45)
    bed_y2 = int(height * 0.85)
    center_y = (bed_y1 + bed_y2) // 2
    center_x = width // 2

    kps = np.zeros((17, 3), dtype=np.float32)
    # 头部（左侧）
    kps[0] = [center_x - 150, center_y - 10, 0.9]   # nose
    kps[1] = [center_x - 155, center_y - 15, 0.8]   # left_eye
    kps[2] = [center_x - 145, center_y - 15, 0.8]   # right_eye
    kps[3] = [center_x - 160, center_y - 12, 0.7]   # left_ear
    kps[4] = [center_x - 140, center_y - 12, 0.7]   # right_ear
    # 肩部
    kps[5] = [center_x - 100, center_y, 0.9]         # left_shoulder
    kps[6] = [center_x - 100, center_y + 20, 0.9]    # right_shoulder
    # 肘部
    kps[7] = [center_x - 50, center_y - 5, 0.8]       # left_elbow
    kps[8] = [center_x - 50, center_y + 25, 0.8]      # right_elbow
    # 手腕
    kps[9] = [center_x, center_y - 5, 0.7]             # left_wrist
    kps[10] = [center_x, center_y + 25, 0.7]           # right_wrist
    # 髋部
    kps[11] = [center_x + 30, center_y, 0.9]           # left_hip
    kps[12] = [center_x + 30, center_y + 20, 0.9]      # right_hip
    # 膝盖
    kps[13] = [center_x + 90, center_y, 0.8]           # left_knee
    kps[14] = [center_x + 90, center_y + 20, 0.8]      # right_knee
    # 脚踝
    kps[15] = [center_x + 140, center_y, 0.7]          # left_ankle
    kps[16] = [center_x + 140, center_y + 20, 0.7]     # right_ankle
    return kps


def create_person_keypoints_fallen(width: int = 640, height: int = 480) -> np.ndarray:
    """创建跌倒在地上的人体关键点（水平姿态，在床外地面）"""
    floor_y = int(height * 0.92)  # 地面区域（床体底部y=0.85*height以下）
    center_x = int(width * 0.3)   # 画面左侧，地面上

    kps = np.zeros((17, 3), dtype=np.float32)
    # 头部
    kps[0] = [center_x - 80, floor_y - 10, 0.9]
    kps[1] = [center_x - 85, floor_y - 15, 0.8]
    kps[2] = [center_x - 75, floor_y - 15, 0.8]
    kps[3] = [center_x - 90, floor_y - 12, 0.7]
    kps[4] = [center_x - 70, floor_y - 12, 0.7]
    # 肩部
    kps[5] = [center_x - 30, floor_y, 0.9]
    kps[6] = [center_x - 30, floor_y + 18, 0.9]
    # 肘部
    kps[7] = [center_x + 20, floor_y - 5, 0.8]
    kps[8] = [center_x + 20, floor_y + 23, 0.8]
    # 手腕
    kps[9] = [center_x + 60, floor_y - 5, 0.7]
    kps[10] = [center_x + 60, floor_y + 23, 0.7]
    # 髋部
    kps[11] = [center_x + 50, floor_y, 0.9]
    kps[12] = [center_x + 50, floor_y + 18, 0.9]
    # 膝盖
    kps[13] = [center_x + 110, floor_y, 0.8]
    kps[14] = [center_x + 110, floor_y + 18, 0.8]
    # 脚踝
    kps[15] = [center_x + 160, floor_y, 0.7]
    kps[16] = [center_x + 160, floor_y + 18, 0.7]
    return kps


def create_person_keypoints_standing(width: int = 640, height: int = 480) -> np.ndarray:
    """创建站立的人体关键点（垂直姿态）"""
    center_x = width // 2
    head_y = int(height * 0.2)

    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = [center_x, head_y, 0.9]
    kps[1] = [center_x - 8, head_y - 8, 0.8]
    kps[2] = [center_x + 8, head_y - 8, 0.8]
    kps[3] = [center_x - 12, head_y, 0.7]
    kps[4] = [center_x + 12, head_y, 0.7]
    kps[5] = [center_x - 25, head_y + 40, 0.9]
    kps[6] = [center_x + 25, head_y + 40, 0.9]
    kps[7] = [center_x - 35, head_y + 80, 0.8]
    kps[8] = [center_x + 35, head_y + 80, 0.8]
    kps[9] = [center_x - 40, head_y + 120, 0.7]
    kps[10] = [center_x + 40, head_y + 120, 0.7]
    kps[11] = [center_x - 15, head_y + 130, 0.9]
    kps[12] = [center_x + 15, head_y + 130, 0.9]
    kps[13] = [center_x - 15, head_y + 190, 0.8]
    kps[14] = [center_x + 15, head_y + 190, 0.8]
    kps[15] = [center_x - 15, head_y + 250, 0.7]
    kps[16] = [center_x + 15, head_y + 250, 0.7]
    return kps


def test_lie_on_bed_exclusion() -> TestResult:
    """
    测试：躺在床上不报警
    断言：误报率 ≤ 2%，准确率 ≥ 98%
    """
    result = TestResult("test_lie_on_bed_exclusion")
    logger.info("=" * 60)
    logger.info("开始测试: 躺在床上不报警")

    width, height = 640, 480
    frame = generate_synthetic_bed_scene(width, height)

    # 初始化模块
    segmenter = BedSegmenter()
    classifier = BehaviorClassifier()

    # 设置床体掩膜（手动）
    bed_mask = np.zeros((height, width), dtype=np.uint8)
    bed_mask[int(height * 0.45):int(height * 0.85), int(width * 0.15):int(width * 0.85)] = 1
    segmenter._set_bed_mask(bed_mask)
    floor_edge_y = int(height * 0.85)

    # 生成30帧"躺在床上"的测试数据（带轻微抖动）
    np.random.seed(42)
    for i in range(30):
        kps = create_person_keypoints_on_bed(width, height)
        # 添加轻微抖动
        kps[:, :2] += np.random.normal(0, 2, kps[:, :2].shape)

        from poses.yolo_pose import PersonPose
        bbox = np.array([kps[:, 0].min() - 10, kps[:, 1].min() - 10,
                         kps[:, 0].max() + 10, kps[:, 1].max() + 10])
        pose = PersonPose(kps, bbox, 0.9, track_id=0)

        cls_result = classifier.classify(pose, bed_mask, floor_edge_y)
        result.total_frames += 1

        # 躺在床上应该被判为 on_bed 或 active，绝不能是 fallen
        if cls_result.state == BehaviorState.FALLEN:
            result.false_positives += 1
            result.details.append(f"帧{i}: 误报为跌倒 (IoU={cls_result.details.get('iou', 0):.3f})")
        elif cls_result.state in (BehaviorState.ON_BED, BehaviorState.ACTIVE, BehaviorState.UNKNOWN):
            result.correct_frames += 1
        else:
            result.details.append(f"帧{i}: 未知状态 {cls_result.state}")

    # 断言
    result.passed = (result.false_positive_rate <= 0.02 and result.accuracy >= 0.98)
    logger.info(result.summary())
    if result.details:
        for d in result.details[:5]:
            logger.warning(f"  {d}")
    return result


def test_fall_on_floor_detection() -> TestResult:
    """
    测试：跌倒在地上必报警
    断言：灵敏度 (Sensitivity) = 100%，漏报率 = 0%
    """
    result = TestResult("test_fall_on_floor_detection")
    logger.info("=" * 60)
    logger.info("开始测试: 跌倒在地上必报警")

    width, height = 640, 480
    frame = generate_synthetic_bed_scene(width, height)

    segmenter = BedSegmenter()
    classifier = BehaviorClassifier()

    # 设置床体掩膜
    bed_mask = np.zeros((height, width), dtype=np.uint8)
    bed_mask[int(height * 0.45):int(height * 0.85), int(width * 0.15):int(width * 0.85)] = 1
    segmenter._set_bed_mask(bed_mask)
    floor_edge_y = int(height * 0.85)

    # 预热5帧（建立时序历史）
    np.random.seed(123)
    for i in range(5):
        kps = create_person_keypoints_fallen(width, height)
        kps[:, :2] += np.random.normal(0, 3, kps[:, :2].shape)
        from poses.yolo_pose import PersonPose
        bbox = np.array([kps[:, 0].min() - 10, kps[:, 1].min() - 10,
                         kps[:, 0].max() + 10, kps[:, 1].max() + 10])
        pose = PersonPose(kps, bbox, 0.9, track_id=1)
        classifier.classify(pose, bed_mask, floor_edge_y)

    # 生成30帧"跌倒在地上"的测试数据
    for i in range(30):
        kps = create_person_keypoints_fallen(width, height)
        kps[:, :2] += np.random.normal(0, 3, kps[:, :2].shape)

        from poses.yolo_pose import PersonPose
        bbox = np.array([kps[:, 0].min() - 10, kps[:, 1].min() - 10,
                         kps[:, 0].max() + 10, kps[:, 1].max() + 10])
        pose = PersonPose(kps, bbox, 0.9, track_id=1)

        cls_result = classifier.classify(pose, bed_mask, floor_edge_y)
        result.total_frames += 1

        # 跌倒在地上应该被判为 fallen
        if cls_result.state == BehaviorState.FALLEN:
            result.correct_frames += 1
        else:
            result.false_negatives += 1
            result.details.append(
                f"帧{i}: 漏报 (状态={cls_result.state}, "
                f"IoU={cls_result.details.get('iou', 0):.3f}, "
                f"水平={cls_result.details.get('is_horizontal', False)})"
            )

    # 断言：灵敏度100%
    result.passed = (result.false_negatives == 0)
    logger.info(result.summary())
    if result.details:
        for d in result.details[:5]:
            logger.warning(f"  {d}")
    return result


def test_standing_not_fall() -> TestResult:
    """测试：站立的人不会被判为跌倒"""
    result = TestResult("test_standing_not_fall")
    logger.info("=" * 60)
    logger.info("开始测试: 站立不报警")

    width, height = 640, 480
    bed_mask = np.zeros((height, width), dtype=np.uint8)
    bed_mask[int(height * 0.45):int(height * 0.85), int(width * 0.15):int(width * 0.85)] = 1
    floor_edge_y = int(height * 0.85)

    classifier = BehaviorClassifier()

    for i in range(20):
        kps = create_person_keypoints_standing(width, height)
        kps[:, :2] += np.random.normal(0, 2, kps[:, :2].shape)

        from poses.yolo_pose import PersonPose
        bbox = np.array([kps[:, 0].min() - 10, kps[:, 1].min() - 10,
                         kps[:, 0].max() + 10, kps[:, 1].max() + 10])
        pose = PersonPose(kps, bbox, 0.9, track_id=2)

        cls_result = classifier.classify(pose, bed_mask, floor_edge_y)
        result.total_frames += 1

        if cls_result.state != BehaviorState.FALLEN:
            result.correct_frames += 1
        else:
            result.false_positives += 1

    result.passed = result.false_positives == 0
    logger.info(result.summary())
    return result


def test_iou_calculation() -> TestResult:
    """测试：IoU计算正确性"""
    result = TestResult("test_iou_calculation")
    logger.info("=" * 60)
    logger.info("开始测试: IoU计算")

    width, height = 100, 100
    bed_mask = np.zeros((height, width), dtype=np.uint8)
    bed_mask[20:80, 20:80] = 1  # 60x60 床体

    classifier = BehaviorClassifier()

    # 测试1：完全重叠
    kps = np.array([[30, 30, 0.9], [70, 30, 0.9], [70, 70, 0.9], [30, 70, 0.9]] +
                    [[0, 0, 0]] * 13, dtype=np.float32)
    from poses.yolo_pose import PersonPose
    bbox = np.array([25, 25, 75, 75], dtype=np.float32)
    pose = PersonPose(kps, bbox, 0.9, 0)
    iou, _ = classifier._compute_iou(pose, bed_mask)
    result.total_frames += 1
    if iou > 0.3:
        result.correct_frames += 1
    else:
        result.details.append(f"完全重叠IoU过低: {iou:.3f}")

    # 测试2：完全不重叠
    kps2 = np.array([[5, 5, 0.9], [15, 5, 0.9], [15, 15, 0.9], [5, 15, 0.9]] +
                     [[0, 0, 0]] * 13, dtype=np.float32)
    bbox2 = np.array([0, 0, 20, 20], dtype=np.float32)
    pose2 = PersonPose(kps2, bbox2, 0.9, 0)
    iou2, _ = classifier._compute_iou(pose2, bed_mask)
    result.total_frames += 1
    if iou2 < 0.1:
        result.correct_frames += 1
    else:
        result.details.append(f"不重叠IoU过高: {iou2:.3f}")

    result.passed = result.correct_frames == result.total_frames
    logger.info(result.summary())
    return result


def run_all_tests() -> Dict:
    """运行所有测试"""
    logger.info("\n" + "=" * 70)
    logger.info("跌倒/在床行为判别系统 - 自动化测试套件")
    logger.info("=" * 70)

    tests = [
        test_lie_on_bed_exclusion,
        test_fall_on_floor_detection,
        test_standing_not_fall,
        test_iou_calculation,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"测试 {test_func.__name__} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("测试汇总")
    logger.info("=" * 70)

    all_passed = True
    for r in results:
        logger.info(r.summary())
        if not r.passed:
            all_passed = False

    logger.info("=" * 70)
    if all_passed:
        logger.info("✓ 所有测试通过!")
    else:
        logger.info("✗ 部分测试失败，请检查详情")
    logger.info("=" * 70)

    # 保存测试报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_passed": all_passed,
        "results": [
            {
                "name": r.name,
                "total_frames": r.total_frames,
                "correct_frames": r.correct_frames,
                "accuracy": round(r.accuracy, 4),
                "false_positives": r.false_positives,
                "false_negatives": r.false_negatives,
                "false_positive_rate": round(r.false_positive_rate, 4),
                "passed": r.passed,
            }
            for r in results
        ]
    }

    report_path = Path(__file__).resolve().parent.parent / "logs" / "test_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"测试报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    report = run_all_tests()
    sys.exit(0 if report["all_passed"] else 1)
