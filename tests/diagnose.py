"""诊断脚本：直接调用处理器，测量每一步耗时"""
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\backend")

from pipeline.video_processor import VideoProcessor

# 初始化处理器
processor = VideoProcessor()
print("加载模型...")
processor.load_models()
print("模型加载完成\n")

# 读取测试视频
cap = cv2.VideoCapture(r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_videos\fall_demo.mp4")

print(f"{'帧':>4} {'分割ms':>8} {'姿态ms':>8} {'判别ms':>8} {'LLMms':>8} {'可视化ms':>8} {'总计ms':>8} {'Mask缓存':>6}")
print("-" * 80)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    # 分步计时
    t0 = time.time()
    bed_mask = processor.bed_segmenter.get_bed_mask(frame)
    t1 = time.time()

    poses = processor.pose_detector.detect(frame)
    t2 = time.time()

    results = processor.classifier.classify_all(poses, bed_mask, processor.bed_segmenter.floor_edge_y)
    t3 = time.time()

    # LLM纠错
    for pose, cls_result in zip(poses, results):
        processor.pose_corrector.correct_classification(pose, cls_result.state, cls_result.details)
    t4 = time.time()

    # 可视化
    annotated = frame.copy()
    annotated = processor.bed_segmenter.visualize(annotated)
    annotated = processor.pose_detector.visualize(annotated, poses)
    annotated = processor.classifier.visualize(annotated, poses, results)
    t5 = time.time()

    seg_ms = (t1 - t0) * 1000
    pose_ms = (t2 - t1) * 1000
    cls_ms = (t3 - t2) * 1000
    llm_ms = (t4 - t3) * 1000
    vis_ms = (t5 - t4) * 1000
    total_ms = (t5 - t0) * 1000

    mask_cached = "是" if processor.bed_segmenter.frame_count > processor.bed_segmenter._last_mask_frame else "否(刚分割)"

    print(f"{i:>4} {seg_ms:>8.1f} {pose_ms:>8.1f} {cls_ms:>8.1f} {llm_ms:>8.1f} {vis_ms:>8.1f} {total_ms:>8.1f} {mask_cached:>6}")

cap.release()

# 检查床体分割器状态
print(f"\n床体分割器状态:")
print(f"  frame_count: {processor.bed_segmenter.frame_count}")
print(f"  _last_mask_frame: {processor.bed_segmenter._last_mask_frame}")
print(f"  bed_mask is None: {processor.bed_segmenter.bed_mask is None}")
print(f"  static_mask_refresh_interval: {processor.bed_segmenter.config['static_mask_refresh_interval']}")
print(f"  floor_edge_y: {processor.bed_segmenter.floor_edge_y}")
