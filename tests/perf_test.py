"""连续帧性能测试：验证床体Mask缓存后的处理速度"""
import cv2
import json
import time
import urllib.request
import sys

video_path = r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_videos\fall_demo.mp4"
cap = cv2.VideoCapture(video_path)

latencies = []
results_summary = []

print("连续帧检测测试（前5帧含SAM分割，后续使用缓存Mask）...")
print(f"{'帧':>4} {'时延ms':>8} {'人数':>4} {'状态':>8} {'跌倒':>4} {'IoU':>6} {'轴线角':>6}")
print("-" * 60)

for frame_idx in range(20):
    ret, frame = cap.read()
    if not ret:
        break

    _, buffer = cv2.imencode('.jpg', frame)
    boundary = "----PerfTestBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="f.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + buffer.tobytes() + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "http://localhost:8564/api/detect/frame",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    elapsed = (time.time() - t0) * 1000

    r = result["result"]
    latencies.append(elapsed)

    states = []
    for p in r["persons"]:
        states.append(p["classification"]["label"])
    state_str = "/".join(states) if states else "无人"

    iou_str = "-"
    angle_str = "-"
    if r["persons"]:
        det = r["persons"][0]["classification"]["details"]
        iou_str = f"{det['iou']:.3f}"
        angle_str = f"{det['body_axis_angle']:.1f}"

    print(f"{frame_idx:>4} {elapsed:>8.1f} {r['person_count']:>4} {state_str:>8} {str(r['fall_detected']):>4} {iou_str:>6} {angle_str:>6}")

    results_summary.append({
        "frame": frame_idx,
        "latency_ms": round(elapsed, 1),
        "persons": r["person_count"],
        "fall": r["fall_detected"],
    })

cap.release()

# 统计（排除前3帧SAM分割预热）
if len(latencies) > 3:
    steady_latencies = latencies[3:]
    print(f"\n{'='*60}")
    print(f"性能统计（排除前3帧SAM预热）")
    print(f"{'='*60}")
    print(f"平均时延: {sum(steady_latencies)/len(steady_latencies):.1f}ms")
    print(f"最小时延: {min(steady_latencies):.1f}ms")
    print(f"最大时延: {max(steady_latencies):.1f}ms")
    print(f"平均FPS: {1000/(sum(steady_latencies)/len(steady_latencies)):.1f}")
    print(f"首帧时延(含SAM): {latencies[0]:.1f}ms")
    print(f"验收标准: 串行时延≤30ms {'✓ 通过' if sum(steady_latencies)/len(steady_latencies) <= 30 else '✗ 未通过'}")

# 保存报告
report = {
    "test": "continuous_frame_performance",
    "total_frames": len(latencies),
    "first_frame_latency_ms": round(latencies[0], 1),
    "steady_state_avg_ms": round(sum(latencies[3:])/len(latencies[3:]), 1) if len(latencies) > 3 else None,
    "steady_state_fps": round(1000/(sum(latencies[3:])/len(latencies[3:])), 1) if len(latencies) > 3 else None,
    "frames": results_summary,
}
with open(r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\logs\performance_test.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n性能报告已保存: logs/performance_test.json")
