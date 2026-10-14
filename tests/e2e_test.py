"""端到端检测测试：从测试视频提取帧，发送给后端API"""
import cv2
import json
import urllib.request
import sys

# 从测试视频提取一帧
video_path = r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_videos\fall_demo.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取测试视频帧")
    sys.exit(1)

print(f"测试帧: {frame.shape}")

# 编码为JPEG
_, buffer = cv2.imencode('.jpg', frame)

# 构造multipart/form-data
boundary = "----TestBoundary1234567890"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + buffer.tobytes() + f"\r\n--{boundary}--\r\n".encode()

# 发送请求
req = urllib.request.Request(
    "http://localhost:8564/api/detect/frame",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

print("发送检测请求...")
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

r = result["result"]
print(f"\n{'='*50}")
print(f"检测结果")
print(f"{'='*50}")
print(f"帧序号: {r['frame_idx']}")
print(f"检测人数: {r['person_count']}")
print(f"跌倒检测: {r['fall_detected']}")
print(f"处理时延: {r['processing_time_ms']}ms")
print(f"FPS: {r['fps']}")
print(f"在床人数: {r['on_bed_count']}")
print(f"活动人数: {r['active_count']}")

for p in r["persons"]:
    cls = p["classification"]
    det = cls["details"]
    print(f"\n  人员#{p['track_id']}: {cls['label']} ({cls['confidence']:.0%})")
    print(f"    置信度: {p['confidence']:.2f}")
    print(f"    IoU: {det['iou']:.3f}")
    print(f"    关键点在床比例: {det['keypoint_in_bed_ratio']:.1%} ({det['keypoint_in_bed_count']}/{det['keypoint_total']})")
    print(f"    人体轴线角度: {det['body_axis_angle']:.1f}°")
    print(f"    是否水平: {det['is_horizontal']}")
    print(f"    臀部Y: {det['hip_y']:.0f}, 地面边缘Y: {det['floor_edge_y']}")
    print(f"    臀部相对高度: {det['hip_relative_height']:.0f}px")
    print(f"    到床体距离: {det['distance_to_bed']:.1f}px")
    if "llm_correction" in det:
        print(f"    LLM纠错: {det['llm_correction']}")

# 保存标注结果图
if "annotated_image" in result:
    import base64
    img_data = result["annotated_image"]
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    out_path = r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_result.jpg"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print(f"\n标注结果图已保存: {out_path}")

print(f"\n{'='*50}")
print("端到端检测测试完成!")
