"""单帧计时调试"""
import cv2, json, urllib.request, time

cap = cv2.VideoCapture(r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_videos\fall_demo.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
print(f"Frame shape: {frame.shape}")

_, buffer = cv2.imencode('.jpg', frame)
boundary = "----DebugBoundary"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="t.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + buffer.tobytes() + f"\r\n--{boundary}--\r\n".encode()

for i in range(3):
    req = urllib.request.Request(
        "http://localhost:8564/api/detect/frame",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    elapsed = (time.time() - t0) * 1000
    dt = result["result"].get("debug_timing", {})
    print(f"Request {i}: roundtrip={elapsed:.0f}ms | "
          f"server_total={dt.get('total_ms')}ms | "
          f"read={dt.get('read_ms')} | decode={dt.get('decode_ms')} | "
          f"process={dt.get('process_ms')} | encode={dt.get('encode_ms')} | "
          f"shape={dt.get('frame_shape')}")
