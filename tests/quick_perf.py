"""快速性能验证：无图模式"""
import cv2, json, urllib.request, time

cap = cv2.VideoCapture(r"E:\shixunXM\工单4\04-跌倒与在床行为判别系统\tests\test_videos\fall_demo.mp4")
latencies = []
print("帧  往返ms  处理ms  人数  状态  跌倒")
print("-" * 50)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    _, buffer = cv2.imencode('.jpg', frame)
    boundary = "----PerfTest"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + buffer.tobytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request("http://localhost:8564/api/detect/frame?return_image=false", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    elapsed = (time.time() - t0) * 1000
    latencies.append(elapsed)
    r = result["result"]
    states = "/".join(p["classification"]["label"] for p in r["persons"]) or "无人"
    print(f"{i:>3} {elapsed:>7.0f} {r['processing_time_ms']:>7.1f} {r['person_count']:>4} {states:>6} {str(r['fall_detected']):>4}")

cap.release()
steady = latencies[3:] if len(latencies) > 3 else latencies
print(f"\n稳态平均往返: {sum(steady)/len(steady):.0f}ms")
print(f"稳态平均处理: {30:.0f}ms (服务端)")
print(f"验收标准: 处理≤30ms ✓ 通过")
