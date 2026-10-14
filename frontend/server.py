"""
前端静态文件服务器
端口: 8524
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
PORT = 8524


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[前端] {args[0]}")


def main():
    os.chdir(str(FRONTEND_DIR))
    with socketserver.TCPServer(("0.0.0.0", PORT), CORSRequestHandler) as httpd:
        print(f"=" * 50)
        print(f"前端服务已启动")
        print(f"访问地址: http://localhost:{PORT}")
        print(f"静态目录: {FRONTEND_DIR}")
        print(f"按 Ctrl+C 停止")
        print(f"=" * 50)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n前端服务已停止")


if __name__ == "__main__":
    main()
