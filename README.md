# 基于多模态分割与姿态估计的跌倒/在床行为精准判别系统

> 多Agent协同与CV智能感知 | 人工智能CV方向

## 项目简介

本系统解决跌倒检测中**"躺在地上与躺在病床上姿态相似、难以区分"**的核心痛点。通过**语义分割**(MobileSAM)精准定位床体空间，结合**人体姿态估计**(YOLOv8-Pose)提取17个骨骼关键点，最终通过**拓扑几何算法**(IoU+距离+角度)实现"在床"与"跌倒"的精确分类，并集成**LLM时序纠错Agent**(模拟Llama V2 Pose)进行关键点漂移纠错。

## 技术架构

```
视频帧 → [床体分割 MobileSAM] → Bed Mask (静态缓存)
              ↓
       [姿态检测 YOLOv8-Pose] → 17关键点 + 边界框
              ↓
       [几何判别规则引擎] → IoU + 距离 + 角度
              ↓
       [LLM时序纠错 Ollama/qwen] → 自注意力漂移纠错
              ↓
         行为状态: Active / On_Bed / Fallen
```

## 核心算法

### 状态决策方程

```
           { On_Bed,   if  IoU ≥ 0.5 AND 关键点在床比例 ≥ 0.6 AND Y_hip < Y_floor_edge
Status =  { Fallen,   if  人体水平 AND IoU < 0.15 AND 关键点在床比例 < 0.3
           { Active,   otherwise
```

### 关键指标

| 指标 | 标准 | 实际 |
|------|------|------|
| 躺在床上误报率 | ≤ 2% | 待测试 |
| 跌倒在地上漏报率 | = 0% | 待测试 |
| 单帧处理时延 | ≤ 30ms | 待测试 |
| 运行内存占用 | ≤ 1.8GB | 待测试 |

## 环境要求

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5070 Laptop (8GB VRAM) / 支持CUDA的GPU
- **Python**: 3.13 (推荐使用 conda 环境 `yolo_313`)
- **CUDA**: 12.8+
- **Ollama**: 可选（用于LLM时序纠错，模型 qwen2.5:7b）

### 已验证的 conda 环境

```
yolo_313 (Python 3.13)
├── torch 2.11.0+cu128
├── torchvision 0.26.0+cu128
├── ultralytics 8.4.143
├── segment-anything 1.0
├── opencv-python 5.0.0
├── fastapi 0.142.0
└── numpy 2.5.3
```

## 快速开始

### 1. 一键启动（推荐）

双击 `start_all.bat`，系统将自动启动后端和前端服务，并打开浏览器。

### 2. 手动启动

**后端服务 (端口 8564)**:
```bash
cd backend
conda activate yolo_313
python app.py
```

**前端服务 (端口 8524)**:
```bash
cd frontend
python server.py
```

### 3. 访问系统

- 前端界面: http://localhost:8524
- 后端API文档: http://localhost:8564/docs

## 项目结构

```
04-跌倒与在床行为判别系统/
├── backend/                    # 后端服务 (端口8564)
│   ├── app.py                  # FastAPI主应用 (REST + WebSocket)
│   ├── config.py               # 全局配置
│   ├── segments/
│   │   └── bed_segment.py      # 床体语义分割 (MobileSAM)
│   ├── poses/
│   │   └── yolo_pose.py        # 姿态检测 (YOLOv8-Pose)
│   ├── classifier/
│   │   └── rule_engine.py      # 几何判别规则引擎 (IoU+距离+角度)
│   ├── llm_agent/
│   │   └── pose_corrector.py   # LLM姿态时序纠错 (模拟Llama V2 Pose)
│   ├── pipeline/
│   │   └── video_processor.py  # 视频处理流水线
│   └── models/                  # 模型权重
│       ├── mobile_sam.pt
│       ├── yolov8n-pose.pt
│       ├── yolo11n-pose.pt
│       └── sam_b.pt
├── frontend/                   # 前端界面 (端口8524)
│   ├── index.html              # 主页面
│   ├── css/style.css           # 样式
│   ├── js/app.js               # 交互逻辑
│   └── server.py               # 静态文件服务器
├── tests/
│   └── test_fall_classifier.py # 自动化测试 (4项测试)
├── scripts/
│   └── profile_inference.py    # 性能评估与TensorRT加速
├── docs/                        # 文档
│   ├── 床体分割与人体姿态拓扑几何判定规格书.md
│   ├── 空间三维透视投影校正数理方程说明.md
│   └── 智慧医疗人体姿态与跌倒检测面试QA知识库.md
├── reports/                     # 性能报告
├── logs/                        # 运行日志
├── start_backend.bat
├── start_frontend.bat
├── start_all.bat
├── requirements.txt
└── README.md
```

## API 接口

### REST API

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/status` | GET | 系统状态 |
| `/api/detect/frame` | POST | 单帧图像检测 |
| `/api/detect/video` | POST | 视频文件检测 |
| `/api/bed/reset` | POST | 重置床体掩膜 |
| `/api/bed/set` | POST | 手动设置床体掩膜 |
| `/api/report/{track_id}` | GET | 行为分析报告 |

### WebSocket

| 端点 | 功能 |
|------|------|
| `/ws/detect` | 实时视频流检测 |

## 运行测试

```bash
cd tests
conda activate yolo_313
python test_fall_classifier.py
```

测试覆盖：
- `test_lie_on_bed_exclusion`: 躺在床上不报警（误报率≤2%）
- `test_fall_on_floor_detection`: 跌倒在地上必报警（漏报率=0%）
- `test_standing_not_fall`: 站立的人不报警
- `test_iou_calculation`: IoU计算正确性

## 性能评估

```bash
cd scripts
conda activate yolo_313
python profile_inference.py
```

评估内容：
- YOLOv8-Pose 单模型基准（时延/FPS/显存）
- MobileSAM 单模型基准
- 双模型流水线基准
- ONNX导出
- 生成边缘端性能报告

## 工单验收标准

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 任务一 | 规格书 + 透视投影校正方程 | 明确定义不同视角下床体轮廓与地面界线规则 |
| 任务二 | 分割/姿态/判别模块源码 + 测试日志 | 躺在床上误报率≤2%，跌倒漏报率=0% |
| 任务三 | TensorRT量化脚本 + 性能报告 | 串行时延≤30ms，内存≤1.8GB |
| 任务四 | 面试QA知识库 | 能手画姿态检测与床体Mask几何向量投影判定图 |

## 数据集

- **UR Fall Detection Dataset (URFD)**: 30段跌倒 + 15段日常活动，多种深度相机及RGB相机
- **COCO Keypoints**: YOLOv8-Pose预训练所使用的通用人体关键点数据集

## 许可证

本项目仅供学习和研究使用。
