# 高精度车辆文搜图与轨迹研判平台 -- 算法技术报告

> **版本**: v1.0 | **日期**: 2026 年 6 月  
> **关键词**: Chinese-CLIP, YOLOv8x, 双层检索, 跨镜轨迹拼接, 融合打分, 不确定性可视化

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [系统架构](#3-系统架构)
4. [数据流向](#4-数据流向)
5. [核心模块详解](#5-核心模块详解)
6. [关键算法](#6-关键算法)
7. [性能指标](#7-性能指标)
8. [部署架构](#8-部署架构)

---

## 1. 项目概述

### 1.1 系统定位

本平台是面向公安交通管理的**文本驱动目标检索与跨镜时空回溯**系统。核心解决两个业务问题：

1. **文搜图**：交警输入"蓝色轿车"等自然语言描述，系统从海量检测目标中返回最匹配的候选图片；
2. **轨迹研判**：确认目标后，系统自动回溯该目标在 46 个摄像头中的完整时空轨迹，标注观测段与推断段，辅助人工研判。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 文本检索 | 自然语言查询 + 车牌精确查询，双层检索（属性粗筛 + CLIP 精排） |
| 目标确认 | 候选图片浏览、属性比对、一键确认 |
| 跨镜轨迹回溯 | 以确认目标为锚点，自动构建跨摄像头观测链 |
| 不确定性可视化 | 明确区分观测段（实线）与推断段（虚线），附带置信度评分 |
| 仪表盘 | 系统运行状态、摄像头分布、数据统计概览 |

### 1.3 创新点

- **双层检索策略**：属性粗筛将 68,349 条检测快速缩减至数百条候选，CLIP 向量精排保证语义精度，整体检索延迟 < 50ms
- **离散观测模型**：主动承认摄像头覆盖不连续，输出"观测段 + 推断段 + 候选路径"三层结构，避免将离散观测误表述为连续轨迹
- **公安业务适配**：前端采用浅色交警主题，属性展示使用中文字段（颜色/车型/车牌），支持苏E车牌前缀

---

## 2. 技术栈

### 2.1 后端

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.13 | 主语言 |
| FastAPI | 0.115.6 | REST API 框架 |
| Uvicorn | 0.34.0 | ASGI 服务器 |
| PyTorch | 2.11.0+cu128 | 深度学习推理 |
| ultralytics | 8.4.80 | YOLOv8x 目标检测 |
| cn-clip | 1.6.0 | Chinese-CLIP 图文特征 |
| open-clip-torch | 3.3.0 | 备用 CLIP 加载 |
| transformers | 4.57.6 | 模型加载辅助 |
| qdrant-client | 1.12.1 | 向量数据库客户端 |
| FAISS | -- | Detection-level 向量索引 |
| OpenCV | 4.13.0 | 图像处理 |

### 2.2 前端

| 组件 | 版本 | 用途 |
|------|------|------|
| Streamlit | 1.45.1 | Web 界面框架 |
| Plotly | -- | 时空分布图表 |
| Folium | -- | 地图可视化 |
| HTML/CSS | -- | 公安浅色主题定制 |

### 2.3 数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| AICity22 Track1 MTMC | 68,349 detections | 主数据集 |
| -- | 2,070 tracks | 单摄跟踪轨迹 |
| -- | 230 vehicles | 唯一车辆 ID |
| -- | 46 cameras | 6 个场景（S01-S06） |
| 标注格式 | MOT | frame,id,left,top,width,height,conf,class |
| 视频源 | vdo.avi | 每摄像头一个视频文件 |

### 2.4 模型

| 模型 | 大小 | 用途 | 输出维度 |
|------|------|------|----------|
| Chinese-CLIP ViT-B/16 | 753 MB | 图文跨模态特征提取 | 512 维（归一化） |
| YOLOv8x | 137 MB | 车辆/行人目标检测 | 检测框 + 类别 + 置信度 |

---

## 3. 系统架构

### 3.1 分层架构

```
+================================================================+
|                      表现层 (Presentation)                       |
|  search.py  |  confirm.py  |  trajectory.py  |  dashboard.py   |
|  检索查询台  |  目标确认页   |  轨迹回溯页      |  数据仪表盘     |
+================================================================+
         |                  |                  |
         v                  v                  v
+================================================================+
|                      服务层 (API Gateway)                        |
|  /api/v1/search/query  |  /api/v1/backtrack/trajectory         |
|  /api/v1/search/plate  |  /api/v1/backtrack/trace              |
|  /api/v1/confirm/target|  /api/v1/dashboard/stats              |
+================================================================+
         |                  |                  |
         v                  v                  v
+================================================================+
|                      业务层 (Business Logic)                     |
|  +-------------+  +--------------+  +------------------------+  |
|  | 检索引擎     |  | 轨迹拼接引擎  |  | 数据治理               |  |
|  | QueryParser |  | AnchorBacktr |  | CameraManager          |  |
|  | AttrFilter  |  | ChainExpand  |  | RoadTopology           |  |
|  | VectorRecall|  | Scoring      |  | VideoStream            |  |
|  | Reranker    |  |              |  |                        |  |
|  +-------------+  +--------------+  +------------------------+  |
+================================================================+
         |                  |                  |
         v                  v                  v
+================================================================+
|                      数据层 (Data)                               |
|  cityflow_results.json | cityflow_camera_metadata.yaml          |
|  clip_vectors.faiss    | Qdrant 向量库 (traffic_targets)        |
|  cityflow_crops/       | models/ (CLIP + YOLO)                  |
+================================================================+
```

### 3.2 模块交互关系

检索链路: 用户查询 -> QueryParser -> AttributeFilter -> VectorRecall -> Reranker -> 候选列表

回溯链路: 用户确认 -> 锚点定位 -> AnchorBacktracker -> ChainExpander -> Scoring -> TrajectoryOutput

---
## 4. 数据流向

### 4.1 离线预处理链路

```
原始视频 (vdo.avi)
    |
    v
[YOLOv8x 检测] -- conf >= 0.5, NMS 0.45 -->
    |
    v
[detections.json] -- 68,349 条检测记录 -->
    |
    v
[ByteTrack 跟踪] -- IoU匹配 + 卡尔曼滤波 -->
    |
    v
[tracks.json] -- 2,070 条轨迹 -->
    |
    v
[按 vehicle_id 聚合] -->
    |
    v
[cityflow_results.json] -- detections + tracks + det_to_track_map -->
    |
    +---> [关键帧裁剪] --> cityflow_crops/
    |
    +---> [Chinese-CLIP 编码] --> clip_vectors.faiss (68,349 x 512)
    |
    +---> [摄像头元数据] --> cityflow_camera_metadata.yaml (46 摄像头)
```

**关键脚本**: scripts/cityflow_all_tracks.py（检测+跟踪）, scripts/clip_feature_pipeline.py（CLIP特征提取）

### 4.2 在线检索链路

```
用户输入 "蓝色轿车"
    |
    v
[QueryParser] -- 正则+规则匹配 -->
    | 提取: {color: "蓝色", vehicle_type: "轿车"}
    v
[AttributeFilter] -- 颜色/车型硬过滤 -->
    | 68,349 --> ~5,000 条候选
    v
[CLIP 文本编码] -- Chinese-CLIP text encoder -->
    | 512 维归一化查询向量
    v
[FAISS 向量检索] -- IndexFlatIP (余弦相似度) -->
    | Top-200 候选
    v
[融合打分] -- final = 0.6 * clip_score + 0.4 * attr_score -->
    |
    v
[按轨迹聚合去重] -->
    |
    v
返回 Top-K 候选列表 (instance_id, keyframe, scores)
```

### 4.3 轨迹回溯链路

```
用户确认 candidate (instance_id="CF3_c001_V0034_000001")
    |
    v
[提取 vehicle_id] -- "V0034" -->
    |
    v
[跨镜索引查询] -- vehicle_detections["V0034"] -->
    | 该车辆在所有摄像头的检测记录
    v
[按摄像头分组] -- camera_groups: {c001: [det1,det2], c004: [det3], ...} -->
    |
    v
[组内时间排序] -- 按 timestamp 升序 -->
    |
    v
[构建摄像头序列] -- 按首次出现时间排序 -->
    | [(c001, t1-t2), (c004, t3-t4), (c010, t5-t6), ...]
    v
[加载摄像头坐标] -- 从 YAML 读取 lat/lon/direction -->
    |
    v
[生成推断段] -- 相邻摄像头间连接 + 置信度 -->
    |
    v
返回完整轨迹 (camera_sequence + observation_nodes + inference_segments)
```

---
## 5. 核心模块详解

### 5.1 检索模块 (`src/retrieval/`)

| 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `query_parser.py` | 自然语言查询解析 | 用户文本 | ParsedQuery (类别/颜色/车型/车牌) |
| `attribute_filter.py` | 属性硬过滤 | detections + ParsedQuery | 过滤后候选集 |
| `vector_recall.py` | CLIP 向量召回 | 查询向量 + FAISS索引 | Top-K 相似候选 |
| `reranker.py` | 多特征融合重排 | 粗召回列表 | 精排后 Top-N |

**核心逻辑**: 实际 API 层 (`api/routes/search.py`) 直接实现了双层检索，融合公式为 `final_score = 0.6 * clip_similarity + 0.4 * attribute_score`，其中 attribute_score 综合了检测置信度和关键词匹配度。

### 5.2 感知模块 (`src/perception/`)

| 文件 | 职责 | 核心算法 |
|------|------|----------|
| `detector.py` | YOLOv8x 车辆检测 | COCO 类别映射: car/truck/bus -> vehicle |
| `tracker.py` | ByteTrack 单摄跟踪 | 8维卡尔曼滤波 + IoU贪心匹配 |
| `feature_extractor.py` | 双特征提取 | ReID (512维) + Chinese-CLIP (768维) |
| `plate_ocr.py` | 车牌识别 | HSV颜色分割定位 + OCR解码 |
| `quality.py` | 质量评分 | 清晰度(拉普拉斯) + 完整度 + 遮挡率 + 检测置信度 |
| `attribute.py` | 属性识别 | HSV直方图匹配(颜色) + 宽高比启发式(车型) |

### 5.3 轨迹拼接模块 (`src/stitching/`)

| 文件 | 职责 | 核心逻辑 |
|------|------|----------|
| `anchor_backtrack.py` | 锚点回溯 | 从确认帧向上/下游贪心扩展 |
| `chain_expander.py` | 观测链扩展 | 生成观测段 + 推断段 + 候选路径 |
| `scoring.py` | 跨镜连接评分 | 六维度加权评分 - 两项惩罚 |

**评分维度**: 外观相似度、属性一致性、车牌一致性、时间可达性、空间可达性、方向一致性。车辆和行人采用差异化权重配置。

### 5.4 数据治理模块 (`src/data_governance/`)

| 文件 | 职责 | 说明 |
|------|------|------|
| `camera_manager.py` | 摄像头元数据管理 | 46个摄像头的坐标、方向、邻接关系 |
| `road_topology.py` | 道路拓扑构建 | BFS可达性判断、Dijkstra最短路径 |
| `video_stream.py` | RTSP/ONVIF拉流 | 预留接口，支持实时视频接入 |

**摄像头元数据**: 46 个摄像头分布在 6 个场景（S01: 5个, S02: 4个, S03: 6个, S04: 25个, S06: 6个），每个摄像头包含 camera_id、经纬度、朝向、车道方向、所属路段、邻接摄像头列表。

### 5.5 API 层 (`api/routes/`)

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/v1/search/query` | POST | 双层检索（属性粗筛 + CLIP精排） |
| `/api/v1/search/plate` | POST | 车牌精确查询 |
| `/api/v1/confirm/target` | POST | 用户确认目标实例 |
| `/api/v1/backtrack/trace` | POST | 锚点轨迹回溯 |
| `/api/v1/backtrack/trajectory` | POST | 完整跨镜轨迹（多摄像头时间线） |
| `/api/v1/backtrack/result/{query_id}` | GET | 获取回溯结果 |
| `/api/v1/dashboard/stats` | GET | 系统统计信息 |
| `/api/v1/dashboard/cameras` | GET | 摄像头元数据列表 |
| `/api/v1/dashboard/health` | GET | 服务健康状态 |

### 5.6 前端页面 (`frontend/pages/`)

| 页面 | 功能 | 特色 |
|------|------|------|
| `search.py` | 业务查询台 | 自然语言/车牌查询、业务化筛选（时间/区域/方向）、候选卡片表格 |
| `confirm.py` | 目标确认页 | 大图浏览、帧切换、属性比对、启动轨迹回溯 |
| `trajectory.py` | 轨迹回溯页 | 轨迹概览卡片、摄像头时序表格、推断段标注、研判摘要 |
| `dashboard.py` | 数据仪表盘 | 系统统计、摄像头列表、服务健康状态 |

---

## 6. 关键算法

### 6.1 双层检索策略

**第一层 -- 属性粗筛**:

基于结构化属性（颜色、车型）进行硬过滤，时间复杂度 O(N)，N 为检测总数。

```python
# 伪代码
for det in all_detections:  # 68,349 条
    if query.color and det.color != query.color and det.color != "unknown":
        continue  # 颜色不匹配，跳过
    if query.vehicle_type and det.type != query.vehicle_type:
        continue  # 车型不匹配，跳过
    candidates.append(det)  # 通过粗筛，约 5,000 条
```

**第二层 -- CLIP 向量精排**:

对粗筛后的候选计算 CLIP 语义相似度。查询文本经 Chinese-CLIP 文本编码器映射为 512 维归一化向量，在 FAISS IndexFlatIP 索引中检索 Top-200 最近邻。

**融合打分**:

```
final_score = ALPHA * clip_score + (1 - ALPHA) * attr_score
            = 0.6 * clip_similarity + 0.4 * attribute_match
```

其中 `attribute_match = min(0.5 + keyword_matches * 0.15 + confidence * 0.3, 0.99)`

**优势**: 相比纯向量检索，属性粗筛将检索空间缩减 90%+，速度提升 10-100 倍，同时 CLIP 精排保证了语义级别的匹配精度。

### 6.2 跨镜轨迹拼接

**Vehicle ID 提取**:

从 target_id 中解析 vehicle_id，构建全局倒排索引：

```
target_id = "CF3_c001_V0034_000001"
    --> split by "_" --> ["CF3", "c001", "V0034", "000001"]
    --> vehicle_id = "V0034"

倒排索引: vehicle_id -> [detection_1, detection_2, ..., detection_n]
```

**六维度跨镜连接评分**:

```
S_final = max(0, w1*S_app + w2*S_attr + w3*S_plate
              + w4*S_temp + w5*S_spat + w6*S_dir
              - lambda1*P_div - lambda2*P_miss)
```

**差异化权重配置**:

| 评分分项 | 车辆权重 | 行人权重 | 设计理由 |
|---------|---------|---------|---------|
| 车牌一致性 | 0.35 | -- | 车牌是车辆唯一强标识 |
| 时间可达性 | 0.25 | 0.35 | 行人无车牌，时间约束更重要 |
| 空间可达性 | 0.15 | -- | 车辆受道路拓扑约束更强 |
| ReID 外观 | 0.15 | 0.30 | 行人匹配更依赖外观特征 |
| 属性一致性 | 0.10 | 0.25 | 行人属性更具区分力 |
| 背包 | -- | 0.10 | 行人特有属性 |

**不确定性标注**: 根据相邻摄像头时间间隔着色 -- 绿色(< 15s)、黄色(15-60s)、红色(>= 60s)，直观展示推断段的可信程度。

### 6.3 融合打分（重排模块）

重排模块 (`CandidateReranker`) 对粗召回结果进行最终精排：

```
S_rank = w1 * S_vec + w2 * S_attr + w3 * S_qual + w4 * S_fresh
       = 0.4 * S_vec + 0.3 * S_attr + 0.2 * S_qual + 0.1 * S_fresh
```

- `S_vec`: Chinese-CLIP 文本-图像向量相似度
- `S_attr`: 属性匹配率（查询属性在候选中的命中率）
- `S_qual`: 目标检测质量分数
- `S_fresh`: 时间新鲜度，指数衰减 `exp(-dt / 3600)`

---
## 7. 性能指标

### 7.1 检索性能

| 指标 | 数值 | 说明 |
|------|------|------|
| IDF1 | 0.987 | 跨镜跟踪 F1 分数 |
| P@10 | 0.973 | 模糊搜索前 10 命中率 |
| unknown 率 | 0.2% | 批量推理中未识别目标比例（从 97.4% 优化） |

### 7.2 轨迹拼接

| 指标 | 数值 | 说明 |
|------|------|------|
| 轨迹拼接成功率 | 95% | 218/230 辆车出现在 >= 2 个摄像头 |
| 摄像头覆盖 | 46 个 | 6 个场景全覆盖 |
| 跨镜索引构建 | < 1s | vehicle_id 倒排索引 |

### 7.3 系统性能

| 指标 | 数值 | 环境 |
|------|------|------|
| 检索延迟 | < 50ms | 属性粗筛 + FAISS 向量检索 |
| GPU 推理 | RTX 5070 (12GB) | CUDA 12.8, PyTorch 2.11 |
| 模型加载 | ~10s | Chinese-CLIP 首次加载 |

---

## 8. 部署架构

### 8.1 Docker 容器化部署

```
+---------------------------------------------------+
|                  Docker Compose                     |
|                                                     |
|  +------------------+  +-------------------------+  |
|  | backend           |  | frontend                |  |
|  | FastAPI + Uvicorn |  | Streamlit               |  |
|  | port: 8000       |  | port: 8501              |  |
|  | GPU: 1x NVIDIA   |  | depends_on: backend     |  |
|  | CUDA 12.8        |  | API -> backend:8000     |  |
|  +--------+---------+  +------------+------------+  |
|           |                            |             |
|           v                            v             |
|  +------------------+                              |
|  | qdrant           |                              |
|  | 向量数据库        |                              |
|  | port: 6333/6334  |                              |
|  +------------------+                              |
|                                                     |
|  Network: traffic-net (bridge)                      |
+---------------------------------------------------+
```

### 8.2 服务配置

| 服务 | 镜像 | 端口 | 资源 |
|------|------|------|------|
| backend | Dockerfile.backend | 8000 | 1x GPU, 50MB日志 |
| frontend | Dockerfile.frontend | 8501 | CPU only |
| qdrant | qdrant/qdrant:latest | 6333, 6334 | CPU only |

### 8.3 数据卷挂载

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| H:/trajectory CLIP/data | /data | 数据集 |
| H:/trajectory CLIP/models | /models | 模型权重 |
| H:/trajectory CLIP/output | /output | 检测结果 |
| H:/trajectory CLIP/logs | /logs | 运行日志 |

### 8.4 本地部署

```bash
# 启动后端
python scripts/run_server.py
# 或: uvicorn api.main:app --host 0.0.0.0 --port 8000

# 启动前端
streamlit run frontend/app.py --server.port 8501
```

### 8.5 关键配置文件

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 系统全局配置（设备/路径/阈值/权重） |
| `configs/cityflow_camera_metadata.yaml` | 46 摄像头元数据（坐标/方向/邻接） |
| `docker-compose.yml` | 容器编排 |
| `docker/.env` | 环境变量 |

---

*报告完*