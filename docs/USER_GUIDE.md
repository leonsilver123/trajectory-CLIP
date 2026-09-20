# 交通风险感知子系统——使用说明书

> ## ⚠️ 历史文档声明（2026-09-21 加注）
>
> 本文写于早期阶段，其中关于**前端**与**模块结构**的描述已过时：
> 前端已由 `frontend/`（Streamlit）统一为 `webapp/`（React + Vite，后端同源托管，
> 无独立容器、无 8501 端口）；`src/backtrack/`、`src/data_governance/video_stream.py`、
> `api/dependencies.py` 已删除；`docker-compose.yml` 现仅剩 backend 一个服务。
> **当前架构以 `CLAUDE.md` 与代码为准。**

---

> **版本**：v1.0.0  
> **适用场景**：苏州市相城区道路交通风险感知与事故回溯  
> **文档日期**：2026 年 6 月

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境要求](#2-环境要求)
3. [安装部署](#3-安装部署)
4. [操作指南](#4-操作指南)
5. [API 接口文档](#5-api-接口文档)
6. [配置说明](#6-配置说明)
7. [常见问题 FAQ](#7-常见问题-faq)
8. [GPU 环境诊断](#8-gpu-环境诊断)
9. [区域配置与部署适配](#9-区域配置与部署适配)
10. [系统设计说明](#10-系统设计说明)

---

## 1. 系统概述

### 1.1 功能简介

交通风险感知子系统是一个面向道路交通场景的**文本驱动目标检索与跨镜时空回溯**系统。系统对多路道路摄像头视频流进行预结构化处理，通过目标检测、单摄跟踪、属性识别、车牌 OCR、质量评分和向量特征提取，生成目标实例库和单摄轨迹库；在此基础上，融合摄像头拓扑、道路方向、合理旅行时间、ReID 外观相似度、属性一致性和车牌一致性构建跨镜候选连接图，生成带置信度的跨摄像头观测链。

核心功能包括：

| 功能模块 | 说明 |
|---------|------|
| **文本检索** | 用户输入"蓝色背包的男人""黑色轿车"或车牌号，系统返回候选目标图片 |
| **目标确认** | 用户从候选图片中确认目标，系统绑定目标实例 |
| **轨迹回溯** | 以确认目标为锚点，在跨镜候选图中回溯经过的摄像头序列、时间节点和候选路径 |
| **仪表盘** | 展示系统运行状态、摄像头分布、轨迹统计等概览信息 |
| **时间轴回放** | 按时间顺序回放目标在各摄像头的观测记录 |

### 1.2 适用场景

- **交通事故回溯**：回溯涉事车辆在事故前后经过的摄像头序列和可能路径
- **重点车辆排查**：通过车牌号或车辆描述查找特定车辆在路网中的活动轨迹
- **风险目标定位**：定位符合文本描述的风险目标（如"红色外套骑摩托车的男子"）
- **人工复核支撑**：为人工研判提供结构化的轨迹证据和置信度参考

### 1.3 系统特色

1. **从图片检索到目标时空回溯**：不仅回答"哪些图片像"，更回答"目标在哪些摄像头出现过、什么时间出现、可能怎么移动"
2. **离散观测链重建**：主动承认道路摄像头覆盖不连续，输出观测段 + 推断段 + 候选路径 + 置信度，避免将离散观测误表述为连续轨迹
3. **交通约束拼接**：融合摄像头拓扑、道路方向、合理旅行时间、车牌一致性等多维交通约束，提升跨镜拼接可靠性

---

## 2. 环境要求

### 2.1 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA GPU，显存 ≥ 4 GB（如 GTX 1650） | NVIDIA GPU，显存 ≥ 8 GB（如 RTX 3070 / A10） |
| **CPU** | 4 核 | 8 核及以上 |
| **内存** | 8 GB | 16 GB 及以上 |
| **磁盘** | 20 GB 可用空间（模型文件约 5 GB） | 50 GB SSD |

### 2.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **操作系统** | Windows 10/11 或 Ubuntu 20.04+ | |
| **Python** | 3.9 ~ 3.11 | 推荐 3.10 |
| **CUDA** | 11.8 或 12.x | GPU 推理必需 |
| **cuDNN** | 与 CUDA 版本匹配 | |
| **Docker** | 20.10+ | Docker 部署时需要 |
| **Docker Compose** | v2.0+ | Docker 部署时需要 |
| **NVIDIA Container Toolkit** | 最新版 | Docker GPU 支持 |

---

## 3. 安装部署

### 3.1 本地部署

#### 3.1.1 Python 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n traffic-risk python=3.10 -y
conda activate traffic-risk
```

#### 3.1.2 依赖安装

```bash
# 进入项目根目录
cd "H:\trajectory CLIP"

# 安装 PyTorch（CUDA 11.8 示例，请根据实际 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装项目包
pip install -e .
```

主要依赖包括：

| 依赖包 | 用途 |
|--------|------|
| `ultralytics` | YOLO 目标检测 |
| `torchreid` / `torchvision` | ReID 特征提取 |
| `transformers` | Chinese-CLIP 图文特征提取 |
| `qdrant-client` | 向量数据库客户端 |
| `fastapi` + `uvicorn` | 后端 API 服务 |
| `streamlit` | 前端可视化界面 |
| `opencv-python` | 图像处理 |
| `pyyaml` | 配置文件解析 |

#### 3.1.3 配置说明

编辑 `configs/default.yaml`，根据实际环境修改以下关键配置：

```yaml
system:
  device: "cuda"              # 无 GPU 改为 "cpu"
  data_dir: "H:/trajectory CLIP/data"
  model_dir: "H:/trajectory CLIP/models"
  output_dir: "H:/trajectory CLIP/output"
  log_dir: "H:/trajectory CLIP/logs"
```

编辑 `configs/camera_metadata.yaml`，配置摄像头元数据（编号、经纬度、朝向、覆盖路段、邻接关系等）。

#### 3.1.4 启动服务

```bash
# 启动后端 API 服务（默认端口 8000）
python scripts/run_server.py
# 或
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 另开终端，启动前端界面（默认端口 8501）
streamlit run frontend/app.py --server.port 8501
```

启动后访问：
- 前端界面：http://localhost:8501
- 后端 API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 3.2 Docker 部署

#### 3.2.1 前置条件

- 安装 Docker 和 Docker Compose
- 安装 NVIDIA Container Toolkit（Linux）或 Docker Desktop GPU 支持（Windows）
- 准备模型文件和摄像头元数据

#### 3.2.2 一键启动

```bash
# 进入项目根目录
cd "H:\trajectory CLIP"

# 构建并启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f
```

#### 3.2.3 服务地址

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:8501 | Streamlit Web 界面 |
| 后端 API | http://localhost:8000 | FastAPI REST API |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| Qdrant | http://localhost:6333 | 向量数据库 |

#### 3.2.4 停止服务

```bash
docker compose down
```

---

## 4. 操作指南

### 4.1 文本检索

系统支持自然语言文本查询和车牌精确查询两种方式。

#### 4.1.1 自然语言查询

1. 在侧边栏选择 **"🔍 目标检索"** 页面
2. 在搜索框中输入目标描述文本，例如：
   - `"蓝色背包的男人"` — 系统将解析为：目标类别=行人，性别=男性，背包=是，背包颜色=蓝色
   - `"黑色轿车"` — 系统将解析为：目标类别=车辆，颜色=黑色，车型=轿车
   - `"白色SUV"` — 系统将解析为：目标类别=车辆，颜色=白色，车型=SUV
3. 点击 **"检索"** 按钮
4. 系统依次执行：查询解析 → 属性过滤 → 图文向量召回 → 候选重排
5. 返回 Top-5 候选目标图片及其属性信息

**查询解析规则**：

| 输入关键词 | 解析结果 |
|-----------|---------|
| 轿车/SUV/卡车/面包车/车辆 | 目标类别=vehicle |
| 行人/男人/女人/男性/女性 | 目标类别=pedestrian |
| 电动车/自行车/摩托车 | 目标类别=non_motor_vehicle |
| 白色/黑色/红色/蓝色等 | 颜色属性 |
| 背包/书包/挎包 | 附属物=背包 |

#### 4.1.2 车牌查询

1. 在搜索框中直接输入车牌号，如 `"苏E12345"`
2. 系统自动识别车牌格式，走精确匹配通道
3. 返回该车牌关联的所有目标实例和关键帧图片

> **提示**：车牌查询比模糊描述可靠得多，建议在有明确车牌信息时优先使用。

### 4.2 目标确认

1. 检索完成后，系统展示候选目标图片列表
2. 每张图片显示：关键帧图片、目标类别、属性信息、检测置信度
3. 浏览候选结果，找到目标后点击 **"确认此目标"** 按钮
4. 系统记录确认的目标实例信息（instance_id、camera_id、timestamp）
5. 自动跳转到轨迹回溯页面

### 4.3 轨迹回溯

#### 4.3.1 基本操作

1. 确认目标后，系统自动以该目标实例为锚点进行轨迹回溯
2. 回溯流程：查找所属 Tracklet → 查询跨镜候选图 → 向上游和下游扩展 → 返回高置信观测链

#### 4.3.2 地图说明

轨迹回溯页面展示目标的空间轨迹信息：

- **观测节点**（实心圆点）：目标在该摄像头中被实际检测到
- **观测段**（实线）：摄像头视野内的真实运动片段，可信度最高
- **推断段**（虚线）：摄像头之间基于拓扑和时间的推断连接，标注置信度
- **候选路径**：两摄像头间有多条可能道路时，显示多条候选路径及各自置信度

#### 4.3.3 输出信息

系统最终返回十类信息：

| 序号 | 信息类型 | 说明 |
|------|---------|------|
| 1 | 候选目标图片 | 检索阶段返回的候选关键帧 |
| 2 | 确认的目标实例 | 用户确认的目标 |
| 3 | 摄像头经过序列 | 目标依次出现的摄像头 ID 列表 |
| 4 | 关键帧 | 每个摄像头的目标关键帧图片 |
| 5 | 时间节点 | 每个观测点的出现时间 |
| 6 | 观测段 | 摄像头视野内真实运动片段 |
| 7 | 推断段 | 摄像头间推断连接及置信度 |
| 8 | 候选路径 | 摄像头间可能的路网路径 |
| 9 | 连接置信度 | 每段跨镜连接的评分 |
| 10 | 拼接证据 | 车牌、属性、ReID、旅行时间等 |

### 4.4 仪表盘使用

仪表盘页面（**📊 仪表盘**）展示系统运行概览：

- **系统统计**：摄像头数量、Tracklet 数量、目标实例数量、候选边数量
- **摄像头列表**：所有摄像头的元数据信息（编号、名称、位置、朝向）
- **服务健康状态**：后端 API、向量数据库、前端服务的运行状态

### 4.5 时间轴回放

时间轴页面（**⏱️ 时间轴回放**）按时间顺序展示目标在各摄像头的观测记录：

- 横向时间轴展示目标经过各摄像头的时间节点
- 每个节点可点击查看关键帧和详细信息
- 支持在不同观测段之间快速跳转

---

## 5. API 接口文档

### 5.1 健康检查

| 项目 | 说明 |
|------|------|
| **URL** | `GET /health` |
| **描述** | 服务健康检查 |
| **响应示例** | `{"status": "ok", "version": "1.0.0"}` |

### 5.2 检索接口

#### 5.2.1 文本查询

| 项目 | 说明 |
|------|------|
| **URL** | `POST /api/v1/search/query` |
| **描述** | 文本查询检索候选目标 |
| **请求体** | 见下表 |

**请求参数**（`SearchRequest`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query_text` | string | 是 | 用户查询文本 |
| `top_k` | int | 否 | 召回数量，默认 20 |
| `target_type` | string | 否 | 目标类别过滤（vehicle/pedestrian/non_motor_vehicle） |

**响应**（`SearchResponse`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_id` | string | 查询唯一 ID |
| `candidates` | list | 候选目标列表 |
| `total_count` | int | 候选总数 |

#### 5.2.2 车牌查询

| 项目 | 说明 |
|------|------|
| **URL** | `POST /api/v1/search/plate` |
| **描述** | 车牌精确查询 |
| **请求体** | `{"plate_number": "苏E12345"}` |

### 5.3 确认接口

| 项目 | 说明 |
|------|------|
| **URL** | `POST /api/v1/confirm/target` |
| **描述** | 用户确认目标实例 |

**请求参数**（`ConfirmRequest`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query_id` | string | 是 | 查询 ID |
| `instance_id` | string | 是 | 确认的目标实例 ID |

**响应**（`ConfirmResponse`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_id` | string | 查询 ID |
| `instance_id` | string | 实例 ID |
| `status` | string | 状态（"confirmed"） |
| `message` | string | 消息 |

### 5.4 回溯接口

#### 5.4.1 轨迹回溯

| 项目 | 说明 |
|------|------|
| **URL** | `POST /api/v1/backtrack/trace` |
| **描述** | 以确认目标为锚点进行轨迹回溯 |

**请求参数**（`BacktrackRequest`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `instance_id` | string | 是 | 确认的目标实例 ID |
| `max_upstream` | int | 否 | 最大上游回溯深度，默认 10 |
| `max_downstream` | int | 否 | 最大下游回溯深度，默认 10 |

**响应**（`BacktrackResponse`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_id` | string | 查询 ID |
| `camera_sequence` | list | 摄像头序列 |
| `observation_nodes` | list | 观测节点列表 |
| `observation_segments` | list | 观测段列表 |
| `inference_segments` | list | 推断段列表 |
| `candidate_paths` | list | 候选路径列表 |
| `overall_confidence` | float | 整体轨迹置信度 |

#### 5.4.2 获取回溯结果

| 项目 | 说明 |
|------|------|
| **URL** | `GET /api/v1/backtrack/result/{query_id}` |
| **描述** | 根据查询 ID 获取已完成的回溯结果 |

### 5.5 仪表盘接口

| 项目 | 说明 |
|------|------|
| **URL** | `GET /api/v1/dashboard/stats` |
| **描述** | 获取系统统计信息（摄像头数、Tracklet 数、实例数、边数） |

| 项目 | 说明 |
|------|------|
| **URL** | `GET /api/v1/dashboard/cameras` |
| **描述** | 获取所有摄像头元数据列表 |

| 项目 | 说明 |
|------|------|
| **URL** | `GET /api/v1/dashboard/health` |
| **描述** | 获取服务健康状态 |

---

## 6. 配置说明

系统配置文件位于 `configs/default.yaml`，以下为各配置项的详细说明：

### 6.1 系统配置（system）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `name` | string | `"TrafficSpatioTemporalBacktrack"` | 系统名称 |
| `version` | string | `"1.0.0"` | 系统版本号 |
| `device` | string | `"cuda"` | 推理设备，可选 `cuda` / `cpu` |
| `data_dir` | string | `"H:/trajectory CLIP/data"` | 数据目录路径 |
| `model_dir` | string | `"H:/trajectory CLIP/models"` | 模型文件目录 |
| `output_dir` | string | `"H:/trajectory CLIP/output"` | 输出文件目录 |
| `log_dir` | string | `"H:/trajectory CLIP/logs"` | 日志文件目录 |
| `log_level` | string | `"INFO"` | 日志级别：DEBUG / INFO / WARNING / ERROR |

### 6.2 摄像头配置（camera）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `metadata_file` | string | `"configs/camera_metadata.yaml"` | 摄像头元数据文件路径 |
| `default_search_radius_km` | float | `5.0` | 默认附近搜索半径（公里） |

### 6.3 目标检测配置（detection）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | string | `"yolov8x"` | YOLO 模型名称 |
| `confidence_threshold` | float | `0.5` | 检测置信度阈值 |
| `nms_threshold` | float | `0.45` | NMS 阈值 |
| `target_classes` | list | `[vehicle, pedestrian, non_motor_vehicle]` | 检测目标类别 |
| `input_size` | list | `[640, 640]` | 输入图像尺寸 |

### 6.4 单摄跟踪配置（tracking）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `algorithm` | string | `"bytetrack"` | 跟踪算法 |
| `max_age` | int | `30` | 最大丢失帧数（超过则终止轨迹） |
| `min_hits` | int | `3` | 最小命中帧数（少于此数不输出） |
| `iou_threshold` | float | `0.3` | IoU 匹配阈值 |

### 6.5 属性识别配置（attribute）

| 配置项 | 说明 |
|--------|------|
| `vehicle_attributes` | 车辆识别属性列表：`color`（车身颜色）、`vehicle_type`（车辆类型） |
| `pedestrian_attributes` | 行人识别属性列表：`gender`（性别外观）、`clothing_color`（上衣颜色）、`bag`（是否背包）、`bag_color`（背包颜色） |

### 6.6 车牌 OCR 配置（plate_ocr）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用车牌 OCR |
| `model_path` | string | `null` | OCR 模型路径，null 使用预训练模型 |
| `confidence_threshold` | float | `0.7` | 识别置信度阈值 |

### 6.7 特征提取配置（feature）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `reid.model` | string | `"osnet_x1_0"` | ReID 模型 |
| `reid.vector_dim` | int | `512` | ReID 向量维度 |
| `reid.batch_size` | int | `32` | ReID 批处理大小 |
| `clip.model` | string | `"CN-CLIP-ViT-L-14"` | Chinese-CLIP 模型 |
| `clip.vector_dim` | int | `768` | CLIP 向量维度 |
| `clip.batch_size` | int | `16` | CLIP 批处理大小 |

### 6.8 质量评分配置（quality）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `min_score` | float | `0.3` | 最低质量阈值 |
| `weights.clarity` | float | `0.3` | 清晰度权重（基于拉普拉斯方差） |
| `weights.completeness` | float | `0.3` | 完整度权重（检测框在画面内的比例） |
| `weights.occlusion` | float | `0.2` | 遮挡率权重（基于边缘梯度分析） |
| `weights.detection_conf` | float | `0.2` | 检测置信度权重 |

### 6.9 跨镜拼接配置（stitching）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_time_gap_seconds` | int | `600` | 最大时间间隔（秒） |
| `min_appearance_score` | float | `0.6` | 最低外观相似度阈值 |
| `spatial_search_radius_km` | float | `10.0` | 空间搜索半径（公里） |
| `weights.vehicle` | dict | 见配置文件 | 车辆评分权重：plate=0.35, temporal=0.25, topology=0.15, reid=0.15, attribute=0.10 |
| `weights.pedestrian` | dict | 见配置文件 | 行人评分权重：temporal=0.35, reid=0.30, attribute=0.25, bag=0.10 |
| `penalties.path_divergence` | float | `0.1` | 路径分叉惩罚系数 |
| `penalties.observation_missing` | float | `0.05` | 观测缺失惩罚系数 |

### 6.10 文本检索配置（retrieval）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `top_k` | int | `20` | 召回数量 |
| `vector_db` | string | `"qdrant"` | 向量数据库类型 |
| `qdrant.host` | string | `"localhost"` | Qdrant 地址 |
| `qdrant.port` | int | `6333` | Qdrant 端口 |
| `qdrant.collection_name` | string | `"traffic_targets"` | Qdrant 集合名 |
| `clip_model` | string | `"CN-CLIP-ViT-L-14"` | 检索用 CLIP 模型 |
| `rerank_top_n` | int | `5` | 重排后返回数量 |

### 6.11 轨迹回溯配置（backtrack）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_upstream_depth` | int | `10` | 最大上游回溯深度 |
| `max_downstream_depth` | int | `10` | 最大下游回溯深度 |
| `min_confidence_threshold` | float | `0.5` | 最低置信度阈值 |
| `max_results` | int | `50` | 最大返回结果数 |

### 6.12 API 配置（api）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `host` | string | `"0.0.0.0"` | 监听地址 |
| `port` | int | `8000` | 监听端口 |
| `workers` | int | `2` | 工作进程数 |
| `cors_origins` | list | `["*"]` | CORS 允许的源 |
| `debug` | bool | `false` | 调试模式 |

---

## 7. 常见问题 FAQ

### Q1：启动后端时报 "YOLO 模型加载失败"

**A**：请确认 `models/` 目录下存在 `yolov8x.pt` 模型文件。首次使用时需下载预训练权重。如果网络受限，可手动下载后放入 `models/` 目录。

### Q2：前端显示 "后端 API 离线（演示模式）"

**A**：请确认后端 API 服务已启动（默认端口 8000）。前端通过 `http://localhost:8000` 访问后端，如后端未启动则自动进入演示模式，使用模拟数据展示界面。

### Q3：Qdrant 连接失败

**A**：Docker 部署时确认 `docker compose ps` 中 qdrant 容器状态为 running。本地部署时需手动启动 Qdrant：`docker run -p 6333:6333 qdrant/qdrant:latest`。如不需要向量检索，系统会自动回退到内存暴力检索模式。

### Q4：GPU 推理报 "CUDA out of memory"

**A**：减小批处理大小（修改 `configs/default.yaml` 中 `feature.reid.batch_size` 和 `feature.clip.batch_size`），或使用更小的模型（如 `yolov8s` 替代 `yolov8x`）。

### Q5：跨镜拼接结果中没有候选边

**A**：可能原因：(1) 摄像头元数据未正确配置，检查 `camera_metadata.yaml` 中的邻接关系；(2) 外观相似度阈值过高，尝试降低 `stitching.min_appearance_score`；(3) Tracklet 数量不足，确认视频结构化流程正常运行。

### Q6：车牌识别率低

**A**：系统支持 EasyOCR 和 PaddleOCR 两种 OCR 后端。请安装其中一种：`pip install easyocr` 或 `pip install paddleocr`。如均未安装，将使用内置简化版，识别率会显著下降。

### Q7：如何添加新的摄像头？

**A**：在 `configs/camera_metadata.yaml` 中添加摄像头条目（包含 camera_id、name、latitude、longitude、direction、covered_road_segment、lane_direction），并在 `adjacency` 和 `road_segments` 中更新邻接关系和路段定义。也可通过 `CameraManager.register_camera()` 接口动态注册。

### Q8：CPU 模式下系统能否正常运行？

**A**：可以，但推理速度会显著下降。将 `configs/default.yaml` 中 `system.device` 设为 `"cpu"` 即可。建议 CPU 模式下仅用于功能验证，不建议用于大规模数据处理。

### Q9：如何调整跨镜拼接的松紧度？

**A**：通过以下参数调节：(1) `stitching.min_appearance_score`：降低则更多候选对通过外观过滤；(2) `stitching.max_time_gap_seconds`：增大则允许更大时间间隔的连接；(3) 调整 `stitching.weights` 中各分项权重，如增大 `plate` 权重使车牌匹配更重要。

### Q10：系统支持哪些目标类别？

**A**：当前支持三类：`vehicle`（机动车）、`pedestrian`（行人）、`non_motor_vehicle`（非机动车）。机动车有车牌和道路约束，拼接更可靠；行人无强身份信息，依赖时空可达性和外观特征。

### Q11：Docker 部署时 GPU 不可用

**A**：确认已安装 NVIDIA Container Toolkit，并在 `docker-compose.yml` 中配置了 GPU 资源预留。运行 `docker compose run backend nvidia-smi` 检查 GPU 是否可见。

### Q12：如何导出轨迹结果？

**A**：系统通过 `TrackManager.save_to_json()` 接口将 Tracklet 数据持久化为 JSON 文件，默认保存至 `output/` 目录。回溯结果包含在 `TrajectoryResult` 数据结构中，可通过 API 接口获取或直接从内存中导出。

---

## 8. GPU 环境诊断

系统提供一键 GPU 环境诊断工具，可自动检测 NVIDIA 驱动、CUDA、PyTorch GPU 支持、Docker NVIDIA Runtime 等关键组件，并在检测失败时给出修复命令。

### 8.1 Windows 一键诊断

双击运行 `scripts/check_gpu_env.bat` 即可执行完整诊断。

### 8.2 命令行诊断

```bash
# 基础诊断
python scripts/check_gpu_env.py

# 详细诊断（含显存详情）
python scripts/check_gpu_env.py --verbose
```

### 8.3 诊断输出示例

正常情况：

```
==================================================
  GPU 环境诊断报告
==================================================

[✓] NVIDIA 驱动: 581.15
[✓] CUDA 版本: 13.0
[✓] PyTorch CUDA: PyTorch 2.11.0+cu128 + CUDA 12.8
[✓] GPU 设备: NVIDIA GeForce RTX 5070 (12.0GB)
[✓] Docker NVIDIA Runtime: 已配置

--------------------------------------------------
状态: 所有检查通过，GPU 环境就绪
```

异常情况（会自动给出修复命令）：

```
[✗] PyTorch CUDA: torch.cuda.is_available() = False
    → 修复: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 8.4 诊断项目说明

| 检测项 | 说明 | 失败时处理 |
|---------|------|------------|
| NVIDIA 驱动 | 检测 nvidia-smi 是否可用 | 安装/更新 NVIDIA 驱动 |
| CUDA 版本 | 检测 CUDA Toolkit 版本 | 安装 CUDA Toolkit |
| PyTorch CUDA | 检测 PyTorch 是否能使用 GPU | 重新安装带 CUDA 支持的 PyTorch |
| GPU 设备 | 检测 GPU 型号和显存 | 确认显存 ≥ 4GB |
| Docker NVIDIA Runtime | 检测 Docker GPU 支持 | 安装 NVIDIA Container Toolkit |

---

## 9. 区域配置与部署适配

系统默认配置为苏州市相城区，部署到其他区域时只需修改配置文件中的区域参数。

### 9.1 修改区域配置

编辑 `configs/default.yaml` 中的 `region` 段：

```yaml
region:
  name: "苏州市相城区"          # 改为目标区域名称
  center_lat: 31.37             # 目标区域中心纬度
  center_lon: 120.62            # 目标区域中心经度
  plate_prefix: "苏E"           # 目标区域车牌前缀（如"京A"、"沪C"、"粤B"）
  default_coordinate:           # 目标区域默认坐标
    latitude: 31.3621
    longitude: 120.6182
```

### 9.2 更新摄像头元数据

编辑 `configs/camera_metadata.yaml`，将摄像头列表、路段定义、邻接关系替换为目标区域的实际数据。每个摄像头需提供：

| 字段 | 说明 | 示例 |
|------|------|------|
| `camera_id` | 唯一编号 | `CAM_001` |
| `name` | 摄像头名称/位置描述 | `XX路-YY路路口东` |
| `latitude` | 纬度 | `31.3621` |
| `longitude` | 经度 | `120.6182` |
| `direction` | 朝向角度（正北 0°，顺时针） | `90.0` |
| `covered_road_segment` | 所属路段 ID | `SEG_001` |
| `lane_direction` | 车道方向 | `eastbound` |

### 9.3 部署检查清单

- [ ] 修改 `configs/default.yaml` 中 `region` 配置
- [ ] 替换 `configs/camera_metadata.yaml` 为实际摄像头数据
- [ ] 运行 `python scripts/check_gpu_env.py` 确认 GPU 环境就绪
- [ ] 确认模型权重已下载（首次运行时 ultralytics 自动下载 YOLO 权重）

---

## 10. 系统设计说明

本节说明系统在架构层面的关键设计决策及其对功能的影响，帮助用户正确理解系统输出的含义。

### 10.1 摄像头覆盖不连续（离散观测模型）

**设计决策**：系统面向道路监控摄像头，这类摄像头天然分布在路口和卡口，覆盖范围不连续。

**对输出的影响**：
- 摄像头之间的轨迹为**推断段**，非真实观测，存在不确定性
- 系统输出明确区分「观测段」（实线）和「推断段」（虚线），避免误导
- 推断段附带置信度评分，供人工研判参考

### 10.2 属性识别采用轻量级方案

**设计决策**：颜色识别基于 HSV 直方图匹配，车型识别基于宽高比启发式规则，以平衡推理速度和精度。

**对输出的影响**：
- 属性识别精度不及深度学习分类器，但推理速度快
- 在光照充足、遮挡较少的场景下效果较好
- 跨镜拼接时属性一致性仅作为辅助证据（权重可配置）

### 10.3 车牌 OCR 支持多后端

**设计决策**：内置简化版 OCR 用于定位车牌区域，精确字符识别依赖 EasyOCR 或 PaddleOCR。

**使用建议**：
- 生产环境建议安装 EasyOCR：`pip install easyocr`
- 或安装 PaddleOCR：`pip install paddleocr`
- 未安装时系统仍可运行，但车牌识别率会下降

### 10.4 行人跨镜拼接可靠性较低

**设计决策**：行人无车牌等强身份标识，跨镜拼接主要依赖时空可达性和外观特征。

**对输出的影响**：
- 行人轨迹的跨镜连接置信度通常低于车辆
- 长距离拼接误匹配率较高，建议结合人工复核
- 可通过调整 `stitching.weights.pedestrian` 参数控制拼接松紧度

### 10.5 单摄跟踪采用 ByteTrack 风格方案

**设计决策**：使用 IoU 匹配 + 卡尔曼滤波的轻量级跟踪，不依赖外观特征关联。

**对输出的影响**：
- 密集场景和频繁遮挡场景下可能出现 ID 切换
- 可通过 `tracking.max_age` 和 `tracking.min_hits` 参数调节跟踪灵敏度

### 10.6 城区速度模型简化

**设计决策**：旅行时间估算使用固定速度范围 [20, 80] km/h，未接入实时路况。

**对输出的影响**：
- 时间估算为合理范围而非精确值
- 在信号灯密集的城区可能偏乐观，在快速路段可能偏保守
- 速度范围可通过代码中的 `DEFAULT_MIN_SPEED` / `DEFAULT_MAX_SPEED` 调整

---

> **文档维护**：本文档随系统版本更新同步维护。如有疑问或建议，请联系项目开发团队。
