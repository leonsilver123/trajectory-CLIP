# 部署与启动说明（二期 T11）

> 生成：2026-09-12　全部步骤**本机实测通过**，失败过的坑一并记在 §5。

---

## 一、一条命令起全栈（后端 API + 前端 UI）

```bash
# 从项目根目录 H:\trajectory-CLIP 执行
H:/trajectory-CLIP/.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

打开 <http://localhost:8000> 即是前端页面，`/docs` 是 Swagger。

> **`--host` 怎么选**：
> - `127.0.0.1`（**默认推荐**）—— 仅本机可访问，局域网内其它机器连不上。
>   可用 `curl http://<你的局域网IP>:8000/health` 应返回 `000`（不可达）来验证。
> - `0.0.0.0` —— 监听所有网卡，**同局域网内任何人都能访问**（且本服务无鉴权）。
>   仅在确实需要多人/跨机访问时使用，并自行评估暴露风险。

> 🔴 **本服务没有任何鉴权，不要直接暴露到公网。**
>
> 两个容易忽略的暴露面（PLAN3-E2）：
> 1. **容器部署默认就是全网卡**：`Dockerfile.backend` 显式传 `--host 0.0.0.0`，
>    所以 `docker compose up` 之后，同网段任何人都能访问 8000 端口。
> 2. **`cors_origins` 默认 `["*"]`**：这**不是**安全措施 —— 它只影响浏览器跨域，
>    挡不住直连（curl / 脚本）。收紧它并不能替代鉴权。
>
> 若必须对外提供服务，最小方案是在前面加一层带认证的反向代理（网关 Basic Auth
> 或内网 ACL），而不是依赖本服务自身的配置。仓库里的 `docker/nginx.conf`
> 是一份反代样例，可按需在其上启用认证。

> **静态资源暴露面（已收窄）**：`/static` **只**挂载图片目录白名单
> （`api/main.py::_STATIC_IMAGE_DIRS`：aicity22_crops、aicity22_frames、
> cityflow_crops、crops、demo_detections、pipeline_test）。
> 早先这里挂的是整个 `output/`，会让 `cityflow_results.json`（全量数据）、
> `datastore/*.parquet`、`meta.sqlite` 可被直接下载。新增目录必须在白名单里显式登记。

`api/main.py::_mount_frontend()` 会把 `webapp/dist/` 挂到根路径：
- `/assets/*` → Vite 产物（JS/CSS），标准静态文件服务
- 其余路径 → 命中 `dist/` 真实文件就返回，否则回 `index.html`（SPA 兜底）
  故**直接刷新** `/backtrack`、`/trajectory`、`/dashboard` 不会 404
- `/api/*`、`/static/*` 下的未知路径**保持 404 语义**，不被前端路由吞掉
- **`dist/` 不存在时自动跳过**，API 行为与从前完全一致（向后兼容）

> ⚠️ **必须用 venv 的 python**。本机 `python` 解析到 `C:\Python314\python.exe`（base 环境），
> 那里没有 faiss，直接 `python -m uvicorn` 会 ImportError。见 §5。

---

## 二、首次部署：构建前端

```bash
cd webapp
npm install          # node_modules 已存在时可跳过
npm run build        # = tsc -b && vite build，产物在 webapp/dist/
```

产物实测：`dist/index.html`(437B) + `dist/assets/index-<hash>.js`(≈2.44MB) + `index-<hash>.css`(≈19KB)。

**开发模式**（改前端时用，走 vite 热更新，`/api` 由 vite proxy 转发到后端）：

```bash
cd webapp && npm run dev      # 默认 :5173
```

---

## 三、端到端联调验证（实测命令与预期输出）

```bash
BASE=http://127.0.0.1:8000

# 0. 健康检查
curl -s $BASE/health                     # {"status":"ok","version":"1.0.0"}

# 1. 文本检索 → 拿 query_id 与候选
curl -s -X POST $BASE/api/v1/search/query \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"白色轿车","top_k":20}'

# 2. 确认目标（用上一步的 query_id + 某个候选的 instance_id）
curl -s -X POST $BASE/api/v1/confirm/target \
  -H 'Content-Type: application/json' \
  -d '{"query_id":"<query_id>","instance_id":"<instance_id>"}'

# 3. 跨镜回溯
curl -s -X POST $BASE/api/v1/backtrack/trace \
  -H 'Content-Type: application/json' \
  -d '{"query_id":"<query_id>","instance_id":"<instance_id>"}'

# 4. 前端与其深链接（都应 200 + text/html）
for p in / /backtrack /trajectory /dashboard; do
  curl -s -o /dev/null -w "$p -> %{http_code}\n" $BASE$p
done

# 5. 仪表盘
curl -s $BASE/api/v1/dashboard/stats
curl -s $BASE/api/v1/dashboard/cameras
```

### 3.1 真实浏览器渲染验证（本机已实测）

前端是 SPA，`curl` 只能证明 HTML/JS/CSS 可服务，**证明不了 React 真的挂载并取到数据**。
本机有 Chrome，可直接用 headless 模式验证（**无需装 playwright/selenium**）：

```bash
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
"$CHROME" --headless --disable-gpu --no-sandbox \
  --virtual-time-budget=15000 --dump-dom "$BASE/dashboard" > /tmp/dash.html
# 关键点：--virtual-time-budget 必须给够，否则 dump 发生在 fetch 完成前
```

**判据**：dump 出的 DOM 里应出现**真实统计数据**。本轮实测（2026-09-12）
`/dashboard` 渲染出的文本为：
`摄像头总数 46 · 在线：-- · 目标实例数（检测记录）68349 · 轨迹数（tracklet）926`，
且 **`2384`/`2070` 出现 0 次**（这两个是已修复缺陷 E6 的虚报值，见 `PLAN2.md`）。
四页 `/` `/search` `/backtrack` `/trajectory` 均渲染出 antd 布局、无白屏。

**本轮实测基线**（2026-09-12，uvicorn 8021 端口）：

| 步骤 | 结果 |
|---|---|
| `GET /health` | 200 `{"status":"ok","version":"1.0.0"}` |
| `POST /search/query` `{"query_text":"白色轿车"}` | 200，20 个候选，属性真实（`{'车型':'轿车','颜色':'白色'}`） |
| `POST /confirm/target` | 200，`status="confirmed"` |
| `POST /backtrack/trace` | 200，强身份，摄像头序列 `['c040','c036','c039','c038']`，4 节点 / 3 推断段 |
| `GET /backtrack` `/trajectory` `/dashboard` | 均 200，`text/html`（SPA 兜底生效） |
| `GET /dashboard/stats` | 200，`camera_count=46`、`tracklet_count=926`、`instance_count=68349` |
| `GET /dashboard/cameras` | 200，46 路，带真实 GPS/方向/场景/检测计数 |

---

## 四、配置与数据依赖

| 项 | 路径 | 说明 |
|---|---|---|
| 总配置 | `configs/default.yaml` | 阈值/权重唯一来源；`TRAFFIC__SECTION__KEY` 环境变量可覆盖 |
| 摄像头权威元数据 | `configs/cityflow_camera_metadata.yaml` | 46 路，含真实 GPS/方向/场景/分辨率 |
| ⚠️ 精简索引 | `configs/camera_metadata.yaml` | **只有 `scenes` 段、没有 `cameras` 段**，别单独拿它取 GPS（曾是缺陷 E6 的根因） |
| 检测与轨迹 | `output/cityflow_results.json` | 68349 检测 / 926 单摄轨迹 |
| 特征索引 | `output/clip_vectors.faiss`（68349×512）、`output/reid/reid_vectors.npy`（68349×2048） | 与 detections 行序对齐 |
| 模型权重 | `models/clip_cn_vit-b-16.pt`、`models/veri_sbs_R50-ibn.pth`、`models/pulc/…`、`yolov8x.pt` | — |

---

## 五、本机踩过的坑（**起服务/跑测试必读**）

1. **`python` 不是 venv 的** —— 解析到 `C:\Python314\python.exe`，无 faiss。
   一律显式用 `H:/trajectory-CLIP/.venv/Scripts/python.exe`。

2. **不要设 `PYTHONNOUSERSITE`** —— 会让 CLIP 静默归零（`clip_score` 全 0 而不报错），
   排查起来极费时。只用 `PYTHONPATH`。

3. **中文日志乱码** —— 控制台默认编码非 UTF-8，加 `PYTHONIOENCODING=utf-8`。

4. **pytest 起不来** —— 用户级插件 `langsmith` 依赖不全会导致测试会话直接失败。
   加 `-p no:langsmith_plugin`：
   ```bash
   H:/trajectory-CLIP/.venv/Scripts/python.exe -m pytest -q -p no:langsmith_plugin
   ```
   **当前基线：5 failed / 406 passed**（5 条均为历史遗留的 `test_camera_manager` GPS/nearby 用例，
   与前端/检索改动无关）。**任何超过 5 条失败的改动都算回归。**

5. **base 环境 `typing_extensions` 遮蔽** —— `F:\Anaconda_envs\envs\py313\Lib\typing_extensions.py`
   （旧版）会遮蔽 venv 的 4.16.0，导致 `fastapi/anyio` 导入失败（起服务同样挂）。
   已在 venv 的 `sitecustomize.py` 里把 venv site-packages 提到 `sys.path` 最前修好。
   **不要**用 `PYTHONNOUSERSITE` 去修这个问题（见第 2 条）。

---

## 六、Docker（未实测 —— 本机未起 Docker）

```bash
docker compose up -d      # 只有一个服务：backend :8000
```

**前端不再单独起容器。** 2026-09-21 起前端统一为 `webapp/`（React），
其构建产物由后端同源托管 —— 访问 `http://<host>:8000/` 即是完整界面。
原先的 Streamlit 服务（`frontend/`、`Dockerfile.frontend`、8501 端口）已删除，
Qdrant 服务也已注释停用（零消费者，见 `docker-compose.yml` 内的说明）。

已验证 / 未验证，分开说：

| 项 | 状态 |
|---|---|
| `docker-compose.yml` 可解析、无悬空 `depends_on` | ✅ 已用 yaml 解析校验 |
| `services` 仅剩 `backend` | ✅ 同上 |
| `Dockerfile.backend` 的 `COPY . .` 会带上 `webapp/dist/`（2.4MB） | ✅ 已核对 `.dockerignore` 未排除它 ⇒ 容器起来即有 UI |
| `.dockerignore` 排除 `node_modules/` 与 `third_party/` | ✅ 已补（此前未排除，`COPY . .` 会把数百 MB 依赖塞进构建上下文） |
| **镜像真的能构建并跑起来** | ❌ **未实测**（本机没起 Docker）。首次部署请自行验一遍 |

> ⚠️ `models/` 与 `output/` 被 `.dockerignore` 排除，镜像内**不存在**模型与索引，
> 必须由 compose 用 volume 挂载进去；只排除不挂载会让 CLIP 加载与 FAISS 读取失败并静默降级。
>
> ⚠️ 记得先构建前端产物再 build 镜像：`cd webapp && npm ci && npm run build`。
> 不构建的话 `dist/` 不存在，后端会**跳过**前端托管、只提供 API（这是有意的向后兼容行为）。
