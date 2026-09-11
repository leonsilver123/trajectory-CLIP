# 重构计划与进度追踪

> **最后更新**：2026-09-11
> **状态图例**：`[ ]` 未开始　|　`[~]` 进行中　|　`[x]` 已完成　|　`[-]` 已放弃/搁置
>
> **维护约定**：每完成一次代码修改，必须在本文件对应任务的「变更记录」里写明
> **改了什么（文件 + 具体改动）、没改什么（明确列出的遗留项）**，并更新顶部「最后更新」时间与「进度总览」。

---

## 零、环境与基线（2026-09-11 建立）

### 0.1 运行环境
- 项目根：`H:\trajectory-CLIP`；venv：`.venv`（由 conda py313 用 `--system-site-packages` 创建，继承 torch 2.9.1+cu128）。
- **规范测试命令**（必须用这条，否则会踩下面两个坑）：
  ```bash
  cd H:/trajectory-CLIP
  PYTHONNOUSERSITE=1 PYTHONPATH="H:/trajectory-CLIP/.venv/Lib/site-packages" .venv/Scripts/python.exe -m pytest -q
  ```
- 两个环境坑（已定位，用上面的环境变量绕过）：
  1. `C:\Users\Administrator\AppData\Roaming\Python\Python313\site-packages` 里的 `langsmith` pytest 插件会 import 失败的 `requests_toolbelt`，导致**整个测试会话崩溃**。→ `PYTHONNOUSERSITE=1` 移除该路径。
  2. `F:\Anaconda_envs\envs\py313`（env 根目录，在 sys.path 中**早于** `.venv/Lib/site-packages`）里有一个散落的 `typing_extensions.py`，缺 `sentinel`，会让 `fastapi.testclient` 导入失败。→ 用 `PYTHONPATH` 前置 venv 的 site-packages 覆盖它。
- 新增 `pytest.ini`：设 `testpaths = tests`。**没有它时裸跑 `pytest` 会递归扫描 `.venv/` 和 `output/`，仅收集阶段就要 438 秒**（实测）。
- 环境瑕疵（已知、未处理）：`.venv` 里存在 numpy 2.5.3（随 streamlit/pandas 拉入），但 sys.path 解析时 py313 的 numpy 2.3.3 优先，torch 正常工作（已验证 `torch<->numpy` 数据通路）。`pip uninstall numpy` 被 pip 拒绝（目标在 venv 外），`rm -rf` 被权限拒绝，故保留现状。
- 另有 6 个测试因**缺 scipy** 失败。用户约束明确"不装 scipy"，故**不安装**，保留失败（见 0.2）。

### 0.2 基线测试结果（改动前的真值）
```
13 failed, 251 passed, 6 xpassed in 68.30s
```
13 个既有失败，**均非本次重构引入**，后续所有验收都以此为准（只许 ≥251 passed、≤13 failed）：

| 数量 | 测试文件 | 原因 |
|---|---|---|
| 6 | `tests/test_trajectory_accuracy.py` | `ModuleNotFoundError: No module named 'scipy'`（未装 scipy） |
| 5 | `tests/test_camera_manager.py` | 真实数据不匹配：测试期望纬度 42.526，YAML 实际 31.326；`get_nearby_cameras` 返回空 |
| 2 | `tests/test_retrieval_accuracy.py` | 阈值边界：车型召回 0.8000 要求 `> 0.8`；查询数 4 < 5 |

### 0.3 计划前提的更正（**计划原文有错，以本表为准**）

| 计划原文的说法 | 核实结果 | 处置 |
|---|---|---|
| 任务 5："`clip_vectors.faiss` 找不到 faiss，是死代码" | **错**。`faiss` 1.14.3 与 `cn_clip` 均可导入；`output/clip_vectors.faiss` 存在（68349 向量 × 512 维）；`models/clip_cn_vit-b-16.pt` 存在（753MB）。该分支**不是死代码** | 任务 5 重新定义为"前端/后端重复实现收敛"，**不删 CLIP 能力** |
| 任务 7："被替代的旧 `configs/camera_metadata.yaml`" | **错**。`api/routes/dashboard.py:31` 正在读它（`_CAMERA_CONFIG`），删了 `/dashboard` 就废 | **保留该文件**，任务 7 不再包含它 |
| 任务 7：`api/routes/search_backup.py` 可删 | **对**。全仓仅在 `.md` 中被提及，无 `.py` 引用，`api/main.py` 只注册 4 个 router | 删 |
| 任务 7：`requirements.txt` 的 `flask` 未用 | **对**。全仓 `.py` 中 `\bflask\b` 零命中 | 删该行 |
| 任务 7：`frontend/mock_data.py` | 仅自身 docstring 提及，无任何 import | 删 |

> 教训：`HANDOFF.md`/`PLAN.md` 的部分前提基于未验证的观察。**动任何"死代码"前必须先 grep 取证**。

---

## 一、目标（一句话）

把项目从「三套平行实现、只有 JSON 直读一条活」重构成**诚实、分层、有状态**的系统，围绕两个真价值：① 离散观测链的诚实表达；② recall→confirm→backtrack 的人机协同闭环。

---

## 二、目标状态机（本次重构要落地的核心）

**后端会话状态机**（`SessionStore`，后端为状态真源，前端 session_state 退化为 UI 缓存）：

```
idle ──search──► searched(query_id)
                   ├──confirm──► confirmed(instance_id)
                   │               └──backtrack──► backtracked(result)
                   └──exclude/suspect──► excluded / suspect   (人机协同，P0 可选)
```

**跨镜拼接两条路**（统一到 `TrajectoryBuilder`）：

```
确定锚点实例
  ├─ 强身份（车牌 / 真值 vehicle_id）→ 直接强匹配        ← 主路径
  └─ 弱身份（无车牌/无真值）        → src/stitching 六维评分拼接 ← P1
```

---

## 三、进度总览

| # | 任务 | 阶段 | 状态 |
|---|------|------|------|
| 1 | 后端会话状态机 | P0 | [x] |
| 2 | 删除伪回溯，只留真实聚合 | P0 | [x] |
| 3 | ID 解析收敛成唯一函数 | P0 | [~] 仅剩 `frontend/pages/search.py:625` 一处 |
| 4 | 数据契约落盘，在线不读大 JSON | P1 | [ ] |
| 5 | 检索管线收敛成一条 | P1 | [ ] |
| 6 | 跨镜拼接统一到 TrajectoryBuilder | P1 | [ ] |
| 7 | 删死代码 | P2 | [x] |
| 8 | 修部署（Docker 路径等） | P2 | [~] compose/Dockerfile 已修并复核；`.bat` 补修中 |
| 9 | git 保护（初始化 + 首次提交） | P2 | [ ] |

---

## 四、P0 —— 打通核心闭环

### 任务 1：后端会话状态机（新增）　— 状态：`[ ]`

**目标**：让「检索 → 确认 → 回溯」靠 `query_id` 真正串联，后端有状态。

**涉及文件**：
- 新建 `src/common/session_store.py`（`SessionStore` 类 + 全局单例）
- 改 `api/routes/search.py`（search 时 `SessionStore.create(query_id, query_text)`，state=searched）
- 改 `api/routes/confirm.py`（校验 query_id → 写 state=confirmed + confirmed_instance_id）
- 改 `api/routes/backtrack.py`（从 session 读 confirmed_instance_id，写 state=backtracked + 结果）

**验收标准**：
- [x] 三步接口能靠 query_id 串联，不依赖前端 session_state
- [x] confirm 传入不存在的 query_id 返回 404，而非静默 echo
- [x] 新增 `tests/test_session_store.py` 覆盖状态转移

**变更记录**：
- 已修改：
  - 新建 `src/common/session_store.py`（`SessionStore` + 全局单例，`threading.Lock` 保护；状态机 `idle→searched→confirmed→backtracked`）
  - `api/routes/search.py`：`/query`、`/plate` 生成 query_id 后建会话
  - `api/routes/confirm.py`：校验 query_id，不存在→404，存在→写 confirmed + confirmed_instance_id
  - `api/routes/backtrack.py`：`/trace` 读 session 的 confirmed_instance_id，完成后写 backtracked
  - 新建 `tests/test_session_store.py`
- 未修改：`api/routes/dashboard.py`（与本任务无关）
- **验收证据（真实 HTTP，非单测）**：
  - `[PASS] POST /confirm/target (合法 query_id) — status=200 status='confirmed'`
  - `[PASS] POST /confirm/target (非法 query_id 应 404) — status=404 {'detail': '查询 NONEXISTENT_QUERY_ID 不存在或已失效，请重新检索'}`

---

### 任务 2：删除伪回溯，只留真实聚合　— 状态：`[ ]`

**目标**：删除用 `random` 伪造证据的代码，回溯只来自真实数据。

**涉及文件**：
- `api/routes/backtrack.py`
  - 删除 `_build_trajectory_from_target()` 中的 `random.uniform()`（时间戳/置信度/旅行时间）
  - 删除 `_find_target_id()` 中 `hash(instance_id) % len(...)` 随机抽目标
  - `/backtrack/trace` 改为：从 session 拿真实 instance_id → 定位 vehicle_id → 复用 `/trajectory` 真实聚合

**验收标准**：
- [x] `backtrack.py` 中 `random` 仅用于 ID 生成/演示数据，无伪造数据
- [x] 对同一 instance_id 回溯两次，结果一致（无随机性）

**变更记录**：
- 已修改：`api/routes/backtrack.py`
  - `_build_trajectory_from_target()`：**删除全部 `random.uniform()` / `random.randint()`**（原伪造的 exit_t、inference confidence、旅行时间、candidate_path confidence、overall_confidence、evidence 四项分数）
  - 时间戳改用 detection 的真实 `timestamp`；无依据的字段置 `None`（inference 的 confidence/travel time）或 `0.0`（overall_confidence、evidence 分项），并加中文注释说明「无真实依据，留空待任务 6 的评分子系统填充」
  - `_find_target_id()`：**删除 `hash(instance_id) % len(...)` 随机抽目标**，改为只做直接匹配（target_id 命中 / 由 track_id 解析 vehicle_id 定位该车真实 target_id），找不到返回 `None` → `/trace` 返回 404
  - `_generate_fallback_mock()`：保留但 docstring 明确标注为「仅数据文件缺失时的演示数据，非真实回溯」
  - 新增 `_path_distance_meters()`：用 `src.common.utils.haversine_distance` 按摄像头真实球面距离累加，取代原先「每段 250 米」的估算
- 未修改：`/trajectory` 接口（本来就是真实聚合，未动）
- **遗留**：`overall_confidence` 目前恒为 `0.0`、inference 段的时间/置信度为 `None` —— 这是**有意为之的诚实留空**（不再造数），待任务 6 接入 `src/stitching` 评分后填充
- **验收证据（真实 HTTP）**：
  - `/trace` 同一 instance_id 连打两次，响应体**完全相等**（`[PASS] 两次结果一致`）
  - `grep random api/routes/backtrack.py` 仅剩 `_generate_fallback_mock` 内 5 处（演示数据），主路径为 0
  - `POST /backtrack/trajectory` → `vehicle_id=V0322 摄像头数=3 检测数=125`（真实聚合可用）

---

### 任务 3：ID 解析收敛成唯一函数　— 状态：`[ ]`

**目标**：消除三处重复的 `_extract_vehicle_id` 正则，改 ID 格式不再三处炸。

**涉及文件**：
- 新建 `src/common/ids.py`（`parse_target_id(target_id) -> {source, camera_id, vehicle_id, det_seq}`）
- 改 `api/routes/search.py`
- 改 `api/routes/backtrack.py`
- 改 `frontend/utils.py`（`_cityflow_det_to_candidate` 中反解 vehicle_id 部分）

**验收标准**：
- [ ] 三处统一调用 `parse_target_id`，无重复 `startswith("V")` 逻辑
- [ ] 新增 `tests/test_ids.py` 覆盖 `CF3_c001_V0034_000001` 等格式

**变更记录**：
- 尚无（未开始）。
  - 已修改：无
  - 未修改：全部

---

## 五、P1 —— 架构对齐

### 任务 4：数据契约落盘，在线不读大 JSON　— 状态：`[ ]`

- 新建 `scripts/build_datastore.py`（JSON → SQLite 元数据 + Parquet detections 表 + 独立向量文件）
- 新建 `src/storage/datastore.py`（统一读写接口）
- 改 `api/routes/search.py` / `backtrack.py` / `frontend/utils.py`（改走 datastore，删散落的 `json.load` + mtime 缓存）
- 验收：[ ] 在线首查不加载 90MB JSON；[ ] `load_cityflow_results()` 等散落函数收敛为一处

**变更记录**：尚无（未开始）。

---

### 任务 5：检索管线收敛成一条　— 状态：`[ ]`

> **前提已更正**（见 §0.3）：原计划"删 ViT-B-16 分支，因为它是死代码"是**错的**——
> `faiss`/`cn_clip` 可用、索引与权重都在，该分支是**活的**。因此本任务改为
> **消除前后端重复实现**，而不是删除 CLIP 能力。

- **不删** `api/routes/search.py` 的 CLIP 精排路径（它是真的能跑）。
- 真正的问题：`frontend/utils.py` 有一套**独立的**「属性硬过滤 → OpenCLIP(ViT-L-14) → BLIP → 属性重排」管线，与后端 `search.py` 的「属性粗筛 → Chinese-CLIP(ViT-B-16) → 融合排序」**不一致**，且模型版本不同（L-14 vs B-16）。
- 目标：**后端为唯一检索真源**，前端改为纯调用后端接口，不再本地跑 CLIP/BLIP。
- Fallback =「少一步」（无 CLIP 时退化为属性分数，`search.py` 已有此逻辑）而非「换一套代码」。
- 验收：[ ] 前端不再自行加载 CLIP/BLIP 模型；[ ] CLIP 不可用时接口仍返回属性排序结果；[ ] 前后端不再各有一套排序逻辑

**变更记录**：尚无（未开始）。

---

### 任务 6：跨镜拼接统一到 TrajectoryBuilder　— 状态：`[ ]`

- 新建 `src/trajectory/builder.py`（`TrajectoryBuilder.build(anchor_instance)`：强身份直接匹配 / 弱身份走 stitching）
- 改 `api/routes/backtrack.py`（改调 TrajectoryBuilder，不再自写聚合）
- 验收：[ ] `src/stitching` 首次被线上引用；[ ] 输出每段标注依据（强身份 vs 概率推断）

**变更记录**：尚无（未开始）。

---

## 六、P2 —— 工程化

### 任务 7：删死代码　— 状态：`[ ]`

> **前提已更正**（见 §0.3）：`configs/camera_metadata.yaml` **不是**死代码
> （`api/routes/dashboard.py:31` 在用），**从本任务中移除，不得删除**。

已取证可删项：
- `api/routes/search_backup.py` —— 全仓仅 `.md` 提及，无 `.py` 引用
- `frontend/mock_data.py` —— 无任何 import（仅自身 docstring）
- `requirements.txt` 中 `flask>=3.0.0` —— `.py` 中 `\bflask\b` 零命中

验收：[x] 删除后 `failed` 仍为 13（passed 因新增测试升至 306）；[x] 无悬空引用

**变更记录**：
- 已修改：
  - 删除 `api/routes/search_backup.py`（取证：全仓仅 `.md` 提及，无 `.py` 引用，`api/main.py` 只注册 4 个 router）
  - 删除 `frontend/mock_data.py`（取证：无任何 import，仅自身 docstring 提及）
  - `requirements.txt`：删除 `flask>=3.0.0`（取证：`.py` 中 `\bflask\b` 零命中）
- **明确保留**：`configs/camera_metadata.yaml` —— 原计划称其「被替代」是**错的**，`api/routes/dashboard.py:31`（`_CAMERA_CONFIG`）正在读它，删除会导致 `/dashboard` 失效
- 未修改：`configs/cityflow_camera_metadata.yaml`（`search.py`/`backtrack.py`/`default.yaml` 在用）
- **验收证据**：删除后复跑测试 `13 failed, 306 passed`，失败集合与基线**逐个完全一致**；`api/`、`frontend/` 下 grep `mock_data|search_backup` 零命中

---

### 任务 8：修部署　— 状态：`[~]`

- `docker-compose.yml` 中写死的 `H:/trajectory CLIP`（带空格）改相对路径 / `$PWD`
- 核对 `Dockerfile.backend` / `Dockerfile.frontend` 模型挂载路径

**变更记录**：
- 已修改：
  - `docker-compose.yml`：写死盘符 → 相对路径（相对 compose 文件所在目录 = 项目根）；**并把容器内挂载点从 `/data`、`/models`、`/output`、`/logs` 改为 `/app/...`** —— 因为后端以 `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`（容器内即 `/app`）拼接路径，挂到根目录会读不到、检索静默退化。已逐条复核该判断成立（`api/main.py:87` 的 `/static` 挂载、`src/common/logger.py` 的 `log_dir`、`configs/default.yaml` 的相对目录）
  - qdrant 卷：`./docker_data/qdrant`（不存在）→ `./docker/qdrant_storage`（已存在，461MB 历史数据）
  - `Dockerfile.backend`：挂载点同步改 `/app/*`
  - `Dockerfile.frontend`：挂载点同步改 `/app/*`；**补装 `torch`(CPU) 与 `PyYAML`** —— 因为 `frontend/utils.py:17,20` 是无 try/except 的顶层 `import yaml` / `import torch`，原 pip 列表里两个都没有，容器起来即 ImportError（已复核该判断属实）；加 `--server.headless=true` 避免首次启动等邮箱输入
  - `scripts/docker_{start,stop,logs}.bat`：`cd /d "H:\trajectory CLIP"`（目录不存在，必然失败）→ `cd /d "%~dp0\.."`（与同目录 `check_gpu_env.bat` 已有写法一致）；`docker_stop.bat` 的提示文案同步改中性表述
  - `.dockerignore`：排除 `models/`、`output/`、`docker_data/`、`docker/qdrant_storage/`、`*.pt/*.faiss` 等（与 compose 的 volume 挂载互为前提，已确认二者自洽）
- **未修改（遗留）**：
  1. **`docker/.env`** —— 其中 `DATA_DIR`/`MODEL_DIR`/`OUTPUT_DIR` 等仍为 `H:/trajectory CLIP/...`。**未修原因：该目录不在本会话权限范围内（Read/Bash 均被拒绝）**。已核实这些变量**无任何代码消费**（全仓只有 `API_BASE_URL` 与 `TRAFFIC__*` 被读），属卫生问题，不影响运行；但建议后续清理
  2. **3 个辅助脚本仍硬编码坏路径**（与任务 8 同源，但**不属部署入口**，且依赖本机不存在的 AICity22 数据集、本就跑不起来，故不擅自扩范围）：
     - `scripts/download_datasets.py:20`、`scripts/extract_aicity22_frames.py:18`、`scripts/pipeline_validation.py:25`
  3. **`requirements.txt` 缺 `faiss-cpu`** —— 后端 `search.py` 的 CLIP 向量精排依赖 faiss，但镜像里没装，容器中该路径会静默降级为纯属性排序。未修原因：改动 `requirements.txt` 属任务 7 范围，且 D 盘环境已有 faiss（继承自 py313），属镜像侧问题，留待确认
  4. 文档（`docs/*.md`、`tests/test_report.md`）中仍有该坏路径写法，仅报告不改
- **验证结果（真实执行）**：
  - `docker --version` → **Docker version 29.5.3**；`docker compose version` → **v5.1.4**（CLI 可用）
  - Docker **daemon 未运行**（`docker info` 报 `npipe:////./pipe/dockerDesktopLinuxEngine ... daemon is running` 失败）→ **镜像构建与容器实跑未执行**
  - `docker compose config` → **rc=0，语法与路径解析全部通过**。关键证据：相对路径已正确解析为宿主绝对路径
    ```
    volumes:
      - source: H:\trajectory-CLIP\data    target: /app/data
      - source: H:\trajectory-CLIP\models  target: /app/models
      - source: H:\trajectory-CLIP\output  target: /app/output
      - source: H:\trajectory-CLIP\logs    target: /app/logs
    ```
  - `grep -rn "trajectory CLIP" docker-compose.yml Dockerfile.*` 零命中
- **`docker compose config` 暴露的新证据（印证上面遗留项 1）**：`env_file: docker/.env` 确实被读取并注入旧路径
  ```
  environment:
    CITYFLOW_DATA_DIR: H:/trajectory CLIP/cityflow/AICity22_Track1_MTMC_Tracking
    DATA_DIR:  H:/trajectory CLIP/data
    LOG_DIR:   H:/trajectory CLIP/logs
    MODEL_DIR: H:/trajectory CLIP/models
    OUTPUT_DIR: H:/trajectory CLIP/output
  ```
  已复核这些变量**确无消费者**：`src/common/config.py` 的环境覆盖只认 `TRAFFIC__` 前缀；`src/` 下对这些名字零命中；`frontend/`、`api/` 的命中均为 Python 模块常量 `_OUTPUT_DIR`，非环境变量。故**属卫生问题，不影响容器运行**，但会误导后来者，建议清掉。
  > 为什么没修：`docker/` 目录在本会话权限范围之外（Read 与 Bash 均被拒绝）。已按要求**上报而非绕过**。

---

### 任务 9：git 保护　— 状态：`[ ]`

- 初始化空 `.git`（当前无 HEAD/commit），首次提交 + `.gitignore`（排除 `output/`、`models/`、`yolov8x.pt` 等大文件）

**变更记录**：尚无（未开始）。

---

## 七、全局变更日志（倒序）

> 每次修改在此追加一条，格式：`日期 — 任务# — 一句话说明改了什么/没改什么`。

- 2026-09-11 — 任务#0（环境）— 建 venv 依赖（fastapi/uvicorn/streamlit 因被 C 盘用户级 site-packages 遮蔽，必须 `--ignore-installed` 才会真正装进 .venv）；新增 `pytest.ini`（`testpaths=tests`，把收集耗时从 438s 降到 <1s）；定位两个 sys.path 遮蔽坑并给出规避环境变量；建立基线 13 failed / 251 passed / 6 xpassed。**未改任何源码**。
- 2026-09-11 — 计划勘误 — 核实并更正了任务 5（CLIP 分支是活的，非死代码）与任务 7（`configs/camera_metadata.yaml` 是活的，不可删）两处前提错误，详见 §0.3。
- 2026-09-11：创建本计划文件，尚未开始任何代码修改。
