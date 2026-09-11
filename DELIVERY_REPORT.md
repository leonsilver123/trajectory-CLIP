# 交付报告 — 交通风险感知子系统重构

> 完成时间：2026-09-12
> 进度唯一真源：`PLAN.md`（含逐任务变更记录与全部取证）
> 本文只报结论与证据，细节见 `PLAN.md`

---

## 一、总览

`PLAN.md` 的 **9 个任务全部完成**。

| # | 任务 | 状态 | 一句话结果 |
|---|------|------|-----------|
| 1 | 后端会话状态机 | ✅ | 靠 `query_id` 真正串联；非法 query_id 由静默 echo 改为 **404** |
| 2 | 删除伪回溯 | ✅ | 主路径 `random` 归零；同一 instance_id 两次回溯**响应体完全相等** |
| 3 | ID 解析收敛 | ✅ | 4 处重复反解收敛进 `src/common/ids.py` |
| 4 | 数据契约落盘 | ✅ | 统一 datastore（12.4MB vs 102.6MB）；回退路径已实测 |
| 5 | 检索管线收敛 | ✅ | 前端本地 CLIP/BLIP 管线删除（**-592 行**），检索只剩后端一处 |
| 6 | 跨镜拼接统一 | ✅ | **`src/stitching` 首次线上生效**；`/trace` 由单摄像头变为跨摄像头 |
| 7 | 删死代码 | ✅ | 删 3 项；`camera_metadata.yaml` 经核实后**保留** |
| 8 | 修部署 | ✅ | `docker compose config` rc=0；挂载点 `/app/*` 修正 |
| 9 | git 保护 | ✅ | 5 次提交，大文件全部挡住 |

**回归控制（全程红线）**：基线 `13 failed, 251 passed` → 终态 `13 failed, 358 passed`。
失败集合用 `comm` 与基线**逐个比对完全一致**（零新增失败）；`passed` 增加 107 个全部来自新增测试。

---

## 二、做了什么（含关键证据）

### 任务 1/2/3 —— 打通核心闭环（P0）

- 新增 `src/common/session_store.py`：线程安全的状态机（`idle→searched→confirmed→backtracked`，另有 `excluded`/`suspect`），后端成为状态真源。
- `api/routes/backtrack.py` **删除全部伪造数据**：`random.uniform()` 造的时间戳/置信度/旅行时间、`hash()` 随机抽目标全部清除；时间戳改用真实 detection 时间，算不出来的**诚实留空**而非造数。
- 新增 `src/common/ids.py`，替换 4 处重复的 vehicle_id 反解。

**验证（真实 HTTP，非单测）**：
```
[PASS] POST /confirm/target (非法 query_id 应 404) — status=404
[PASS] POST /backtrack/trace 两次结果一致 — 一致
[PASS] POST /backtrack/trajectory — vehicle_id=V0322 摄像头数=3 检测数=125
```
`grep 'startswith("V")'` 全仓 **零命中**。

### 任务 5/6 —— 架构对齐（P1）

- **任务 5**：`frontend/utils.py` **1374 → 703 行**，删除整套本地检索实现（OpenCLIP ViT-L-14 + BLIP + 自读 JSON 的属性重排）。前端检索改为**只有后端一个真源**。
- **任务 6**：新增 `src/trajectory/builder.py`（1343 行），强身份按 `vehicle_id` 聚合、弱身份走 `src/stitching` 六维评分（离线 2700 行代码**首次被线上引用**）。

**`/trace` 的实测变化（这是本次最直观的改进）**：
```
改前: camera_sequence = ['c040']                          ← 单摄像头
      inference_segments = []                             ← 恒空
      overall_confidence = 0.0                            ← 硬编码

改后: camera_sequence = ['c004','c005','c003','c002','c001']   ← 跨 5 摄像头
      inference_segments = 4 条（含六维得分明细）            ← 有真实内容
      overall_confidence = 0.3411                          ← 真实计算
      每段新增 basis 标注（strong_identity / probabilistic_inference）
```
`/trajectory` **向后兼容**：原有 9 个顶层键全在，`basis` 为新增字段。

### 任务 4 —— 数据契约落盘（P1）

统一读取层 `src/storage/datastore.py`，5 处散落 JSON 读取点全部收敛。**叠加式设计**：datastore 存在则用，不存在则回退 JSON。

**回退路径已实测演练**（这是本任务最重要的安全保证）：
```
改名前: /dashboard/health → data_source="datastore"
mv 走 : data_source="json"  →  全量冒烟 14/14 通过（跑在回退路径上）
mv 回 : data_source="datastore"
```

### 任务 7/8/9 —— 工程化（P2）

- 删 `api/routes/search_backup.py`、`frontend/mock_data.py`、`requirements.txt` 的 flask（均经 grep 取证无引用）。
- `docker-compose.yml` 写死且**根本不存在**的 `H:/trajectory CLIP`（带空格）改相对路径；容器内挂载点由 `/models` 等改为 **`/app/*`**（因为代码以 `_PROJECT_ROOT` 拼路径，挂错位置会读不到、静默退化）；`Dockerfile.frontend` 补装 `torch`/`PyYAML`（顶层无 try/except 的 import，缺则容器起来即崩）。
- `git init` + 5 次提交；`.gitignore` 挡住 `output/`/`models/`/`cityflow/`/`qdrant_storage` 等大目录。

---

## 三、没做什么（明确列出）

1. **两套 CLIP 特征空间未统一** —— 见遗留 #1。需重跑特征提取管线，成本高。
2. **前端仍系统性"编造数据"** —— 见遗留 #2，共 9 处 + 编造描述 + 伪造经纬度。
3. **弱身份拼接路径准确率差** —— 见遗留 #3。通路打通但匹配错误。
4. **上游 `_count_possible_paths` 无界 DFS 未修** —— 见遗留 #4，只做了绕过。
5. **`docker/.env` 未修改** —— 该文件所在目录**不在本会话权限范围内**（Read 与 Bash 均被拒），已按要求上报而非绕过。其中变量经核实**无任何代码消费**，属卫生问题。
6. **`configs/camera_metadata.yaml` 未删除** —— 经核实它仍被 `dashboard.py` 引用（原计划称其"可删"不成立，但我的"删了就废"也说过头了，准确说法是"被引用但非 load-bearing"）。结论：不删。
7. **3 个辅助脚本仍硬编码坏路径** —— `scripts/download_datasets.py:20`、`extract_aicity22_frames.py:18`、`pipeline_validation.py:25`。与任务 8 同源但不属部署入口，且依赖本机不存在的 AICity22 数据集，故未擅自扩范围。
8. **Docker 镜像未实际构建** —— Docker daemon 未运行（CLI 可用，`docker compose config` 通过）。**以上 Docker 改动为静态审查 + 配置解析级验证，未经容器实跑**。
9. **`output/` 下临时文件未清理** —— 删除被权限系统拒绝。
10. **未更新 `CLAUDE.md`/`HANDOFF.md`** —— `CLAUDE.md:80` 仍记载已删除的 `search_backup.py`，现已过时。

---

## 四、验证结果（真实输出）

### 4.1 测试
```
基线（任何改动之前）: 13 failed, 251 passed, 6 xpassed
终态                : 13 failed, 358 passed, 6 xpassed

失败集合 comm 比对: 完全一致（零新增失败）
```
13 个失败**全部是既有问题**，与本次重构无关，且按要求**未修**：

| 数量 | 文件 | 原因 |
|---|---|---|
| 6 | `test_trajectory_accuracy.py` | `ModuleNotFoundError: scipy`（用户约束不装 scipy） |
| 5 | `test_camera_manager.py` | 测试期望纬度 42.526，YAML 实际 31.326；`get_nearby_cameras` 返回空 |
| 2 | `test_retrieval_accuracy.py` | 阈值边界：车型召回 0.8000 要求 `> 0.8`；查询数 4 < 5 |

### 4.2 前后端冒烟测试
`scripts/smoke_test.py`（本次新增，含真实拉起 Streamlit 探活）：
```
15/15 全部通过
```
覆盖：健康检查、文本检索、车牌检索、确认（含**非法 query_id 必须 404**）、跨镜轨迹、`/trace` **两次调用响应体完全一致**、仪表盘三接口、前端导入/编译、**真实 Streamlit 启动并响应健康检查**。

### 4.3 端到端数据流验证
```
/search/query   → 200，候选带真实 clip_score 0.4275（CLIP 融合生效）
/search/plate   → 200
/confirm/target → 200 confirmed（合法 query_id）/ 404（非法）
/backtrack/trajectory → 200 vehicle_id=V0034 摄像头数=5 检测数=206
/backtrack/trace      → 200 跨 5 摄像头 + 4 条推断段 + basis 标注
/dashboard/{stats,cameras,health} → 均 200
/dashboard/health → data_source="datastore"
```

### 4.4 性能（实测）
| 项 | 改前 | 改后 |
|---|---|---|
| 数据读取（线上 `load_results()` 热读中位） | 1.250s | **0.482s（约 2.6×）** |
| 落盘体积 | 102.63 MB | **12.37 MB（小 8.3×）** |

> 说明：原型阶段曾测得「28 倍」，那是**裸 parquet 读取**的对比，**对线上不成立** —— 线上必须物化 68349 个同结构 dict，这部分成本两条路都要付。**以 2.6× 为准。**

---

## 五、遗留问题（按优先级）

### 🔴 #1 项目存在两套不兼容的 CLIP 特征空间
| 来源 | 维度 | 模型 | 覆盖率 | 谁在用 |
|---|---|---|---|---|
| `output/clip_vectors.faiss` | 512 | ViT-B-16 | **100%**（68349） | `api/routes/search.py`（检索主路径） |
| 检测内联 `clip_image_vector` | 768 | ViT-L-14 | **1.39%**（949） | `src/trajectory/builder.py` |
| `configs/default.yaml:124` | 声明 768 | 声明 ViT-L-14 | — | **与实际检索路径不符** |

维度都不同，**向量不可混用**。导致拼接的外观分对 98.6% 的检测只能是中性 0.5。**需重跑特征提取管线统一**，成本较高。

### 🔴 #2 前端仍系统性"编造数据"（与任务 2 同类，只删了后端那一半）
| 位置 | 行为 |
|---|---|
| `frontend/utils.py:164,180` | 无依据时**凭空填** `0.8`/`0.85` 置信度 |
| `frontend/utils.py:417,444,466,497`、`pages/timeline.py:439,531,602` | 共 9 处凭空填入 `0.9`/`0.7`/`0.5` |
| `frontend/utils.py:193-194` | 凭空写 `entry_description="从画面进入"`，**而后端已诚实返回 `""`** |
| `frontend/pages/map_view.py:73-74` | 摄像头无坐标时**按索引伪造经纬度**，好让点画在地图上 |

**为什么重要**：这些编造值**掩盖了本轮刚加好的诚实兜底** —— `confidence_color()` 在 `None` 时返回"未知"，但上游在值到达之前**已把 `None` 换成 `0.9`**，于是"未知"路径永远触发不了，界面显示一个看着像真实测量的数字。**与项目"诚实"目标直接冲突，建议列为后续第一优先级。**

### 🟠 #3 弱身份拼接路径「通路已连通、但匹配是错的」
实测 `mode="stitch"` 产出 `c004→V0001`、`c002→V0008`，**都不是目标车 V0034**。两个根因（均有实测支撑）：
- 内联 CLIP 区分度弱（不同车 cos 0.88–0.90，本车 0.942）
- **同一辆车 `vehicle_type` 跨镜标注不一致**，导致硬属性否决**惩罚正确轨迹、奖励错误车辆**

**数据质量问题**：`V0034` 的属性标注跨摄像头不一致，**甚至单个摄像头内部自相矛盾**（c001 混 SUV/轿车，c003 混 面包车/轿车）。这**同时损害检索的属性过滤与拼接的属性评分**，非代码可修。
**结论：生产使用应依赖强身份（或车牌），弱身份结果当前不可信。**

### 🟠 #4 上游 `src/stitching/scoring.py` 的 `_count_possible_paths` 是无界 DFS
实测**单对摄像头 4–23 秒**（c001→c002 23.16s）。当前在 `builder.py` 内用子类加上界绕过（线上降至 0.8s），**上游文件未改** → **隐患仍在，任何其他调用方都会踩**。

### 🟡 #5 其他
- `/trace` 的 `inference_segments[].basis` 标为 `strong_identity`，语义上推断段更宜标 `probabilistic_inference`
- `evidence.linkage_confidence` 为 `None`（未填充）
- `api_backtrack` 在 `frontend/` **只被 import、从未被调用** —— 前端实际走 `/trajectory`，故本轮 `/trace` 的跨镜改进**不会自动反映到前端**
- 前端筛选面板「场景/摄像头」下拉框**不生效**（只收集和显示，从未进入过滤逻辑；后端也不收这两个参数）—— 既有缺陷
- 评分维度大面积为空：`reid` **全空**（但 `weights.vehicle.reid=0.15` 权重仍生效，实际挂在 CLIP 分上，**存在语义偏差**）；`plate` **全空**（导致 `is_valid` 校验形同虚设）
- `docker/.env`、3 个辅助脚本仍有 `H:/trajectory CLIP` 坏路径
- Docker 镜像内缺 `cn_clip`/`faiss`（`requirements.txt` 未声明）→ **容器内 CLIP 精排会静默退化**
- `output/` 下遗留临时文件与 `_backup_task4/`
- `CLAUDE.md:80` 记载已过时

---

## 六、如何运行

### 6.1 环境（**两个变量有讲究，务必看**）
```bash
cd H:/trajectory-CLIP
```
**起服务**（关键：**不要**加 `PYTHONNOUSERSITE=1`）：
```bash
PYTHONPATH="H:/trajectory-CLIP/.venv/Lib/site-packages" \
    .venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
**跑测试**（测试额外需要一个变量禁掉坏插件）：
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="H:/trajectory-CLIP/.venv/Lib/site-packages" \
    .venv/Scripts/python.exe -m pytest -q
```
**为什么**：
- `PYTHONPATH` 前置 venv 的 site-packages，用来压住 conda env 根目录里一个缺 `sentinel` 的残缺 `typing_extensions`（否则 fastapi/anyio 起不来）。
- **绝不能加 `PYTHONNOUSERSITE=1`**：`cn_clip` 只装在用户级 site-packages 里，加了这个变量会把它一起移除，**CLIP 检索会静默退化成纯属性排序**（`clip_score` 恒 0.0，接口仍返回 200，极难察觉）。**本轮我就踩过这个坑**，详见 `PLAN.md` §0.1。
- 测试的插件变量是因为用户级目录里有个坏掉的 `langsmith` pytest 插件会 import 缺失的 `requests_toolbelt`，导致整个测试会话崩溃。

### 6.2 前端
```bash
.venv/Scripts/python.exe -m streamlit run frontend/app.py --server.port 8501
```

### 6.3 冒烟测试
服务起来后另开终端：
```bash
.venv/Scripts/python.exe scripts/smoke_test.py --frontend
```
（脚本自带子进程环境构造，**无需**额外设变量）

### 6.4 Docker
```bash
docker compose up -d --build     # 或 scripts/docker_start.bat
```
首次到新机器需跑一次 `python scripts/build_datastore.py`（**不跑也能用**，走 JSON 回退，只是慢）。

---

## 七、附：本次复核中被推翻的判断（诚实性记录）

重构过程中，**子 agent 与实测推翻了我多处判断**，均已更正并记入 `PLAN.md`：

| # | 我原先的判断 | 实际情况 | 谁发现 |
|---|---|---|---|
| 1 | `search.py` 的 CLIP 分支是死代码（承 PLAN 原文） | **错**。faiss/cn_clip/索引/权重俱在，路径可用 | 我派活前 grep 取证 |
| 2 | `configs/camera_metadata.yaml` 删了就废 | **说过头**。它被引用但非 load-bearing（YAML 无 `cameras:` 键，走兜底分支） | 子 agent 非破坏性 A/B |
| 3 | `.dockerignore` 被覆盖、原内容可能丢失 | **错**。未覆盖，是我把观测时刻搞错 | 子 agent 举证（Write 成功即证明原文件不存在） |
| 4 | JSON 里没有 clip_vector | **错**。有 949 条 768 维内联向量 | 子 agent 实测反证 |
| 5 | `actual_travel_time` 是唯一需兜底的字段 | **错**。仅在 `auto` 下成立；`stitch` 下 `observation_nodes[].confidence` 同样为 None | 子 agent 拒绝照单接受，自行实测发现第二处崩溃 |
| 6 | 环境命令（含 `PYTHONNOUSERSITE=1`） | **错**。会静默关掉整个 CLIP 检索路径 | 我自己实测 `clip_score=0.0` 后查日志定位 |
| 7 | datastore 加速 28× | **误导**。线上实际 2.6× | 子 agent 拒绝引用我的数字，坚持自测 |

**附带说明**：子 agent 还主动把两处"不是成功、只是能跑"如实标注为遗留（弱身份匹配错误、上游 DFS 未修），并主动区分了「真实数据实测复现」与「注入 null 的合成验证」两类证据。这些没有被美化。
