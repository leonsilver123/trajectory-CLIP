# 重构计划与进度追踪

> **最后更新**：2026-09-12 —— **9 个任务全部完成**，交付报告见 `DELIVERY_REPORT.md`
>
> **终态**：pytest `13 failed, 358 passed`（失败集合与基线逐条一致，0 回归）；前后端冒烟 `15/15`。
> git：`a155f0a` → `ab4d339` → `7a4d5a5` → `de60914` → `3d50f01`。
> **状态图例**：`[ ]` 未开始　|　`[~]` 进行中　|　`[x]` 已完成　|　`[-]` 已放弃/搁置
>
> **维护约定**：每完成一次代码修改，必须在本文件对应任务的「变更记录」里写明
> **改了什么（文件 + 具体改动）、没改什么（明确列出的遗留项）**，并更新顶部「最后更新」时间与「进度总览」。

---

## 零、环境与基线（2026-09-11 建立）

### 0.1 运行环境
- 项目根：`H:\trajectory-CLIP`；venv：`.venv`（由 conda py313 用 `--system-site-packages` 创建，继承 torch 2.9.1+cu128）。
- **规范命令（起服务）**：
  ```bash
  cd H:/trajectory-CLIP
  PYTHONPATH="H:/trajectory-CLIP/.venv/Lib/site-packages" .venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
  ```
- **规范命令（跑测试）**：测试额外需要禁掉一个坏插件，故多一个变量：
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="H:/trajectory-CLIP/.venv/Lib/site-packages" .venv/Scripts/python.exe -m pytest -q
  ```
- 三个环境坑与解法（已定位并实测）：
  1. `F:\Anaconda_envs\envs\py313`（env 根目录，在 sys.path 中**早于** `.venv/Lib/site-packages`）里有一个散落的 `typing_extensions.py`，缺 `sentinel`，会让 `fastapi` 导入失败。→ 用 **`PYTHONPATH` 前置 venv 的 site-packages** 覆盖它。
  2. 用户级目录 `C:\Users\Administrator\AppData\Roaming\Python\Python313\site-packages` 里有个坏掉的 `langsmith` pytest 插件（import 缺失的 `requests_toolbelt`），会让**整个测试会话崩溃**。→ 测试时加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。
  3. ⚠️ **`cn_clip` 只存在于上述用户级目录**（venv 里没有，`requirements.txt` 里也没声明）。因此**绝不能用 `PYTHONNOUSERSITE=1`** —— 那会把它一起移除，导致 CLIP 检索**静默退化**为纯属性排序（候选 `clip_score` 恒为 `0.0`，后端日志报 `No module named 'cn_clip'`）。
- ⚠️ **本节曾记录过一条错误命令**（含 `PYTHONNOUSERSITE=1`）。它在测试场景下看似正常（测试照样全绿，因为检索会优雅降级），但**起服务时会静默关掉 CLIP**。主 agent 是在实测 `/search/query` 发现候选 `clip_score` 全为 `0.0`、再查后端日志才定位到的。**已更正并改用上面的命令复跑全部验收**：pytest 仍为 `13 failed, 335 passed`，失败集合与基线逐条一致；改用正确命令后 CLIP 恢复（`Chinese-CLIP ViT-B-16 模型加载成功, device=cuda`，候选 `clip_score` 为真实的 0.4275 / 0.4279 / 0.4213，融合 `final = 0.6*clip + 0.4*attr` 真实生效）。
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

> **关于"基线数字对不上"的说明**：并行子 agent 曾各自报出 `19 failed / 245 passed`，与我给的 `13 failed / 251 passed` 不一致。
> 两者都是**真值，只是测量时刻不同**：`13/251` 是**任何 agent 动手之前**测得；`19/245` 是子 agent 在**任务 1/2 已改完 `backtrack.py`/`confirm.py` 之后**才跑的，此时 `tests/test_api.py` 里 6 条断言旧行为的用例已从「通过」变为「失败」。
> 校验：`13 + 6 = 19`，`251 - 6 = 245` —— 正好吻合。所以**不存在矛盾，也不是谁测错了**。
> 后续所有验收统一以「失败集合与基线逐个一致」为准（用 `comm` 比对），而不是比单个数字。

### 0.3 计划前提的更正（**计划原文有错，以本表为准**）

| 计划原文的说法 | 核实结果 | 处置 |
|---|---|---|
| 任务 5："`clip_vectors.faiss` 找不到 faiss，是死代码" | **错**。`faiss` 1.14.3 与 `cn_clip` 均可导入；`output/clip_vectors.faiss` 存在（68349 向量 × 512 维）；`models/clip_cn_vit-b-16.pt` 存在（753MB）。该分支**不是死代码** | 任务 5 重新定义为"前端/后端重复实现收敛"，**不删 CLIP 能力** |
| 任务 7："被替代的旧 `configs/camera_metadata.yaml`" | **错**（但我的更正第一版也过头了，见下）。`api/routes/dashboard.py:31` 确实在读它（`_CAMERA_CONFIG`） | **保留该文件**，任务 7 不再包含它 |
| 任务 7：`api/routes/search_backup.py` 可删 | **对**。全仓仅在 `.md` 中被提及，无 `.py` 引用，`api/main.py` 只注册 4 个 router | 删 |
| 任务 7：`requirements.txt` 的 `flask` 未用 | **对**。全仓 `.py` 中 `\bflask\b` 零命中 | 删该行 |
| 任务 7：`frontend/mock_data.py` | 仅自身 docstring 提及，无任何 import | 删 |

> 教训：`HANDOFF.md`/`PLAN.md` 的部分前提基于未验证的观察。**动任何"死代码"前必须先 grep 取证**。

### 0.4 主 agent 自身判断的两处更正（子 agent 提出，我已复核）

**更正 A：`configs/camera_metadata.yaml` 是「被引用但非 load-bearing」，不是我说的「删了就废」**
- 我最初写「删了 `/dashboard` 就废」——**说过头了**。
- 实测（`yaml.safe_load` 复核）：该文件**只有 `scenes:` 一个顶层键，没有 `cameras:` / `camera_list:`**。因此 `api/routes/dashboard.py` 的 `_load_camera_metadata()` 里 `data.get("cameras", data.get("camera_list", []))` 拿到 `[]`，直接落进第 99-121 行「从 results.json 推断」的兜底分支。
- 子 agent 做了非破坏性 A/B（只在内存改路径、未动磁盘）：带文件与不带文件均返回 **2384 条、首条 `000001`、`identical: True`**。
- **准确结论**：它仍被活代码路径引用（`dashboard.py:31`；另在 `camera_manager.py:38`、`road_topology.py:37` 的用法示例里出现），但**不是 load-bearing**——删掉不会让接口失效，只会静默切到兜底分支。**保留依然正确**（仍被引用 + 原计划理由不成立），但保留理由要写成这个准确的版本。
- 处置不变：**文件未删**。

**更正 C：我一度把「唯一需要兜底的字段」收窄错了，漏掉一个实测可达的崩溃**
- 我向子 agent 传达「`inference_segments[].actual_travel_time` 是唯一需要新增兜底的地方」——**该结论只在 `auto` 模式下成立**。
- 来源：任务 6 的 agent 给了一份 null 清单，但它**只测了 `auto`**；我误当作全量结论转达。
- 子 agent **没有照单接受，改为自己实测 `mode="stitch"`**，发现两种模式的 null 集合**不同**。**主 agent 已独立复核，确认子 agent 正确**：
  ```
  auto   : inference_segments[].actual_travel_time = None（4 段中 3 段）
  stitch : observation_nodes[].confidence = [0.7822, None, 0.7779]  ← 真正的 None
           identity.vehicle_id / evidence.vehicle_id = None
           inference_segments[].actual_travel_time = [4.8, 3.201]   ← 反而不是 None
  ```
- **后果**：`observation_nodes[].confidence` 会被当数字格式化（`trajectory.py:547`、`timeline.py:491`、`timeline.py:593`、`confidence_color/label`），子 agent **实测复现出第二处崩溃**（PDF 与 Excel 导出各一处）。若按我的错误收窄执行，**这处会被漏掉**。
- 处置：保留子 agent 的完整 sweep。它同时把改动**如实分档**为「A 档 = 实测可达的真修复」与「B 档 = 理论风险 / 零行为变更的防御性兜底」（B 档附 17 组新旧对比，全部 `SAME`），**这个分档被原样保留在交付报告中**。

**更正 B：`.dockerignore` 并未被覆盖，是我看错了**
- 我先前称「该文件在派活前已存在，被 Agent B 覆盖、原内容可能丢失」——**这是错的**。
- Agent B 的证据链：其 `Write` 返回的是 `File created successfully`；而本 harness 对「未先 Read 就覆盖已存在文件」是**直接拒绝**的——能写入即证明当时文件不存在。
- 复核时间线：`.dockerignore` mtime `22:52`（B 写入）、我新建的 `.gitignore` mtime `22:55`。而我那次 `ls -a .gitignore .dockerignore` 的观测结果是「`.gitignore` 不存在、`.dockerignore` 存在」，这只可能发生在 **22:52–22:55 之间**，即 **B 写入之后**。也就是说我看到的是 B 已创建的文件，而非它的前身。
- **结论：无内容丢失，无需追查。** 我先前的判断把观测时刻搞错了。

---

### 0.5 贯穿性问题：项目里存在**两套不兼容的 CLIP 特征空间**（新发现，未修）

重构过程中发现一个比原计划描述的更严重的版本不一致，**实测取证如下**：

| 来源 | 维度 | 模型 | 覆盖 | 谁在用 |
|---|---|---|---|---|
| `output/clip_vectors.faiss` | **512** | Chinese-CLIP **ViT-B-16** | 68349 / 68349（100%） | `api/routes/search.py`（活的检索路径） |
| `detection.clip_image_vector`（内联） | **768** | Chinese-CLIP **ViT-L-14** | 949 / 68349（**1.39%**，全部 `source=="track3"`） | `src/trajectory/builder.py`（任务 6 新增，用于 `_score_appearance`） |
| `configs/default.yaml:124-125` | 声明 **768** | 声明 **CN-CLIP-ViT-L-14** | — | 配置声明，与 `search.py` 实际不符 |

**问题**：配置声明 L-14/768，而线上检索实际用 B-16/512 —— **两者不是同一个特征空间，向量不可混用**（维度都不同）。`CLAUDE.md` 已记载该脱节，但此前无人量化。

**影响与处置**：
- 新 builder 用的是**内联 768 维**（1.39% 覆盖），故 `_score_appearance` 仅对 1.39% 的检测是真实计算，其余约 98.6% 回退中性 0.5 —— 这直接削弱了弱身份路径的判别力（见任务 6 遗留 1）。
- 检索路径用 512 维索引，**覆盖 100%**，是本项目 CLIP 能力的主载体。
- **本次未统一**（属上游数据重建问题：需要重跑特征提取管线，成本高且不在 9 个任务范围内）。**列入交付报告头号遗留项**。

> 附：主 agent 曾向子 agent 断言「JSON 里没有 clip_vector」——**该断言错误**，子 agent 实测反证并正确使用了内联向量。已更正。

---

### 0.6 ⚠️ 尚未清理的「编造数据」点（与任务 2 同类问题，**未修**）

任务 2 删掉了后端的伪回溯（`random` 伪造时间戳/置信度），但**同类"编造看似合理的数值"的做法在前端仍系统性存在**，且**任务 2 的原则并未覆盖到它们**：

**实测 grep 取证 —— `frontend/` 下 9 处在数据缺失时凭空填入置信度：**
```
frontend/pages/timeline.py:439   node.get("confidence", 0.9)
frontend/pages/timeline.py:531   node.get("confidence", 0.9)
frontend/pages/timeline.py:602   inf.get("confidence", 0.7)
frontend/utils.py:164            ... .get("quality_score", 0.8)
frontend/utils.py:180            first_frame.get("confidence", 0.85)
frontend/utils.py:417            node.get("confidence", 0.9)
frontend/utils.py:444            src.get("confidence", 0.9)
frontend/utils.py:466            seg.get("confidence", 0.7)
frontend/utils.py:497            path.get("confidence", 0.5)
```

**为什么这条重要（不只是洁癖）**：
1. 这些**编造值会掩盖诚实留空**。本轮刚给 `confidence_color()` / `confidence_label()` 加了 `None` 处理（返回中性色 / `"未知"`，不冒充"低置信度"），但上述站点在**值到达它们之前就已经把 `None` 换成了 `0.9`** —— 于是"未知"这条诚实路径**永远触发不了**，UI 上显示的是一个看起来像真实测量的 `0.9`。
2. 与任务 2 删掉的后端伪回溯**性质完全相同**：都是"没有依据时造一个像样的数字"。**只删了后端那一半。**
3. `map_view.py:73-74`（摄像头无坐标时按索引伪造 lat/lon 偏移）同属此类。

**同类还有两处"编造观测"**（主 agent 读代码确认）：
- `frontend/utils.py:193-194`：`_convert_trajectory_response()` 对每个摄像头凭空写 `entry_description: "从画面进入"` / `exit_description: "从画面离开"` —— 而任务 2 已让后端在这些无真实标注的字段上**诚实返回空串 `""`**。前端适配器**又把描述编了回来**，等于断言了一个并未观测到的事实。
- `frontend/pages/map_view.py:73-74`：摄像头无坐标时按索引伪造 lat/lon 偏移，好让点能画在地图上。

**另一个结构性发现**：`api_backtrack`（即 `/trace` 端点）在 `frontend/` 里**只被 import、从未被调用**（`confirm.py:19`、`search.py:20` 是 import 语句）。前端实际走的是 `api_trajectory`（`/trajectory` 端点）+ `_convert_trajectory_response` 适配。这解释了为什么本轮给 `/trace` 带来的跨镜改进**不会自动反映到前端**，也解释了前端那两处格式化崩溃为何"函数级可复现、UI 级到不了"。

**未修原因**：超出 PLAN.md 九项任务范围，HANDOFF §9.2 明确要求"发现额外问题写进遗留项，不擅自扩范围"。**列入交付报告遗留项，并建议作为后续第一优先级**（与项目目标"诚实"直接冲突）。

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
| 3 | ID 解析收敛成唯一函数 | P0 | [x] 全仓 4 处已收敛，`startswith("V")` 零命中 |
| 4 | 数据契约落盘，在线不读大 JSON | P1 | [x] datastore 落盘（12.4MB vs 102.6MB）；回退路径已实测 |
| 5 | 检索管线收敛成一条 | P1 | [x] 前端本地 CLIP/BLIP 管线已删（-592 行），检索只剩后端一处 |
| 6 | 跨镜拼接统一到 TrajectoryBuilder | P1 | [x] `/trace` 已由单摄像头变为跨摄像头；`src/stitching` 首次线上生效 |
| 7 | 删死代码 | P2 | [x] |
| 8 | 修部署（Docker 路径等） | P2 | [x] compose/Dockerfile/.bat 已修并复核；`docker/.env` 因权限未改 |
| 9 | git 保护（初始化 + 首次提交） | P2 | [x] 首提交 `a155f0a`（180 文件）；波次 2 后需再提交一次 |

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
- **遗留（重要，已转交任务 6）**：
  1. `overall_confidence` 目前恒为 `0.0`、inference 段的时间/置信度为 `None` —— 这是**有意为之的诚实留空**（不再造数），待任务 6 接入 `src/stitching` 评分后填充
  2. ⚠️ **`/trace` 目前必然返回「退化的单摄像头结果」**。子 agent 实测：`cityflow_results.json` 有 **68349 条 detection、68349 个互不相同的 `target_id`**，**每个 `target_id` 只出现在 1 个摄像头里**；跨镜身份**只存在于 `vehicle_id` 上**（如 `V0034` 覆盖 c001–c005 共 206 条）。而 `/trace` 是按 `target_id` 聚合的 → 恒为 1 摄像头、1 观测节点、`inference_segments` 恒空。
     而**前端主流程走的正是 `/trace`** —— 也就是说「跨镜回溯」这条主线此前实际是空的。
     本任务要求的「`/trace` 改为：从 session 拿 instance_id → 定位 vehicle_id → **复用 `/trajectory` 的真实聚合**」**未做**（当时指令只列了删 `random`）。**已转交任务 6**（强身份走 vehicle 级聚合、弱身份走 stitching）
  3. `_load_camera_metadata_from_yaml()` 在 camera 已存在时**只 merge `name`，不 merge `latitude`/`longitude`** → `/trace` 的 lat/lon 恒为 `None` → 本任务新加的 `_path_distance_meters()`（haversine）恒返回 `0.0`。**已转交任务 6** 一并修（`src/stitching` 的 `_score_spatial` 也依赖真实坐标）
  4. 前端格式化风险：`frontend/pages/trajectory.py:882-884`（PDF）、`:1053-1055`（Excel）对 `candidate_paths` 的 `confidence`/`estimated_time` 直接 `f"{x:.0%}"`，遇 `None` 会 TypeError。故这两项置 `0.0` 而非 `None`。**若任务 6 把它们改成 `None`，前端需加 `or 0` 兜底**（`inference_segments` 转非空后 `trajectory.py:433/588/855-856/1032-1033`、`timeline.py:604/611-612` 同理）
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
- [x] 全部反解点统一调用 `src.common.ids`，无重复 `startswith("V")` 逻辑
- [x] 新增 `tests/test_ids.py` 覆盖 `CF3_c001_V0034_000001` 等格式

**变更记录**：
- 已修改：
  - 新建 `src/common/ids.py`：`parse_target_id` / `parse_track_id` / `extract_vehicle_id`。实现用正则（`_VEHICLE_RE = ^V\d+$`）而非 `startswith`，语义等价且更严；畸形输入返回空值不抛异常（保持原容错语义）
  - `api/routes/backtrack.py`：删 `_extract_vehicle_id`，`/trajectory` 里的 `track_id.split("_")` 段一并改调 ids
  - `frontend/utils.py`：`_cityflow_det_to_candidate` 里 6 行反解循环 → 1 行
  - `frontend/pages/search.py`（**计划清单漏掉的第 4 处**）：`_build_traj_from_local` 里 6 行循环 → `extract_vehicle_id()`
  - 新建 `tests/test_ids.py`（29 例）
- **验收证据**：`grep -rn 'startswith("V")' --include=*.py api/ frontend/ src/ scripts/ tests/` → **零命中**（ids.py 用的是正则，本就不含该字面量）；29 例新测试全过
- **已知行为差异（1 例，已如实记录）**：畸形输入 `V0034_c001_V0035_000001`（第 1 段是车辆号，违反 ID 契约）旧代码取 `V0034`、新代码按契约取第 3 段 `V0035`。17 组对照中其余 16 组完全一致。该形态在真实数据中不存在（source 段恒为 `CF3` 等），双方都是"垃圾进垃圾出"，未强行对齐

---

## 五、P1 —— 架构对齐

### 任务 4：数据契约落盘，在线不读大 JSON　— 状态：`[x]`

- 新建 `scripts/build_datastore.py`（JSON → SQLite 元数据 + Parquet detections 表 + 独立向量文件）
- 新建 `src/storage/datastore.py`（统一读写接口）
- 改 `api/routes/search.py` / `backtrack.py` / `frontend/utils.py`（改走 datastore，删散落的 `json.load` + mtime 缓存）
- 验收：[x] 在线首查不加载 90MB JSON；[x] `load_cityflow_results()` 等散落函数收敛为一处

**变更记录**：
- 已修改：
  - 新建 `src/storage/datastore.py`（**唯一读取层**，对外 `load_results/load_detections/get_summary/get_stats/data_source/has_data/reset_cache`）、`src/storage/__init__.py`
  - 新建 `scripts/build_datastore.py`（离线构建 + 校验，含 `--verify-only`）
  - 新建 `tests/test_datastore.py`（23 例）
  - **5 处散落读取点全部收敛**：`api/routes/search.py`、`api/routes/dashboard.py`（`/health` 新增 `data_source` 字段，**只加不删**）、`src/trajectory/builder.py`、`frontend/utils.py`、`frontend/home.py`
  - **数据布局**：`output/datastore/{detections.parquet, tracks.parquet, det_image_vectors.npy, det_text_vectors.npy, det_vector_rows.npy, det_text_vector_rows.npy, meta.sqlite}`，**12.37 MB vs 源 JSON 102.63 MB（小 8.3 倍）**
- **核心安全设计（叠加式，已实测）**：datastore **存在则用、不存在则回退 JSON**，行为与改造前一致。**主 agent 独立演练过回退**：
  ```
  改名前 : /dashboard/health → data_source="datastore"
  mv 走  : data_source="json"  results_loaded=true
           全量冒烟 14/14 通过（跑在回退路径上）
  mv 回  : data_source="datastore"
  ```
  演练包在 bash `trap` 内，目录不可能被遗留为改名状态（已确认 7 个文件齐全）
- **实测性能（子 agent 自测，**纠正了主 agent 的误导性数字**）**：
  - 主 agent 原型给出的「28 倍」是**裸 parquet 读取**的对比，**对线上不成立**。
  - 子 agent 实测线上 `load_results()`：datastore 热读中位 **0.482s** vs JSON 直读 **1.250s** vs 裸 `json.load` **1.288s** → **实际约 2.6 倍**。
  - 原因（子 agent 解释，主 agent 认可）：线上必须物化 68349 个同结构 dict，这份成本两条路都要付；datastore 省掉的只是 JSON 词法解析的约 0.77s。
  - **子 agent 明确拒绝引用我给的原型数字，坚持自测** —— 这正是要求的诚实标准。
- **保真度取舍（子 agent 主动说明）**：Parquet 无法区分「键不存在」与「键值为 null」，而源数据在 `bbox_size`/`color_analysis`/`dominant_color_rgb`（949 行）与 tracks 的 `attributes`（1779 行）上依赖该区分。为保持 `==` 级等价，对真缺键的列用 Arrow null bitmap，对「键在值为 null」的列另写存在性列表；代价约 +0.15s/次、+1.17MB。**不这么做会返回源数据里没有的 `None` 键 = 破坏契约**
- **主 agent 独立复核**：
  - pytest `13 failed, 358 passed`（+23 = 新增测试），失败集合与基线 `comm` 逐个一致（0 回归）
  - 读取点收敛 grep：`api/`、`src/`、`frontend/` 下**除 `src/storage/datastore.py` 外无人 `open()` 该 JSON**；唯一那处 `json.load` 只有两个调用方（回退分支 + 离线工具包装）
  - `/dashboard/health` 实测 `data_source="datastore"`
  - 冒烟 **15/15**（含真实拉起 Streamlit）
- **遗留**：
  1. **`output/datastore/` 被 `.gitignore` 忽略** → 是**本机产物，不进版本库**。部署到新机器需跑一次 `scripts/build_datastore.py`（**不跑也能用**，走 JSON 回退，只是慢）
  2. `output/` 下留有子 agent 的临时文件（`_proto/`、`_fallback_check.py`、`_bench_read.py`、若干验收证据 txt、`_backup_task4/`），删除被**权限系统拒绝**，未清理（都在 `output/` 内且已被 gitignore，不影响运行）
  3. 子 agent 的观察（已记录未修）：`/trace` 的 `inference_segments[].basis` 标为 `strong_identity` 而非 `probabilistic_inference` —— 语义上推断段更宜标后者。`backtrack.py` 在禁改清单内，未动

---

### 任务 5：检索管线收敛成一条　— 状态：`[x]`

> **前提已更正**（见 §0.3）：原计划"删 ViT-B-16 分支，因为它是死代码"是**错的**——
> `faiss`/`cn_clip` 可用、索引与权重都在，该分支是**活的**。因此本任务改为
> **消除前后端重复实现**，而不是删除 CLIP 能力。

- **不删** `api/routes/search.py` 的 CLIP 精排路径（它是真的能跑）。
- 真正的问题：`frontend/utils.py` 有一套**独立的**「属性硬过滤 → OpenCLIP(ViT-L-14) → BLIP → 属性重排」管线，与后端 `search.py` 的「属性粗筛 → Chinese-CLIP(ViT-B-16) → 融合排序」**不一致**，且模型版本不同（L-14 vs B-16）。
- 目标：**后端为唯一检索真源**，前端改为纯调用后端接口，不再本地跑 CLIP/BLIP。
- Fallback =「少一步」（无 CLIP 时退化为属性分数，`search.py` 已有此逻辑）而非「换一套代码」。
- 验收：[x] 前端不再自行加载 CLIP/BLIP 模型；[x] CLIP 不可用时接口仍返回属性排序结果；[x] 前后端不再各有一套排序逻辑

**变更记录**：
- 已修改：
  - `frontend/utils.py`（**1374 → 703 行，删 592 行**）：删掉整套本地检索引擎 —— `build_search_results_from_cityflow()`、`_get_clip_extractor()`（CN-CLIP-ViT-L-14）、`_get_blip_extractor()` / `_blip_score_image_text()`（Salesforce/blip-itm-base-coco）、`_compute_attr_consistency()`、`_batch_cosine_similarity()`、`_cosine_similarity()`、`_cityflow_det_to_candidate()`、`_parse_cityflow_query()` 及其专用同义词表；连带清理失效的 `import numpy as np`
  - `frontend/pages/search.py`：`_do_search()` 改为**唯一真源 = 后端 `POST /api/v1/search/query`**；删除 `if has_cityflow_results(): build_search_results_from_cityflow(...)` 本地兜底分支。后端不可用时**如实渲染警告块并返回 None**，不再静默换一套算法；同时抑制「数据集不含该目标」的二次误导提示
  - 新建 `tests/test_frontend_retrieval.py`（6 例）：源码级断言 `frontend/` 无 CLIP/BLIP 加载标识、已删函数确不存在、轨迹展示函数仍在、`_do_search` 的 AST 只调用 `api_search` 且失败分支走警告
- **保留**：`import torch` / `import yaml`（按 Docker 部署约定保留，注释说明 torch 已无使用点）；`load_cityflow_results()` / `has_cityflow_results()` / `resolve_image_path()` 及轨迹展示逻辑（属任务 4 范围）
- **验收证据**：`grep` 已删函数在 `api/`/`src/`/`tests/` 下**无实际调用点**（仅剩 `utils.py` 的说明性注释与 `test_frontend_retrieval.py` 的断言）；`import frontend.utils, frontend.pages.search` OK；全量 pytest `13 failed, 334 passed`，失败集合与基线逐个一致
- **遗留（子 agent 上报，未修，不擅自扩范围）**：
  1. ⚠️ **筛选面板的「场景 / 摄像头」下拉框实际不生效**。已复核：`cf_scene`/`cf_camera` 只在 `search.py:376,381` 被收集、`317-320` 当标签显示、`563,565` 判空，**从未进入 `_apply_filters()`**；后端 `/search/query` 也不接受这两个参数。属**既有缺陷**（后端在线时改造前后一样无效，以前只有本地兜底分支会用到它们），现因本地分支删除而更明显。未修原因：补客户端后置过滤会削减 `top_k` 甚至返回 0 条，属**引入新行为**，超出本任务范围
  2. 前端仍有 4 处直读 107MB `output/cityflow_results.json`（均**非检索**）：`frontend/utils.py:656`、`frontend/pages/search.py:625/697`、`frontend/pages/dashboard.py:26`、**`frontend/home.py:25,59`**（首页统计）→ 交任务 4
  3. `_CITYFLOW_CROPS_DIR`（`utils.py:647`）改动前即无引用，属既有死代码，未动
  4. `search.py:970` 的 `cand.get("data_source") == "cityflow"` 分支现已不可能命中（后端候选不带该字段），无害死分支，未动
  5. ⚠️ **`frontend/pages/map_view.py:73-74`：摄像头无坐标时，按索引伪造 lat/lon 偏移** —— 这是**与任务 2 删掉的伪回溯同一性质的"编造数据"**（为了让点能画在地图上而凭空造坐标）。子 agent 如实上报，因超出「只修 None 格式化」范围**未改**。**列入交付报告遗留项**，建议后续按任务 2 的原则处理（无坐标就如实标注不可定位，而不是造一个假坐标）
  6. 子 agent 的诚实边界声明（值得保留）：`candidate_paths` / `observation_nodes` 等的 `None` 兜底是**用注入 null 的方式验证**的（合成验证），**不是**真实数据实测复现；而 `actual_travel_time` 相关几处的崩溃是**在真实 `/trace` 响应上实测复现**的。两类证据被明确区分，未混为一谈

---

### 任务 6：跨镜拼接统一到 TrajectoryBuilder　— 状态：`[x]`

- 新建 `src/trajectory/builder.py`（`TrajectoryBuilder.build(anchor_instance)`：强身份直接匹配 / 弱身份走 stitching）
- 改 `api/routes/backtrack.py`（改调 TrajectoryBuilder，不再自写聚合）
- 验收：[x] `src/stitching` 首次被线上引用；[x] 输出每段标注依据（强身份 vs 概率推断）

**变更记录**：
- 已修改：
  - 新建 `src/trajectory/__init__.py`、`src/trajectory/builder.py`（1343 行）。**强身份**（有真实 `vehicle_id`）→ 提为公共实现，按 vehicle 聚合跨摄像头序列；**弱身份**（无车牌/无真值，需显式 `mode="stitch"`）→ 调用 `src/stitching`（`CandidateEdgeGenerator` + `CrossCameraScorer` + `ObservationChainBuilder`），并写了 JSON detection → `src/common/data_models` dataclass 的适配层
  - `api/routes/backtrack.py`：`/trace`、`/trajectory` 全部委托 `TrajectoryBuilder`，不再自写聚合；`_generate_fallback_mock()` 原样保留
  - 新建 `tests/test_trajectory_builder.py`（23 例）
- **主 agent 独立复核（真实 HTTP，非采信自述）**：
  - `/trace` 由**单摄像头**变为**跨摄像头**：`camera_sequence=['c004','c005','c003','c002','c001']`（改前恒为 1 个摄像头），`inference_segments` 由**恒空**变为有内容
  - 每段带 `basis` / `basis_text`：强身份路径 `basis="strong_identity"`、`confidence=1.0`；弱身份路径 `basis="probabilistic_inference"`
  - `overall_confidence` 由硬编码 `0.0` 变为真实值（强身份 `0.3411`）
  - `evidence` 新增 `identity_basis` / `identity_certainty` / `linkage_confidence` / `link_count` / `direction_consistency` / `vehicle_id`
  - **`src/stitching` 确已被线上引用**：`src/trajectory/builder.py:55-57` 三条 import；`mode="stitch"` 实测 200（0.8s）、`evidence.source="src.stitching"`、`identity_basis="probabilistic_inference"`、段置信度 `0.7822/0.7779`、`overall_confidence=0.7801`
  - 两次 `/trace` sha256 完全一致（无随机性回归）；`random` 仅存于 `_generate_fallback_mock`
  - 全量 pytest `13 failed, 335 passed`，失败集合与基线逐个一致
- **遗留（子 agent 如实上报，主 agent 已复核）**：
  1. ⚠️ **弱身份路径「通路已连通、准确率差」**：实测 `mode="stitch"` 得到 `c004→c001→c002`，而强身份真值是 `c004→c005→c003→c002→c001` —— **两者不一致**，弱路径未还原真实链。子 agent 归因为「CLIP 区分度弱 + 同一车辆 `vehicle_type` 跨镜标注不一致」。**故弱身份结果不可信，仅证明通路打通**；生产使用应依赖强身份（或车牌）
  2. 本数据集**每条 detection 都带真实 `vehicle_id`**，故 `auto` 模式恒走强身份；弱身份分支必须显式 `mode="stitch"` 才触发
  3. `src/stitching/scoring.py` 的 `_count_possible_paths` 是**无界 DFS**，实测单对摄像头耗 4–23 秒。子 agent 在 `builder.py` 里用**子类加上界绕过**，**未改上游** `src/stitching`（避免动公共模块）。上游隐患仍在
  4. `evidence.linkage_confidence` 当前为 `None`（未填充）
  5. 上一轮遗留的前端格式化风险：本次 `candidate_paths` 的 `confidence`/`estimated_time`/`distance_meters` 均为**真实数值**（`0.3411`/`15.3`/`213.0`），不会触发 `None` 崩溃，且已加回归测试 `test_candidate_path_numeric_fields` 守约。**但 `inference_segments[].actual_travel_time` 为 `None`**（本数据集 timestamp 是各摄像头视频内时间且跨镜重叠，「首现−末现」常为负，故诚实留空而非取绝对值凑数）→ **已实测确认会令 `frontend/pages/trajectory.py:1033` 抛 `TypeError`**，已派修（见任务 5 遗留追加项）
  6. **评分维度大面积为空（子 agent 逐项披露，主 agent 已复核）**：
     - `reid`：**全空**。`avg_reid_vector` 恒为 `None`，外观分完全由 CLIP 支撑。但 `stitching.weights.vehicle.reid = 0.15` 的权重**仍然生效**——即权重实际挂在 CLIP 分上，**这是真实的语义偏差**（配置项名与其实指不符）
     - `plate`：**全空**。数据集无车牌字段，`_score_plate` 对双方均返回中性 0.5，导致 `is_valid = score > 0 and plate_score > 0` 这道校验形同虚设
     - `appearance`：**仅 1.39% 为真实计算**（见 §0.5），其余回退中性 0.5
     - `temporal` / `spatial`：强身份路径上基本为退化值（跨镜时间窗重叠）
     - `entry_description` / `exit_description`：仍为空串（无真实标注）
  7. **上游性能隐患未修**：`src/stitching/scoring.py` 的 `_count_possible_paths` 是**无界 DFS**，实测单对摄像头 **4–23 秒**（c001→c002 23.16s、c003→c004 12.08s）。子 agent 在 `builder.py` 内**用子类加上界覆盖**（`BoundedCrossCameraScorer`，返回值与上游一致），线上实测降至 0.8s。**上游文件未改**（避免动公共模块）→ **隐患仍在，任何其他调用方都会踩**
  8. **弱身份路径匹配错误（子 agent 主动如实标注为"通路已连通、准确率差"，非成功）**：V0034 的链跑出 `c004→CF3_TRACK_c004_V0001`、`c002→CF3_TRACK_c002_V0008`，**两段都不是 V0034 自己的轨迹**。根因两条且均有实测支撑：① 内联 CLIP 区分度弱（不同车 cos 0.88–0.90，本车 0.942）；② 同一辆车 `vehicle_type` **跨镜标注不一致**（V0034 在 c001 标 SUV、c002/c005 标轿车、c003/c004 标面包车），导致 `CandidateEdgeGenerator` 的硬属性否决**惩罚正确轨迹（attr=0.5）而奖励错误车辆（attr=1.0）**。属数据/上游算法问题，**未修**。
     > **主 agent 已独立复核该根因②**（脚本按 `vehicle_id × camera` 交叉统计，实测 `V0034`）：
     > `c001` 混杂 SUV/轿车 且两种颜色；`c002` 混杂 轿车/SUV/面包车 且三种颜色；`c003` 混杂 面包车/轿车 且三种颜色；`c004` 基本全为面包车；`c005` 混杂 轿车/SUV 且两种颜色。
     > **同一辆车不仅跨摄像头标注不一致，单个摄像头内部的检测标注也自相矛盾** → 说明属性标注本身噪声极大。这**同时损害了检索的 `_attribute_filter` 与拼接的 `_score_attribute`**，属**数据质量问题**，非代码可修

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

### 任务 9：git 保护　— 状态：`[x]`

- 初始化空 `.git`（当前无 HEAD/commit），首次提交 + `.gitignore`（排除 `output/`、`models/`、`yolov8x.pt` 等大文件）

**变更记录**：
- 已修改：
  - `git init`，设置仓库级 `user.name` / `user.email`（仅本仓库，不动全局配置）
  - 新建 `.gitignore`：排除 `output/`、`models/`、`cityflow/`（含 AICity/VisDrone 数据集 zip）、`data/`、`docker_data/`、**`docker/qdrant_storage/`（461MB）**、`.venv/`、`*.pt`、`*.faiss`、`*.parquet`、`*.sqlite` 等；`docker/` 只排除 `qdrant_storage/` 子目录，保留其中的 `init_qdrant.py`、`nginx.conf` 源码
  - 首次提交 `a155f0a`（**180 个文件，最大文件仅 92KB** —— 已逐项核对，大文件均被正确挡住）
- **为什么先做**：HANDOFF §8 指出此前无 HEAD/commit，任何改动**无版本保护**；波次 2 还要大改 `api/routes/backtrack.py` 与 `frontend/`，先建快照才能出问题时回滚
- **遗留**：本提交是在波次 2 进行中打的检查点，包含部分波次 2 的中间状态；**需在波次 2 验收后追加一次提交**，才是干净终态

---

## 七、全局变更日志（倒序）

> 每次修改在此追加一条，格式：`日期 — 任务# — 一句话说明改了什么/没改什么`。

- 2026-09-11 — 任务#4 — datastore 落盘完成：新增 `src/storage/datastore.py`（唯一读取层）+ `scripts/build_datastore.py` + `tests/test_datastore.py`；5 处散落 JSON 读取点全部收敛；`output/datastore/` 12.37MB vs 源 JSON 102.63MB。**叠加式设计（存在则用、不存在回退 JSON），回退路径经主 agent 独立演练：`mv` 走后 `data_source="json"` 且全量冒烟 14/14 通过，`mv` 回后 `data_source="datastore"`**。实测加速 **2.6 倍**（子 agent 自测，**纠正了主 agent 原型给出的误导性 28 倍**——那只是裸 parquet 读取，线上需物化 68349 个同结构 dict）。pytest `13 failed, 358 passed` 失败集合与基线一致；冒烟 **15/15**。
- 2026-09-11 — ⚠️ 复核中发现子 agent 一次 `git stash` 未 pop，任务 4 成果一度全部躺在 stash 里。已只读核查确认 9 个文件（1454 insertions）完整可恢复，**未擅自操作其 stash**，交由该 agent 自行 pop；恢复后复跑验证通过。**教训：用 `git stash` 取基线是全局单例操作，用完不 pop 等于清空工作区；应改用 `git show HEAD:<path>` 或 `git worktree`。**
- 2026-09-11 — ⚠️ **主 agent 自纠**：先前在 §0.1 给出的规范命令含 `PYTHONNOUSERSITE=1`，会把**只存在于用户级 site-packages 的 `cn_clip` 一并移除**，导致 CLIP 检索**静默退化为纯属性排序**（`clip_score` 恒 0.0，接口仍 200，极难察觉）。这是在实测 `/search/query` 时发现候选 `clip_score` 全为 `0.0`、查后端日志见到 `No module named 'cn_clip'` 才定位的。**更正为只用 `PYTHONPATH`**（测试另加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`），并**用新命令复跑全部验收**：pytest 仍 `13 failed, 335 passed`（失败集合与基线一致），CLIP 恢复（`ViT-B-16 模型加载成功, device=cuda`，`clip_score` 真实值 0.4275 等，融合 `0.6*clip+0.4*attr` 生效）。**结论：此前所有验收数字在两套命令下一致，结论不变；但"CLIP 路径是否真的在跑"这一项此前未被真正验证过，现已验证并确认可用。**
- 2026-09-11 — 波次2 追加修复 — 主 agent 实测发现：任务 6 让 `inference_segments` 非空后，`actual_travel_time`（诚实为 `None`）会令 `frontend/pages/trajectory.py` 的 Excel 导出抛 `TypeError`（`dict.get(k, 0)` 兜不住「键存在、值为 None」）。已修复：`frontend/utils.py` 新增 `is_number`/`safe_number`/`format_number`（缺依据显示 `--` 而非编造 0），共 **29 处**格式化点改为显式判空，另给出 7 处「判定为非风险」的依据清单。验证：pytest `13 failed, 335 passed` 失败集合与基线一致；冒烟 **15/15**。
- 2026-09-11 — 波次2 — 任务 5 完成（`frontend/utils.py` 1374→703 行，删净本地 CLIP/BLIP 管线，检索收敛为后端一处）；任务 6 完成（新增 `src/trajectory/builder.py`，`src/stitching` **首次线上生效**；`/trace` 由单摄像头退化为**跨摄像头**：`['c040']` → `['c004','c005','c003','c002','c001']`，`inference_segments` 由恒空变为有内容，`overall_confidence` 由硬编码 0.0 变为真实 0.3411，每段新增 `basis` 标注）。pytest `13 failed, 335 passed`，失败集合与基线逐条一致；**冒烟 15/15**（含真实拉起 Streamlit 并探活 `/_stcore/health`）。
- 2026-09-11 — 波次2 新发现 — ① **项目存在两套不兼容的 CLIP 特征空间**（512/ViT-B-16 覆盖 100% vs 768/ViT-L-14 覆盖 1.39%），配置声明与实际不符，见 §0.5；② **属性标注噪声极大**：同一车辆 `vehicle_type`/`color` 跨摄像头乃至单摄像头内部自相矛盾（已实测 `V0034`），同时损害检索过滤与拼接评分；③ 实测确认 `actual_travel_time=None` 会令前端 Excel 导出抛 `TypeError`，已派修。
- 2026-09-11 — 任务#9 — `git init` + 首提交 `a155f0a`（180 文件，最大 92KB），建立版本保护；`.gitignore` 排除 output/models/cityflow/qdrant_storage 等大目录。**未改任何源码**。
- 2026-09-11 — 波次1验收 — 任务 1/2/3/7 完成并经**独立复核**（不采信子 agent 自述）：pytest `13 failed, 306 passed`，失败集合与基线用 `comm` 逐个比对**完全一致**（零回归）；真实 HTTP 冒烟 `14/14 通过`，其中 confirm 非法 query_id 返回 404、`/trace` 两次调用结果完全一致（无随机性）、`/trajectory` 真实聚合出 `vehicle_id=V0322 摄像头3 检测125`。任务 8 的 `docker compose config` rc=0 验证挂载路径解析正确。
- 2026-09-11 — 任务#0（环境）— 建 venv 依赖（fastapi/uvicorn/streamlit 因被 C 盘用户级 site-packages 遮蔽，必须 `--ignore-installed` 才会真正装进 .venv）；新增 `pytest.ini`（`testpaths=tests`，把收集耗时从 438s 降到 <1s）；定位两个 sys.path 遮蔽坑并给出规避环境变量；建立基线 13 failed / 251 passed / 6 xpassed。**未改任何源码**。
- 2026-09-11 — 计划勘误 — 核实并更正了任务 5（CLIP 分支是活的，非死代码）与任务 7（`configs/camera_metadata.yaml` 是活的，不可删）两处前提错误，详见 §0.3。
- 2026-09-11：创建本计划文件，尚未开始任何代码修改。
