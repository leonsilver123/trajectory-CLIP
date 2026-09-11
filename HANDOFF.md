# HANDOFF — 重构任务交接（新终端接手用）

> 生成时间：2026-09-11
> 本文件自包含，新会话无需依赖之前的对话上下文。

---

## 0. 一句话现状

**代码 0 修改**。已完成环境探查、已在 H 盘建好 venv。下一步：装依赖 → 启动 4 个子 agent 执行 `PLAN.md` 的 9 个任务 → 验收 → smoke test → 交付报告。

---

## 1. 任务目标（用户意图）

- **overnight 跑完 `PLAN.md` 全部 9 个任务**（能全做就全做）。
- 可调用**最多 4 个子 agent**，允许并行；但要注意串并行、避免文件冲突。
- **每完成一个完整任务后先验收**，再开下一个。
- **整体做完后**在前后端做 **smoke test**。
- 最终写一份**交付报告**。
- 每次修改都要把「改了什么 / 没改什么」写进 `PLAN.md` 的变更记录。

---

## 2. 环境现状（务必按此操作，别再重复探查）

| 项 | 值 |
|---|---|
| 项目根 | `H:\trajectory-CLIP` |
| 系统 Python 3.14 | `C:\Python314\python.exe` —— **裸，无依赖，不要用** |
| conda `py313` | `D:\Anaconda_envs\envs\py313\python.exe`（Python 3.13.9） |
| py313 已有 | **torch 2.9.1+cu128（CUDA 可用）、numpy 2.3.3、PIL** |
| py313 缺失 | fastapi / pytest / yaml / cv2 / pandas / sklearn / scipy / requests，**且自身无 pip** |
| **已建 venv** | **`H:\trajectory-CLIP\.venv`** ✅ 已创建 |

**venv 创建命令（已执行成功）**：
```bash
D:/Anaconda_envs/envs/py313/python.exe -m venv --system-site-packages .venv
```
→ 用 `--system-site-packages` 从 py313 **继承 torch（无需重装几个 GB）**。已验证：
```
.venv/Scripts/python.exe -c "import torch,numpy"
# torch 2.9.1+cu128 cuda True | numpy 2.3.3
```

**待办（上一步被用户打断，尚未执行）——安装轻量依赖**：
```bash
cd H:/trajectory-CLIP
.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  fastapi "uvicorn[standard]" pytest requests pyyaml pydantic httpx streamlit
```
（conda 用的是 USTC 镜像 `mirrors.ustc.edu.cn`，pip 用清华源即可；如失败换阿里/USTC 源）

**重要约束**：
- **环境必须保留在 H 盘**（用户明确要求），不要污染 C/D 盘全局环境。
- **不装重依赖**：`cn_clip` / `faiss` / `ultralytics` / `cv2` / `pandas` / `sklearn` / `scipy` —— 核心重构不需要它们。设计上：
  - 检索走**无 CLIP 的降级路径**（属性/文本匹配）；
  - 回溯走**真实 vehicle_id 聚合**（只依赖 `output/cityflow_results.json`，数据在，107MB）。
- 数据文件：`output/cityflow_results.json`（107MB，**存在**）。

---

## 3. 代码修改状态

- **0 修改** —— 尚未动任何源码。
- 已新建文档：
  - `PLAN.md` —— 9 任务计划 + 进度追踪（**每步都要更新它**）
  - `CLAUDE.md` —— 项目结构说明
- `HANDOFF.md` —— 本文件

---

## 4. 计划摘要（详见 `PLAN.md`）

| # | 任务 | 阶段 |
|---|------|------|
| 1 | 后端会话状态机（新建 `src/common/session_store.py`，改 search/confirm/backtrack 路由） | P0 |
| 2 | 删除伪回溯（`api/routes/backtrack.py` 的 random 伪造 + hash 随机抽目标） | P0 |
| 3 | ID 解析收敛（新建 `src/common/ids.py`，替换三处 `_extract_vehicle_id`） | P0 |
| 4 | 数据契约落盘（SQLite/Parquet + datastore，在线不读大 JSON） | P1 |
| 5 | 检索管线收敛成一条（删死代码分支，下沉到后端） | P1 |
| 6 | 跨镜拼接统一到 `TrajectoryBuilder`（接上 `src/stitching`） | P1 |
| 7 | 删死代码（search_backup.py / 旧 configs / flask / mock_data 评估） | P2 |
| 8 | 修 Docker（`docker-compose.yml` 写死的 `H:/trajectory CLIP` 带空格） | P2 |
| 9 | git 保护（初始化空 `.git`，首次提交 + .gitignore） | P2 |

---

## 5. 建议的 agent 编排（串并行，避免文件冲突）

**波次 1（并行 3 个 agent，文件不重叠）**
- **Agent A —— P0 全部（任务 1+2+3）**：三者都改 `api/routes/search.py`、`confirm.py`、`backtrack.py` + `src/common/`，**强重叠，必须在同一 agent 内串行**。
- **Agent B —— 任务 8（Docker）**：只改 `docker-compose.yml` / `Dockerfile.backend` / `Dockerfile.frontend`，与 A/C 无重叠。
- **Agent C —— 任务 7（死代码）**：只删文件（`api/routes/search_backup.py`、被替代的旧 `configs/camera_metadata.yaml`、`requirements.txt` 的 flask、评估 `frontend/mock_data.py`）。**删前先 grep 确认无引用**。

**波次 2（等 Agent A 完成后，1 个 agent 串行）**
- **Agent D —— P1（任务 4+5+6）**：都改 `search.py`/`backtrack.py`/`utils.py`，**串行**。

**波次 3（主 agent 自己做）**
- 任务 9（git）+ 前后端 smoke test + 交付报告。

---

## 6. 关键项目情报（省去重新探索）

- **三套代码，只有「JSON 直读」一条活**：`src/` 是设计稿（未接入服务），`api/` 是实际服务，`frontend/utils.py` 是前端降级实现。
- `api/routes/search.py` 用 CN-CLIP **ViT-B-16** + `output/clip_vectors.faiss`（**该文件不存在 = 死代码**）；`frontend/utils.py` 用 **ViT-L-14 + BLIP**。两套不一致。
- `api/routes/backtrack.py`：
  - `/trace` 用 `random.uniform()` **伪造**时间戳/置信度/旅行时间，`_find_target_id()` 用 `hash()` **随机抽**目标 → **伪回溯，要删**。
  - `/trajectory` 按 **vehicle_id 真实聚合** → **这才是可信路径**。
- `confirm` 接口**无状态**（只 echo `status="confirmed"`）。
- **ID 体系**：`target_id`=`CF3_c001_V0034_000001`；`vehicle_id`=`V0034`（解析"V+数字"部分）；`track_id`=`CF3_TRACK_c001_V0034`。三处重复反解逻辑要收敛。
- 已知文档与代码严重脱节（`RESUME_PROJECT.md` 的指标对应的是未接入的 `src/` 设计稿）。

---

## 7. 立即执行的下一步

1. 装依赖（见 §2 的命令）。
2. 跑基线：`.venv/Scripts/python.exe -m pytest`（确认现有测试状态；`tests/` 测的是 `src/` 纯逻辑模块）。
3. 启动波次 1 的 3 个 agent（`subagent_type: general-purpose`）。
4. 每个任务完成后：跑验证 → 更新 `PLAN.md` 变更记录（改了什么/没改什么）→ 再开下一个。
5. 全部完成后：后端 `uvicorn api.main:app` + curl `/health` 及检索/确认/回溯流程；前端 import/启动检查 → 写交付报告。

---

## 8. 备注

- `PLAN.md` 是**进度唯一真源**，接手后先读它。
- 当前 `.git` 是**空的**（无 HEAD/commit），所以任何代码改动**无版本保护**，任务 9 之前请小心操作。

---

## 9. 工作规矩（必须遵守）

### 9.1 沟通与风格
- **全程中文**回复与文档。
- 代码/文档延续项目现有风格：**中文 docstring + 中文注释**，英文标识符；注释密度贴合周围代码，不额外堆砌。

### 9.2 改代码的红线
- **不破坏现有可运行路径**：改接口必须向后兼容，保留原函数签名，除非 `PLAN.md` 明确要求修改。
- **打死不动 ID 格式**：`target_id`(`CF3_c001_V0034_000001`) / `track_id` / `vehicle_id` 是所有模块的隐性契约，改一处会连锁炸三处；若确需改，必须在同一次内改完所有引用点。
- **改前先 grep**：动任何符号前先确认引用点；删文件/函数前先确认无 `import`。
- **scope 控制**：只做 `PLAN.md` 内的事，不做"顺手重构"；发现额外问题写进 `PLAN.md` 遗留项，**不擅自扩范围**。
- **每次改动后代码可 import**：至少 `python -c "import <模块>"` 能通过。

### 9.3 验证与诚实（硬要求）
- **每个任务完成必须验收**（跑测试 / import 检查 / 接口调用），**不验收不开下一个**。
- **如实汇报**：测试失败就贴失败输出；某步跳过就明说跳过；没实现的功能**不许写"已完成"**。
- **smoke test 必须真跑**，不能只声称跑过。

### 9.4 `PLAN.md` 维护（进度唯一真源）
- 每步做完：状态 `[ ]→[~]→[x]`，勾选验收项，并在「变更记录」写清**改了哪些文件、具体改了什么、还有什么没改**。
- 同步更新顶部「最后更新」时间与「进度总览」表；追加一条全局变更日志。

### 9.5 子 agent 编排
- 上限 **4 个**；**文件不重叠才并行**，重叠的必须在**同一个 agent 内串行**。
- 子 agent 的 prompt 必须**自包含**（子 agent 不带主会话上下文）：写清任务范围、要改的具体文件、约束、验收标准、报告格式。
- 子 agent 完成后，主 agent **复核其改动**（读 diff / 跑验证）再计入 `PLAN.md`；**不许直接采信子 agent 的自述**。

### 9.6 环境与文件
- 一切命令用 **H 盘 venv**：`.venv/Scripts/python.exe ...`。
- **环境只放 H 盘**，不污染 C/D 盘全局环境。
- **不装重依赖**（`cn_clip`/`faiss`/`ultralytics`/`cv2`/`pandas`/`sklearn`/`scipy`）。
- 不移动、不提交大文件：`output/`、`models/`、`*.pt`、107MB 的 `cityflow_results.json`。

### 9.7 交付
- 最终交付报告须含：**做了什么 / 没做什么 / 验证结果（真实输出）/ 遗留问题 / 如何运行**。
- 「没做什么」和「遗留问题」不得省略——不允许只报喜。
