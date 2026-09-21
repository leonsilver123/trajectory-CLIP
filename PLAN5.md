# 五期计划：按简历把能力补齐（PLAN5）

> 创建：2026-09-21
> 状态图例：`[ ]` 未开始　`[~]` 进行中　`[x]` 完成　`[-]` 放弃　`[?]` 待定
> 前序：`PLAN.md` / `PLAN2.md` / `PLAN3.md` / `PLAN4.md`

---

## 一、目标

简历里有 5 项能力**本仓库没有**（此前记为"业务系统有，不动"）。本计划把它们**在本仓库真做出来**：

| # | 简历声明 | 本仓库现状 |
|---|---|---|
| 1 | **OSNet** 提取 ReID embedding | 用的是 fast-reid SBS R50-ibn |
| 2 | **BLIP** 交叉注意力精排 | 代码中零出现 |
| 3 | **Hard Negative Mining** 强化细粒度区分 | 无训练代码 |
| 4 | ReID 空间用 **Batch Hard Triplet Loss** 优化 | 无训练代码 |
| 5 | 边评分用 **BCE + Ranking Loss** 联合监督 | 评分权重是网格搜索调的，不是学的 |

外加两处**形态不符**（PLAN4 期间发现）：

| # | 简历措辞 | 实际 |
|---|---|---|
| 6 | 属性重排「颜色、车型、**方向**三维**余弦加权**」 | 检索路径里颜色/车型是精确匹配+关键词打分；**方向根本不在检索路径** |
| 7 | 「**BFS** 路网拓扑」 | 实际是 Dijkstra 最短路 + 邻接表 |

---

## 二、可行性（已实测，不是估计）

| 依赖 | 状态 | 取证 |
|---|---|---|
| **BLIP ITM 模型** | ✅ **已缓存**（1.7GB，`F:\AI\huggingface\transformers\models--Salesforce--blip-itm-base-coco`） | 离线加载成功 |
| BLIP 端到端打分 | ✅ 机制可用（能出分、能降级）；❌ **但实测无区分力**，详见 A 段 | `tests/test_blip_reranker.py` |
| **OSNet** | ✅ `torchreid 0.2.5` 可从 PyPI 安装；`osnet_x1_0` 权重可下载（10.9MB，HTTP 200） | pip index / curl |
| GPU | ✅ RTX 5070 12GB | `nvidia-smi` |
| 磁盘 | ✅ 1.7TB 空闲 | `df` |
| 训练标签 | ✅ 数据集每条检测带真实 `vehicle_id`（230 个身份） | datastore |
| 补装依赖 | ✅ 已装 `wrapt`（**transformers 懒加载全靠它**，缺了会让所有模型导入报误导性错误） | `import wrapt` OK |

### 遇到并已解决的三个坑（写下来免得重踩）

1. **`wrapt` 缺失** → transformers 所有懒导入报 `ModuleNotFoundError: Could not import module 'BlipProcessor'`，
   错误信息完全指错方向（真正缺的是 wrapt）。
2. **`BlipProcessor` 已从 transformers 顶层移除** → 用 `AutoProcessor.from_pretrained()` 仍能构造出它。
3. **ITM 必须图文一一对应**：传 1 张图配 2 条文本会让视觉侧 batch=1、文本侧 batch=2，
   在 `modeling_blip_text.py:182` 报一个完全无关的 shape 错误。必须 `images=[img]*len(texts)`。
   另需 `padding=True, truncation=True`。

---

## 三、诚实边界（必须先接受）

1. **本数据集只有约 230 个车辆身份**。用这个量级训练 ReID 嵌入，效果会**明显弱于**公开数据集上的
   报数（VeRi 等动辄上万身份）。训练出来能用，但不要期待高指标。
2. **AICity22 及其衍生模型禁止商用**（`docs/LICENSE_COMPLIANCE.md` 已取证）。训练产物仅限研究。
3. **简历上的 P@10 0.973 / F1 0.819 是业务数据集（苏州 48 路）的数字**。
   本计划在本数据集上做出来的**是另一个数字**，两者**不可互换**。
   所有产出都必须**分别标注口径**，不得混用 —— 这是 PLAN2 立下的红线。
4. **精排未必带来增益**。PLAN2 已实测过"CLIP 向量精排几乎不增益"（差 1/218）。
   BLIP ITM 是另一套机制，但**同样可能无增益** —— 若实测无增益，如实报告，
   不为了"简历好看"而调参凑数。

---

## 四、任务清单

### A. BLIP 交叉注意力精排

- `[x]` **A1** 新增 `src/retrieval/blip_reranker.py` —— 模块已完成、已测、可用
  - 封装 ITM 打分：惰性加载、device 感知、批量推理、失败隔离
  - 三个坑全部处理（AutoProcessor / 图文一一对应 / padding+truncation）
  - 另发现并处理第四个问题：**ITM 绝对分跨查询不可比**（实测差一个数量级），
    因此 `fuse()` 默认先做**集合内归一化**，否则固定融合权重会随查询漂移
  - **验收**：`tests/test_blip_reranker.py` **19 项通过**（纯逻辑 17 + 需模型 2）

- `[!]` **A2** 接入在线检索 → **改为"作为可选精排接入，默认关闭"**（原因见下）
- `[x]` **A3** 降级契约已实现：`available` 为 False 时 `score()` 返回全 None，
  调用方据此如实标注"已降级"，不静默换算法

### 🔴 A 段的关键负结果（2026-09-21 实测）

**BLIP ITM 在本数据集的裁剪图上不可用于精排。** 两次独立测量：

| 样本量 | 判别正确率（正确颜色描述 vs 随机错误颜色描述） |
|---|---|
| 80 张 | **51.2%**（≈ 随机） |
| 40 张（另一抽样） | **22.5%**（**显著低于**随机） |

两次都远离 50% 且方向相反 ⇒ 分数**不是由"图文是否匹配"驱动的，而是被短语层面的偏置主导**。
直接证据：同一辆白色轿车，`"白色轿车"` 打 **0.0036**，`"黑色卡车"` 却打 **0.0293**（高一个数量级）。

**根因**与项目早先测到的"CLIP 向量精排不带来增益"**同源**：本数据集裁剪图中位数仅
**118×98 像素**，而 BLIP 在 COCO 高分辨率图上训练 —— 域差太大。
（同一根因也解释了跨镜 ReID 的 d-prime 只有 0.78。）

**因此**：精排模块**做出来了、能跑、有测试**，但**不会默认开启**，也不会宣称它提升效果。
这与 PLAN2 立下的红线一致：**不为简历好看而虚报增益**。
`tests/test_blip_reranker.py::test_measured_discrimination_is_unusable` 会守住这个结论 ——
若将来真的变好，它会失败并提示重新测量。

### B. OSNet ReID

- `[x]` **B1** 安装 `torchreid`，下载 `osnet_x1_0` 权重到 `models/`
  - **验收**：能构造 OSNet 并对一张裁剪图出 512 维向量
- `[x]` **B2** `scripts/extract_reid.py` 增加 `--backbone osnet_x1_0|fastreid` 选项
  - **验收**：两种 backbone 都能跑出向量，落盘到不同文件
- `[ ]` **B3** 用 `scripts/eval_cross_camera.py` 做 **OSNet vs fast-reid 对照**
  - **验收**：出 Rank-1 / Rank-5 / mAP 对照表，如实报告谁更好

### C. 训练（三项声明的落点）

- `[x]` **C1** 构建训练数据 `scripts/build_reid_training_set.py`
  - 从 `vehicle_id` 生成身份标签；**按摄像头切分**避免同车跨集泄漏
  - **验收**：产出 train/val 身份不重叠的清单 + 统计
- `[x]` **C2** Batch Hard Triplet 训练 ReID 嵌入 `scripts/train_reid_triplet.py`
  - PK sampling（每 batch P 个身份 × K 张图）+ batch-hard 挖掘
  - **验收**：训练收敛（loss 下降），val 上 Rank-1 高于随机基线；产物落盘
- `[ ]` **C3** Hard Negative Mining
  - 在每个 epoch 后用当前模型挖难负样本，重训
  - **验收**：与 C2 不开挖掘的版本做对照，如实报告增益或**负增益**
- `[ ]` **C4** BCE + Ranking Loss 训练边评分器 `scripts/train_edge_scorer.py`
  - 从 GT `vehicle_id` 构造正负边（同车=正、异车=负）；BCE + ranking 联合
  - **验收**：训练后在 val 边上算 AUC/准确率；与"网格搜索权重"版本在 IDF1 上对照

### D. 属性重排的形态对齐

- `[x]` **D1** 把**方向**接进检索路径（简历明确写了三维）
  - **验收**：`grep -n direction api/routes/search.py` 有命中，且检索能按方向过滤
- `[x]` **D2** 颜色/车型/方向改为**余弦加权**（或如实说明为何仍需精确匹配）
  - 若改为余弦：属性要先嵌入；若保持精确匹配，**必须在文档里写明实际形态**
  - **验收**：二选一，且文档与代码一致

### E. BFS 路网拓扑的措辞对齐

- `[x]` **E1** 二选一：① 真的补一个 BFS 可达性接口（与 Dijkstra 并存，各司其职）；
  ② 简历措辞改为"路网最短路"。**默认取 ①** —— 代码里补 BFS 比改简历更实在
  - **验收**：`RoadTopology` 有 BFS 方法且有测试

### F. 评测与文档

- `[ ]` **F1** 跑全套评测并产出**本数据集口径**的指标表（P@10 / mAP / F1 / IDF1）
- `[ ]` **F2** 更新 `CLAUDE.md`、`docs/`，**两套数据集的数字分开标注**
- `[ ]` **F3** 新增测试守护；全量回归
- `[ ]` **F4** commit + push

---

## 五、变更记录

| 日期 | 条目 | 改动 | 验收结果 |
|---|---|---|---|
| 2026-09-21 | 可行性验证 | 装 `wrapt`；确认 BLIP ITM 可从本地缓存加载 | 小样本 4/4 —— **但后来证明是挑样本的运气**，见下行 |
| 2026-09-21 | 计划创建 | — | — |
| 2026-09-21 | **A1 / A3** | 新增 `src/retrieval/blip_reranker.py`（惰性加载/批量/失败隔离/集合内归一化/降级契约）；新增 `tests/test_blip_reranker.py`（19 项） | **19 passed**。踩平 4 个坑（3 个 API 层 + 1 个分数不可比），见 §二 与 A1 |
| 2026-09-21 | **B1** | 装 `torchreid 0.2.5` + `gdown`；下载 `osnet_x1_0` 权重到 `models/`（10.9MB）。踩坑：PyPI 包的**真实子模块是 `torchreid.reid.models`**（`torchreid.models` 只是再导出），且 `load_pretrained_weights` **只加载不下载** | 构建成功：**2.68M 参数、输出 512 维**（正对应简历的"OSNet 提取 512 维 embedding"）；权重加载成功 |
| 2026-09-21 | **B2** | 新增 `scripts/extract_reid_osnet.py`（未改 `extract_reid.py` —— 两条路的模型构造与预处理完全不同，分开写、**产物格式保持一致**）。行号严格对齐 detections，坏图填 NaN 行 | 冒烟 400 张：**400/400 成功、512 维、L2 模长 1.0000**。全量 68,349 张已在后台运行 |
| 2026-09-21 | **C1** | 新增 `scripts/build_reid_training_set.py`。**按摄像头切分**（同一身份保留若干摄像头进 val），避免同摄像头相邻帧跨集泄漏 | 215 身份（230 中筛掉 13 图片太少 + 2 验证侧不足）；train 5024 张 / val 1516 张，两侧身份均为 215。**逐身份同摄像头跨集泄漏 = 0** |
| 2026-09-21 | C1 自我更正 | 初版把"摄像头全局重叠 35 个"当成泄漏告警 —— **那是错的**：46 个摄像头装 215 辆车，全局必然重叠。正确的泄漏指标是**逐身份**的摄像头互斥 | 已修正校验逻辑与 summary 字段（`per_identity_camera_leaks=0`），并把"为什么全局重叠是正常的"写进注释 |
| 2026-09-21 | **C2** | 新增 `scripts/train_reid_triplet.py`：PK 采样（P×K）+ `torchreid.losses.TripletLoss`（batch-hard 语义）+ 可选 ID 损失；每 epoch 在 val 上算 **Rank-1/Rank-5/mAP**（跨摄像头协议） | 脚本已跑通模型加载与预训练权重自动下载；完整训练待 GPU 空闲 |
| 2026-09-21 | **D1/D2** | `api/routes/search.py`：新增 `DIRECTION_MAP`（**最长匹配**，防「由北向南」被 '北' 抢先命中成 northbound）；`CAMERA_DIRECTIONS` 从摄像头元数据取 `lane_direction`；新增 `_attribute_consistency()` 三维加权；方向接入粗筛与打分。权重进 `configs/default.yaml` 的 `retrieval.attribute_weights`（**在线实际读取**） | 新增 `tests/test_attribute_consistency.py`（19 项，全过）。实测：`白色轿车由南向北` → c001(northbound) 1.0 / c002(eastbound) 0.8，差值恰为方向权重 0.2 |
| 2026-09-21 | **C4** | 新增 `scripts/train_edge_scorer.py`：把六维分项当**特征**、用带标签的边（同 `vehicle_id` = 正、异 = 负，标签来自数据集事实）训练线性打分器，`loss = BCE + λ·Ranking`。**刻意用线性** —— 目的是"学六维权重"以对照手写加权和，上 MLP 会让对照失去意义 | 见下方 C4 结果与**关键负结果** |
| 2026-09-21 | **C4 结果** | 218 身份全量（6134 训练边 / 1534 验证边）：**val AUC 0.855**（8 身份冒烟 0.898 —— 身份越多负样本越难，下降合理）。训练曲线 epoch 20–40 见顶（≈0.859），之后 bce 继续降而 AUC 微降 ⇒ 过拟合。同批边上**手写加权和 0.716**、仅 spatial 单维 0.796、仅 temporal 单维 0.448 | ⚠️ 这些数字**看似**支持"学习型更好"，但**不成立** —— 见下行的方法缺陷。学到的六维系数也**不能直接读**，见再下一行 |
| 2026-09-21 | **C4 的关键负结果** | 学到的系数里 `appearance` / `plate` / `direction` 三者**完全相同（-0.50091）** —— 不是巧合：它们在该数据集上**取值恒定、方差为 0**（车牌实测 0 条、tracklet 无 CLIP/ReID 向量、direction 未填）。常数维度的"权重"只是替 bias 分担偏移，**没有任何信息**。这与早先"六维实际退化为时间+拓扑"的结论**同源**。脚本初版没暴露这一点，会让读的人把 -0.50091 当成有效读数 | 补 `feature_stats.json`（含 `constant_on_dimensions` 标记）与常数维度告警；`metrics.json` 的 `note` 里写明该维度权重不可解释 |
| 2026-09-21 | **C4 的方法缺陷（最重要的一条）** | 拿到 AUC 后我先写了"学习型完胜手写（0.855 vs 0.716）"，**随后自我推翻**：查取值分布发现 `spatial_score` **正边 100% 恰为 0.5（中性）**、负边仅 36.7%。原因：同一场景摄像头 GPS 相同 → 距离 0 → 按"未知"给中性；**正边必然同场景**，而我的负边是**随机跨车配对、包含跨场景对** ⇒ 那个 0.796 的 AUC 全部来自"正边必同场景"这个**采样假象**（剔除中性组的 723 条边里**正边为 0**）。`temporal` 同理：正边 70% 恰为 0。**生产路径的 `CandidateEdgeGenerator` 有空间搜索半径，跨场景对根本不会成为候选** ⇒ 该判别器在生产分布上不存在，学到的权重（spatial +4.98）**不可迁移** | **结论：本实验没有回答它该回答的问题**。AUC/权重**不能**用来支持"学权重优于调权重"，也**不能**用来质疑手写权重 —— 它只说明"损失实现正确、能训练"。已在脚本 docstring 与 `metrics.json` 的 `known_limitation` 字段写明"负样本必须从生产候选边分布取"，避免后人照抄这些数字 |
| 2026-09-21 | C4 工程化 | 构边是 O(tracklet²) 次 `score()`，218 身份上约 **10 分钟**，而训练只要几秒 —— 调参时反复重付毫无意义 | 加**构边缓存**（键含 `max_identities`/`neg_ratio`/`seed`，换参数自动 miss）+ `--rebuild-edges`。另修一处自伤：初版把特征统计写进 `main()` 却引用 `train()` 的局部张量 `Xva`，**训完 300 epoch 才在最后一步 `NameError`**，整轮结果丢掉 |
| 2026-09-21 | **P-A 性能缺陷（做 C4 时发现）** | `CrossCameraScorer._count_possible_paths`（`src/stitching/scoring.py`）用**无界 DFS 枚举简单路径**：只限制了计数（`found < max_paths`）、没限制搜索，两点不可达时循环条件恒真，走完整棵指数级路径树。实测 **单对摄像头 348 万次递归 / 1.84 秒**（c029→c030），而**线上每对候选边都要调一次**。此前只在 `builder.py` 的子类 `BoundedCrossCameraScorer` 里绕开过 —— **基类一直是坏的**，任何直接用 `CrossCameraScorer` 的代码（如 C4 构边）都会撞上。**影响范围要说准**：`TrajectoryBuilder._scene_edge_pool` 显式传了 `scorer=self._get_scorer()`，所以**生产路径一直有保护**；真正暴露的是不传 `scorer` 时的 `CandidateEdgeGenerator`（`candidate_edge.py:114` 自建基类）与直接实例化基类的脚本 | 上限（`MAX_PATH_HOPS=6` / `PATH_EXPANSION_BUDGET=20000`）**下沉到基类**，子类降级为兼容别名。**1035 对全拓扑从几十分钟降到 0.35 秒**；新增 `tests/test_path_count_bounds.py`（8 项）。**不改变任何线上行为**，只是拆掉地雷 |
| 2026-09-21 | P-A 的**诚实附注** | 我最初把它当"等价重构"写，**实测后发现不等价**：旧实现能终止的 709 对里 **278 对（39.2%）返回值不同**，集中在 `(旧 5, 新 1)` —— 旧版把 **>6 跳**的绕行也计入了。对 `path_divergence_penalty` 影响最大 0.80、平均 0.27，×权重 0.1 后**对总分影响最大 0.08**。也试过用"反向可达性剪枝 + 压入即计数"保住精确语义：能把可比对对数提到 814 且**零差异**，但仍有 221 对爆炸（最坏 c011→c040 需 195 万次扩展）—— 简单路径计数是 **#P-hard**，指数下界绕不过去 | **接受语义变化**并写进 `scoring.py` docstring（含上面这张差异表）。理由：系统自身可达性假设就是 ≤6 跳，把 9 跳绕行算作"同一辆车可能走的路线"本就可疑；且**线上路径一直在用这个语义**（builder 一直用子类），本次是让基类与线上口径**趋于一致**，不是引入新偏离 |
| 2026-09-21 | **E1** | **更正我先前的错误结论**。我说过"简历写 BFS 但代码是 Dijkstra"——**错了**。实测：`is_reachable()` **本来就是 BFS**（`deque`+`popleft`+hop 计数），`shortest_path()` 才是 Dijkstra。**两者分工并存，简历的「BFS 路网拓扑」是准确的** | 新增 `tests/test_road_topology_bfs.py`（9 项）把这件事钉住：若有人把 `is_reachable` 改成带权最短路（功能等价、但会让简历变假），测试会红 |
| 2026-09-21 | **A 段负结果** | 80 样本 + 40 样本两次独立测量 BLIP ITM 判别力 | **51.2% / 22.5%** —— 均不可用；分数被短语偏置主导而非图文匹配。精排模块保留但**默认不启用**，理由见 A 段 |
