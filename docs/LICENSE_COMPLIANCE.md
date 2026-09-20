# 许可合规与商用边界说明（二期 T11）

> 生成：2026-09-12
> 依据：全部**逐条实测**（许可原文直接从 PDF 提取、包许可从已安装 metadata 读取），
> 不引用二手结论。凡未经验证的都明确标注「**未验证**」。

---

## 〇、一句话结论（先说不好的）

**本仓库 = 可商业化的软件骨架 + 仅限学术/演示的数据与模型 + 一个必须先解决的代码级许可阻塞。**

比一期诊断更严重的一点是本轮**新发现**的：**`ultralytics`（YOLOv8）是 AGPL-3.0**，
它影响的不是"数据能不能商用"，而是**"这份代码本身能不能闭源商用"**——
这比 AICity22 的数据限制更难绕过（数据可以换客户自己的，代码的 copyleft 换不掉）。

---

## 一、数据许可：AICity22 / NVIDIA（**逐条原文取证**）

原文取自 `cityflow/AICity22_Track1_MTMC_Tracking/Dataset License AIC2022.pdf`
（3 页，本次重新提取正文核对，非转述）：

**(1) 许可授予（§1）**
> (ii) Upon the conclusion of a challenge or event … you may use the DATASET,
> including models developed using the DATASET, for **non-commercial, academic purposes only**

**(2) 限制（§2）**
> a. You may not use the DATASET or any models developed using the DATASET for any
> **commercial or production purpose**.
> b. You may not remove copyright or other proprietary notices …
> c. Except as expressly authorized under Section 1, you may not copy, sell, rent,
> **sublicense, transfer or distribute** the DATASET, or share with others.
> d. Unless authorized in writing by NVIDIA, you may not indicate or suggest that any content
> (including any models) created with the DATASET is **sponsored or endorsed by NVIDIA**.

**(3) ⚠️ 本轮新发现的硬约束（§1(iii)）**
> While the DATASET is not generally distributable, you may distribute a minimal portion of the
> DATASET as part of research papers … provided that … **no unredacted faces or license plates
> are reproduced or otherwise displayed**.

**这一条此前未被记录，但它直接约束本项目的前端与演示材料**：
系统的裁剪图里同时含有**行人人脸**和**车辆牌照**，且系统自带车牌 OCR 功能。
任何对外分发（论文、截图、演示视频、PPT）**必须对人脸与车牌打码**，
否则即便在学术口径下也违约。

### 1.1 受此约束的资产（清单）

| 资产 | 是否受 §2(a) 非商用约束 |
|---|---|
| `output/aicity22_crops/` 68349 张裁剪图 | ✅ 是（含人脸/车牌 → 还受 §1(iii) 打码约束） |
| `output/cityflow_results.json` 全部检测/轨迹 | ✅ 是 |
| `output/clip_vectors.faiss`、`output/reid/reid_vectors.npy` 等**特征** | ✅ 是（衍生数据） |
| **任何用本数据训练/微调的模型**（含 T12 若执行的微调产物） | ✅ 是（§2(a) 逐字包含"models developed using the DATASET"） |
| `configs/cityflow_camera_metadata.yaml` 的**摄像头 GPS 坐标** | ✅ 是（§Description-2 明确 GPS 属 DATASET "Metadata"） |
| 轨迹的**自然语言描述** | ✅ 是（同上，"natural language descriptions for vehicle tracks"） |
| **`src/`、`api/`、`webapp/` 本项目自有代码** | ❌ 否 —— 这是自研代码，许可由本项目自定 |

→ 即：**"数据/模型层不可商用，软件层可商用"这个判断依然成立**，但见第二节。

---

## 二、代码与第三方组件许可（**本轮新发现的重点**）

### 2.1 🔴 `ultralytics`（YOLOv8）= AGPL-3.0 —— 商业化的**代码级**阻塞

实测（从已安装包的 metadata 读取，非记忆）：

```
ultralytics   ver=8.4.80   License='AGPL-3.0'
              Classifier: 'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)'
```

**AGPL-3.0 的后果（与本项目的两种交付方式都冲突）**：
- **分发软件**：整个组合作品须以 AGPL 开源 ⇒ 无法闭源售卖；
- **SaaS 提供**：AGPL §13 要求向通过网络交互的用户**提供完整对应源码**。

**暴露面（实测 grep 结果）**：

| 位置 | 是否 import ultralytics | 说明 |
|---|---|---|
| `src/perception/detector.py:84-85`（`from ultralytics import YOLO`） | ✅ 是 | 属 `src/` 包，随产品一起交付 |
| `scripts/`（6 个文件） | ✅ 是 | 离线预处理 |
| **`api/`（部署中的服务路径）** | ❌ **否** | 服务只读预计算 JSON + FAISS，**不跑 YOLO** |
| `webapp/`（前端，仅静态产物，不含推理依赖） | ❌ 否 | — |

→ **重要区别（已实测证实，非推断）**：**正在运行的服务（`api/`）本身不触发 AGPL**。

取证方式：在干净进程里 `import api.main`，检查 `sys.modules` 是否出现 ultralytics ——
结果 **`ultralytics loaded? False`**，且新引入的模块中提及
ultralytics/perception/cv2 的为 **NONE**。
唯一可能牵进来的 `api/dependencies.py:59`（惰性 import `FeatureExtractor`）
经 grep 确认**没有任何 route 或 `api/main.py` 引用它 —— 是死代码**。

但只要**把整个仓库作为产品交付**、或在交付物里包含 `src/perception/detector.py`，
AGPL 就会被触发。这个边界必须在商务材料里说清楚，不能含糊。

**三条出路（按推荐度）**：

1. **换掉检测器（推荐）**：检测在离线管线里是**通用能力**，本项目的价值在检索/回溯层。
   换成宽松许可的检测器（如 **YOLOX / RT-DETR / torchvision 检测器**，Apache-2.0 或 BSD）
   即可彻底解除阻塞。成本：改 `src/perception/detector.py` 一处 + 重跑一次离线检测。
2. **购买 Ultralytics Enterprise License**（商业授权，需询价）。
3. **隔离**：检测只在客户自有环境、由客户自行获取 ultralytics 完成，交付物不含它。
   （法律上最含糊，**不推荐作为唯一方案**。）

### 2.2 其他组件

| 组件 | 许可 | 验证方式 | 商用影响 |
|---|---|---|---|
| `fast-reid`（ReID，**按需 clone 不入库**） | **Apache-2.0** | 上游仓库 `JDAI-CV/fast-reid` 的 LICENSE；本仓库不再 vendored 该源码树 | ✅ 无阻塞 |
| `paddlepaddle` | **Apache-2.0** | 已安装 metadata classifier | ✅ 无阻塞 |
| `torch` | **BSD-3-Clause** | 已安装 metadata | ✅ 无阻塞 |
| `faiss-cpu` / `fastapi` / `pydantic` | metadata **未声明** | 实测 License 字段为空 | ⚠️ 需逐个补核（上游均为 MIT/BSD/Apache，但**本机无法证实**） |
| `cn-clip` 1.6.0 | metadata **未声明** | License 字段为空，仅 Home-page=`github.com/OFA-Sys/Chinese-CLIP` | ⚠️ **未验证**；上游仓库据称为 MIT，但本项目**未取证**，商务使用前必须核实 |
| PULC `vehicle_attribute_infer` 权重 | 未验证 | 本地仅有模型文件 | ⚠️ PaddlePaddle 生态通常 Apache-2.0，**但权重未取证** |
| `yolov8x.pt` 权重 | 随 ultralytics，**AGPL-3.0** | 与 2.1 同源 | 🔴 同上 |

> **方法学说明**：表中"未声明/未验证"不是"有问题"，而是**本项目没有证据**。
> 按一期红线 1（不虚报），这里如实标注为待核，不写成"MIT/Apache 可商用"。

---

## 三、可落地的商用路径

| 路径 | 需要做什么 |
|---|---|
| **A. 客户数据交付（主路径）** | 用客户自有摄像头与标注重建索引 → 数据层 §2(a) 不再适用；软件层仍需先解决 §2.1 的 AGPL |
| **B. 学术/演示交付** | 可继续用 AICity22，但对外材料**必须对人脸与车牌打码**（§1(iii)），且注明 AICity22/NVIDIA 出处、不得暗示 NVIDIA 背书（§2(d)） |
| **C. 换检测器后开源** | 把 ultralytics 换成 Apache-2.0 检测器，则软件层可自主选择闭源商用 |

**推荐组合：C 的换检测器动作 + A 的客户数据交付** —— 两者都不依赖额外采购。

---

## 四、对外表述红线（**不许说什么**）

1. ❌ "可直接商用" —— 当前交付物含 AGPL 组件与仅学术可用的数据/模型。
2. ❌ "已获 NVIDIA 授权/认可" —— §2(d) 明令禁止暗示背书。
3. ❌ 发布含**未打码人脸或车牌**的截图/演示 —— §1(iii) 明令禁止。
4. ❌ 把 T1/T7 的**有条件口径**（如属性 52.6%）当作主指标对外宣传 —— 主口径是 28.4%/37.2%。
5. ✅ 可以说："自研的检索与跨镜回溯软件栈；数据与特征仅限学术演示；
   商用需替换为客户自有数据并解决检测器许可。"

---

## 五、待办（未完成项，如实列出）

- [ ] 逐个核实 `faiss` / `fastapi` / `pydantic` / `cn-clip` / PULC 权重的许可（本机 metadata 未声明）
- [ ] 若走商用路径 C：评估替换 `src/perception/detector.py` 检测器的改动量与精度影响
- [ ] 前端演示模式加入人脸/车牌打码开关（§1(iii) 要求）
