# 本机开发环境检查报告

**检查时间**: 2026-06-27  
**检查范围**: 交通目标时空回溯系统 (H:\trajectory CLIP) 开发环境

---

## 1. 系统信息

| 项目 | 详情 |
|------|------|
| 操作系统 | Windows 11 25H2 (NT 10.0.26200.0) |
| CPU | 12th Gen Intel Core i5-12400F (6核12线程) |
| 内存 | 15.84 GB |
| GPU | NVIDIA GeForce RTX 5070 (12GB VRAM) |
| GPU 驱动 | 581.15 |
| CUDA 版本 (驱动支持) | 13.0 |

## 2. Python 环境

| 项目 | 状态 | 版本/路径 |
|------|------|-----------|
| Python | 已安装 | 3.13.5 |
| pip | 已安装 | 25.1 (来自 D:\Anaconda) |
| conda | 已安装 | 25.5.1 |
| Anaconda | 已安装 | 路径 D:\Anaconda |

## 3. GPU / CUDA 环境

| 项目 | 状态 | 详情 |
|------|------|------|
| nvidia-smi | 正常 | RTX 5070, 驱动 581.15, CUDA 13.0 |
| CUDA Toolkit (nvcc) | **未安装** | nvcc 不在 PATH 中 |
| PyTorch | **已安装 (CUDA)** | 2.11.0+cu128 (CUDA 版，GPU 加速可用) |
| CUDA 可用性 | **可用** | torch.cuda.is_available() = True，设备: NVIDIA GeForce RTX 5070 |

## 4. Docker 环境

| 项目 | 状态 | 版本/详情 |
|------|------|-----------|
| Docker | 已安装 | 29.5.3 |
| Docker Compose | 已安装 | v5.1.4 |
| 存储驱动 | overlay2 | 基于 WSL2 |
| NVIDIA Runtime | 已配置 | runc + nvidia |
| Docker 内存限制 | 7.67 GiB | |

## 5. 缺失工具

| 工具 | 状态 | 说明 |
|------|------|------|
| Node.js | **未安装** | 前端开发需要 |
| Git | **未安装** | 版本控制需要 |
| FFmpeg | **已通过 pip 安装** | ffmpeg-python + imageio-ffmpeg (含二进制) |

## 6. 磁盘空间

| 盘符 | 已用 (GB) | 可用 (GB) | 状态 |
|------|-----------|-----------|------|
| C: | 128.89 | 21.11 | 空间紧张 |
| D: | 179.92 | 23.63 | 空间紧张 |
| E: | 78.60 | 21.41 | 空间紧张 |
| F: | 398.97 | 101.03 | 尚可 |
| **G:** | **0.33** | **1862.65** | **充裕** |

## 7. 已安装的关键 Python 包

| 包名 | 版本 | 状态 |
|------|------|------|
| torch | 2.11.0+cu128 | CUDA 版 (cu128，GPU 加速) |
| torchvision | 0.26.0+cu128 | CUDA 版，与 torch 版本匹配 |
| torchaudio | 2.11.0+cu128 | CUDA 版，与 torch 版本匹配 |
| transformers | 4.57.6 | 已安装 |
| sentence-transformers | 3.3.1 | 已安装 |
| tensorflow | 2.21.0 | 已安装 |
| scikit-learn | 1.6.1 | 正常 |
| scipy | 1.15.3 | 正常 |
| qdrant-client | 1.12.1 | 正常 |
| streamlit | 1.45.1 | 正常 |
| uvicorn | 0.34.0 | 正常 |
| Pillow | 已安装 | 正常 |
| tqdm | 4.67.1 | 正常 |
| opencv-python | 4.13.0 | 新安装 |
| ultralytics | 8.4.80 | 新安装 |
| open-clip-torch | 3.3.0 | 新安装 |
| cn-clip | 1.6.0 | 新安装 (--no-deps, lmdb 版本不兼容) |
| flask | 3.1.0 | 已安装 |
| fastapi | 0.115.6 | 已安装 |
| ffmpeg-python | 0.2.0 | 新安装 |
| imageio-ffmpeg | 0.6.0 | 新安装 (含 FFmpeg 二进制) |

## 8. 问题汇总与修复记录

### 已修复
1. **PyTorch CUDA 版本已修复** → 从 2.12.1+cpu 升级至 2.11.0+cu128 (CUDA 版)，GPU 加速正常
2. **FFmpeg 未安装** → 已通过 pip 安装 ffmpeg-python + imageio-ffmpeg
3. **opencv-python 未安装** → 已安装 4.13.0
4. **ultralytics 未安装** → 已安装 8.4.80
5. **open-clip-torch 未安装** → 已安装 3.3.0
6. **chinese-clip 未安装** → 已安装 1.6.0 (--no-deps)
7. **flask / fastapi** → 已确认安装

### 未解决 / 待处理
1. **CUDA Toolkit (nvcc) 未安装** → conda 频道 SSL 同样失败，需网络恢复后安装
2. **cn-clip lmdb 版本不兼容** → cn_clip 要求 lmdb==1.3.0，当前为 1.6.2，可能影响功能
3. **Git 未安装** → 需要手动安装
4. **Node.js 未安装** → 需要手动安装

### 网络问题说明
- download.pytorch.org: 之前 SSL TLS 握手失败，现已恢复正常（2026-06-28 成功下载 cu128 版本）
- conda 频道 (pytorch/nvidia/conda-forge): 之前 SSL 失败，未再次尝试
- pypi.tuna.tsinghua.edu.cn: 正常工作（仅含 CPU 版 PyTorch）
- mirror.sjtu.edu.cn: 根路径可访问，但 pytorch-wheels 子目录重定向至 download.pytorch.org

### PyTorch CUDA 修复记录 (2026-06-28)
- 原版本: torch 2.12.1+cpu / torchvision 0.27.1+cpu
- 修复方法: 卸载 CPU 版后，从 `https://download.pytorch.org/whl/cu128` 安装 CUDA 版
- 修复后版本: torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / torchaudio 2.11.0+cu128
- 验证结果: `torch.cuda.is_available()` = True，GPU 计算正常（NVIDIA GeForce RTX 5070）
- 注意: SSL 问题已自行解决，download.pytorch.org 可正常访问
