"""
scripts.pre_download_weights - 模型权重预下载脚本

统一管理所有模型权重的下载、检查和缓存:
- YOLOv8x 目标检测模型
- Chinese-CLIP 图文特征模型
- ResNet50 ReID 外观特征模型

所有权重缓存到 models/ 目录，并生成 weights_manifest.json 清单。

使用方式:
    # 检查所有权重状态
    python scripts/pre_download_weights.py --check

    # 下载所有缺失的权重
    python scripts/pre_download_weights.py --download

    # 强制重新下载所有权重
    python scripts/pre_download_weights.py --download --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "weights_manifest.json"


# ============================================================
# 工具函数
# ============================================================

def _ensure_models_dir() -> None:
    """确保 models 目录存在"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _compute_sha256(file_path: Path) -> str:
    """
    计算文件的 SHA256 哈希值

    Args:
        file_path: 文件路径

    Returns:
        SHA256 哈希字符串
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_dir_size(dir_path: Path) -> int:
    """
    计算目录的总大小（字节）

    Args:
        dir_path: 目录路径

    Returns:
        总大小（字节）
    """
    total = 0
    for f in dir_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为可读字符串"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


# ============================================================
# 下载函数
# ============================================================

def download_yolo(target: str) -> bool:
    """
    下载 YOLOv8x 预训练权重

    通过 ultralytics 库下载 yolov8x.pt 模型文件。

    Args:
        target: 目标文件路径（相对于项目根目录）

    Returns:
        下载是否成功
    """
    try:
        from ultralytics import YOLO

        target_path = PROJECT_ROOT / target
        print(f"  正在通过 ultralytics 下载 YOLOv8x...")

        # ultralytics 默认下载到当前工作目录
        # 先切换到 models 目录，让文件下载到正确位置
        old_cwd = os.getcwd()
        os.chdir(str(MODELS_DIR))
        try:
            _ = YOLO("yolov8x.pt")
        finally:
            os.chdir(old_cwd)

        # ultralytics 可能下载到当前目录或 ~/.ultralytics
        # 检查目标路径是否已有文件
        if target_path.exists():
            print(f"  YOLOv8x 权重已就位: {target_path}")
            return True

        # 如果 ultralytics 下载到了项目根目录，移动到 models/
        root_pt = PROJECT_ROOT / "yolov8x.pt"
        if root_pt.exists() and root_pt.resolve() != target_path.resolve():
            import shutil
            shutil.move(str(root_pt), str(target_path))
            print(f"  已将 YOLOv8x 移动到: {target_path}")
            return True

        # 直接通过 YOLO 下载到目标路径
        _ = YOLO(str(target_path))
        if target_path.exists():
            print(f"  YOLOv8x 权重下载完成: {target_path}")
            return True

        print(f"  [警告] YOLOv8x 下载后未在预期路径找到: {target_path}")
        return False

    except Exception as e:
        print(f"  [错误] YOLOv8x 下载失败: {e}")
        return False


def download_chinese_clip(target: str, repo_id: str = "") -> bool:
    """
    下载 Chinese-CLIP 预训练权重

    从 HuggingFace 下载 Chinese-CLIP ViT-Large 模型。

    Args:
        target: 目标目录路径（相对于项目根目录）
        repo_id: HuggingFace 仓库 ID

    Returns:
        下载是否成功
    """
    try:
        from transformers import AutoModel, AutoProcessor

        target_path = PROJECT_ROOT / target
        print(f"  正在从 HuggingFace 下载 Chinese-CLIP: {repo_id}")
        print(f"  目标路径: {target_path}")

        # 下载模型到指定目录
        model = AutoModel.from_pretrained(repo_id)
        processor = AutoProcessor.from_pretrained(repo_id)

        # 保存到目标目录
        model.save_pretrained(str(target_path))
        processor.save_pretrained(str(target_path))

        print(f"  Chinese-CLIP 权重下载完成: {target_path}")
        return True

    except Exception as e:
        print(f"  [错误] Chinese-CLIP 下载失败: {e}")
        return False


def download_resnet50(target: str) -> bool:
    """
    下载 ResNet50 预训练权重（用于 ReID backbone）

    通过 torchvision 加载预训练 ResNet50 并保存权重文件。

    Args:
        target: 目标文件路径（相对于项目根目录）

    Returns:
        下载是否成功
    """
    try:
        import torch
        from torchvision import models

        target_path = PROJECT_ROOT / target
        print(f"  正在通过 torchvision 下载 ResNet50 预训练权重...")

        # 加载预训练模型（会自动下载权重到 torch hub 缓存）
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 保存完整模型权重
        torch.save(resnet.state_dict(), str(target_path))

        print(f"  ResNet50 权重保存完成: {target_path}")
        return True

    except Exception as e:
        print(f"  [错误] ResNet50 下载失败: {e}")
        return False


# ============================================================
# 权重清单
# ============================================================

WEIGHTS: Dict[str, Dict[str, Any]] = {
    "yolov8x": {
        "description": "YOLOv8x 目标检测模型",
        "source": "ultralytics",
        "target": "models/yolov8x.pt",
        "expected_size_mb": 130,  # 预期大小约 130MB
        "download_fn": download_yolo,
        "extra_kwargs": {},
    },
    "chinese_clip_vit_large": {
        "description": "Chinese-CLIP ViT-Large 图文特征模型",
        "source": "huggingface",
        "repo_id": "OFA-Sys/chinese-clip-vit-large-patch14",
        "target": "models/chinese-clip-vit-large-patch14/",
        "expected_size_mb": 1800,  # 预期大小约 1.8GB
        "download_fn": download_chinese_clip,
        "extra_kwargs": {"repo_id": "OFA-Sys/chinese-clip-vit-large-patch14"},
    },
    "resnet50": {
        "description": "ResNet50 ReID 外观特征 backbone",
        "source": "torchvision",
        "target": "models/resnet50_reid.pth",
        "expected_size_mb": 100,  # 预期大小约 100MB
        "download_fn": download_resnet50,
        "extra_kwargs": {},
    },
}


# ============================================================
# 核心逻辑
# ============================================================

def load_manifest() -> Dict[str, Any]:
    """
    加载权重清单文件

    Returns:
        清单字典，文件不存在时返回空字典
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict[str, Any]) -> None:
    """
    保存权重清单文件

    Args:
        manifest: 清单字典
    """
    _ensure_models_dir()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"清单已保存: {MANIFEST_PATH}")


def check_weight(name: str, weight_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查单个权重是否存在

    Args:
        name: 权重名称
        weight_cfg: 权重配置

    Returns:
        检查结果字典
    """
    target = weight_cfg["target"]
    target_path = PROJECT_ROOT / target

    result = {
        "name": name,
        "description": weight_cfg.get("description", ""),
        "target": target,
        "source": weight_cfg.get("source", ""),
        "exists": False,
        "size": 0,
        "size_human": "0 B",
        "sha256": "",
    }

    if target_path.exists():
        if target_path.is_file():
            size = target_path.stat().st_size
            result["exists"] = True
            result["size"] = size
            result["size_human"] = _format_size(size)
            result["sha256"] = _compute_sha256(target_path)
        elif target_path.is_dir():
            # 目录类型（如 Chinese-CLIP）
            size = _get_dir_size(target_path)
            # 检查目录中是否有模型文件
            has_config = (target_path / "config.json").exists()
            has_model = any(target_path.glob("*.safetensors")) or any(
                target_path.glob("*.bin")
            )
            result["exists"] = has_config and has_model
            result["size"] = size
            result["size_human"] = _format_size(size)
    return result


def check_all_weights() -> Dict[str, Dict[str, Any]]:
    """
    检查所有权重的状态

    Returns:
        所有权重的检查结果
    """
    results = {}
    for name, cfg in WEIGHTS.items():
        results[name] = check_weight(name, cfg)
    return results


def download_weight(name: str, weight_cfg: Dict[str, Any]) -> bool:
    """
    下载单个权重

    Args:
        name: 权重名称
        weight_cfg: 权重配置

    Returns:
        下载是否成功
    """
    target = weight_cfg["target"]
    download_fn = weight_cfg["download_fn"]
    extra_kwargs = weight_cfg.get("extra_kwargs", {})

    print(f"\n{'='*60}")
    print(f"下载权重: {name} ({weight_cfg.get('description', '')})")
    print(f"来源: {weight_cfg.get('source', '')}")
    print(f"目标: {target}")
    print(f"{'='*60}")

    _ensure_models_dir()

    start_time = time.time()
    success = download_fn(target=target, **extra_kwargs)
    elapsed = time.time() - start_time

    if success:
        print(f"  耗时: {elapsed:.1f}s")
    return success


def update_manifest_for_weight(name: str, weight_cfg: Dict[str, Any]) -> None:
    """
    更新清单中某个权重的记录

    Args:
        name: 权重名称
        weight_cfg: 权重配置
    """
    manifest = load_manifest()
    check_result = check_weight(name, weight_cfg)

    manifest[name] = {
        "description": weight_cfg.get("description", ""),
        "source": weight_cfg.get("source", ""),
        "target": weight_cfg["target"],
        "exists": check_result["exists"],
        "size_bytes": check_result["size"],
        "size_human": check_result["size_human"],
        "sha256": check_result["sha256"],
        "download_time": datetime.now().isoformat(),
    }

    save_manifest(manifest)


def print_check_results(results: Dict[str, Dict[str, Any]]) -> None:
    """
    打印权重检查结果

    Args:
        results: 检查结果字典
    """
    print(f"\n{'='*70}")
    print(f"{'模型权重检查报告':^70}")
    print(f"{'='*70}")
    print(f"{'名称':<25} {'状态':<8} {'大小':<12} {'来源':<15}")
    print(f"{'-'*70}")

    all_ok = True
    for name, result in results.items():
        status = "✓ 已就绪" if result["exists"] else "✗ 缺失"
        size = result["size_human"] if result["exists"] else "-"
        source = result["source"]
        print(f"{name:<25} {status:<8} {size:<12} {source:<15}")
        if not result["exists"]:
            all_ok = False

    print(f"{'-'*70}")
    if all_ok:
        print("所有模型权重已就绪。")
    else:
        print("部分权重缺失，请运行: python scripts/pre_download_weights.py --download")
    print(f"{'='*70}\n")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """脚本主入口"""
    parser = argparse.ArgumentParser(
        description="模型权重预下载脚本 - 管理 YOLO/Chinese-CLIP/ResNet50 权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/pre_download_weights.py --check        # 仅检查权重状态
  python scripts/pre_download_weights.py --download      # 下载缺失的权重
  python scripts/pre_download_weights.py --download --force  # 强制重新下载所有权重
        """,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查权重状态，不进行下载",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载缺失的模型权重",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载所有权重（忽略已存在的文件）",
    )

    args = parser.parse_args()

    # 如果没有任何参数，打印帮助信息
    if not args.check and not args.download:
        parser.print_help()
        return

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模型缓存目录: {MODELS_DIR}")

    # 检查模式
    if args.check:
        results = check_all_weights()
        print_check_results(results)
        return

    # 下载模式
    if args.download:
        _ensure_models_dir()

        # 先检查当前状态
        results = check_all_weights()
        print_check_results(results)

        success_count = 0
        fail_count = 0
        skip_count = 0

        for name, cfg in WEIGHTS.items():
            result = results[name]

            # 如果已存在且不强制下载，跳过
            if result["exists"] and not args.force:
                print(f"\n[跳过] {name}: 已存在 ({result['size_human']})")
                skip_count += 1
                continue

            # 下载
            ok = download_weight(name, cfg)
            if ok:
                update_manifest_for_weight(name, cfg)
                success_count += 1
            else:
                fail_count += 1

        # 最终汇总
        print(f"\n{'='*70}")
        print(f"下载完成汇总:")
        print(f"  成功: {success_count}")
        print(f"  跳过: {skip_count}")
        print(f"  失败: {fail_count}")
        print(f"{'='*70}")

        # 重新检查并生成最终清单
        final_results = check_all_weights()
        for name, cfg in WEIGHTS.items():
            if final_results[name]["exists"]:
                update_manifest_for_weight(name, cfg)

        print_check_results(final_results)

        if fail_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
