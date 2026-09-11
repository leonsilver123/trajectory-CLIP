#!/usr/bin/env python3
"""
AICity22 Track1 MTMC Tracking 数据集下载与统计脚本
从 HuggingFace 下载 AICity22 城市级多目标跟踪数据集。
数据集路径: cityflow/AICity22_Track1_MTMC_Tracking
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path(r"H:\trajectory CLIP\cityflow")
CITYFLOW_DIR = DATA_ROOT / "AICity22_Track1_MTMC_Tracking"
REPORT_PATH = DATA_ROOT / "data_report.txt"

# AICity22 数据集配置
DATASET_CONFIG = {
    "repo_id": "AICity2022-Challenge/AICity2022-Track1-MTMC",
    "name": "AICity22-Track1-MTMC",
    "description": "AICity2022 城市级多目标跟踪数据集 (46摄像头, 6场景)",
    "local_dir": CITYFLOW_DIR,
    "scenes": {
        "S01": {"cameras": [f"c{i:03d}" for i in range(1, 6)], "split": "train"},
        "S02": {"cameras": [f"c{i:03d}" for i in range(6, 10)], "split": "validation"},
        "S03": {"cameras": [f"c{i:03d}" for i in range(10, 16)], "split": "train"},
        "S04": {"cameras": [f"c{i:03d}" for i in range(16, 41)], "split": "train"},
        "S05": {"cameras": [f"c{i:03d}" for i in [10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36]], "split": "validation"},
        "S06": {"cameras": [f"c{i:03d}" for i in range(41, 47)], "split": "test"},
    },
    "fps": 10,
    "video_format": "vdo.avi",
    "annotation_format": "MOTChallenge (10 fields)",
    "total_cameras": 46,
    "total_scenes": 6,
}


def ensure_dirs():
    """创建必要的目录结构"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    CITYFLOW_DIR.mkdir(parents=True, exist_ok=True)
    # 创建场景目录结构
    for scene_id, info in DATASET_CONFIG["scenes"].items():
        split = info["split"]
        scene_dir = CITYFLOW_DIR / split / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for cam_id in info["cameras"]:
            cam_dir = scene_dir / cam_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            (cam_dir / "gt").mkdir(parents=True, exist_ok=True)
            (cam_dir / "det").mkdir(parents=True, exist_ok=True)


def download_file_from_hf(repo_id: str, filename: str, local_dir: Path) -> bool:
    """从 HuggingFace 下载单个文件"""
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:
        print(f"  [错误] 下载 {filename} 失败: {e}")
        return False


def download_cityflow_dataset():
    """下载 AICity22 Track1 MTMC 数据集"""
    print(f"\n{'='*60}")
    print(f"下载数据集: {DATASET_CONFIG['name']}")
    print(f"{'='*60}")

    repo_id = DATASET_CONFIG["repo_id"]
    local_dir = DATASET_CONFIG["local_dir"]

    ensure_dirs()

    # 下载辅助文件 (cam_timestamp, cam_framenum, cam_loc, list_cam.txt)
    aux_files = [
        "list_cam.txt",
        "ReadMe.txt",
    ]
    aux_dirs = ["cam_timestamp", "cam_framenum", "cam_loc"]

    print("\n[1/3] 下载辅助文件...")
    for fname in aux_files:
        download_file_from_hf(repo_id, fname, local_dir)

    for dir_name in aux_dirs:
        # 尝试下载该目录下的文件
        for scene_id in DATASET_CONFIG["scenes"]:
            fname = f"{dir_name}/{scene_id}.txt"
            download_file_from_hf(repo_id, fname, local_dir)

    # 下载视频和标注文件
    print("\n[2/3] 下载视频和标注文件...")
    total_cameras = 0
    for scene_id, info in DATASET_CONFIG["scenes"].items():
        split = info["split"]
        for cam_id in info["cameras"]:
            # 视频文件
            video_path = f"{split}/{scene_id}/{cam_id}/vdo.avi"
            download_file_from_hf(repo_id, video_path, local_dir)
            # GT 标注
            gt_path = f"{split}/{scene_id}/{cam_id}/gt/gt.txt"
            download_file_from_hf(repo_id, gt_path, local_dir)
            # 检测结果
            det_path = f"{split}/{scene_id}/{cam_id}/det/det.txt"
            download_file_from_hf(repo_id, det_path, local_dir)
            total_cameras += 1

    print(f"\n[3/3] 下载完成，共处理 {total_cameras} 个摄像头")

    # 验证下载结果
    video_count = sum(1 for _ in local_dir.rglob("vdo.avi"))
    gt_count = sum(1 for _ in local_dir.rglob("gt.txt"))
    print(f"  视频文件数: {video_count}")
    print(f"  GT 标注文件数: {gt_count}")

    return video_count > 0


# ============================================================
# 统计与报告
# ============================================================

def get_dir_size(path: Path) -> int:
    """计算目录总大小(字节)"""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f} MB"
    else:
        return f"{size_bytes / (1024*1024*1024):.2f} GB"


def analyze_dataset() -> dict:
    """分析 AICity22 Track1 MTMC 数据集"""
    local_dir = DATASET_CONFIG["local_dir"]

    stats = {
        "name": DATASET_CONFIG["name"],
        "description": DATASET_CONFIG["description"],
        "path": str(local_dir),
        "total_size": get_dir_size(local_dir),
        "video_count": 0,
        "gt_count": 0,
        "det_count": 0,
        "scene_count": 0,
        "camera_count": 0,
        "scenes": {},
        "annotation_format": DATASET_CONFIG["annotation_format"],
        "fps": DATASET_CONFIG["fps"],
        "video_format": DATASET_CONFIG["video_format"],
    }

    for scene_id, info in DATASET_CONFIG["scenes"].items():
        split = info["split"]
        scene_dir = local_dir / split / scene_id
        scene_stats = {
            "cameras": [],
            "split": split,
            "video_count": 0,
            "gt_count": 0,
        }
        for cam_id in info["cameras"]:
            cam_dir = scene_dir / cam_id
            video_path = cam_dir / "vdo.avi"
            gt_path = cam_dir / "gt" / "gt.txt"
            det_path = cam_dir / "det" / "det.txt"

            if video_path.exists():
                scene_stats["video_count"] += 1
                stats["video_count"] += 1
            if gt_path.exists():
                scene_stats["gt_count"] += 1
                stats["gt_count"] += 1
            if det_path.exists():
                stats["det_count"] += 1

            if video_path.exists() or gt_path.exists():
                scene_stats["cameras"].append(cam_id)
                stats["camera_count"] += 1

        if scene_stats["cameras"]:
            stats["scenes"][scene_id] = scene_stats
            stats["scene_count"] += 1

    return stats


def generate_report(stats: dict):
    """生成统计报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 60)
    lines.append("AICity22 Track1 MTMC 数据集统计报告")
    lines.append(f"生成时间: {now}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"数据集: {stats['name']}")
    lines.append(f"描述: {stats['description']}")
    lines.append(f"路径: {stats['path']}")
    lines.append(f"总大小: {format_size(stats['total_size'])}")
    lines.append(f"视频文件数: {stats['video_count']}")
    lines.append(f"GT 标注文件数: {stats['gt_count']}")
    lines.append(f"检测结果文件数: {stats['det_count']}")
    lines.append(f"场景数: {stats['scene_count']}")
    lines.append(f"摄像头数: {stats['camera_count']}")
    lines.append(f"帧率: {stats['fps']} FPS")
    lines.append(f"视频格式: {stats['video_format']}")
    lines.append(f"标注格式: {stats['annotation_format']}")
    lines.append("")

    lines.append("场景分布:")
    for scene_id, scene_stats in sorted(stats.get("scenes", {}).items()):
        lines.append(f"  {scene_id} ({scene_stats['split']}): "
                     f"{len(scene_stats['cameras'])} 个摄像头, "
                     f"{scene_stats['video_count']} 个视频, "
                     f"{scene_stats['gt_count']} 个GT")

    lines.append("")
    lines.append("注: 数据来自 AICity2022 Challenge Track1 MTMC Tracking。")
    lines.append(f"报告生成时间: {now}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n报告已保存到: {REPORT_PATH}")
    print(report_text)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("AICity22 Track1 MTMC 数据集下载与统计工具")
    print("=" * 60)

    ensure_dirs()

    # 下载数据集
    try:
        ok = download_cityflow_dataset()
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    if not ok:
        print("数据集下载失败。请检查网络连接或手动下载。")
        print(f"数据集地址: https://huggingface.co/datasets/{DATASET_CONFIG['repo_id']}")

    # 统计
    print("\n\n" + "=" * 60)
    print("统计数据集信息...")
    print("=" * 60)

    stats = analyze_dataset()
    generate_report(stats)

    print("\n完成！")


if __name__ == "__main__":
    main()
