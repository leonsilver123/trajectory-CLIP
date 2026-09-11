"""
src.common.utils - 通用工具函数

提供系统中各模块共享的通用工具函数，包括:
- 时间戳处理
- 坐标计算
- 向量操作
- ID 生成
- 文件操作
"""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from src.common.logger import get_logger

logger = get_logger("common.utils")


# ============================================================
# ID 生成
# ============================================================

def generate_id(prefix: str = "") -> str:
    """
    生成唯一 ID

    Args:
        prefix: ID 前缀，如 "INST", "TRK", "EDGE"

    Returns:
        唯一 ID 字符串

    示例:
        generate_id("INST")  # "INST_a1b2c3d4e5f6"
    """
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def generate_deterministic_id(*parts: str) -> str:
    """
    生成确定性 ID (相同输入产生相同输出)

    用于需要幂等性的场景，如根据摄像头ID和时间戳生成 Tracklet ID。

    Args:
        *parts: 用于生成 ID 的字符串片段

    Returns:
        确定性 ID 字符串
    """
    combined = "_".join(parts)
    hash_value = hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]
    return hash_value


# ============================================================
# 时间处理
# ============================================================

def parse_timestamp(timestamp_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    解析时间戳字符串

    Args:
        timestamp_str: 时间戳字符串
        fmt: 时间格式

    Returns:
        datetime 对象
    """
    return datetime.strptime(timestamp_str, fmt)


def format_timestamp(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化 datetime 为字符串

    Args:
        dt: datetime 对象
        fmt: 时间格式

    Returns:
        格式化后的时间字符串
    """
    return dt.strftime(fmt)


def time_diff_seconds(t1: datetime, t2: datetime) -> float:
    """
    计算两个时间点之间的秒数差

    Args:
        t1: 第一个时间点
        t2: 第二个时间点

    Returns:
        秒数差 (t2 - t1)
    """
    return (t2 - t1).total_seconds()


def is_reasonable_travel_time(
    distance_meters: float,
    time_seconds: float,
    min_speed_kmh: float = 5.0,
    max_speed_kmh: float = 120.0,
) -> bool:
    """
    判断旅行时间是否在合理范围内

    基于路段距离和速度约束，判断两个摄像头之间的旅行时间是否合理。

    Args:
        distance_meters: 路段距离(米)
        time_seconds: 旅行时间(秒)
        min_speed_kmh: 最低速度(km/h)，默认 5 km/h
        max_speed_kmh: 最高速度(km/h)，默认 120 km/h

    Returns:
        是否在合理时间范围内
    """
    if time_seconds <= 0 or distance_meters <= 0:
        return False
    min_time = distance_meters / (max_speed_kmh * 1000 / 3600)
    max_time = distance_meters / (min_speed_kmh * 1000 / 3600)
    return min_time <= time_seconds <= max_time


# ============================================================
# 坐标与距离计算
# ============================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    计算两个经纬度坐标之间的球面距离(米)

    使用 Haversine 公式。

    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度

    Returns:
        球面距离(米)
    """
    R = 6371000  # 地球半径(米)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def direction_angle_diff(angle1: float, angle2: float) -> float:
    """
    计算两个角度之间的差值(考虑环形)

    Args:
        angle1: 第一个角度(度)
        angle2: 第二个角度(度)

    Returns:
        角度差(度)，范围 [0, 180]
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)


# ============================================================
# 向量操作
# ============================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        vec1: 向量 1
        vec2: 向量 2

    Returns:
        余弦相似度 [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def batch_cosine_similarity(
    query: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    """
    批量计算查询向量与图库向量的余弦相似度

    Args:
        query: 查询向量 (dim,)
        gallery: 图库向量矩阵 (n, dim)

    Returns:
        相似度数组 (n,)
    """
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    gallery_norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery_normalized = gallery / (gallery_norms + 1e-8)
    return gallery_normalized @ query_norm


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    L2 归一化向量

    Args:
        vec: 输入向量

    Returns:
        归一化后的向量
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# ============================================================
# 文件操作
# ============================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_keyframe_save_dir(camera_id: str, base_dir: str) -> Path:
    """
    获取关键帧保存目录

    Args:
        camera_id: 摄像头 ID
        base_dir: 基础目录

    Returns:
        关键帧保存目录路径
    """
    save_dir = Path(base_dir) / "keyframes" / camera_id
    ensure_dir(save_dir)
    return save_dir


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    列出目录下的文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名过滤列表，如 [".jpg", ".png"]

    Returns:
        文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    files = list(dir_path.iterdir())
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return sorted(files)
