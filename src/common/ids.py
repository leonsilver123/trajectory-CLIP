"""
src.common.ids - 目标 ID 解析

项目内共有三种 ID，格式由数据生产端约定，此处统一解析，避免各处重复实现：

    target_id  : CF3_c001_V0034_000001   detection 级（单条检测）
    track_id   : CF3_TRACK_c001_V0034    轨迹级（单摄像头内的一段轨迹）
    vehicle_id : V0034                   车辆真实身份（跨摄像头不变）

使用方式:
    from src.common.ids import parse_target_id, parse_track_id, extract_vehicle_id

    parse_target_id("CF3_c001_V0034_000001")
    # {'source': 'CF3', 'camera_id': 'c001', 'vehicle_id': 'V0034', 'det_seq': '000001'}

    extract_vehicle_id("CF3_TRACK_c001_V0034")   # 'V0034'

所有函数对畸形输入（空串、None、段数不对、V 后非数字）返回空值而不抛异常。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# 各字段的合法形式
_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")   # CF3
_CAMERA_RE = re.compile(r"^c\d+$", re.IGNORECASE)    # c001
_VEHICLE_RE = re.compile(r"^V\d+$")                  # V0034
_SEQ_RE = re.compile(r"^\d+$")                       # 000001

# 轨迹 ID 的中间固定段
_TRACK_MARKER = "TRACK"


def _empty_target() -> Dict[str, Optional[str]]:
    """target_id 解析失败时的空结果"""
    return {
        "source": None,
        "camera_id": None,
        "vehicle_id": None,
        "det_seq": None,
    }


def _empty_track() -> Dict[str, Optional[str]]:
    """track_id 解析失败时的空结果"""
    return {
        "source": None,
        "camera_id": None,
        "vehicle_id": None,
    }


def _split(value: Any) -> list:
    """按 '_' 切分；非字符串或空串返回空列表"""
    if not isinstance(value, str) or not value:
        return []
    return value.split("_")


def parse_target_id(target_id: Any) -> Dict[str, Optional[str]]:
    """
    解析 detection 级 target_id

    Args:
        target_id: 形如 "CF3_c001_V0034_000001" 的字符串

    Returns:
        {'source', 'camera_id', 'vehicle_id', 'det_seq'}；
        段数不对或任一段非法时各字段均为 None（不抛异常）

    示例:
        parse_target_id("CF3_c001_V0034_000001")
        # {'source': 'CF3', 'camera_id': 'c001', 'vehicle_id': 'V0034', 'det_seq': '000001'}
    """
    parts = _split(target_id)
    if len(parts) != 4:
        return _empty_target()

    source, camera_id, vehicle_id, det_seq = parts
    if not (
        _SOURCE_RE.match(source)
        and _CAMERA_RE.match(camera_id)
        and _VEHICLE_RE.match(vehicle_id)
        and _SEQ_RE.match(det_seq)
    ):
        return _empty_target()

    return {
        "source": source,
        "camera_id": camera_id,
        "vehicle_id": vehicle_id,
        "det_seq": det_seq,
    }


def parse_track_id(track_id: Any) -> Dict[str, Optional[str]]:
    """
    解析轨迹级 track_id

    Args:
        track_id: 形如 "CF3_TRACK_c001_V0034" 的字符串

    Returns:
        {'source', 'camera_id', 'vehicle_id'}；
        段数不对、缺少 TRACK 标记或任一段非法时各字段均为 None（不抛异常）

    示例:
        parse_track_id("CF3_TRACK_c001_V0034")
        # {'source': 'CF3', 'camera_id': 'c001', 'vehicle_id': 'V0034'}
    """
    parts = _split(track_id)
    if len(parts) != 4 or parts[1] != _TRACK_MARKER:
        return _empty_track()

    source, _, camera_id, vehicle_id = parts
    if not (
        _SOURCE_RE.match(source)
        and _CAMERA_RE.match(camera_id)
        and _VEHICLE_RE.match(vehicle_id)
    ):
        return _empty_track()

    return {
        "source": source,
        "camera_id": camera_id,
        "vehicle_id": vehicle_id,
    }


def _scan_vehicle_id(value: Any) -> str:
    """
    容错兜底：逐段扫描出第一个 V+数字 的段

    仅用于 target_id / track_id 两种标准格式都不匹配时，保持既有容错语义。
    注意 "CF3" 里的 3 不满足「以 V 开头且其后全为数字」，不会被误判。
    """
    for part in _split(value):
        if _VEHICLE_RE.match(part):
            return part
    return ""


def extract_vehicle_id(value: Any) -> str:
    """
    从 target_id 或 track_id 中提取 vehicle_id

    Args:
        value: target_id（CF3_c001_V0034_000001）或 track_id（CF3_TRACK_c001_V0034）

    Returns:
        vehicle_id（如 "V0034"），无法识别时返回空串

    示例:
        extract_vehicle_id("CF3_c001_V0034_000001")  # 'V0034'
        extract_vehicle_id("CF3_TRACK_c001_V0034")   # 'V0034'
        extract_vehicle_id("INST_TEST_001")          # ''
    """
    parsed = parse_target_id(value)
    if parsed["vehicle_id"]:
        return parsed["vehicle_id"]

    parsed = parse_track_id(value)
    if parsed["vehicle_id"]:
        return parsed["vehicle_id"]

    return _scan_vehicle_id(value)
