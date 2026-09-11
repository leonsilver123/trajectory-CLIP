"""
src.common.config - 配置加载模块

从 YAML 文件加载系统配置，支持环境变量覆盖，提供全局单例访问。
同时包含 AICity22 Track1 MTMC Tracking 数据集常量与路径工具函数。

使用方式:
    from src.common.config import get_config
    config = get_config()
    device = config.get("system.device")

    # AICity22 数据集常量
    from src.common.config import DATASET_ROOT, SCENE_CAMERA_MAP, get_video_path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Config:
    """
    系统配置管理器

    从 YAML 文件加载配置，支持通过环境变量覆盖。
    环境变量命名规则: TRAFFIC__SECTION__KEY (双下划线分隔)

    示例:
        config = Config("configs/default.yaml")
        device = config.get("system.device")  # "cuda"
        config.set("system.device", "cpu")
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        初始化配置

        Args:
            config_path: YAML 配置文件路径，默认为 configs/default.yaml
        """
        self._config_path = config_path or self._find_default_config()
        self._data: Dict[str, Any] = {}
        self._load()
        self._apply_env_overrides()

    def _find_default_config(self) -> str:
        """查找默认配置文件"""
        # 从项目根目录查找
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / "configs" / "default.yaml"
        if default_path.exists():
            return str(default_path)
        raise FileNotFoundError(
            f"未找到默认配置文件: {default_path}。"
            "请确保 configs/default.yaml 存在。"
        )

    def _load(self) -> None:
        """从 YAML 文件加载配置"""
        config_path = Path(self._config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

    def _apply_env_overrides(self) -> None:
        """
        应用环境变量覆盖

        环境变量格式: TRAFFIC__SECTION__KEY
        例如: TRAFFIC__SYSTEM__DEVICE=cpu 会覆盖 system.device 配置
        """
        prefix = "TRAFFIC__"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # 解析环境变量键为配置路径
                parts = env_key[len(prefix):].lower().split("__")
                if len(parts) >= 2:
                    section = parts[0]
                    key = parts[1]
                    if section in self._data:
                        # 尝试转换为合适的类型
                        self._data[section][key] = self._cast_value(
                            self._data[section].get(key), env_value
                        )

    @staticmethod
    def _cast_value(original: Any, value: str) -> Any:
        """
        根据原始值类型转换环境变量字符串

        Args:
            original: 原始配置值(用于推断类型)
            value: 环境变量字符串值

        Returns:
            转换后的值
        """
        if original is None:
            return value
        if isinstance(original, bool):
            return value.lower() in ("true", "1", "yes")
        if isinstance(original, int):
            try:
                return int(value)
            except ValueError:
                return value
        if isinstance(original, float):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值(支持点分隔路径)

        Args:
            key_path: 配置键路径，如 "system.device" 或 "detection.confidence_threshold"
            default: 键不存在时的默认值

        Returns:
            配置值

        示例:
            config.get("system.device")  # "cuda"
            config.get("nonexistent.key", "default_value")
        """
        keys = key_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值(支持点分隔路径)

        Args:
            key_path: 配置键路径
            value: 要设置的值
        """
        keys = key_path.split(".")
        data = self._data
        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        data[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取整个配置段

        Args:
            section: 配置段名称

        Returns:
            该配置段的字典
        """
        return self._data.get(section, {})

    @property
    def data(self) -> Dict[str, Any]:
        """获取完整配置数据"""
        return self._data.copy()

    def __repr__(self) -> str:
        return f"Config(path={self._config_path}, sections={list(self._data.keys())})"


# ============================================================
# 全局单例
# ============================================================

_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取全局配置单例

    首次调用时加载配置，后续调用返回已加载的配置实例。

    Args:
        config_path: 配置文件路径(仅首次调用有效)

    Returns:
        Config 实例
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reset_config() -> None:
    """重置全局配置(主要用于测试)"""
    global _global_config
    _global_config = None


# ============================================================
# AICity22 Track1 MTMC Tracking 数据集常量
# ============================================================

# 数据集根目录（相对于项目根目录）
DATASET_ROOT = "cityflow/AICity22_Track1_MTMC_Tracking"

# 场景-摄像头映射（更正后）
SCENE_CAMERA_MAP: Dict[str, List[str]] = {
    "S01": [f"c{i:03d}" for i in range(1, 6)],        # c001-c005, train
    "S02": [f"c{i:03d}" for i in range(6, 10)],        # c006-c009, validation
    "S03": [f"c{i:03d}" for i in range(10, 16)],       # c010-c015, train
    "S04": [f"c{i:03d}" for i in range(16, 41)],       # c016-c040, train
    "S05": [                                            # validation (跨场景复用)
        "c010",
        *[f"c{i:03d}" for i in range(16, 30)],          # c016-c029
        *[f"c{i:03d}" for i in range(33, 37)],          # c033-c036
    ],
    "S06": [f"c{i:03d}" for i in range(41, 47)],       # c041-c046, test
}

# 数据划分
SPLIT_SCENES: Dict[str, List[str]] = {
    "train": ["S01", "S03", "S04"],
    "validation": ["S02", "S05"],
    "test": ["S06"],
}

# 摄像头帧率 (FPS)，默认 10，c015 为 8
CAMERA_FPS: Dict[str, int] = {cam: 10 for cams in SCENE_CAMERA_MAP.values() for cam in cams}
CAMERA_FPS["c015"] = 8

# 所有唯一摄像头 ID 列表 (c001-c046, 共 46 个)
ALL_CAMERA_IDS: List[str] = sorted(set(
    cam for cams in SCENE_CAMERA_MAP.values() for cam in cams
))


def get_scene_for_camera(camera_id: str) -> Optional[str]:
    """获取摄像头所属场景 ID（如 'S01'）"""
    for scene, cams in SCENE_CAMERA_MAP.items():
        if camera_id in cams:
            return scene
    return None


def get_split_for_camera(camera_id: str) -> Optional[str]:
    """获取摄像头所属数据划分（train/validation/test）"""
    scene = get_scene_for_camera(camera_id)
    if scene is None:
        return None
    for split, scenes in SPLIT_SCENES.items():
        if scene in scenes:
            return split
    return None


def get_video_path(split: str, scene: str, camera_id: str) -> str:
    """
    构建摄像头视频文件路径

    Args:
        split: 数据划分 (train/validation/test)
        scene: 场景 ID (如 'S01')
        camera_id: 摄像头 ID (如 'c001')

    Returns:
        视频文件相对路径，如 'cityflow/AICity22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi'
    """
    return f"{DATASET_ROOT}/{split}/{scene}/{camera_id}/vdo.avi"


def get_frame_path(split: str, scene: str, camera_id: str, frame: int) -> str:
    """
    构建提取后帧图片路径

    Args:
        split: 数据划分
        scene: 场景 ID
        camera_id: 摄像头 ID
        frame: 帧编号 (1-based)

    Returns:
        帧图片相对路径，如 '.../train/S01/c001/img1/000001.jpg'
    """
    return f"{DATASET_ROOT}/{split}/{scene}/{camera_id}/img1/{frame:06d}.jpg"


def get_scene_map_path(scene: str) -> str:
    """
    获取场景级地图 PNG 路径

    Args:
        scene: 场景 ID (如 'S01')

    Returns:
        地图 PNG 相对路径，如 'cityflow/AICity22_Track1_MTMC_Tracking/cam_loc/S01.png'
    """
    # S03/S04/S05 共用 S0345.png
    if scene in ("S03", "S04", "S05"):
        return f"{DATASET_ROOT}/cam_loc/S0345.png"
    return f"{DATASET_ROOT}/cam_loc/{scene}.png"


def get_cam_timestamp_path(scene: str) -> str:
    """获取场景的摄像头时间戳文件路径"""
    return f"{DATASET_ROOT}/cam_timestamp/{scene}.txt"


def get_cam_framenum_path(scene: str) -> str:
    """获取场景的摄像头帧数文件路径"""
    return f"{DATASET_ROOT}/cam_framenum/{scene}.txt"


def get_gt_path(split: str, scene: str) -> str:
    """
    获取场景的 gt.txt 标注路径

    gt.txt 格式 (MOT 10字段): frame, ID, left, top, width, height, 1, -1, -1, -1
    """
    return f"{DATASET_ROOT}/{split}/{scene}/gt/gt.txt"


def load_cam_timestamps(scene: str) -> Dict[str, float]:
    """
    加载场景的摄像头时间戳偏移

    Args:
        scene: 场景 ID

    Returns:
        {camera_id: timestamp_offset_seconds} 字典
    """
    path = get_cam_timestamp_path(scene)
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    result[parts[0]] = float(parts[1])
    except FileNotFoundError:
        pass
    return result


def load_cam_framenums(scene: str) -> Dict[str, int]:
    """
    加载场景的摄像头总帧数

    Args:
        scene: 场景 ID

    Returns:
        {camera_id: total_frames} 字典
    """
    path = get_cam_framenum_path(scene)
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    result[parts[0]] = int(parts[1])
    except FileNotFoundError:
        pass
    return result
