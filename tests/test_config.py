"""
tests/test_config.py - 配置系统测试

测试目标: 验证 src/common/config.py 配置加载与 AICity22 数据集常量
覆盖:
  - Config 加载 default.yaml
  - get() 方法读取嵌套配置
  - set() 方法覆盖配置
  - AICity22 常量验证（SCENE_CAMERA_MAP, ALL_CAMERA_IDS, CAMERA_FPS）
  - 辅助函数测试（get_scene_for_camera, get_split_for_camera 等）
"""

import pytest
from pathlib import Path

from src.common.config import (
    Config,
    get_config,
    reset_config,
    DATASET_ROOT,
    SCENE_CAMERA_MAP,
    SPLIT_SCENES,
    ALL_CAMERA_IDS,
    CAMERA_FPS,
    get_scene_for_camera,
    get_split_for_camera,
    get_video_path,
    get_frame_path,
    get_scene_map_path,
    get_cam_timestamp_path,
    get_cam_framenum_path,
    get_gt_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_global_config():
    """每个测试前重置全局配置单例"""
    reset_config()
    yield
    reset_config()


# ============================================================
# Config 类测试
# ============================================================

class TestConfig:
    """Config 配置管理器测试"""

    def test_load_default_config(self):
        """测试加载默认配置文件"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert config.data is not None
        assert "system" in config.data
        assert "detection" in config.data

    def test_get_top_level_key(self):
        """测试读取顶级配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        system = config.get("system")
        assert isinstance(system, dict)
        assert "name" in system

    def test_get_nested_key(self):
        """测试读取嵌套配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        device = config.get("system.device")
        assert device in ("cuda", "cpu")

    def test_get_version(self):
        """测试读取版本号"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        version = config.get("system.version")
        assert version == "1.0.0"

    def test_get_nonexistent_key_returns_default(self):
        """测试读取不存在的键返回默认值"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        val = config.get("nonexistent.key", "fallback")
        assert val == "fallback"

    def test_get_nonexistent_key_returns_none(self):
        """测试读取不存在的键默认返回 None"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        val = config.get("nonexistent.deeply.nested.key")
        assert val is None

    def test_set_override(self):
        """测试 set() 覆盖配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        config.set("system.device", "cpu")
        assert config.get("system.device") == "cpu"

    def test_set_new_key(self):
        """测试 set() 创建新键"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        config.set("test_section.test_key", "test_value")
        assert config.get("test_section.test_key") == "test_value"

    def test_get_section(self):
        """测试获取配置段"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        detection = config.get_section("detection")
        assert isinstance(detection, dict)
        assert "model" in detection

    def test_get_section_nonexistent(self):
        """测试获取不存在的配置段"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        section = config.get_section("nonexistent_section")
        assert section == {}

    def test_config_file_not_found(self):
        """测试配置文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_config_repr(self):
        """测试 Config 的 repr"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        r = repr(config)
        assert "Config" in r
        assert "default.yaml" in r

    def test_detection_threshold(self):
        """测试检测阈值配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        threshold = config.get("detection.confidence_threshold")
        assert isinstance(threshold, float)
        assert 0 < threshold < 1

    def test_stitching_weights(self):
        """测试拼接权重配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        weights = config.get("stitching.weights.vehicle")
        assert isinstance(weights, dict)
        assert "plate" in weights
        assert "temporal" in weights

    def test_camera_total(self):
        """测试摄像头总数配置"""
        config = Config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        total = config.get("camera.total_cameras")
        assert total == 46


# ============================================================
# 全局单例测试
# ============================================================

class TestGlobalConfig:
    """全局配置单例测试"""

    def test_get_config_singleton(self):
        """测试全局配置是单例"""
        c1 = get_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        c2 = get_config()
        assert c1 is c2

    def test_reset_config(self):
        """测试重置配置"""
        c1 = get_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        reset_config()
        c2 = get_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert c1 is not c2


# ============================================================
# AICity22 常量测试
# ============================================================

class TestAICity22Constants:
    """AICity22 数据集常量测试"""

    def test_dataset_root(self):
        """测试数据集根路径"""
        assert DATASET_ROOT == "cityflow/AICity22_Track1_MTMC_Tracking"

    def test_scene_camera_map_scenes(self):
        """测试场景-摄像头映射包含所有场景"""
        expected_scenes = {"S01", "S02", "S03", "S04", "S05", "S06"}
        assert set(SCENE_CAMERA_MAP.keys()) == expected_scenes

    def test_scene_camera_map_s01(self):
        """测试 S01 场景有 5 个摄像头"""
        assert len(SCENE_CAMERA_MAP["S01"]) == 5
        assert SCENE_CAMERA_MAP["S01"] == ["c001", "c002", "c003", "c004", "c005"]

    def test_scene_camera_map_s04(self):
        """测试 S04 场景有 25 个摄像头"""
        assert len(SCENE_CAMERA_MAP["S04"]) == 25
        assert SCENE_CAMERA_MAP["S04"][0] == "c016"
        assert SCENE_CAMERA_MAP["S04"][-1] == "c040"

    def test_scene_camera_map_s06(self):
        """测试 S06 场景有 6 个摄像头"""
        assert len(SCENE_CAMERA_MAP["S06"]) == 6
        assert SCENE_CAMERA_MAP["S06"] == ["c041", "c042", "c043", "c044", "c045", "c046"]

    def test_all_camera_ids_count(self):
        """测试总摄像头数（注意 S05 复用其他场景摄像头，唯一数应为 46）"""
        assert len(ALL_CAMERA_IDS) == 46

    def test_all_camera_ids_sorted(self):
        """测试摄像头 ID 已排序"""
        assert ALL_CAMERA_IDS == sorted(ALL_CAMERA_IDS)

    def test_all_camera_ids_range(self):
        """测试摄像头 ID 范围 c001-c046"""
        assert ALL_CAMERA_IDS[0] == "c001"
        assert ALL_CAMERA_IDS[-1] == "c046"

    def test_camera_fps_default(self):
        """测试默认帧率为 10"""
        assert CAMERA_FPS["c001"] == 10
        assert CAMERA_FPS["c010"] == 10

    def test_camera_fps_c015(self):
        """测试 c015 特殊帧率"""
        assert CAMERA_FPS["c015"] == 8

    def test_split_scenes(self):
        """测试数据划分"""
        assert "train" in SPLIT_SCENES
        assert "validation" in SPLIT_SCENES
        assert "test" in SPLIT_SCENES
        assert SPLIT_SCENES["train"] == ["S01", "S03", "S04"]
        assert SPLIT_SCENES["test"] == ["S06"]


# ============================================================
# 辅助函数测试
# ============================================================

class TestHelperFunctions:
    """辅助函数测试"""

    def test_get_scene_for_camera_s01(self):
        """测试 S01 摄像头场景查询"""
        assert get_scene_for_camera("c001") == "S01"
        assert get_scene_for_camera("c005") == "S01"

    def test_get_scene_for_camera_s04(self):
        """测试 S04 摄像头场景查询"""
        assert get_scene_for_camera("c016") == "S04"
        assert get_scene_for_camera("c040") == "S04"

    def test_get_scene_for_camera_unknown(self):
        """测试未知摄像头返回 None"""
        assert get_scene_for_camera("c099") is None
        assert get_scene_for_camera("unknown") is None

    def test_get_split_for_camera_train(self):
        """测试训练集摄像头"""
        assert get_split_for_camera("c001") == "train"  # S01
        assert get_split_for_camera("c010") == "train"  # S03
        assert get_split_for_camera("c016") == "train"  # S04

    def test_get_split_for_camera_validation(self):
        """测试验证集摄像头"""
        assert get_split_for_camera("c006") == "validation"  # S02

    def test_get_split_for_camera_test(self):
        """测试测试集摄像头"""
        assert get_split_for_camera("c041") == "test"  # S06

    def test_get_split_for_camera_unknown(self):
        """测试未知摄像头返回 None"""
        assert get_split_for_camera("c099") is None

    def test_get_video_path(self):
        """测试视频路径构建"""
        path = get_video_path("train", "S01", "c001")
        assert path == "cityflow/AICity22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi"

    def test_get_frame_path(self):
        """测试帧图片路径构建"""
        path = get_frame_path("train", "S01", "c001", 1)
        assert path == "cityflow/AICity22_Track1_MTMC_Tracking/train/S01/c001/img1/000001.jpg"

    def test_get_scene_map_path_s01(self):
        """测试 S01 场景地图路径"""
        path = get_scene_map_path("S01")
        assert path.endswith("S01.png")

    def test_get_scene_map_path_s03_shared(self):
        """测试 S03/S04/S05 共用地图"""
        path_s03 = get_scene_map_path("S03")
        path_s04 = get_scene_map_path("S04")
        path_s05 = get_scene_map_path("S05")
        assert "S0345.png" in path_s03
        assert "S0345.png" in path_s04
        assert "S0345.png" in path_s05

    def test_get_cam_timestamp_path(self):
        """测试时间戳文件路径"""
        path = get_cam_timestamp_path("S01")
        assert "cam_timestamp/S01.txt" in path

    def test_get_cam_framenum_path(self):
        """测试帧数文件路径"""
        path = get_cam_framenum_path("S01")
        assert "cam_framenum/S01.txt" in path

    def test_get_gt_path(self):
        """测试 gt.txt 路径"""
        path = get_gt_path("train", "S01")
        assert path.endswith("gt/gt.txt")
        assert "train/S01" in path
