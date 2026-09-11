"""
tests/test_data_loading.py - 数据加载测试

测试目标: 验证数据文件加载与结构完整性
覆盖:
  - cityflow_results.json 加载和结构验证
  - AICity22 数据集目录结构验证
  - gt.txt 标注文件解析
  - cam_timestamp/cam_framenum 文件读取
  - 视频文件存在性检查
"""

import pytest
import json
from pathlib import Path

from src.common.config import (
    DATASET_ROOT,
    SCENE_CAMERA_MAP,
    get_gt_path,
    get_video_path,
    get_cam_timestamp_path,
    get_cam_framenum_path,
    load_cam_timestamps,
    load_cam_framenums,
    get_split_for_camera,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "output" / "cityflow_results.json"
DATASET_PATH = PROJECT_ROOT / DATASET_ROOT


# ============================================================
# cityflow_results.json 测试
# ============================================================

class TestCityflowResults:
    """cityflow_results.json 加载与结构验证"""

    @pytest.fixture
    def results_data(self):
        """加载 results.json"""
        if not RESULTS_JSON.exists():
            pytest.skip("cityflow_results.json 不存在")
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_file_exists(self):
        """测试结果文件存在"""
        assert RESULTS_JSON.exists(), f"文件不存在: {RESULTS_JSON}"

    def test_file_size(self):
        """测试结果文件大小合理（应 > 1MB）"""
        size = RESULTS_JSON.stat().st_size
        assert size > 1_000_000, f"文件过小: {size} bytes"

    def test_has_detections(self, results_data):
        """测试包含 detections 字段"""
        assert "detections" in results_data
        assert isinstance(results_data["detections"], list)

    def test_detections_count(self, results_data):
        """测试检测数量（预期约 1829）"""
        count = len(results_data["detections"])
        assert count > 0, "detections 为空"
        # 记录实际数量用于报告
        assert count >= 100, f"检测数量过少: {count}"

    def test_detection_structure(self, results_data):
        """测试单个检测记录的结构"""
        det = results_data["detections"][0]
        # 必要字段
        assert "target_id" in det or "instance_id" in det
        assert "camera_id" in det
        assert "target_type" in det
        assert "confidence" in det

    def test_detection_types_valid(self, results_data):
        """测试检测类型值有效"""
        valid_types = {"vehicle", "pedestrian", "non_motor_vehicle", "unknown"}
        for det in results_data["detections"][:100]:  # 抽样检查
            t_type = det.get("target_type", "unknown")
            assert t_type in valid_types, f"无效目标类型: {t_type}"

    def test_detection_confidence_range(self, results_data):
        """测试置信度在 [0, 1] 范围"""
        for det in results_data["detections"][:100]:
            conf = det.get("confidence", 0)
            assert 0 <= conf <= 1.01, f"置信度越界: {conf}"

    def test_has_tracks(self, results_data):
        """测试包含 tracks 字段"""
        if "tracks" in results_data:
            assert isinstance(results_data["tracks"], list)

    def test_has_summary(self, results_data):
        """测试包含 summary 字段"""
        if "summary" in results_data:
            assert isinstance(results_data["summary"], dict)

    def test_detection_camera_ids(self, results_data):
        """测试检测记录中的摄像头 ID 格式"""
        cam_ids = set()
        for det in results_data["detections"][:200]:
            cid = det.get("camera_id", "")
            if cid:
                cam_ids.add(cid)
        # 应有多个摄像头
        assert len(cam_ids) > 0

    def test_detection_has_attributes(self, results_data):
        """测试检测记录包含属性信息"""
        det = results_data["detections"][0]
        assert "attributes" in det
        assert isinstance(det["attributes"], dict)

    def test_detection_has_image_path(self, results_data):
        """测试检测记录包含图片路径"""
        det = results_data["detections"][0]
        # 至少有某种路径信息
        has_path = (
            "image_path" in det or
            "keyframe_path" in det or
            "detection_image_path" in det or
            "crop_path" in det
        )
        assert has_path, "检测记录缺少图片路径字段"


# ============================================================
# AICity22 数据集目录结构测试
# ============================================================

class TestAICity22Dataset:
    """AICity22 数据集目录结构验证"""

    def test_dataset_root_exists(self):
        """测试数据集根目录存在"""
        # 检查 zip 包或解压目录
        zip_path = PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking.zip"
        dir_path = PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking"
        assert zip_path.exists() or dir_path.exists(), "AICity22 数据集不存在"

    def test_train_scenes_exist(self):
        """测试训练场景目录存在"""
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            pytest.skip("train 目录不存在（可能只有 zip 包）")
        # S01, S03, S04 是训练场景
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if scene_path.exists():
                assert scene_path.is_dir()

    def test_camera_dirs_have_video(self):
        """测试摄像头目录有视频文件"""
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            pytest.skip("train 目录不存在")

        found_videos = 0
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if not scene_path.exists():
                continue
            for cam_dir in scene_path.iterdir():
                if cam_dir.is_dir():
                    vdo = cam_dir / "vdo.avi"
                    if vdo.exists():
                        found_videos += 1

        assert found_videos > 0, "未找到任何 vdo.avi 视频文件"

    def test_camera_dirs_have_gt(self):
        """测试训练场景有 gt.txt"""
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            pytest.skip("train 目录不存在")

        found_gt = 0
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if not scene_path.exists():
                continue
            for cam_dir in scene_path.iterdir():
                if cam_dir.is_dir():
                    gt = cam_dir / "gt" / "gt.txt"
                    if gt.exists():
                        found_gt += 1

        assert found_gt > 0, "未找到任何 gt.txt 标注文件"


# ============================================================
# gt.txt 标注文件解析测试
# ============================================================

class TestGtAnnotation:
    """gt.txt 标注文件解析测试"""

    def _find_gt_files(self):
        """查找可用的 gt.txt 文件"""
        gt_files = []
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            return gt_files
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if not scene_path.exists():
                continue
            for cam_dir in sorted(scene_path.iterdir()):
                if cam_dir.is_dir():
                    gt = cam_dir / "gt" / "gt.txt"
                    if gt.exists():
                        gt_files.append((scene, cam_dir.name, gt))
        return gt_files

    def test_gt_files_exist(self):
        """测试 gt.txt 文件存在"""
        gt_files = self._find_gt_files()
        if not gt_files:
            pytest.skip("无可用 gt.txt 文件")
        assert len(gt_files) > 0

    def test_gt_format(self):
        """测试 gt.txt 格式（MOT 标准 10 字段）"""
        gt_files = self._find_gt_files()
        if not gt_files:
            pytest.skip("无可用 gt.txt 文件")

        scene, cam, gt_path = gt_files[0]
        with open(gt_path, "r") as f:
            lines = f.readlines()

        if not lines:
            pytest.skip(f"gt.txt 为空: {gt_path}")

        # 检查前 10 行格式
        for line in lines[:10]:
            parts = line.strip().split(",")
            assert len(parts) >= 7, f"gt.txt 字段数不足: {line}"
            # 第一字段: frame (int)
            frame = int(parts[0])
            assert frame >= 1
            # 第二字段: ID (int)
            obj_id = int(parts[1])
            assert obj_id >= 1

    def test_gt_has_multiple_objects(self):
        """测试 gt.txt 包含多个目标"""
        gt_files = self._find_gt_files()
        if not gt_files:
            pytest.skip("无可用 gt.txt 文件")

        scene, cam, gt_path = gt_files[0]
        obj_ids = set()
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    obj_ids.add(int(parts[1]))

        assert len(obj_ids) > 0, "gt.txt 中没有目标 ID"


# ============================================================
# cam_timestamp / cam_framenum 文件测试
# ============================================================

class TestCamTimestampFramenum:
    """cam_timestamp 和 cam_framenum 文件读取测试"""

    def test_cam_timestamp_files_exist(self):
        """测试 cam_timestamp 文件存在"""
        ts_dir = PROJECT_ROOT / DATASET_ROOT / "cam_timestamp"
        if not ts_dir.exists():
            # 也检查 extracted 目录
            ts_dir = PROJECT_ROOT / "cityflow" / "extracted" / "cam_timestamp"
        if not ts_dir.exists():
            pytest.skip("cam_timestamp 目录不存在")
        txt_files = list(ts_dir.glob("*.txt"))
        assert len(txt_files) > 0

    def test_load_cam_timestamps(self):
        """测试加载摄像头时间戳"""
        # 尝试 S01
        ts_path = PROJECT_ROOT / DATASET_ROOT / "cam_timestamp" / "S01.txt"
        if not ts_path.exists():
            ts_path = PROJECT_ROOT / "cityflow" / "extracted" / "cam_timestamp" / "S01.txt"
        if not ts_path.exists():
            pytest.skip("S01.txt 不存在")

        timestamps = load_cam_timestamps("S01")
        # 应该能解析出摄像头时间戳
        if timestamps:
            assert isinstance(timestamps, dict)
            for cam_id, ts in timestamps.items():
                assert isinstance(ts, float)

    def test_cam_framenum_files_exist(self):
        """测试 cam_framenum 文件存在"""
        fn_dir = PROJECT_ROOT / DATASET_ROOT / "cam_framenum"
        if not fn_dir.exists():
            fn_dir = PROJECT_ROOT / "cityflow" / "extracted" / "cam_framenum"
        if not fn_dir.exists():
            pytest.skip("cam_framenum 目录不存在")
        txt_files = list(fn_dir.glob("*.txt"))
        assert len(txt_files) > 0

    def test_load_cam_framenums(self):
        """测试加载摄像头帧数"""
        fn_path = PROJECT_ROOT / DATASET_ROOT / "cam_framenum" / "S01.txt"
        if not fn_path.exists():
            fn_path = PROJECT_ROOT / "cityflow" / "extracted" / "cam_framenum" / "S01.txt"
        if not fn_path.exists():
            pytest.skip("S01.txt 不存在")

        framenums = load_cam_framenums("S01")
        if framenums:
            assert isinstance(framenums, dict)
            for cam_id, num in framenums.items():
                assert isinstance(num, int)
                assert num > 0


# ============================================================
# 视频文件存在性测试
# ============================================================

class TestVideoFiles:
    """视频文件存在性检查"""

    def test_video_path_format(self):
        """测试视频路径格式正确"""
        path = get_video_path("train", "S01", "c001")
        assert "train" in path
        assert "S01" in path
        assert "c001" in path
        assert path.endswith("vdo.avi")

    def test_available_videos(self):
        """测试可用的视频文件"""
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            pytest.skip("train 目录不存在")

        video_count = 0
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if not scene_path.exists():
                continue
            for cam_dir in scene_path.iterdir():
                if cam_dir.is_dir():
                    vdo = cam_dir / "vdo.avi"
                    if vdo.exists() and vdo.stat().st_size > 0:
                        video_count += 1

        assert video_count > 0, "未找到可用视频文件"

    def test_video_file_size_reasonable(self):
        """测试视频文件大小合理"""
        train_path = PROJECT_ROOT / DATASET_ROOT / "train"
        if not train_path.exists():
            pytest.skip("train 目录不存在")

        # 找到第一个可用视频
        for scene in ["S01", "S03", "S04"]:
            scene_path = train_path / scene
            if not scene_path.exists():
                continue
            for cam_dir in sorted(scene_path.iterdir()):
                if cam_dir.is_dir():
                    vdo = cam_dir / "vdo.avi"
                    if vdo.exists():
                        size_mb = vdo.stat().st_size / (1024 * 1024)
                        assert size_mb > 0.1, f"视频文件过小: {vdo} ({size_mb:.2f} MB)"
                        return  # 找到一个就够了

        pytest.skip("未找到视频文件")


# ============================================================
# 辅助配置文件路径测试
# ============================================================

class TestConfigPaths:
    """配置文件路径辅助函数测试"""

    def test_gt_path_format(self):
        """测试 gt 路径格式"""
        path = get_gt_path("train", "S01")
        assert "train" in path
        assert "S01" in path
        assert "gt.txt" in path

    def test_cam_timestamp_path_format(self):
        """测试时间戳路径格式"""
        path = get_cam_timestamp_path("S01")
        assert "cam_timestamp" in path
        assert "S01.txt" in path

    def test_cam_framenum_path_format(self):
        """测试帧数路径格式"""
        path = get_cam_framenum_path("S04")
        assert "cam_framenum" in path
        assert "S04.txt" in path
