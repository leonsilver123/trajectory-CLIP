"""
tests/test_camera_manager.py - 摄像头管理测试

测试目标: 验证 src/data_governance/camera_manager.py 功能
覆盖:
  - CameraManager 初始化（加载 cityflow_camera_metadata.yaml）
  - get_camera() 获取摄像头元数据
  - get_cameras_by_scene() 按场景获取
  - get_nearby_cameras() 空间查询
  - is_topologically_reachable() 拓扑可达性
  - 46 个摄像头完整性检查

注意: cityflow_camera_metadata.yaml 中字段名为 'scene' 而非 'scene_id'，
      CameraManager._load_metadata() 使用 cam_data.get("scene_id") 读取，
      这可能导致 scene_id 为 None（已知 bug）。
"""

import pytest
from pathlib import Path

from src.data_governance.camera_manager import CameraManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_METADATA_PATH = str(PROJECT_ROOT / "configs" / "cityflow_camera_metadata.yaml")


@pytest.fixture
def manager():
    """创建 CameraManager 实例"""
    return CameraManager(CAMERA_METADATA_PATH)


# ============================================================
# 初始化测试
# ============================================================

class TestCameraManagerInit:
    """CameraManager 初始化测试"""

    def test_init_loads_cameras(self, manager):
        """测试初始化加载了摄像头"""
        assert manager.get_camera_count() > 0

    def test_init_camera_count(self, manager):
        """测试摄像头总数为 46"""
        assert manager.get_camera_count() == 46

    def test_init_file_not_found(self):
        """测试文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError):
            CameraManager("/nonexistent/path/metadata.yaml")

    def test_camera_ids_list(self, manager):
        """测试摄像头 ID 列表"""
        ids = manager.camera_ids
        assert len(ids) == 46
        assert "c001" in ids
        assert "c046" in ids


# ============================================================
# get_camera 测试
# ============================================================

class TestGetCamera:
    """get_camera 接口测试"""

    def test_get_existing_camera(self, manager):
        """测试获取存在的摄像头"""
        cam = manager.get_camera("c001")
        assert cam is not None
        assert cam.camera_id == "c001"

    def test_get_camera_has_name(self, manager):
        """测试摄像头有名称"""
        cam = manager.get_camera("c001")
        assert cam.name == "Scene01-Camera01"

    def test_get_camera_has_gps(self, manager):
        """测试摄像头有 GPS 坐标"""
        cam = manager.get_camera("c001")
        assert cam.latitude is not None
        assert cam.longitude is not None
        assert abs(cam.latitude - 42.526) < 0.01

    def test_get_camera_has_direction(self, manager):
        """测试摄像头有朝向"""
        cam = manager.get_camera("c001")
        assert cam.direction == 0.0

    def test_get_camera_has_road_segment(self, manager):
        """测试摄像头有路段信息"""
        cam = manager.get_camera("c001")
        assert cam.covered_road_segment == "SEG_S01_001"

    def test_get_camera_has_lane_direction(self, manager):
        """测试摄像头有车道方向"""
        cam = manager.get_camera("c001")
        assert cam.lane_direction == "northbound"

    def test_get_nonexistent_camera(self, manager):
        """测试获取不存在的摄像头返回 None"""
        cam = manager.get_camera("c099")
        assert cam is None

    def test_camera_scene_id_field(self, manager):
        """测试摄像头 scene_id 字段是否正确加载

        YAML 里字段名是 'scene'，CameraManager 读的是 'scene_id'；
        `camera_manager.py` 已写成 `cam_data.get("scene_id") or cam_data.get("scene")`
        兼容两种键名，因此这里应当是**正常通过**的测试，不再是 xfail。
        """
        cam = manager.get_camera("c001")
        assert cam.scene_id == "S01", (
            f"scene_id 应为 'S01' 但得到 {cam.scene_id}"
        )


# ============================================================
# get_cameras_by_scene 测试
# ============================================================

class TestGetCamerasByScene:
    """get_cameras_by_scene 接口测试"""

    def test_get_cameras_by_scene_s01(self, manager):
        """测试按场景 S01 获取摄像头"""
        cams = manager.get_cameras_by_scene("S01")
        assert len(cams) == 5
        cam_ids = {c.camera_id for c in cams}
        assert cam_ids == {"c001", "c002", "c003", "c004", "c005"}

    def test_get_cameras_by_scene_s06(self, manager):
        """测试按场景 S06 获取摄像头"""
        cams = manager.get_cameras_by_scene("S06")
        assert len(cams) == 6

    def test_get_cameras_by_scene_empty_for_unknown(self, manager):
        """测试未知场景返回空列表"""
        cams = manager.get_cameras_by_scene("S99")
        assert len(cams) == 0


# ============================================================
# get_nearby_cameras 测试
# ============================================================

class TestGetNearbyCameras:
    """get_nearby_cameras 空间查询测试"""

    def test_nearby_cameras_returns_results(self, manager):
        """测试附近摄像头查询返回结果"""
        # 使用 c001 的坐标查询
        cams = manager.get_nearby_cameras(42.526, -90.7236, radius_km=1.0)
        assert len(cams) > 0

    def test_nearby_cameras_includes_self(self, manager):
        """测试附近摄像头包含自身"""
        cams = manager.get_nearby_cameras(42.526, -90.7236, radius_km=1.0)
        cam_ids = [c.camera_id for c in cams]
        assert "c001" in cam_ids

    def test_nearby_cameras_sorted_by_distance(self, manager):
        """测试附近摄像头按距离排序"""
        cams = manager.get_nearby_cameras(42.526, -90.7236, radius_km=5.0)
        # 至少返回多个结果
        if len(cams) >= 2:
            # 第一个应该是距离最近的（c001 本身距离为 0）
            assert cams[0].camera_id == "c001"

    def test_nearby_cameras_small_radius(self, manager):
        """测试极小半径只返回自身

        注意：查询点必须用 c001 的**精确**坐标（42.525678, -90.723601）。
        早先这里写的是四舍五入后的 42.526，与 c001 实际相距约 35.8 米
        （Δlat 0.000322° × 111320 m/°），却用 1 米半径去查，
        于是必然返回空列表 —— 那是测试期望写错，不是 get_nearby_cameras 的 bug。
        """
        cams = manager.get_nearby_cameras(42.525678, -90.723601, radius_km=0.001)
        # c001 与本查询点重合，距离为 0，必须被返回且排在首位
        assert len(cams) >= 1, f"半径 1m 内应至少含 c001，实际返回 {len(cams)} 个"
        assert cams[0].camera_id == "c001"

    def test_nearby_cameras_large_radius(self, manager):
        """测试大半径返回所有摄像头"""
        cams = manager.get_nearby_cameras(42.5, -90.7, radius_km=100.0)
        assert len(cams) == 46

    def test_get_cameras_by_region(self, manager):
        """测试按车道方向筛选"""
        northbound = manager.get_cameras_by_region("northbound")
        assert len(northbound) > 0
        for cam in northbound:
            assert cam.lane_direction == "northbound"


# ============================================================
# 拓扑可达性测试
# ============================================================

class TestTopology:
    """拓扑可达性与邻接关系测试"""

    def test_adjacent_cameras_s01(self, manager):
        """测试 S01 场景内相邻摄像头"""
        neighbors = manager.get_adjacent_cameras("c001")
        assert len(neighbors) > 0
        assert "c002" in neighbors

    def test_are_adjacent_true(self, manager):
        """测试相邻摄像头判断"""
        assert manager.are_adjacent("c001", "c002") is True

    def test_are_adjacent_false(self, manager):
        """测试不相邻摄像头判断"""
        # c001 和 c041 距离很远，不应相邻
        assert manager.are_adjacent("c001", "c041") is False

    def test_is_topologically_reachable_same(self, manager):
        """测试自身可达"""
        assert manager.is_topologically_reachable("c001", "c001") is True

    def test_is_topologically_reachable_adjacent(self, manager):
        """测试相邻摄像头可达"""
        assert manager.is_topologically_reachable("c001", "c002") is True

    def test_is_topologically_reachable_cross_scene(self, manager):
        """测试跨场景可达性"""
        # S01 和 S03 之间应该拓扑可达（通过中间摄像头）
        result = manager.is_topologically_reachable("c001", "c010", max_hops=10)
        assert result is True

    def test_is_topologically_reachable_s06_isolated(self, manager):
        """测试 S06 与其他场景的连通性"""
        # S06 (c041-c046) 通过 c044/c045 连接到 S02 (c006/c007)
        result = manager.is_topologically_reachable("c041", "c001", max_hops=10)
        assert result is True

    def test_is_topologically_reachable_unknown_camera(self, manager):
        """测试未知摄像头不可达"""
        assert manager.is_topologically_reachable("c001", "c099") is False

    def test_road_segments_loaded(self, manager):
        """测试路段数据已加载"""
        segments = manager.road_segments
        assert len(segments) > 0
        assert "SEG_S01_001" in segments


# ============================================================
# register_camera 测试
# ============================================================

class TestRegisterCamera:
    """动态注册摄像头测试"""

    def test_register_new_camera(self, manager):
        """测试注册新摄像头"""
        cam = manager.register_camera(
            camera_id="c_test",
            name="Test Camera",
            latitude=42.5,
            longitude=-90.7,
            scene_id="S_TEST",
        )
        assert cam.camera_id == "c_test"
        assert manager.get_camera("c_test") is not None
        assert manager.get_camera_count() == 47

    def test_register_duplicate_raises(self, manager):
        """测试注册重复摄像头抛出异常"""
        with pytest.raises(ValueError, match="已存在"):
            manager.register_camera(
                camera_id="c001",
                name="Duplicate",
            )

    def test_register_with_adjacency(self, manager):
        """测试注册带邻接关系的摄像头"""
        manager.register_camera(
            camera_id="c_test2",
            name="Test Camera 2",
            adjacent_cameras=["c001", "c002"],
        )
        neighbors = manager.get_adjacent_cameras("c_test2")
        assert "c001" in neighbors
        assert "c002" in neighbors
