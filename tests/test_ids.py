"""
tests/test_ids.py - ID 解析测试

测试目标: 验证 src/common/ids.py 对 target_id / track_id 的统一解析
覆盖:
  - parse_target_id: CF3_c001_V0034_000001 正常解析
  - parse_track_id: CF3_TRACK_c001_V0034 正常解析
  - extract_vehicle_id: 从两种 ID 中取 vehicle_id
  - 畸形输入（空串 / None / 段数不对 / V 后非数字）返回空值而不抛异常
"""

import pytest

from src.common.ids import extract_vehicle_id, parse_target_id, parse_track_id


class TestParseTargetId:
    """target_id 解析"""

    def test_parse_valid_target_id(self):
        """标准 target_id 解析出四个字段"""
        parsed = parse_target_id("CF3_c001_V0034_000001")
        assert parsed == {
            "source": "CF3",
            "camera_id": "c001",
            "vehicle_id": "V0034",
            "det_seq": "000001",
        }

    def test_parse_other_camera_and_vehicle(self):
        """不同摄像头/车辆编号同样可解析"""
        parsed = parse_target_id("CF3_c016_V1234_000042")
        assert parsed["camera_id"] == "c016"
        assert parsed["vehicle_id"] == "V1234"
        assert parsed["det_seq"] == "000042"

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",
            None,
            12345,
            "CF3_c001_V0034",              # 段数少
            "CF3_c001_V0034_000001_extra",  # 段数多
            "CF3_c001_X0034_000001",       # 车辆段不以 V 开头
            "CF3_c001_VABCD_000001",       # V 后面不是数字
            "CF3_c001_0034_000001",        # 缺 V 前缀
            "CF3_c001_V0034_abcdef",       # 序号不是数字
            "CF3_001_V0034_000001",        # 摄像头段不合法
        ],
    )
    def test_parse_malformed_returns_empty(self, bad_input):
        """畸形输入返回空值，不抛异常"""
        parsed = parse_target_id(bad_input)
        assert parsed == {
            "source": None,
            "camera_id": None,
            "vehicle_id": None,
            "det_seq": None,
        }


class TestParseTrackId:
    """track_id 解析"""

    def test_parse_valid_track_id(self):
        """标准 track_id 解析出三个字段"""
        parsed = parse_track_id("CF3_TRACK_c001_V0034")
        assert parsed == {
            "source": "CF3",
            "camera_id": "c001",
            "vehicle_id": "V0034",
        }

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",
            None,
            "CF3_c001_V0034_000001",     # 这是 target_id，不是 track_id
            "CF3_TRACK_c001",            # 段数少
            "CF3_TRACK_c001_V0034_0001",  # 段数多
            "CF3_c001_V0034_TRACK",      # 缺少 TRACK 标记
            "CF3_TRACK_c001_VABCD",      # V 后面不是数字
        ],
    )
    def test_parse_malformed_returns_empty(self, bad_input):
        """畸形输入返回空值，不抛异常"""
        parsed = parse_track_id(bad_input)
        assert parsed == {"source": None, "camera_id": None, "vehicle_id": None}


class TestExtractVehicleId:
    """vehicle_id 提取（target_id / track_id 通用）"""

    def test_from_target_id(self):
        """从 target_id 提取"""
        assert extract_vehicle_id("CF3_c001_V0034_000001") == "V0034"

    def test_from_track_id(self):
        """从 track_id 提取"""
        assert extract_vehicle_id("CF3_TRACK_c001_V0034") == "V0034"

    def test_source_digits_not_mistaken(self):
        """'CF3' 中的 3 不会被误判为 vehicle_id"""
        assert extract_vehicle_id("CF3") == ""
        assert extract_vehicle_id("CF3_c001") == ""
        assert extract_vehicle_id("V3") == "V3"

    @pytest.mark.parametrize("bad_input", ["", None, 12345, "INST_TEST_001", "unknown"])
    def test_malformed_returns_empty_string(self, bad_input):
        """无法识别时返回空串"""
        assert extract_vehicle_id(bad_input) == ""

    def test_consistent_between_id_types(self):
        """同一车辆的 target_id 与 track_id 提取结果一致"""
        target_id = "CF3_c001_V0034_000001"
        track_id = f"CF3_TRACK_{parse_target_id(target_id)['camera_id']}_" \
                   f"{parse_target_id(target_id)['vehicle_id']}"
        assert extract_vehicle_id(target_id) == extract_vehicle_id(track_id)
