"""
tests.test_attribute_consistency - 三维属性一致性重排（PLAN5-D）

简历里这一条写的是「属性一致性约束重排（颜色、车型、方向三维余弦加权）」。
本文件守住**实际实现的三维加权**，并把与措辞的差异写在明处：

| 简历措辞 | 实际实现 | 说明 |
|---|---|---|
| 颜色、车型、**方向** | ✅ 三维都有 | 方向取自摄像头 `lane_direction`（检测不带方向字段） |
| **余弦**加权 | 三态取值后**加权平均** | 见下 |

## 为什么不是"余弦"

颜色/车型/方向都是**类别值**，不是连续向量。对类别值算余弦需要先把它们嵌入
向量空间；而"白色"与"白色"的余弦恒为 1，与精确匹配完全等价 ——
多一层嵌入不增加任何信息，只增加一个可出错的环节。

因此实现为：每一维给出三态取值（1.0 相符 / 0.0 不符 / 0.5 无证据），
再按权重求平均。**加权是实的，余弦是措辞差异**，已在文档中写明。
（业务系统若真做了属性嵌入，"余弦"才成立；本仓库没有那份嵌入。）
"""

from __future__ import annotations

import pytest

from api.routes.search import (
    ATTRIBUTE_WEIGHTS,
    CAMERA_DIRECTIONS,
    _attribute_consistency,
    _extract_query_features,
)


# ============================================================
# 方向提取
# ============================================================


class TestDirectionExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("由南向北的白色SUV", "northbound"),
        ("由北向南行驶的黑色轿车", "southbound"),
        ("向东的卡车", "eastbound"),
        ("向西的面包车", "westbound"),
        ("白色轿车向北", "northbound"),
        ("轿车朝南", "southbound"),
    ])
    def test_extracts_direction(self, text, expected):
        assert _extract_query_features(text)["direction"] == expected

    def test_longest_match_prevents_reversal(self):
        """**关键**：「由北向南」不得被 '北' 抢先命中成 northbound（那是反向错误）

        与车型的「皮卡 vs 卡车」是同一类 bug —— 按顺序取首个命中就会中招。
        """
        f = _extract_query_features("由北向南行驶的黑色轿车")
        assert f["direction"] == "southbound", (
            f"'由北向南' 被解析成 {f['direction']} —— 方向反了"
        )

    @pytest.mark.parametrize("text", ["白色轿车", "蓝色SUV", "黑色卡车"])
    def test_no_direction_returns_none(self, text):
        """查询没提方向时必须是 None，不能猜一个"""
        assert _extract_query_features(text)["direction"] is None


# ============================================================
# 三态取值与加权
# ============================================================


@pytest.fixture
def two_cameras():
    """取两个**方向不同**的摄像头（同方向的对照没有意义）"""
    by_dir: dict[str, list] = {}
    for cam, d in CAMERA_DIRECTIONS.items():
        by_dir.setdefault(d, []).append(cam)
    dirs = sorted(by_dir)
    if len(dirs) < 2:
        pytest.skip("摄像头方向种类不足")
    return by_dir[dirs[0]][0], dirs[0], by_dir[dirs[1]][0], dirs[1]


class TestThreeStateValues:
    def test_all_match_scores_one(self, two_cameras):
        cam, d, _, _ = two_cameras
        det = {"camera_id": cam, "attributes": {"color": "白色", "vehicle_type": "轿车"}}
        f = _extract_query_features("白色轿车")
        assert _attribute_consistency(det, f) == pytest.approx(1.0)

    def test_mismatch_scores_zero_on_that_dim(self, two_cameras):
        """颜色不符时，该维贡献 0，其余维照常"""
        cam, _, _, _ = two_cameras
        det = {"camera_id": cam, "attributes": {"color": "黑色", "vehicle_type": "轿车"}}
        f = _extract_query_features("白色轿车")
        score = _attribute_consistency(det, f)
        # 颜色权重 0.4 归零，车型 0.4 命中 -> 0.4 / 0.8 = 0.5
        assert score == pytest.approx(0.5)

    def test_unknown_detection_value_is_neutral(self, two_cameras):
        """检测该维未知时应为中性 0.5，**不是 0**（未知 ≠ 不符）"""
        cam, _, _, _ = two_cameras
        det = {"camera_id": cam, "attributes": {"vehicle_type": "轿车"}}  # 无颜色
        f = _extract_query_features("白色轿车")
        score = _attribute_consistency(det, f)
        # 颜色中性 0.4*0.5 + 车型命中 0.4 = 0.6 ; 总权重 0.8 -> 0.75
        assert score == pytest.approx(0.75)

    def test_unspecified_dimension_is_dropped_not_neutral(self, two_cameras):
        """查询没问的维度要**整维剔除**，不能按中性分算进去

        否则"没问方向"会给每个候选加同一个常数偏移，白白压缩区分度 ——
        这与 `stitching.reweight_missing_dimensions` 是同一原则。
        """
        cam, _, _, _ = two_cameras
        det = {"camera_id": cam, "attributes": {"color": "白色", "vehicle_type": "轿车"}}
        with_dir = _extract_query_features("白色轿车")
        # 两者都是满分，但若方向被按中性算进去，总分会被压到 0.9
        assert _attribute_consistency(det, with_dir) == pytest.approx(1.0)


class TestDirectionDimension:
    def test_direction_discriminates(self, two_cameras):
        """方向相符的摄像头得分应高于不符的"""
        cam_a, dir_a, cam_b, dir_b = two_cameras
        attrs = {"color": "白色", "vehicle_type": "轿车"}
        f = _extract_query_features(f"白色轿车{dir_a}")
        a = _attribute_consistency({"camera_id": cam_a, "attributes": attrs}, f)
        b = _attribute_consistency({"camera_id": cam_b, "attributes": attrs}, f)
        assert a > b, f"方向相符({dir_a}) 得分 {a} 未高于不符({dir_b}) {b}"
        # 差值应恰为方向权重
        assert a - b == pytest.approx(ATTRIBUTE_WEIGHTS["direction"])

    def test_unknown_camera_direction_is_neutral(self):
        """摄像头方向未知时按中性处理，不误杀"""
        det = {"camera_id": "__no_such_camera__",
               "attributes": {"color": "白色", "vehicle_type": "轿车"}}
        f = _extract_query_features("白色轿车由南向北")
        score = _attribute_consistency(det, f)
        assert score is not None and 0.0 < score < 1.0


class TestNoAttributesSpecified:
    def test_returns_none(self, two_cameras):
        """查询没指定任何可核对属性时返回 None（调用方据此不用这一项）

        **不能返回 0 或 0.5** —— 那会让"没什么可核对的查询"被当成"全都不匹配"。
        """
        cam, _, _, _ = two_cameras
        det = {"camera_id": cam, "attributes": {"color": "白色"}}
        f = _extract_query_features("找一辆车")
        assert _attribute_consistency(det, f) is None

    def test_weights_are_positive_and_configured(self):
        """三维权重存在且为正 —— 全为 0 会让这一维永远失效"""
        for k in ("color", "vehicle_type", "direction"):
            assert k in ATTRIBUTE_WEIGHTS
            assert ATTRIBUTE_WEIGHTS[k] > 0

    def test_weights_sum_to_one(self):
        """权重和应为 1，便于把一致性分解释成 [0,1] 的匹配度"""
        assert sum(ATTRIBUTE_WEIGHTS.values()) == pytest.approx(1.0)
