"""
tests/test_path_count_bounds.py - 路径计数必须「有界」（性能回归守护）

## 守的是什么

`CrossCameraScorer._count_possible_paths` 枚举两摄像头之间的简单路径数，
用来算路径分叉惩罚 `penalty = 1 - 1/num_paths`。简单路径的数量在图上是
**指数级**的，而旧实现只限制了**计数**（`found < max_paths`）、没有限制
**搜索**，于是：

- 两点不可达时 `found` 恒为 0，循环条件恒真，DFS 走完整棵指数级的路径树
- **实测**单对摄像头 348 万次递归、1.84 秒（c029→c030）
- 而**线上每一对候选边**都要调它一次 ⇒ 接口超时

修复方式是在基类上加 `MAX_PATH_HOPS` / `PATH_EXPANSION_BUDGET` 两个上限。

## 为什么必须钉在**基类**上

历史上这两个上限只写在 `src/trajectory/builder.py` 的
`BoundedCrossCameraScorer` 子类里 —— 那只救了走 builder 的那条路，
任何直接用 `CrossCameraScorer` 的代码（例如 `scripts/train_edge_scorer.py`
构边）依然会撞上爆炸。本文件的第一条用例就是防这个回退。

## 注意：这不是等价重构

修复会改变返回值（709 对可比对中 278 对不同，详见 `scoring.py` 的 docstring）。
所以这里**不**断言"与旧实现相等"——那会是假的；只断言**有界**与**语义方向**。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology
from src.stitching.scoring import CrossCameraScorer

_META = ROOT / "configs" / "cityflow_camera_metadata.yaml"

# 修复前实测最慢的几对（c001→c002 达 23.2s、c029→c030 达 1.84s）
_PATHOLOGICAL = [
    ("c001", "c002"),
    ("c001", "c003"),
    ("c029", "c030"),
    ("c029", "c031"),
]


@pytest.fixture(scope="module")
def scorer() -> CrossCameraScorer:
    return CrossCameraScorer(CameraManager(str(_META)), RoadTopology(str(_META)))


def test_base_class_defines_bounds(scorer):
    """
    上限必须定义在**基类**上 —— 这是本条回归的核心

    历史 bug：上限只在 `BoundedCrossCameraScorer` 子类里，基类没有。
    """
    assert hasattr(CrossCameraScorer, "MAX_PATH_HOPS"), (
        "基类缺少 MAX_PATH_HOPS —— 上限又跑回子类里了，"
        "直接用 CrossCameraScorer 的调用方会再次组合爆炸"
    )
    assert hasattr(CrossCameraScorer, "PATH_EXPANSION_BUDGET")
    assert CrossCameraScorer.MAX_PATH_HOPS > 0
    assert CrossCameraScorer.PATH_EXPANSION_BUDGET > 0


def test_bounds_are_on_base_not_only_subclass(scorer):
    """
    子类不应再覆盖该方法 —— 两处各写一份实现会再次漂移

    `BoundedCrossCameraScorer` 降级为兼容别名后，应与基类**共用**同一个函数对象。
    """
    from src.trajectory.builder import BoundedCrossCameraScorer

    assert (
        BoundedCrossCameraScorer._count_possible_paths
        is CrossCameraScorer._count_possible_paths
    ), "BoundedCrossCameraScorer 又自己实现了一份路径计数，两边会漂移"


def test_pathological_pairs_are_fast(scorer):
    """
    修复前这几对是秒级（最慢 23.2s），现在必须是亚毫秒级

    断言留了很宽的余量（1 秒），只要不退回指数枚举就不会触发。
    """
    for src, tgt in _PATHOLOGICAL:
        t0 = time.perf_counter()
        value = scorer._count_possible_paths(src, tgt)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, (
            f"{src}->{tgt} 耗时 {elapsed:.3f}s —— 路径计数退回指数枚举了"
        )
        assert value >= 1, "路径数下限为 1（不可达时也返回 1，与旧语义一致）"


def test_all_pairs_complete_quickly(scorer):
    """
    全拓扑 1035 对必须整体跑完

    修复前：单对最慢 23.2s，全量跑一遍要几十分钟（本文件作者实测 7 分钟超时）。
    修复后：全量应在毫秒级。这里给 5 秒的极宽松上限。
    """
    ids = sorted(scorer.road_topology.camera_ids)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    assert len(pairs) > 900, f"摄像头太少（{len(ids)} 个），用例失去意义"

    t0 = time.perf_counter()
    for a, b in pairs:
        scorer._count_possible_paths(a, b)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"{len(pairs)} 对耗时 {elapsed:.2f}s —— 单对平均 "
        f"{elapsed / len(pairs) * 1000:.2f}ms，路径计数退回指数枚举了"
    )


def test_count_is_capped_at_five(scorer):
    """返回值上限仍是 5（惩罚公式 `1 - 1/num_paths` 依赖这个口径）"""
    ids = sorted(scorer.road_topology.camera_ids)
    for a in ids:
        for b in ids:
            if a != b:
                assert scorer._count_possible_paths(a, b) <= 5


def test_returns_one_for_unknown_camera(scorer):
    """不在邻接表中的摄像头返回 1（旧语义，保持不变）"""
    assert scorer._count_possible_paths("c001", "不存在的摄像头") == 1
    assert scorer._count_possible_paths("不存在的摄像头", "c001") == 1


def test_hop_cap_actually_prunes(scorer):
    """
    钉住"跳数上限真的在起作用"这个事实

    如果哪天有人把 MAX_PATH_HOPS 调到很大（或去掉），本用例会失败并提醒：
    那样虽然返回值更接近旧实现，但会把指数爆炸放回来。
    """
    assert CrossCameraScorer.MAX_PATH_HOPS <= 8, (
        "MAX_PATH_HOPS 被放大了 —— 跳数上限是压住组合爆炸的主要手段，不要随意提高"
    )


def test_documented_divergence_is_real(scorer):
    """
    钉住"这不是等价重构"这个**已知事实**，防止有人再写错文档

    `scoring.py` 的 docstring 记着：旧实现能终止的 709 对里有 278 对返回值不同
    （集中在 `(旧 5, 新 1)`，即旧版把 >6 跳的绕行也计入了）。
    这里用一对具体样本确认差异确实存在 —— 若哪天变成相等，
    说明上限被放得很宽，应回头更新那段文档而不是让文档继续撒谎。
    """
    # c001->c014：旧实现（无跳数上限）给 5，有界版给 1
    assert scorer._count_possible_paths("c001", "c014") == 1
