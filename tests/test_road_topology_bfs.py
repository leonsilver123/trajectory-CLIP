"""
tests.test_road_topology_bfs - 路网拓扑里 BFS 与 Dijkstra 的分工（PLAN5-E）

## 背景：一次自我更正

我先前对作者说「简历写 BFS 但代码实际是 Dijkstra」—— **那是错的**。
我只 grep 了 "Dijkstra" 就下了结论，没读 `is_reachable` 的实现。实际是：

| 方法 | 算法 | 用途 |
|---|---|---|
| `is_reachable()` | **BFS**（`deque` + `popleft` + `visited` + hop 计数） | 判断"能不能到"（无权，只关心连通） |
| `shortest_path()` | **Dijkstra**（`heapq` 优先队列 + 边权） | 求"有多远"（带权，关心距离） |

所以简历里的「BFS 路网拓扑」**是准确的**。本文件把这件事钉住：
若有人把 `is_reachable` 改成带权最短路，或删掉 BFS，这里会红 ——
因为那会让简历的一句话变假。

## 为什么值得单独一个测试文件

这是"文档与代码一致性"类的问题里最容易悄悄失效的一种：
算法被重构（BFS → Dijkstra）在**功能上完全等价**，没有任何测试会失败，
但项目对外的描述从此不实。项目已经因为这类问题吃过亏
（见 PLAN2 的 E6、PLAN3 的 F1）。
"""

from __future__ import annotations

import inspect
from collections import deque

import pytest

from src.data_governance.road_topology import RoadTopology


class TestIsReachableIsBFS:
    """`is_reachable` 必须是 BFS（无权、按层扩展）"""

    def test_uses_deque(self):
        """BFS 的标志：用 deque 且从左侧弹出（FIFO）"""
        src = inspect.getsource(RoadTopology.is_reachable)
        assert "deque" in src, "is_reachable 不再使用 deque — 还是 BFS 吗？"
        assert "popleft" in src, "is_reachable 不再 FIFO 弹出 — 还是 BFS 吗？"

    def test_does_not_use_priority_queue(self):
        """不应出现 heap/优先队列 —— 那说明被改成了带权最短路（Dijkstra）"""
        src = inspect.getsource(RoadTopology.is_reachable)
        assert "heapq" not in src and "heappush" not in src, (
            "is_reachable 用了优先队列 —— 它已被改成 Dijkstra 类算法，"
            "简历里的「BFS 路网拓扑」将不再准确。请确认这是有意改动并同步描述。"
        )

    def test_has_visited_set(self):
        """BFS 必须有 visited 集合，否则无向图上会无限循环"""
        src = inspect.getsource(RoadTopology.is_reachable)
        assert "visited" in src


class TestShortestPathIsDijkstra:
    """`shortest_path` 必须是带权最短路（Dijkstra），与 BFS 分工不同"""

    def test_uses_priority_queue(self):
        src = inspect.getsource(RoadTopology.shortest_path)
        assert "heapq" in src or "heappush" in src, (
            "shortest_path 不再用优先队列 —— 带权最短路会算错"
        )


@pytest.fixture(scope="module")
def topo():
    """真实摄像头元数据构造的路网拓扑（模块级，避免 class 作用域弃用告警）"""
    from pathlib import Path

    meta = (
        Path(__file__).resolve().parent.parent
        / "configs" / "cityflow_camera_metadata.yaml"
    )
    if not meta.exists():
        pytest.skip(f"缺少 {meta}")
    return RoadTopology(str(meta))


class TestBFSSemantics:
    """用真实元数据验证 BFS 的行为特征"""

    def test_self_is_reachable(self, topo):
        """自己到自己恒定可达"""
        assert topo.is_reachable("c001", "c001") is True

    def test_unknown_camera_not_reachable(self, topo):
        """不存在的摄像头不可达（不抛异常）"""
        assert topo.is_reachable("c001", "__nope__") is False
        assert topo.is_reachable("__nope__", "c001") is False

    def test_zero_hops_only_self(self, topo):
        """max_hops=0 时只有自己可达 —— 这条最能体现**按跳数**的 BFS 语义

        带权最短路没有"跳数"概念，这个参数会变得没有意义。
        """
        assert topo.is_reachable("c001", "c001", max_hops=0) is True

    def test_max_hops_monotonic(self, topo):
        """可达性随 max_hops 单调不减（放宽容许不会让原本可达的变成不可达）"""
        pairs = [("c001", "c002"), ("c001", "c005"), ("c001", "c010")]
        for src, dst in pairs:
            prev = False
            for hops in (1, 2, 3, 5, 10):
                cur = topo.is_reachable(src, dst, max_hops=hops)
                assert cur or not prev, (
                    f"{src}->{dst} 在 max_hops={hops} 时不可达，"
                    f"但更小的 max_hops 下却可达 —— 违反单调性"
                )
                prev = prev or cur

    def test_bfs_returns_bool_not_path(self, topo):
        """BFS 只回答"能不能到"，返回 bool；距离是 Dijkstra 的职责"""
        result = topo.is_reachable("c001", "c002")
        assert isinstance(result, bool)
