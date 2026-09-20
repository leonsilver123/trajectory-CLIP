"""
tests.test_stitching_dependency_contract - 跨镜评分对上游的依赖契约（PLAN3-D7）

背景：`src/stitching/` 里有 11 处

    except (NotImplementedError, AttributeError):
        ... 走宽松兜底

这种写法本意是"上游没实现就宽松处理"，但它同时吞掉了 **AttributeError ——
即方法改名/拼写错误**。一旦 `RoadTopology.is_reachable` 被改名，
评分不会报错，而是静默地认为"一切都可达"，指标悄悄变差却无人知晓。

现在这 11 处都加了 `logger.debug(..., exc_info=True)`，不再静默。
本文件进一步把"兜底分支是死的"这件事**钉成契约**：
断言上游依赖确实实现了 stitching 需要的每一个方法。
这样方法一旦被改名，这里会直接红，而不是退化成宽松兜底。
"""

from __future__ import annotations

import pytest

from src.data_governance.camera_manager import CameraManager
from src.data_governance.road_topology import RoadTopology

# stitching 实际调用的上游方法（由 grep 从 src/stitching/*.py 反推）
_ROAD_TOPOLOGY_METHODS = (
    "is_reachable",
    "get_segment_distance",
    "shortest_path",
)
_CAMERA_MANAGER_METHODS = (
    "get_camera",
    "is_topologically_reachable",
)


class TestDependencyContract:
    """上游依赖必须实现 stitching 需要的全部方法"""

    @pytest.mark.parametrize("method", _ROAD_TOPOLOGY_METHODS)
    def test_road_topology_implements(self, method):
        assert hasattr(RoadTopology, method), (
            f"RoadTopology 缺少 {method}() —— src/stitching 会静默走宽松兜底，"
            f"而不是报错。若确实改了名，请同步更新 src/stitching/ 的调用点。"
        )
        assert callable(getattr(RoadTopology, method))

    @pytest.mark.parametrize("method", _CAMERA_MANAGER_METHODS)
    def test_camera_manager_implements(self, method):
        assert hasattr(CameraManager, method), (
            f"CameraManager 缺少 {method}() —— src/stitching 会静默走宽松兜底。"
        )
        assert callable(getattr(CameraManager, method))


class TestFallbacksAreDormant:
    """兜底分支当前不应被触发（上游都实现了）"""

    def test_real_instances_answer_without_fallback(self, caplog):
        """用真实实例调用一遍，不应产生"走宽松兜底"的日志"""
        import logging
        from pathlib import Path

        meta = (
            Path(__file__).resolve().parent.parent
            / "configs" / "cityflow_camera_metadata.yaml"
        )
        if not meta.exists():
            pytest.skip(f"缺少摄像头元数据 {meta}")

        mgr = CameraManager(str(meta))
        topo = RoadTopology(str(meta))

        cams = list(getattr(mgr, "_cameras", {}).keys())
        if len(cams) < 2:
            pytest.skip("摄像头元数据不足，跳过")
        a, b = cams[0], cams[1]

        with caplog.at_level(logging.DEBUG):
            topo.is_reachable(a, b)
            topo.get_segment_distance(a, b)
            mgr.get_camera(a)
            mgr.is_topologically_reachable(a, b)

        hits = [r.message for r in caplog.records if "走宽松兜底" in str(r.message)]
        assert not hits, (
            f"兜底分支被触发了 {len(hits)} 次 —— 说明某个上游方法实际不可用，"
            f"评分正在静默降级：{hits[:3]}"
        )
