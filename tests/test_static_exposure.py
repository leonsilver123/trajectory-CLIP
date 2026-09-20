"""
tests.test_static_exposure - /static 暴露面测试（PLAN3-E1）

背景：早先 `api/main.py` 把整个 `output/` 挂到 `/static`，于是
`cityflow_results.json`（75MB 全量数据）、`datastore/*.parquet`、`meta.sqlite`
以及各类调试日志都能被直接 HTTP 下载。

修复方式是**逐目录图片白名单**（`_STATIC_IMAGE_DIRS`）。本文件锁定两条不变量：

1. 非图片产物必须 404（回归守护 —— 谁再把整个 output/ 挂上去就会红）
2. 白名单内的真实图片必须 200（防止"修安全问题顺手把功能改坏"）
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import _STATIC_IMAGE_DIRS, create_app

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _ROOT / "output"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _first_image(dirname: str) -> Path | None:
    """取白名单目录下的第一张图（用于验证功能未被改坏）"""
    base = _OUTPUT / dirname
    if not base.is_dir():
        return None
    for p in base.rglob("*.jpg"):
        return p
    return None


# ============================================================
# 不变量 1：非图片产物必须不可达
# ============================================================


class TestNonImageArtifactsAreNotExposed:
    """数据文件与内部产物不得经 /static 暴露"""

    @pytest.mark.parametrize(
        "path",
        [
            "/static/cityflow_results.json",
            "/static/results.json",
            "/static/datastore/meta.sqlite",
            "/static/datastore/detections.parquet",
            "/static/datastore/tracks.parquet",
            "/static/datastore/det_image_vectors.npy",
        ],
    )
    def test_returns_404(self, client, path):
        assert client.get(path).status_code == 404, f"{path} 不应可达"

    def test_datastore_dir_itself_is_not_a_mount(self, client):
        """datastore 目录没有被挂载（列目录也应 404）"""
        assert client.get("/static/datastore/").status_code == 404

    def test_probing_nonexistent_data_file_is_404(self, client):
        """随便猜一个数据文件名也应 404，不得因兜底路由而返回 index.html"""
        resp = client.get("/static/not_a_real_artifact.json")
        assert resp.status_code == 404


# ============================================================
# 不变量 2：白名单内的图片必须可达
# ============================================================


class TestWhitelistedImagesStayReachable:
    """修复安全问题时不得把图片服务一起改坏"""

    def test_at_least_one_image_dir_present(self):
        """本仓库应至少有一个图片目录 —— 否则下面的测试会静默全跳过"""
        present = [d for d in _STATIC_IMAGE_DIRS if (_OUTPUT / d).is_dir()]
        if not (_OUTPUT / "aicity22_crops").is_dir():
            pytest.skip("output/ 数据产物不存在，跳过")
        assert present, f"白名单目录一个都不存在: {_STATIC_IMAGE_DIRS}"

    @pytest.mark.parametrize("dirname", ["aicity22_crops", "aicity22_frames"])
    def test_first_image_is_served(self, client, dirname):
        img = _first_image(dirname)
        if img is None:
            pytest.skip(f"{dirname}/ 无图片，跳过")
        rel = img.relative_to(_OUTPUT).as_posix()
        resp = client.get(f"/static/{rel}")
        assert resp.status_code == 200, f"/static/{rel} 应返回 200，实际 {resp.status_code}"
        assert resp.headers["content-type"].startswith("image/")

    def test_all_whitelist_entries_are_image_dirs(self):
        """白名单里不应混进数据目录 —— 防的是「顺手把 datastore 加进去」"""
        forbidden = {"datastore", "attributes", "reid", "logs"}
        assert not (set(_STATIC_IMAGE_DIRS) & forbidden), (
            f"白名单包含数据目录: {set(_STATIC_IMAGE_DIRS) & forbidden}"
        )
