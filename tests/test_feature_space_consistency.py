"""
tests.test_feature_space_consistency - 特征空间一致性（PLAN3-B3）

背景：这个项目**真的因此出过事**。库里一度同时存在两套 CLIP 向量：

  - 949 条检测内联的是 768 维（CN-CLIP ViT-L-14 时代的产物）
  - 而在线检索的 FAISS 索引是 512 维（ViT-B-16）

两者维度不同、不可混用，直接算余弦会抛 `ValueError`。
修复过程见 `scripts/unify_clip_949.py` + `scripts/merge_clip_unified.py`。

**但修复只做了一半**：图像向量已重编码为 512，文本向量仍是 768
（`output/datastore/det_text_vectors.npy` = (949, 768)）。
文本向量当前**没有任何在线消费者**（唯一的向量使用者是
`builder.py` 读 `clip_image_vector`），所以暂时无害 ——
但它是一个随时可能被误用的陷阱。

本文件把"特征空间必须一致"钉成可执行的不变量：
一旦有人再引入不同维度的向量，或把文本向量当图像向量用，这里会红。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _ROOT / "output"
_DATASTORE = _OUTPUT / "datastore"

faiss = pytest.importorskip("faiss", reason="需要 faiss 才能读取在线索引维度")

# 在线检索实际使用的图像向量维度（ViT-B-16）
_EXPECTED_CLIP_DIM = 512


def _load_index_dim(path: Path):
    idx = faiss.read_index(str(path))
    return idx.d, idx.ntotal


@pytest.fixture(scope="module")
def index_info():
    path = _OUTPUT / "clip_vectors.faiss"
    if not path.exists():
        pytest.skip(f"缺少在线索引 {path}")
    return _load_index_dim(path)


class TestOnlineIndexMatchesExpectedSpace:
    def test_online_index_is_512_dim(self, index_info):
        """在线检索索引必须是 512 维（ViT-B-16）—— 换模型必须同步换索引"""
        dim, ntotal = index_info
        assert dim == _EXPECTED_CLIP_DIM, (
            f"在线索引维度 {dim} != 期望 {_EXPECTED_CLIP_DIM}。"
            f"若确实换了 CLIP 模型，请同步重建 output/clip_vectors.faiss "
            f"并更新 configs/default.yaml 的 feature.clip.vector_dim"
        )
        assert ntotal > 0


class TestDatastoreVectorsMatchOnlineIndex:
    """datastore 里参与在线打分的向量必须与检索索引同维"""

    def test_image_vectors_dim_matches_index(self, index_info):
        """内联图像向量（跨镜外观打分用）必须与在线索引同维

        这条是最关键的：`builder.py` 用 `clip_image_vector` 算跨镜外观相似度，
        而检索用 FAISS 索引 —— 两者若不同维，跨镜评分会在运行时抛异常。
        """
        path = _DATASTORE / "det_image_vectors.npy"
        if not path.exists():
            pytest.skip(f"缺少 {path}")

        arr = np.load(path, mmap_mode="r")
        index_dim, _ = index_info
        assert arr.shape[1] == index_dim, (
            f"内联图像向量是 {arr.shape[1]} 维，而在线索引是 {index_dim} 维 —— "
            f"正是 2026-09 那次的 768/512 混库问题。"
            f"修复方式见 scripts/unify_clip_949.py"
        )

    def test_text_vectors_are_not_silently_mixed(self):
        """文本向量若仍是 768 维，必须与图像向量不同维 —— 这是已知的半数迁移

        本测试**不要求**文本向量是 512（它是无人使用的历史产物），
        而是把这个状态显式记录下来：一旦有人把它当成图像向量用，
        维度不等会在运行时炸；这里先一步把事实写清楚。
        """
        path = _DATASTORE / "det_text_vectors.npy"
        if not path.exists():
            pytest.skip("文本向量已移除（B3 的清理目标），跳过")

        arr = np.load(path, mmap_mode="r")
        img_path = _DATASTORE / "det_image_vectors.npy"
        if not img_path.exists():
            pytest.skip("缺少图像向量，无法比较")

        img = np.load(img_path, mmap_mode="r")
        assert arr.shape[0] == img.shape[0], "文本/图像向量的行数应一一对应"

        if arr.shape[1] != img.shape[1]:
            # 已知状态：文本 768 / 图像 512。此时唯一的要求是"没人混用"。
            assert arr.shape[1] == 768 and img.shape[1] == 512, (
                f"出现了预期之外的维度组合: 文本 {arr.shape[1]} / 图像 {img.shape[1]}；"
                f"请确认是否有新的向量来源混入"
            )


class TestNoOnlineConsumerOfTextVectors:
    """文本向量在线路径上不得有消费者（有的话必须同维，见上一个测试）"""

    def test_only_build_script_and_pipeline_reference_text_vectors(self):
        """只有构建脚本与离线特征管线可以引用文本向量"""
        allowed = {
            "src/storage/datastore.py",
            "src/storage/__init__.py",
            "scripts/build_datastore.py",
            "scripts/clip_feature_pipeline.py",
            "scripts/merge_clip_unified.py",
            "scripts/import_baseline_detections.py",
        }
        offenders = []
        for base in ("api", "src/trajectory", "src/stitching", "frontend"):
            for py in (_ROOT / base).rglob("*.py"):
                rel = py.relative_to(_ROOT).as_posix()
                if rel in allowed:
                    continue
                text = py.read_text(encoding="utf-8", errors="ignore")
                if "clip_text_vector" in text or "det_text_vectors" in text:
                    offenders.append(rel)

        assert not offenders, (
            f"以下在线模块引用了文本向量: {offenders}。"
            f"若确实要用，必须先把维度统一为 512（见 PLAN3-B3）"
        )
