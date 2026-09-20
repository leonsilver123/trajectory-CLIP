"""
tests.test_frontend_no_fabrication - 前端不得编造数据、不得自算分数

## 这个文件守的是什么

项目有两条硬约束，都是有代价换来的：

1. **前端不编造数据**。`frontend/`（Streamlit，已于 2026-09-21 删除）曾有三处编造：
   写死的待办数字（23/8/5/12）、7 条凭空捏造的活动记录（假车牌、假文件名、假告警），
   以及把 `cameras_online` 直接设成 `cameras_total`（等于声称"全部在线"，
   而离线视频数据集根本没有在线状态这一事实）。

2. **前端不自算分数**。两套前端各自实现过一次检索与打分，导致同一句话
   在两个界面给出不同排序 —— 那是两个真相。收敛后唯一真源是后端接口。

`webapp/`（React + TypeScript）是**幸存的那一套前端**，本文件把上面两条约束
钉在它身上。此前这些断言针对 `frontend/`（见已删除的 `test_frontend_retrieval.py`），
Streamlit 下线后迁移到这里，避免约束随被删代码一起消失。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WEBAPP_SRC = _ROOT / "webapp" / "src"


def _webapp_files(suffix: str = ".ts") -> list[Path]:
    if not _WEBAPP_SRC.is_dir():
        return []
    return [
        p for p in _WEBAPP_SRC.rglob(f"*{suffix}")
        if "node_modules" not in p.parts
    ]


def _all_sources() -> list[tuple[Path, str]]:
    out = []
    for p in _webapp_files(".ts") + _webapp_files(".tsx"):
        out.append((p, p.read_text(encoding="utf-8", errors="replace")))
    return out


@pytest.fixture(scope="module", autouse=True)
def _require_webapp():
    if not _WEBAPP_SRC.is_dir():
        pytest.skip("webapp/src 不存在，跳过")


# ============================================================
# 约束 1：不得编造数据
# ============================================================


class TestNoFabricatedContent:
    """不得出现凭空捏造的业务数据"""

    @pytest.mark.parametrize(
        "marker",
        [
            "京A",              # 假车牌（frontend/ 曾用过）
            "INST_a3f2c1",      # 假实例 ID
            "每日工作简报",       # 假导出文件名
            "异常停留车辆",       # 假告警文案
        ],
    )
    def test_no_fake_records(self, marker):
        offenders = [str(p.relative_to(_ROOT)) for p, t in _all_sources() if marker in t]
        assert not offenders, f"webapp 出现编造内容 {marker!r}: {offenders}"

    def test_no_hardcoded_business_numbers_in_api_layer(self):
        """API 封装层不得出现字面量业务数字（说明白了就是写死的假数据）

        只检查 `src/api/` —— 展示层出现数字是正常的（列宽、百分比等）。
        """
        api_dir = _WEBAPP_SRC / "api"
        if not api_dir.is_dir():
            pytest.skip("webapp/src/api 不存在")
        offenders = []
        for p in api_dir.rglob("*.ts"):
            text = p.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"//[^\n]*", "", text)
            text = re.sub(r"/\*(?:.|\n)*?\*/", "", text)
            # 16 进制颜色、HTTP 状态码、超时毫秒数是正常的，不算业务数字
            stripped = re.sub(r"0x[0-9a-fA-F]+|\b\d{3,4}\b|\b\d+_000\b", "", text)
            for m in re.finditer(r"(?<![\w.\"'])(\d{2,})(?![\w\"'])", stripped):
                val = int(m.group(1))
                if val in (10, 20, 100, 1000):
                    continue
                offenders.append(f"{p.relative_to(_ROOT)}: {val}")
        assert not offenders, f"API 层出现可疑硬编码数字: {offenders[:8]}"


# ============================================================
# 约束 2：不得自算分数 / 不得本地加载模型
# ============================================================


class TestNoLocalModelOrScoring:
    """检索与打分只能来自后端"""

    @pytest.mark.parametrize(
        "marker",
        [
            "open_clip",
            "BlipForImageTextRetrieval",
            "BlipProcessor",
            "Salesforce/blip",
            "CN-CLIP-ViT-L-14",
            "onnxruntime",
            "@xenova/transformers",     # 前端侧 transformers.js 也不该出现
        ],
    )
    def test_no_model_loading_markers(self, marker):
        """前端不得自己加载模型（注释里提到名字不算，只看非注释行）"""
        offenders = []
        for path, text in _all_sources():
            for lineno, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith(("//", "*", "/*")):
                    continue
                if marker in line:
                    offenders.append(f"{path.relative_to(_ROOT)}:{lineno}")
        assert not offenders, f"webapp 出现本地模型加载标识 {marker!r}: {offenders}"

    def test_no_rerank_or_score_computation(self):
        """不得在前端重新实现排序/打分（应与后端返回的 final_score 对齐即可）"""
        forbidden = [
            "function rerank",
            "const rerank",
            "computeScore",
            "computeRank",
            "cosineSimilarity",
        ]
        offenders = []
        for path, text in _all_sources():
            for name in forbidden:
                if name in text:
                    offenders.append(f"{path.relative_to(_ROOT)}: {name}")
        assert not offenders, f"webapp 出现前端侧打分/重排实现: {offenders}"

    def test_endpoints_module_is_thin_and_has_no_fallback(self):
        """api/endpoints.ts 必须明示"不做兜底伪造"（防止后人加回本地数据源）"""
        ep = _WEBAPP_SRC / "api" / "endpoints.ts"
        if not ep.exists():
            pytest.skip("webapp/src/api/endpoints.ts 不存在")
        text = ep.read_text(encoding="utf-8")
        assert "兜底" in text or "伪造" in text, (
            "endpoints.ts 应保留『不做数据补全/兜底伪造』的说明 —— "
            "这条约定一旦没人记得，编造数据就会重新长回来"
        )
