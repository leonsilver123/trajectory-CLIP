"""
tests.test_query_parser_divergence - 两套查询解析器的差异（PLAN3-B1）

## 背景

项目里并存两套查询解析：

- **在线路径**：`api/routes/search.py` 的 `_extract_query_features()`（路由内联实现）
- **设计路径**：`src/retrieval/query_parser.py` 的 `QueryParser`（1884 行的完整包，
  有 34 项测试覆盖，但**不被在线代码 import**）

B1 原本的计划是"让路由改调 `src/retrieval`，删掉内联实现"。

## 但这个计划被实测否决了

用评测里那 32 条**真实查询**（`output/_retrieval_hit_clip_fixed.json`）做对比：

| 维度 | 结果 |
|---|---|
| 颜色判定 | **32/32 完全一致** |
| 车型判定 | **10/32 不一致** |

差异分三类，而且**两边互不包含**：

| 查询 | 路由内联 | QueryParser | 谁更好 |
|---|---|---|---|
| `灰色MPV` | `MPV` ✅ | `None` ❌ | 路由 |
| `绿色公交车` | `None` ❌ | `公交车` ✅ | QueryParser |
| `红色两厢车` | `两厢车` | `轿车` | 语义不同，需人判断 |

既然**没有任何一方是另一方的超集**，直接切换必然丢失能力
（切到 QueryParser 会丢掉 MPV 这一类；MPV 在数据里是真实存在的车型）。

因此 B1 的结论是：**暂不合并**，并把这份差异**钉成测试** ——
谁改动任一解析器，这里都会红，迫使做一次有意识的决定，而不是悄悄漂移。

## 一个好消息

`皮卡 / 卡车` 那个历史 bug（"皮卡"被"卡车"的关键词 '卡' 抢先匹配）
**在两套实现里都已修好**（两侧都返回 `皮卡`）。下面有专门的回归用例守住它。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from api.routes.search import _extract_query_features
from src.retrieval.query_parser import QueryParser

_ROOT = Path(__file__).resolve().parent.parent
_EVAL_JSON = _ROOT / "output" / "_retrieval_hit_clip_fixed.json"


@pytest.fixture(scope="module")
def parser() -> QueryParser:
    # QueryParser 逐条打 INFO 日志，测试里关掉以免淹没有效输出
    logging.disable(logging.INFO)
    yield QueryParser()
    logging.disable(logging.NOTSET)


def _route(text: str) -> tuple:
    f = _extract_query_features(text)
    return f.get("color"), f.get("vehicle_type")


def _rich(parser: QueryParser, text: str) -> tuple:
    r = parser.parse(text)
    return r.color, r.vehicle_type


# ============================================================
# 已修 bug 的回归守护（两套实现都必须对）
# ============================================================


class TestPikaRegressionBothImplementations:
    """「皮卡」不得被「卡车」的 '卡' 抢先匹配 —— 两套实现都要守住"""

    @pytest.mark.parametrize("text,expected", [
        ("皮卡", "皮卡"),
        ("白色皮卡", "皮卡"),
        ("卡车", "卡车"),
        ("白色卡车", "卡车"),
    ])
    def test_route_gets_vehicle_type_right(self, text, expected):
        _, vtype = _route(text)
        assert vtype == expected, f"路由内联解析 {text!r} 得到 {vtype!r}"

    @pytest.mark.parametrize("text,expected", [
        ("皮卡", "皮卡"),
        ("白色皮卡", "皮卡"),
        ("卡车", "卡车"),
        ("白色卡车", "卡车"),
    ])
    def test_query_parser_gets_vehicle_type_right(self, parser, text, expected):
        _, vtype = _rich(parser, text)
        assert vtype == expected, f"QueryParser 解析 {text!r} 得到 {vtype!r}"


# ============================================================
# 已知差异：钉住，防止悄悄漂移
# ============================================================


class TestKnownDivergence:
    """两套实现的已知差异 —— 改动任一侧都会让这里红

    实测（32 条真实查询）：分歧恰好 3 类、共 10 条，全部在 vehicle_type，颜色 0 分歧。
    """

    def test_mpv_only_route_detects(self, parser):
        """MPV：路由内联能识别，QueryParser 不能（3/32 条）

        因此**不能**把路由切到 QueryParser —— 会丢掉 MPV 这一类。
        """
        assert _route("灰色MPV")[1] == "MPV", "路由内联应识别 MPV"
        assert _rich(parser, "灰色MPV")[1] is None, (
            "QueryParser 现在能识别 MPV 了？那说明差异已消失，"
            "可以重新评估 B1 的合并方案（见本文件开头）"
        )

    def test_bus_only_query_parser_detects(self, parser):
        """公交车：QueryParser 能识别，路由内联不能（2/32 条）"""
        assert _route("绿色公交车")[1] is None, (
            "路由内联现在能识别公交车了？差异已变化，请重新评估 B1"
        )
        assert _rich(parser, "绿色公交车")[1] == "公交车"

    def test_hatchback_semantics_differ(self, parser):
        """两厢车：路由→两厢车，QueryParser→轿车（5/32 条，语义不同需人判断）

        注意两者都**不是** None，即都做了过滤，只是过滤到不同的桶里 ——
        这类分歧最危险：指标上看不出异常，但召回集合已经不同。
        """
        assert _route("红色两厢车")[1] == "两厢车"
        assert _rich(parser, "红色两厢车")[1] == "轿车"

    def test_divergence_count_is_exactly_three_patterns(self, parser):
        """分歧恰好只有这 3 类 —— 出现第 4 类说明有人在单侧改了逻辑"""
        if not _EVAL_JSON.exists():
            pytest.skip(f"缺少评测记录 {_EVAL_JSON}")
        data = json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
        queries = list(dict.fromkeys(
            q["query_text"] for q in data.get("per_query", []) if q.get("query_text")
        ))

        patterns = set()
        for text in queries:
            rc, rt = _route(text)
            qc, qt = _rich(parser, text)
            if (rc, rt) == (qc, qt):
                continue
            patterns.add((rt, qt))

        known = {("MPV", None), (None, "公交车"), ("两厢车", "轿车")}
        unknown = patterns - known
        assert not unknown, (
            f"出现了新的分歧模式 {unknown}（已知三类: {known}）。"
            f"请确认是修好了一类，还是引入了新的一类，并同步更新本文件与 PLAN3-B1"
        )


class TestColorAgreement:
    """颜色判定两套必须一致（实测 32/32 一致，这里守住它）"""

    def test_colors_agree_on_real_queries(self, parser):
        if not _EVAL_JSON.exists():
            pytest.skip(f"缺少评测记录 {_EVAL_JSON}")
        data = json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
        queries = list(dict.fromkeys(
            q["query_text"] for q in data.get("per_query", []) if q.get("query_text")
        ))
        assert queries, "评测记录里没有查询文本"

        mismatches = [
            (t, _route(t)[0], _rich(parser, t)[0])
            for t in queries
            if _route(t)[0] != _rich(parser, t)[0]
        ]
        assert not mismatches, (
            f"颜色判定出现分歧（实测应为 0 条）: {mismatches[:5]}"
        )

    def test_divergence_is_bounded(self, parser):
        """车型分歧应保持在已知范围（10/32）—— 突然变多说明有人改了单侧逻辑"""
        if not _EVAL_JSON.exists():
            pytest.skip(f"缺少评测记录 {_EVAL_JSON}")
        data = json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
        queries = list(dict.fromkeys(
            q["query_text"] for q in data.get("per_query", []) if q.get("query_text")
        ))

        diffs = [t for t in queries if _route(t)[1] != _rich(parser, t)[1]]
        assert len(diffs) <= 12, (
            f"车型判定分歧从 10 条涨到 {len(diffs)} 条，两套实现正在加速漂移。"
            f"新增分歧: {diffs[:8]}"
        )
