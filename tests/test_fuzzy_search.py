"""
tests/test_fuzzy_search.py - 用户模糊搜索场景匹配准确率测试

测试系统在面对真实用户模糊、口语化、不完整查询时的检索匹配能力。
5类测试场景：
  A. 口语化模糊描述
  B. 不完整属性查询
  C. 近义词/口语表达
  D. 模糊颜色描述
  E. 复合模糊查询
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import pytest

# 项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 抑制项目日志输出（避免日志轮转错误干扰测试输出）
import logging as _logging
_logging.disable(_logging.CRITICAL)

from src.retrieval.query_parser import QueryParser, QueryResult
from src.retrieval.attribute_filter import AttributeFilter
from src.common.data_models import BoundingBox, TargetInstance


# ============================================================
# 数据加载与转换
# ============================================================

DATA_PATH = os.path.join(PROJECT_ROOT, "output", "cityflow_results.json")


def _load_detections() -> List[Dict[str, Any]]:
    """加载 cityflow_results.json 中的检测记录"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["detections"]


def _det_to_instance(det: Dict[str, Any]) -> TargetInstance:
    """将 JSON 检测记录转换为 TargetInstance"""
    bbox = det.get("bbox", [0, 0, 0, 0])
    ts_str = det.get("timestamp", "2020-01-01 00:00:00.000")
    try:
        ts = datetime.fromisoformat(ts_str)
    except Exception:
        ts = datetime(2020, 1, 1)
    attrs = det.get("attributes", {})
    return TargetInstance(
        instance_id=det.get("target_id", ""),
        camera_id=det.get("camera_id", ""),
        timestamp=ts,
        frame_id=det.get("frame_id", 0),
        target_type=det.get("target_type", "vehicle"),
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], confidence=det.get("confidence", 0.5)),
        attributes=attrs,
        plate_number=None,
        plate_confidence=0.0,
        quality_score=det.get("confidence", 0.5),
        reid_vector=None,
        clip_vector=None,
        keyframe_path=det.get("keyframe_path"),
        scene_id=det.get("scene_id"),
    )


# ============================================================
# Ground Truth 构建
# ============================================================

# 颜色语义分组 —— 用于模糊颜色匹配的 ground truth
COLOR_GROUPS = {
    "深色": {"黑色", "灰色", "棕色", "蓝色"},
    "浅色": {"白色", "黄色", "粉色", "金色"},
    "暗色": {"黑色", "灰色", "棕色"},
    "亮色": {"白色", "红色", "黄色", "橙色", "粉色", "金色"},
    "金属色": {"银色", "灰色", "金色"},
    "暖色": {"红色", "橙色", "黄色", "棕色", "金色", "粉色"},
    "冷色": {"蓝色", "绿色", "灰色", "黑色", "紫色"},
}

# 车型语义分组 —— 用于模糊车型匹配的 ground truth
TYPE_GROUPS = {
    "大车": {"卡车", "公交车", "房车"},
    "小车": {"轿车", "两厢车", "跑车", "SUV", "皮卡"},
    "小型客车": {"面包车", "MPV", "房车"},
}

# 近义词分组
SYNONYM_GROUPS = {
    "面包车": {"面包车"},
    "小型客车": {"面包车", "MPV", "房车"},
    "微面": {"面包车"},
    "大货": {"卡车"},
    "重型货车": {"卡车"},
    "大卡车": {"卡车"},
    "轿车": {"轿车", "两厢车"},
    "小车": {"轿车", "两厢车", "跑车"},
    "私家车": {"轿车", "两厢车", "跑车", "SUV"},
    "吉普": {"SUV"},
    "SUV": {"SUV"},
    "越野车": {"SUV"},
}


def _build_ground_truth(
    instances: List[TargetInstance],
    color: str | None = None,
    vehicle_type: str | None = None,
    color_group: str | None = None,
    type_group: str | None = None,
    synonym_key: str | None = None,
) -> Set[str]:
    """
    从实例列表中构建 ground truth 集合。

    返回满足条件的 instance_id 集合。
    """
    result = set()
    for inst in instances:
        attrs = inst.attributes
        inst_color = attrs.get("color", "unknown")
        inst_type = attrs.get("vehicle_type", "unknown")

        # 跳过属性未知的记录
        if inst_color == "unknown" and inst_type == "unknown":
            continue

        match = True

        if color is not None:
            if inst_color == "unknown" or inst_color != color:
                match = False

        if vehicle_type is not None:
            if inst_type == "unknown" or inst_type != vehicle_type:
                match = False

        if color_group is not None:
            allowed = COLOR_GROUPS.get(color_group, set())
            if inst_color == "unknown" or inst_color not in allowed:
                match = False

        if type_group is not None:
            allowed = TYPE_GROUPS.get(type_group, set())
            if inst_type == "unknown" or inst_type not in allowed:
                match = False

        if synonym_key is not None:
            allowed = SYNONYM_GROUPS.get(synonym_key, set())
            if inst_type == "unknown" or inst_type not in allowed:
                match = False

        if match:
            result.add(inst.instance_id)

    return result


def _build_ground_truth_multi(
    instances: List[TargetInstance],
    color: str | None = None,
    colors: List[str] | None = None,
    color_group: str | None = None,
    vehicle_type: str | None = None,
    vehicle_types: List[str] | None = None,
    type_group: str | None = None,
    synonym_key: str | None = None,
) -> Set[str]:
    """多值 ground truth：颜色或车型可以是多个值之一"""
    # 统一为列表形式
    if color and not colors:
        colors = [color]
    if vehicle_type and not vehicle_types:
        vehicle_types = [vehicle_type]

    result = set()
    for inst in instances:
        attrs = inst.attributes
        inst_color = attrs.get("color", "unknown")
        inst_type = attrs.get("vehicle_type", "unknown")

        if inst_color == "unknown" and inst_type == "unknown":
            continue

        match = True

        if colors is not None:
            if inst_color == "unknown" or inst_color not in colors:
                match = False

        if color_group is not None:
            allowed = COLOR_GROUPS.get(color_group, set())
            if inst_color == "unknown" or inst_color not in allowed:
                match = False

        if vehicle_types is not None:
            if inst_type == "unknown" or inst_type not in vehicle_types:
                match = False

        if type_group is not None:
            allowed = TYPE_GROUPS.get(type_group, set())
            if inst_type == "unknown" or inst_type not in allowed:
                match = False

        if synonym_key is not None:
            allowed = SYNONYM_GROUPS.get(synonym_key, set())
            if inst_type == "unknown" or inst_type not in allowed:
                match = False

        if match:
            result.add(inst.instance_id)

    return result


# ============================================================
# 评估指标
# ============================================================

def _precision_at_k(retrieved_ids: List[str], gt_ids: Set[str], k: int) -> float:
    """Precision@K"""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in gt_ids)
    return hits / min(k, len(top_k))


def _recall_at_k(retrieved_ids: List[str], gt_ids: Set[str], k: int) -> float:
    """Recall@K"""
    if not gt_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in gt_ids)
    return hits / len(gt_ids)


def _f1_at_k(retrieved_ids: List[str], gt_ids: Set[str], k: int) -> float:
    """F1@K"""
    p = _precision_at_k(retrieved_ids, gt_ids, k)
    r = _recall_at_k(retrieved_ids, gt_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _mrr(retrieved_ids: List[str], gt_ids: Set[str]) -> float:
    """MRR - Mean Reciprocal Rank"""
    for i, rid in enumerate(retrieved_ids):
        if rid in gt_ids:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# 测试用例定义
# ============================================================

# 每个测试用例: (查询文本, ground_truth构建参数, 模糊级别)
# 模糊级别: "exact" / "light" / "medium" / "heavy"

TEST_CASES_A = [
    # A. 口语化模糊描述
    ("那辆红色的车",       {"color": "红色"},                          "light"),
    ("一辆大车",           {"type_group": "大车"},                      "medium"),
    ("深色的面包车",       {"color_group": "深色", "vehicle_types": ["面包车"]}, "medium"),
    ("白色的小车",         {"color": "白色", "type_group": "小车"},     "light"),
    ("黑色的大家伙",       {"color": "黑色", "type_group": "大车"},     "medium"),
    ("蓝色的货车",         {"color": "蓝色", "vehicle_types": ["卡车"]}, "light"),
    ("有辆灰色的车",       {"color": "灰色"},                           "light"),
    ("红色的那种小车",     {"color": "红色", "type_group": "小车"},     "light"),
    ("一辆银色面包车",     {"color": "银色", "vehicle_types": ["面包车"]}, "light"),
    ("黄色跑车",           {"color": "黄色", "vehicle_types": ["跑车"]}, "light"),
]

TEST_CASES_B = [
    # B. 不完整属性查询
    ("红色的",             {"color": "红色"},                           "light"),
    ("面包车",             {"vehicle_types": ["面包车"]},               "light"),
    ("大车",               {"type_group": "大车"},                      "medium"),
    ("深色的车",           {"color_group": "深色"},                     "medium"),
    ("SUV",                {"vehicle_types": ["SUV"]},                  "exact"),
    ("轿车",               {"vehicle_types": ["轿车", "两厢车"]},       "light"),
    ("白色",               {"color": "白色"},                           "light"),
    ("卡车",               {"vehicle_types": ["卡车"]},                 "exact"),
    ("蓝色的",             {"color": "蓝色"},                           "light"),
    ("公交车",             {"vehicle_types": ["公交车"]},               "exact"),
]

TEST_CASES_C = [
    # C. 近义词/口语表达
    ("面包车",             {"synonym_key": "面包车"},                   "light"),
    ("小型客车",           {"synonym_key": "小型客车"},                 "medium"),
    ("微面",               {"synonym_key": "微面"},                     "heavy"),
    ("大货",               {"synonym_key": "大货"},                     "heavy"),
    ("重型货车",           {"synonym_key": "重型货车"},                 "medium"),
    ("大卡车",             {"synonym_key": "大卡车"},                   "light"),
    ("轿车",               {"synonym_key": "轿车"},                     "light"),
    ("小车",               {"synonym_key": "小车"},                     "medium"),
    ("吉普",               {"synonym_key": "吉普"},                     "heavy"),
    ("SUV",                {"synonym_key": "SUV"},                      "exact"),
    ("越野车",             {"synonym_key": "越野车"},                   "medium"),
    ("私家车",             {"synonym_key": "私家车"},                   "medium"),
]

TEST_CASES_D = [
    # D. 模糊颜色描述
    ("深色车",             {"color_group": "深色"},                     "heavy"),
    ("浅色SUV",            {"color_group": "浅色", "vehicle_types": ["SUV"]}, "heavy"),
    ("暗色的",             {"color_group": "暗色"},                     "heavy"),
    ("亮色的车",           {"color_group": "亮色"},                     "heavy"),
    ("金属色的",           {"color_group": "金属色"},                   "heavy"),
    ("暖色的车",           {"color_group": "暖色"},                     "heavy"),
    ("冷色SUV",            {"color_group": "冷色", "vehicle_types": ["SUV"]}, "heavy"),
    ("红色的车",           {"color": "红色"},                           "exact"),
    ("白色的面包车",       {"color": "白色", "vehicle_types": ["面包车"]}, "exact"),
    ("黑色轿车",           {"color": "黑色", "vehicle_types": ["轿车", "两厢车"]}, "exact"),
]

TEST_CASES_E = [
    # E. 复合模糊查询
    ("路口那辆深色大车",   {"color_group": "深色", "type_group": "大车"}, "heavy"),
    ("红色小车",           {"color": "红色", "type_group": "小车"},      "light"),
    ("c001那边出现的白色面包车", {"color": "白色", "vehicle_types": ["面包车"]}, "light"),
    ("深色面包车",         {"color_group": "深色", "vehicle_types": ["面包车"]}, "medium"),
    ("蓝色大车",           {"color": "蓝色", "type_group": "大车"},      "medium"),
    ("亮色小车",           {"color_group": "亮色", "type_group": "小车"}, "heavy"),
    ("灰色SUV",            {"color": "灰色", "vehicle_types": ["SUV"]},  "exact"),
    ("暖色大车",           {"color_group": "暖色", "type_group": "大车"}, "heavy"),
    ("白色货车",           {"color": "白色", "vehicle_types": ["卡车"]},  "light"),
    ("暗色的小车",         {"color_group": "暗色", "type_group": "小车"}, "heavy"),
]

ALL_CATEGORIES = [
    ("A.口语化描述", TEST_CASES_A),
    ("B.不完整属性", TEST_CASES_B),
    ("C.近义词表达", TEST_CASES_C),
    ("D.模糊颜色",   TEST_CASES_D),
    ("E.复合模糊",   TEST_CASES_E),
]


# ============================================================
# Fixture: 加载数据 & 初始化组件
# ============================================================

@pytest.fixture(scope="module")
def test_env():
    """加载数据、初始化解析器和过滤器"""
    print("\n" + "=" * 70)
    print("  模糊搜索匹配准确率测试")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载 cityflow_results.json ...")
    dets = _load_detections()
    print(f"    总检测记录数: {len(dets)}")

    instances = [_det_to_instance(d) for d in dets]
    known = [i for i in instances if i.attributes.get("color", "unknown") != "unknown"]
    print(f"    有已知属性的记录数: {len(known)}")

    # 初始化组件
    parser = QueryParser()
    af = AttributeFilter()

    env = {
        "instances": instances,
        "known_instances": known,
        "parser": parser,
        "filter": af,
    }
    yield env

    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70)


# ============================================================
# 核心测试逻辑
# ============================================================

def _run_single_query(
    query_text: str,
    gt_params: Dict[str, Any],
    instances: List[TargetInstance],
    parser: QueryParser,
    af: AttributeFilter,
) -> Dict[str, Any]:
    """
    执行单个查询并计算指标。

    返回包含解析结果、过滤结果、指标的字典。
    """
    # 1. 解析查询
    qr: QueryResult = parser.parse(query_text)

    # 2. 构建 ground truth (使用 multi 版本以支持列表参数)
    gt_ids = _build_ground_truth_multi(instances, **gt_params)

    # 3. 属性过滤
    filtered = af.apply(instances, qr)
    retrieved_ids = [inst.instance_id for inst in filtered]

    # 4. 计算指标
    p5 = _precision_at_k(retrieved_ids, gt_ids, 5)
    p10 = _precision_at_k(retrieved_ids, gt_ids, 10)
    p20 = _precision_at_k(retrieved_ids, gt_ids, 20)
    r5 = _recall_at_k(retrieved_ids, gt_ids, 5)
    r10 = _recall_at_k(retrieved_ids, gt_ids, 10)
    r20 = _recall_at_k(retrieved_ids, gt_ids, 20)
    f1_10 = _f1_at_k(retrieved_ids, gt_ids, 10)
    mrr_val = _mrr(retrieved_ids, gt_ids)

    # 解析是否成功提取了有意义的属性
    parse_ok = (qr.color is not None or qr.vehicle_type is not None
                or qr.target_type is not None)

    return {
        "query_text": query_text,
        "parsed_color": qr.color,
        "parsed_vehicle_type": qr.vehicle_type,
        "parsed_target_type": qr.target_type,
        "parse_ok": parse_ok,
        "gt_count": len(gt_ids),
        "retrieved_count": len(retrieved_ids),
        "p5": p5, "p10": p10, "p20": p20,
        "r5": r5, "r10": r10, "r20": r20,
        "f1_10": f1_10,
        "mrr": mrr_val,
    }


def _run_category(
    cat_name: str,
    cases: List[Tuple],
    instances: List[TargetInstance],
    parser: QueryParser,
    af: AttributeFilter,
) -> List[Dict[str, Any]]:
    """运行一个类别下的所有查询"""
    results = []
    for query_text, gt_params, fuzz_level in cases:
        r = _run_single_query(query_text, gt_params, instances, parser, af)
        r["category"] = cat_name
        r["fuzz_level"] = fuzz_level
        results.append(r)
    return results


# ============================================================
# Pytest 测试入口
# ============================================================

class TestFuzzySearch:
    """模糊搜索匹配准确率测试集"""

    def test_category_A_colloquial(self, test_env, capsys):
        """A. 口语化模糊描述"""
        results = _run_category(
            "A.口语化描述", TEST_CASES_A,
            test_env["instances"], test_env["parser"], test_env["filter"],
        )
        self._print_category_results("A.口语化描述", results)
        # 断言：至少一半查询能解析出属性
        parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
        assert parse_rate >= 0.5, f"A类解析成功率过低: {parse_rate:.1%}"

    def test_category_B_incomplete(self, test_env, capsys):
        """B. 不完整属性查询"""
        results = _run_category(
            "B.不完整属性", TEST_CASES_B,
            test_env["instances"], test_env["parser"], test_env["filter"],
        )
        self._print_category_results("B.不完整属性", results)
        parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
        assert parse_rate >= 0.5, f"B类解析成功率过低: {parse_rate:.1%}"

    def test_category_C_synonyms(self, test_env, capsys):
        """C. 近义词/口语表达"""
        results = _run_category(
            "C.近义词表达", TEST_CASES_C,
            test_env["instances"], test_env["parser"], test_env["filter"],
        )
        self._print_category_results("C.近义词表达", results)
        parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
        assert parse_rate >= 0.3, f"C类解析成功率过低: {parse_rate:.1%}"

    def test_category_D_fuzzy_color(self, test_env, capsys):
        """D. 模糊颜色描述"""
        results = _run_category(
            "D.模糊颜色", TEST_CASES_D,
            test_env["instances"], test_env["parser"], test_env["filter"],
        )
        self._print_category_results("D.模糊颜色", results)
        # 模糊颜色查询可能解析率较低，但仍需有一定成功率
        parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
        # 不做硬断言，只报告

    def test_category_E_composite(self, test_env, capsys):
        """E. 复合模糊查询"""
        results = _run_category(
            "E.复合模糊", TEST_CASES_E,
            test_env["instances"], test_env["parser"], test_env["filter"],
        )
        self._print_category_results("E.复合模糊", results)
        parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
        assert parse_rate >= 0.3, f"E类解析成功率过低: {parse_rate:.1%}"

    def test_full_report(self, test_env, capsys):
        """生成完整的模糊搜索准确率报告"""
        all_results = []
        for cat_name, cases in ALL_CATEGORIES:
            results = _run_category(
                cat_name, cases,
                test_env["instances"], test_env["parser"], test_env["filter"],
            )
            all_results.extend(results)

        self._print_full_report(all_results)

    def test_parse_detail_analysis(self, test_env, capsys):
        """查询解析详情分析 —— 暴露系统薄弱环节"""
        print("\n" + "=" * 70)
        print("  查询解析详情分析")
        print("=" * 70)

        all_cases = []
        for cat_name, cases in ALL_CATEGORIES:
            for query_text, gt_params, fuzz_level in cases:
                all_cases.append((cat_name, query_text, gt_params, fuzz_level))

        parse_failures = []
        weak_matches = []

        for cat_name, query_text, gt_params, fuzz_level in all_cases:
            qr = test_env["parser"].parse(query_text)
            gt_ids = _build_ground_truth_multi(test_env["instances"], **gt_params)

            # 记录解析失败
            if not (qr.color or qr.vehicle_type or qr.target_type):
                parse_failures.append((cat_name, query_text, fuzz_level))

            # 记录弱匹配（F1 < 0.1）
            filtered = test_env["filter"].apply(test_env["instances"], qr)
            retrieved_ids = [inst.instance_id for inst in filtered]
            f1 = _f1_at_k(retrieved_ids, gt_ids, 10)
            if 0 < f1 < 0.1 and len(gt_ids) > 0:
                weak_matches.append((cat_name, query_text, fuzz_level,
                                     qr.color, qr.vehicle_type, len(gt_ids), len(retrieved_ids), f1))

        print(f"\n--- 解析失败的查询 ({len(parse_failures)}) ---")
        for cat, q, fl in parse_failures:
            print(f"  [{cat}] \"{q}\" (模糊级别: {fl})")

        print(f"\n--- 弱匹配查询 (F1@10 < 0.1, 共 {len(weak_matches)}) ---")
        for cat, q, fl, pc, pvt, gtc, rc, f1 in weak_matches:
            print(f"  [{cat}] \"{q}\" → color={pc}, type={pvt}, "
                  f"GT={gtc}, retrieved={rc}, F1@10={f1:.3f} (模糊: {fl})")

        if not parse_failures and not weak_matches:
            print("  无解析失败或弱匹配情况")

    def test_precision_recall_detail(self, test_env, capsys):
        """精确查询 vs 模糊查询对比分析"""
        all_results = []
        for cat_name, cases in ALL_CATEGORIES:
            results = _run_category(
                cat_name, cases,
                test_env["instances"], test_env["parser"], test_env["filter"],
            )
            all_results.extend(results)

        # 按模糊级别分组
        level_results = {"exact": [], "light": [], "medium": [], "heavy": []}
        for r in all_results:
            fl = r.get("fuzz_level", "medium")
            if fl in level_results:
                level_results[fl].append(r)

        print("\n" + "=" * 70)
        print("  精确查询 vs 模糊查询对比")
        print("=" * 70)
        print(f"\n{'查询精度':<12} {'查询数':>6} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'MRR':>8} {'衰减率':>8}")
        print("-" * 60)

        baseline_f1 = None
        for level_name, label in [("exact", "精确查询"), ("light", "轻度模糊"),
                                   ("medium", "中度模糊"), ("heavy", "高度模糊")]:
            rs = level_results[level_name]
            if not rs:
                continue
            avg_p10 = sum(r["p10"] for r in rs) / len(rs)
            avg_r10 = sum(r["r10"] for r in rs) / len(rs)
            avg_f1 = sum(r["f1_10"] for r in rs) / len(rs)
            avg_mrr = sum(r["mrr"] for r in rs) / len(rs)

            if baseline_f1 is None:
                baseline_f1 = avg_f1
                decay = "baseline"
            else:
                if baseline_f1 > 0:
                    decay_val = (baseline_f1 - avg_f1) / baseline_f1 * 100
                    decay = f"-{decay_val:.1f}%"
                else:
                    decay = "N/A"

            print(f"{label:<12} {len(rs):>6} {avg_p10:>8.3f} {avg_r10:>8.3f} "
                  f"{avg_f1:>8.3f} {avg_mrr:>8.3f} {decay:>8}")

    # --------------------------------------------------------
    # 报告输出辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _print_category_results(cat_name: str, results: List[Dict[str, Any]]):
        """打印单个类别的结果"""
        print(f"\n--- {cat_name} ---")
        print(f"{'查询':<30} {'解析色':<6} {'解析型':<6} {'GT数':>5} {'召回数':>6} "
              f"{'P@10':>6} {'R@10':>6} {'F1@10':>6} {'MRR':>6}")
        print("-" * 110)

        for r in results:
            color_s = r["parsed_color"] or "-"
            type_s = r["parsed_vehicle_type"] or "-"
            # 截断过长的查询
            q_display = r["query_text"][:28]
            print(f"{q_display:<30} {color_s:<6} {type_s:<6} {r['gt_count']:>5} "
                  f"{r['retrieved_count']:>6} {r['p10']:>6.3f} {r['r10']:>6.3f} "
                  f"{r['f1_10']:>6.3f} {r['mrr']:>6.3f}")

        # 汇总
        n = len(results)
        parse_rate = sum(1 for r in results if r["parse_ok"]) / n * 100
        avg_p10 = sum(r["p10"] for r in results) / n
        avg_r10 = sum(r["r10"] for r in results) / n
        avg_f1 = sum(r["f1_10"] for r in results) / n
        avg_mrr = sum(r["mrr"] for r in results) / n
        print(f"{'[汇总]':<30} {'':<6} {'':<6} {'':<5} {'':<6} "
              f"{avg_p10:>6.3f} {avg_r10:>6.3f} {avg_f1:>6.3f} {avg_mrr:>6.3f}")
        print(f"  解析成功率: {parse_rate:.0f}%")

    @staticmethod
    def _print_full_report(all_results: List[Dict[str, Any]]):
        """打印完整报告"""
        print("\n" + "=" * 70)
        print("  模糊搜索准确率报告")
        print("=" * 70)

        # 按类别汇总
        categories = {}
        for r in all_results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        print(f"\n{'查询类型':<16} {'查询数':>6} {'解析率':>7} {'P@10':>7} {'R@10':>7} {'F1@10':>7} {'MRR':>7}")
        print("-" * 65)

        total_n = 0
        total_parse_ok = 0
        total_p10 = 0.0
        total_r10 = 0.0
        total_f1 = 0.0
        total_mrr = 0.0

        for cat_name, _ in ALL_CATEGORIES:
            rs = categories.get(cat_name, [])
            if not rs:
                continue
            n = len(rs)
            parse_rate = sum(1 for r in rs if r["parse_ok"]) / n * 100
            avg_p10 = sum(r["p10"] for r in rs) / n
            avg_r10 = sum(r["r10"] for r in rs) / n
            avg_f1 = sum(r["f1_10"] for r in rs) / n
            avg_mrr = sum(r["mrr"] for r in rs) / n

            print(f"{cat_name:<16} {n:>6} {parse_rate:>6.0f}% {avg_p10:>7.3f} "
                  f"{avg_r10:>7.3f} {avg_f1:>7.3f} {avg_mrr:>7.3f}")

            total_n += n
            total_parse_ok += sum(1 for r in rs if r["parse_ok"])
            total_p10 += sum(r["p10"] for r in rs)
            total_r10 += sum(r["r10"] for r in rs)
            total_f1 += sum(r["f1_10"] for r in rs)
            total_mrr += sum(r["mrr"] for r in rs)

        print("-" * 65)
        if total_n > 0:
            print(f"{'总体':<16} {total_n:>6} {total_parse_ok/total_n*100:>6.0f}% "
                  f"{total_p10/total_n:>7.3f} {total_r10/total_n:>7.3f} "
                  f"{total_f1/total_n:>7.3f} {total_mrr/total_n:>7.3f}")

        # 逐查询问题诊断
        print(f"\n--- 问题查询诊断 (F1@10 = 0 且 GT > 0) ---")
        problem_count = 0
        for r in all_results:
            if r["f1_10"] == 0 and r["gt_count"] > 0:
                problem_count += 1
                if problem_count <= 20:
                    print(f"  [{r['category']}] \"{r['query_text']}\" → "
                          f"color={r['parsed_color']}, type={r['parsed_vehicle_type']}, "
                          f"GT={r['gt_count']}, retrieved={r['retrieved_count']}")
        if problem_count > 20:
            print(f"  ... 共 {problem_count} 个问题查询")
        elif problem_count == 0:
            print("  无问题查询")

        # 漏召回 & 误召回分析
        print(f"\n--- 典型漏召回/误召回分析 ---")
        for r in all_results:
            if r["gt_count"] > 0 and r["retrieved_count"] > 0:
                # 只看 F1 较低的有代表性的案例
                if 0 < r["f1_10"] < 0.2:
                    print(f"  [{r['category']}] \"{r['query_text']}\"")
                    print(f"    解析: color={r['parsed_color']}, type={r['parsed_vehicle_type']}")
                    print(f"    GT={r['gt_count']}, retrieved={r['retrieved_count']}, "
                          f"P@10={r['p10']:.3f}, R@10={r['r10']:.3f}, F1@10={r['f1_10']:.3f}")
