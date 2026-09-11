"""
tests/test_retrieval_accuracy.py - 检索匹配准确率测试

基于 cityflow_results.json 中的真实检测数据，评估系统检索特定种类/行为车辆的匹配能力。

测试内容:
  A. 属性过滤准确率 (AttributeFilter Precision / Recall)
  B. 文本检索匹配准确率 (Precision@K, Recall@K, F1@K, MRR, mAP)
  C. 重排序效果评估 (重排序前后 P@5/MRR 对比, NDCG@K)

数据来源:
  - output/cityflow_results.json (1829条检测, 1150条轨迹)
  - 属性字段: color, vehicle_type 等来自 CityFlow 数据集真实标注

注意:
  - CLIP 图像向量可用(768维), 但模型权重不可用，无法编码新文本
  - 文本查询使用"伪向量"方案: 以匹配项平均图像向量模拟文本编码
  - 这保证了向量检索路径可测，同时属性过滤完全确定性测试
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.data_models import (
    BoundingBox,
    ParsedQuery,
    RetrievalCandidate,
    TargetInstance,
)
from src.retrieval.query_parser import QueryParser, QueryResult
from src.retrieval.attribute_filter import AttributeFilter
from src.retrieval.reranker import CandidateReranker
from src.retrieval.vector_recall import VectorRecall


# ============================================================
# 数据加载 (模块级, 只加载一次)
# ============================================================

DATA_PATH = PROJECT_ROOT / "output" / "cityflow_results.json"


def _load_cityflow_data() -> dict:
    """加载 cityflow_results.json"""
    if not DATA_PATH.exists():
        pytest.skip(f"数据文件不存在: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_instances_from_detections(detections: List[dict]) -> List[TargetInstance]:
    """从 cityflow_results.json 的 detections 构建 TargetInstance 列表"""
    instances = []
    for det in detections:
        bbox = det.get("bbox", [0, 0, 100, 100])
        ts_str = det.get("timestamp", "2020-01-01 00:00:00.000")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except (ValueError, TypeError):
            ts = datetime(2020, 1, 1)

        # CLIP 向量
        clip_vec = None
        if det.get("clip_image_vector"):
            clip_vec = np.array(det["clip_image_vector"], dtype=np.float32)

        # 属性
        attrs = det.get("attributes", {})
        # 构建标准属性
        std_attrs = {}
        if "color" in attrs:
            std_attrs["color"] = attrs["color"]
        if "vehicle_type" in attrs:
            std_attrs["vehicle_type"] = attrs["vehicle_type"]
        if "color_refined" in attrs:
            std_attrs["color_refined"] = attrs["color_refined"]

        inst = TargetInstance(
            instance_id=det.get("target_id", f"DET_{len(instances)}"),
            camera_id=det.get("camera_id", "c001"),
            timestamp=ts,
            frame_id=det.get("frame_id", 0),
            target_type=det.get("target_type", "vehicle"),
            bbox=BoundingBox(
                x1=bbox[0], y1=bbox[1],
                x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3],
                confidence=det.get("confidence", 0.9),
            ),
            attributes=std_attrs,
            plate_number=None,
            plate_confidence=0.0,
            quality_score=det.get("confidence", 0.9),
            reid_vector=None,
            clip_vector=clip_vec,
            keyframe_path=det.get("keyframe_path"),
            scene_id=det.get("scene_id"),
        )
        instances.append(inst)
    return instances


def _get_ground_truth(
    instances: List[TargetInstance],
    color: str = None,
    vehicle_type: str = None,
    camera_id: str = None,
    scene_id: str = None,
) -> Set[str]:
    """获取满足条件的实例 ID 集合 (ground truth)"""
    gt = set()
    for inst in instances:
        if color and inst.attributes.get("color") != color:
            continue
        if vehicle_type and inst.attributes.get("vehicle_type") != vehicle_type:
            continue
        if camera_id and inst.camera_id != camera_id:
            continue
        if scene_id and inst.scene_id != scene_id:
            continue
        gt.add(inst.instance_id)
    return gt


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def raw_data():
    """加载原始 JSON 数据"""
    return _load_cityflow_data()


@pytest.fixture(scope="module")
def all_instances(raw_data):
    """从 detections 构建 TargetInstance 列表"""
    return _build_instances_from_detections(raw_data["detections"])


@pytest.fixture(scope="module")
def instance_index(all_instances):
    """instance_id → TargetInstance 索引"""
    return {inst.instance_id: inst for inst in all_instances}


@pytest.fixture(scope="module")
def parser():
    return QueryParser()


@pytest.fixture(scope="module")
def attr_filter():
    return AttributeFilter()


@pytest.fixture(scope="module")
def reranker():
    return CandidateReranker(top_n=10)


# ============================================================
# 查询集定义 (基于真实数据分布)
# ============================================================

# 从数据统计中得出的查询集
QUERY_SET = [
    # 颜色查询 (6个)
    {"text": "白色轿车", "color": "白色", "vehicle_type": "轿车", "type": "颜色+车型"},
    {"text": "黑色SUV", "color": "黑色", "vehicle_type": "SUV", "type": "颜色+车型"},
    {"text": "红色跑车", "color": "红色", "vehicle_type": "跑车", "type": "颜色+车型"},
    {"text": "蓝色卡车", "color": "蓝色", "vehicle_type": "卡车", "type": "颜色+车型"},
    {"text": "灰色面包车", "color": "灰色", "vehicle_type": "面包车", "type": "颜色+车型"},
    {"text": "棕色皮卡", "color": "棕色", "vehicle_type": "皮卡", "type": "颜色+车型"},
    {"text": "黄色公交车", "color": "黄色", "vehicle_type": "公交车", "type": "颜色+车型"},
    {"text": "绿色房车", "color": "绿色", "vehicle_type": "房车", "type": "颜色+车型"},

    # 纯车型查询 (5个)
    {"text": "轿车", "color": None, "vehicle_type": "轿车", "type": "车型"},
    {"text": "SUV", "color": None, "vehicle_type": "SUV", "type": "车型"},
    {"text": "卡车", "color": None, "vehicle_type": "卡车", "type": "车型"},
    {"text": "面包车", "color": None, "vehicle_type": "面包车", "type": "车型"},
    {"text": "公交车", "color": None, "vehicle_type": "公交车", "type": "车型"},

    # 纯颜色查询 (5个)
    {"text": "白色车辆", "color": "白色", "vehicle_type": None, "type": "颜色"},
    {"text": "黑色车辆", "color": "黑色", "vehicle_type": None, "type": "颜色"},
    {"text": "红色车辆", "color": "红色", "vehicle_type": None, "type": "颜色"},
    {"text": "蓝色车辆", "color": "蓝色", "vehicle_type": None, "type": "颜色"},
    {"text": "灰色车辆", "color": "灰色", "vehicle_type": None, "type": "颜色"},

    # 场景/位置查询 (4个)
    {"text": "S01场景中的车辆", "color": None, "vehicle_type": None, "scene_id": "S01", "type": "场景"},
    {"text": "c001摄像头中的车辆", "color": None, "vehicle_type": None, "camera_id": "c001", "type": "位置"},
]


# ============================================================
# 指标计算工具函数
# ============================================================

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K: 前K个结果中正确匹配的比例"""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / min(k, len(retrieved_ids)) if retrieved_ids else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K: 所有正确结果中被前K个覆盖的比例"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """F1@K: Precision和Recall的调和平均"""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Average Precision: 每个相关项处的Precision的平均"""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Reciprocal Rank: 第一个正确结果排名的倒数"""
    if not relevant_ids:
        return 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """DCG@K: Discounted Cumulative Gain"""
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 因为 log2(1)=0
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain"""
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    # 理想排序: 所有相关项排在前面
    ideal_retrieved = list(relevant_ids) + [rid for rid in retrieved_ids if rid not in relevant_ids]
    idcg = dcg_at_k(ideal_retrieved, relevant_ids, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# A. 属性过滤准确率测试
# ============================================================

class TestAttributeFilterAccuracy:
    """属性过滤准确率测试 - 基于真实数据的确定性过滤"""

    def test_color_filter_precision_recall(self, all_instances, attr_filter, parser):
        """测试颜色过滤的 Precision 和 Recall"""
        color_queries = [
            ("白色车辆", "白色"),
            ("黑色车辆", "黑色"),
            ("红色车辆", "红色"),
            ("蓝色车辆", "蓝色"),
            ("灰色车辆", "灰色"),
        ]
        report_lines = ["\n=== A. 属性过滤准确率报告 ==="]
        report_lines.append(f"{'颜色':<8} {'查询数':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'过滤结果数':<10}")
        report_lines.append("-" * 60)

        total_p, total_r, total_f1, count = 0, 0, 0, 0

        for text, color in color_queries:
            query = parser.parse(text)
            # 确保颜色被正确解析
            query.color = color
            query.target_type = "vehicle"

            # Ground truth: 所有该颜色的车辆
            gt_ids = _get_ground_truth(all_instances, color=color)

            # 过滤
            filtered = attr_filter.apply(all_instances, query)
            filtered_ids = {inst.instance_id for inst in filtered}

            # 计算指标
            tp = len(filtered_ids & gt_ids)
            precision = tp / len(filtered_ids) if filtered_ids else 0.0
            recall = tp / len(gt_ids) if gt_ids else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            total_p += precision
            total_r += recall
            total_f1 += f1
            count += 1

            report_lines.append(
                f"{color:<8} {1:<6} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {len(filtered_ids):<10}"
            )

        if count > 0:
            report_lines.append("-" * 60)
            report_lines.append(
                f"{'平均':<8} {count:<6} {total_p/count:<10.4f} {total_r/count:<10.4f} {total_f1/count:<10.4f}"
            )

        print("\n".join(report_lines))

        # 断言: 颜色过滤的 Precision 应该较高 (允许模糊匹配带来的额外结果)
        assert total_p / count > 0.5, f"颜色过滤平均 Precision 过低: {total_p/count:.4f}"

    def test_vehicle_type_filter_precision_recall(self, all_instances, attr_filter, parser):
        """测试车型过滤的 Precision 和 Recall"""
        type_queries = [
            ("轿车", "轿车"),
            ("SUV", "SUV"),
            ("卡车", "卡车"),
            ("面包车", "面包车"),
            ("公交车", "公交车"),
        ]

        report_lines = ["\n--- 车型过滤准确率 ---"]
        report_lines.append(f"{'车型':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'GT数':<8} {'结果数':<8}")
        report_lines.append("-" * 60)

        total_p, total_r, count = 0, 0, 0

        for text, vtype in type_queries:
            query = parser.parse(text)
            query.vehicle_type = vtype
            query.target_type = "vehicle"

            gt_ids = _get_ground_truth(all_instances, vehicle_type=vtype)
            filtered = attr_filter.apply(all_instances, query)
            filtered_ids = {inst.instance_id for inst in filtered}

            tp = len(filtered_ids & gt_ids)
            precision = tp / len(filtered_ids) if filtered_ids else 0.0
            recall = tp / len(gt_ids) if gt_ids else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            total_p += precision
            total_r += recall
            count += 1

            report_lines.append(
                f"{vtype:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {len(gt_ids):<8} {len(filtered_ids):<8}"
            )

        if count > 0:
            report_lines.append("-" * 60)
            report_lines.append(f"{'平均':<10} {total_p/count:<10.4f} {total_r/count:<10.4f}")

        print("\n".join(report_lines))
        assert total_r / count > 0.8, f"车型过滤平均 Recall 过低: {total_r/count:.4f}"

    def test_combined_filter_accuracy(self, all_instances, attr_filter, parser):
        """测试组合查询(颜色+车型)的过滤准确率"""
        combined_queries = [
            ("白色轿车", "白色", "轿车"),
            ("黑色SUV", "黑色", "SUV"),
            ("红色跑车", "红色", "跑车"),
            ("蓝色卡车", "蓝色", "卡车"),
            ("灰色面包车", "灰色", "面包车"),
        ]

        report_lines = ["\n--- 组合查询(颜色+车型)过滤准确率 ---"]
        report_lines.append(f"{'查询':<14} {'Precision':<10} {'Recall':<10} {'F1':<10} {'GT数':<8} {'结果数':<8}")
        report_lines.append("-" * 65)

        total_p, total_r, count = 0, 0, 0

        for text, color, vtype in combined_queries:
            query = parser.parse(text)
            query.color = color
            query.vehicle_type = vtype
            query.target_type = "vehicle"

            gt_ids = _get_ground_truth(all_instances, color=color, vehicle_type=vtype)
            filtered = attr_filter.apply(all_instances, query)
            filtered_ids = {inst.instance_id for inst in filtered}

            tp = len(filtered_ids & gt_ids)
            precision = tp / len(filtered_ids) if filtered_ids else 0.0
            recall = tp / len(gt_ids) if gt_ids else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            total_p += precision
            total_r += recall
            count += 1

            report_lines.append(
                f"{text:<14} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {len(gt_ids):<8} {len(filtered_ids):<8}"
            )

        if count > 0:
            report_lines.append("-" * 65)
            report_lines.append(f"{'平均':<14} {total_p/count:<10.4f} {total_r/count:<10.4f}")

        print("\n".join(report_lines))
        # 组合查询应该更精确
        assert total_p / count > 0.5, f"组合过滤平均 Precision 过低: {total_p/count:.4f}"


# ============================================================
# B. 检索匹配准确率测试 (使用真实 CLIP 向量)
# ============================================================

class TestRetrievalMatchingAccuracy:
    """
    检索匹配准确率测试

    使用真实 CLIP 图像向量进行向量检索评估。
    由于模型权重不可用，无法编码新文本，采用以下策略:
    - 对每个查询，找出所有匹配的 ground truth 实例
    - 用 ground truth 实例的平均 CLIP 向量作为"查询向量"
    - 在全部实例中进行向量检索 (余弦相似度)
    - 评估检索结果的 Precision@K, Recall@K, F1@K, MRR, mAP

    这评估了 CLIP 特征空间中的视觉相似性检索能力。
    """

    def _build_vector_index(self, all_instances: List[TargetInstance]) -> Tuple[np.ndarray, List[str]]:
        """构建向量矩阵和 ID 列表"""
        vectors = []
        ids = []
        for inst in all_instances:
            if inst.has_clip:
                vectors.append(inst.clip_vector)
                ids.append(inst.instance_id)
        return np.stack(vectors), ids

    def _query_by_vector(
        self,
        query_vec: np.ndarray,
        vec_matrix: np.ndarray,
        ids: List[str],
        top_k: int = 50,
    ) -> List[str]:
        """使用余弦相似度进行向量检索, 返回排序后的 ID 列表"""
        # 归一化
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        gallery_norm = vec_matrix / (np.linalg.norm(vec_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = gallery_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [ids[i] for i in top_indices]

    def test_clip_vector_retrieval_accuracy(self, all_instances):
        """测试 CLIP 向量检索的准确率"""
        vec_matrix, ids = self._build_vector_index(all_instances)
        id_to_inst = {inst.instance_id: inst for inst in all_instances}

        # 定义查询集: (查询名, 颜色, 车型)
        queries = [
            ("白色轿车", "白色", "轿车"),
            ("黑色SUV", "黑色", "SUV"),
            ("红色跑车", "红色", "跑车"),
            ("蓝色卡车", "蓝色", "卡车"),
            ("灰色面包车", "灰色", "面包车"),
            ("棕色皮卡", "棕色", "皮卡"),
            ("黄色公交车", "黄色", "公交车"),
            ("绿色房车", "绿色", "房车"),
            ("白色两厢车", "白色", "两厢车"),
            ("黑色旅行车", "黑色", "旅行车"),
            ("红色MPV", "红色", "MPV"),
            ("灰色轿车", "灰色", "轿车"),
        ]

        results_by_type: Dict[str, List[dict]] = {}

        for query_name, color, vtype in queries:
            # Ground truth
            gt_ids = _get_ground_truth(all_instances, color=color, vehicle_type=vtype)
            if not gt_ids:
                continue

            # 构建查询向量: ground truth 实例的平均 CLIP 向量
            gt_vectors = []
            for gid in gt_ids:
                inst = id_to_inst.get(gid)
                if inst and inst.has_clip:
                    gt_vectors.append(inst.clip_vector)

            if not gt_vectors:
                continue

            query_vec = np.mean(np.stack(gt_vectors), axis=0)

            # 向量检索
            retrieved = self._query_by_vector(query_vec, vec_matrix, ids, top_k=50)

            # 计算指标
            metrics = {
                "query": query_name,
                "gt_size": len(gt_ids),
                "p@5": precision_at_k(retrieved, gt_ids, 5),
                "p@10": precision_at_k(retrieved, gt_ids, 10),
                "p@20": precision_at_k(retrieved, gt_ids, 20),
                "r@5": recall_at_k(retrieved, gt_ids, 5),
                "r@10": recall_at_k(retrieved, gt_ids, 10),
                "r@20": recall_at_k(retrieved, gt_ids, 20),
                "f1@5": f1_at_k(retrieved, gt_ids, 5),
                "f1@10": f1_at_k(retrieved, gt_ids, 10),
                "mrr": reciprocal_rank(retrieved, gt_ids),
                "ap": average_precision(retrieved, gt_ids),
            }
            qtype = f"{color}+{vtype}"
            results_by_type.setdefault(qtype, []).append(metrics)

        # 输出报告
        report_lines = ["\n=== B. CLIP 向量检索匹配准确率报告 ==="]
        report_lines.append(
            f"{'查询':<14} {'GT数':<6} {'P@5':<7} {'P@10':<7} {'P@20':<7} "
            f"{'R@10':<7} {'F1@10':<7} {'MRR':<7} {'AP':<7}"
        )
        report_lines.append("-" * 90)

        all_metrics = []
        for qtype, metrics_list in results_by_type.items():
            for m in metrics_list:
                report_lines.append(
                    f"{m['query']:<14} {m['gt_size']:<6} {m['p@5']:<7.4f} {m['p@10']:<7.4f} "
                    f"{m['p@20']:<7.4f} {m['r@10']:<7.4f} {m['f1@10']:<7.4f} "
                    f"{m['mrr']:<7.4f} {m['ap']:<7.4f}"
                )
                all_metrics.append(m)

        if all_metrics:
            # 汇总
            avg_p5 = np.mean([m["p@5"] for m in all_metrics])
            avg_p10 = np.mean([m["p@10"] for m in all_metrics])
            avg_p20 = np.mean([m["p@20"] for m in all_metrics])
            avg_r10 = np.mean([m["r@10"] for m in all_metrics])
            avg_f1_10 = np.mean([m["f1@10"] for m in all_metrics])
            avg_mrr = np.mean([m["mrr"] for m in all_metrics])
            avg_map = np.mean([m["ap"] for m in all_metrics])

            report_lines.append("-" * 90)
            report_lines.append(
                f"{'平均':<14} {'':<6} {avg_p5:<7.4f} {avg_p10:<7.4f} "
                f"{avg_p20:<7.4f} {avg_r10:<7.4f} {avg_f1_10:<7.4f} "
                f"{avg_mrr:<7.4f} {avg_map:<7.4f}"
            )
            report_lines.append(f"\n总查询数: {len(all_metrics)}, 总 GT 实例范围: "
                              f"{min(m['gt_size'] for m in all_metrics)}-"
                              f"{max(m['gt_size'] for m in all_metrics)}")

        print("\n".join(report_lines))

        # 断言: MRR 应该 > 0.3 (第一个正确结果不应排太极)
        if all_metrics:
            avg_mrr_val = np.mean([m["mrr"] for m in all_metrics])
            assert avg_mrr_val > 0.3, f"平均 MRR 过低: {avg_mrr_val:.4f}"

    def test_color_only_vector_retrieval(self, all_instances):
        """测试纯颜色查询的向量检索准确率"""
        vec_matrix, ids = self._build_vector_index(all_instances)
        id_to_inst = {inst.instance_id: inst for inst in all_instances}

        color_queries = ["白色", "黑色", "红色", "蓝色", "灰色", "黄色", "绿色", "棕色"]
        results = []

        for color in color_queries:
            gt_ids = _get_ground_truth(all_instances, color=color)
            if len(gt_ids) < 3:
                continue

            gt_vectors = []
            for gid in gt_ids:
                inst = id_to_inst.get(gid)
                if inst and inst.has_clip:
                    gt_vectors.append(inst.clip_vector)
            if not gt_vectors:
                continue

            query_vec = np.mean(np.stack(gt_vectors), axis=0)
            retrieved = self._query_by_vector(query_vec, vec_matrix, ids, top_k=50)

            results.append({
                "query": color,
                "gt_size": len(gt_ids),
                "p@5": precision_at_k(retrieved, gt_ids, 5),
                "p@10": precision_at_k(retrieved, gt_ids, 10),
                "p@20": precision_at_k(retrieved, gt_ids, 20),
                "r@10": recall_at_k(retrieved, gt_ids, 10),
                "f1@10": f1_at_k(retrieved, gt_ids, 10),
                "mrr": reciprocal_rank(retrieved, gt_ids),
                "ap": average_precision(retrieved, gt_ids),
            })

        report_lines = ["\n--- 纯颜色查询向量检索 ---"]
        report_lines.append(f"{'颜色':<8} {'GT数':<6} {'P@5':<7} {'P@10':<7} {'P@20':<7} {'R@10':<7} {'F1@10':<7} {'MRR':<7}")
        report_lines.append("-" * 70)
        for r in results:
            report_lines.append(
                f"{r['query']:<8} {r['gt_size']:<6} {r['p@5']:<7.4f} {r['p@10']:<7.4f} "
                f"{r['p@20']:<7.4f} {r['r@10']:<7.4f} {r['f1@10']:<7.4f} {r['mrr']:<7.4f}"
            )
        if results:
            report_lines.append("-" * 70)
            report_lines.append(
                f"{'平均':<8} {'':<6} {np.mean([r['p@5'] for r in results]):<7.4f} "
                f"{np.mean([r['p@10'] for r in results]):<7.4f} "
                f"{np.mean([r['p@20'] for r in results]):<7.4f} "
                f"{np.mean([r['r@10'] for r in results]):<7.4f} "
                f"{np.mean([r['f1@10'] for r in results]):<7.4f} "
                f"{np.mean([r['mrr'] for r in results]):<7.4f}"
            )
        print("\n".join(report_lines))
        assert len(results) >= 5, "颜色查询数不足"

    def test_vehicle_type_only_vector_retrieval(self, all_instances):
        """测试纯车型查询的向量检索准确率"""
        vec_matrix, ids = self._build_vector_index(all_instances)
        id_to_inst = {inst.instance_id: inst for inst in all_instances}

        type_queries = ["轿车", "SUV", "卡车", "面包车", "跑车", "公交车", "皮卡"]
        results = []

        for vtype in type_queries:
            gt_ids = _get_ground_truth(all_instances, vehicle_type=vtype)
            if len(gt_ids) < 3:
                continue

            gt_vectors = []
            for gid in gt_ids:
                inst = id_to_inst.get(gid)
                if inst and inst.has_clip:
                    gt_vectors.append(inst.clip_vector)
            if not gt_vectors:
                continue

            query_vec = np.mean(np.stack(gt_vectors), axis=0)
            retrieved = self._query_by_vector(query_vec, vec_matrix, ids, top_k=50)

            results.append({
                "query": vtype,
                "gt_size": len(gt_ids),
                "p@5": precision_at_k(retrieved, gt_ids, 5),
                "p@10": precision_at_k(retrieved, gt_ids, 10),
                "p@20": precision_at_k(retrieved, gt_ids, 20),
                "r@10": recall_at_k(retrieved, gt_ids, 10),
                "f1@10": f1_at_k(retrieved, gt_ids, 10),
                "mrr": reciprocal_rank(retrieved, gt_ids),
                "ap": average_precision(retrieved, gt_ids),
            })

        report_lines = ["\n--- 纯车型查询向量检索 ---"]
        report_lines.append(f"{'车型':<8} {'GT数':<6} {'P@5':<7} {'P@10':<7} {'P@20':<7} {'R@10':<7} {'F1@10':<7} {'MRR':<7}")
        report_lines.append("-" * 70)
        for r in results:
            report_lines.append(
                f"{r['query']:<8} {r['gt_size']:<6} {r['p@5']:<7.4f} {r['p@10']:<7.4f} "
                f"{r['p@20']:<7.4f} {r['r@10']:<7.4f} {r['f1@10']:<7.4f} {r['mrr']:<7.4f}"
            )
        if results:
            report_lines.append("-" * 70)
            report_lines.append(
                f"{'平均':<8} {'':<6} {np.mean([r['p@5'] for r in results]):<7.4f} "
                f"{np.mean([r['p@10'] for r in results]):<7.4f} "
                f"{np.mean([r['p@20'] for r in results]):<7.4f} "
                f"{np.mean([r['r@10'] for r in results]):<7.4f} "
                f"{np.mean([r['f1@10'] for r in results]):<7.4f} "
                f"{np.mean([r['mrr'] for r in results]):<7.4f}"
            )
        print("\n".join(report_lines))
        assert len(results) >= 5, "车型查询数不足"


# ============================================================
# C. 重排序效果评估
# ============================================================

class TestRerankerEffectiveness:
    """
    重排序效果评估

    对比重排序前后的排名变化:
    - 使用向量检索获得初始候选列表 (模拟粗召回)
    - 使用 Reranker 进行重排
    - 对比重排前后的 P@5, MRR, NDCG@K
    """

    def _build_vector_index(self, all_instances):
        vectors, ids = [], []
        for inst in all_instances:
            if inst.has_clip:
                vectors.append(inst.clip_vector)
                ids.append(inst.instance_id)
        return np.stack(vectors), ids

    def _vector_recall_candidates(
        self, query_vec, vec_matrix, ids, all_instances, top_k=20
    ) -> List[RetrievalCandidate]:
        """模拟向量召回, 返回 RetrievalCandidate 列表"""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        gallery_norm = vec_matrix / (np.linalg.norm(vec_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = gallery_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        id_to_inst = {inst.instance_id: inst for inst in all_instances}
        candidates = []
        for rank, idx in enumerate(top_indices):
            inst = id_to_inst[ids[idx]]
            candidates.append(RetrievalCandidate(
                instance=inst,
                text_score=float(similarities[idx]),
                attribute_match_score=0.0,
                combined_score=float(similarities[idx]),
                rank=rank + 1,
            ))
        return candidates

    def test_reranking_improvement(self, all_instances, reranker):
        """测试重排序对检索效果的提升"""
        vec_matrix, ids = self._build_vector_index(all_instances)
        id_to_inst = {inst.instance_id: inst for inst in all_instances}

        queries = [
            ("白色轿车", "白色", "轿车"),
            ("黑色SUV", "黑色", "SUV"),
            ("红色跑车", "红色", "跑车"),
            ("蓝色卡车", "蓝色", "卡车"),
            ("灰色面包车", "灰色", "面包车"),
            ("棕色皮卡", "棕色", "皮卡"),
            ("黄色公交车", "黄色", "公交车"),
            ("绿色房车", "绿色", "房车"),
        ]

        before_metrics = []
        after_metrics = []

        report_lines = ["\n=== C. 重排序效果评估报告 ==="]
        report_lines.append(
            f"{'查询':<14} {'P@5前':<7} {'P@5后':<7} {'MRR前':<7} {'MRR后':<7} "
            f"{'NDCG@5前':<10} {'NDCG@5后':<10} {'提升':<8}"
        )
        report_lines.append("-" * 85)

        for query_name, color, vtype in queries:
            gt_ids = _get_ground_truth(all_instances, color=color, vehicle_type=vtype)
            if len(gt_ids) < 2:
                continue

            # 构建查询向量
            gt_vectors = []
            for gid in gt_ids:
                inst = id_to_inst.get(gid)
                if inst and inst.has_clip:
                    gt_vectors.append(inst.clip_vector)
            if not gt_vectors:
                continue

            query_vec = np.mean(np.stack(gt_vectors), axis=0)

            # 向量召回
            candidates = self._vector_recall_candidates(
                query_vec, vec_matrix, ids, all_instances, top_k=20
            )

            # 重排序前的排序 (按向量分数)
            before_ids = [c.instance.instance_id for c in candidates]
            before_p5 = precision_at_k(before_ids, gt_ids, 5)
            before_mrr = reciprocal_rank(before_ids, gt_ids)
            before_ndcg5 = ndcg_at_k(before_ids, gt_ids, 5)

            # 构造 QueryResult 用于重排
            query_result = QueryResult(
                query_text=query_name,
                query_type="vehicle",
                target_type="vehicle",
                color=color,
                vehicle_type=vtype,
            )

            # 重排序 (复制 candidates 避免修改原始数据)
            import copy
            candidates_copy = copy.deepcopy(candidates)
            reranked = reranker.rerank(candidates_copy, query_result, top_n=20)

            after_ids = [c.instance.instance_id for c in reranked]
            after_p5 = precision_at_k(after_ids, gt_ids, 5)
            after_mrr = reciprocal_rank(after_ids, gt_ids)
            after_ndcg5 = ndcg_at_k(after_ids, gt_ids, 5)

            improvement = after_p5 - before_p5

            before_metrics.append({"p5": before_p5, "mrr": before_mrr, "ndcg5": before_ndcg5})
            after_metrics.append({"p5": after_p5, "mrr": after_mrr, "ndcg5": after_ndcg5})

            report_lines.append(
                f"{query_name:<14} {before_p5:<7.4f} {after_p5:<7.4f} "
                f"{before_mrr:<7.4f} {after_mrr:<7.4f} "
                f"{before_ndcg5:<10.4f} {after_ndcg5:<10.4f} "
                f"{improvement:+.4f}"
            )

        if before_metrics:
            avg_before_p5 = np.mean([m["p5"] for m in before_metrics])
            avg_after_p5 = np.mean([m["p5"] for m in after_metrics])
            avg_before_mrr = np.mean([m["mrr"] for m in before_metrics])
            avg_after_mrr = np.mean([m["mrr"] for m in after_metrics])
            avg_before_ndcg = np.mean([m["ndcg5"] for m in before_metrics])
            avg_after_ndcg = np.mean([m["ndcg5"] for m in after_metrics])

            report_lines.append("-" * 85)
            report_lines.append(
                f"{'平均':<14} {avg_before_p5:<7.4f} {avg_after_p5:<7.4f} "
                f"{avg_before_mrr:<7.4f} {avg_after_mrr:<7.4f} "
                f"{avg_before_ndcg:<10.4f} {avg_after_ndcg:<10.4f} "
                f"{avg_after_p5 - avg_before_p5:+.4f}"
            )

            report_lines.append(f"\n重排序效果分析:")
            report_lines.append(f"  P@5 变化: {avg_before_p5:.4f} → {avg_after_p5:.4f} "
                              f"({'提升' if avg_after_p5 > avg_before_p5 else '下降'} "
                              f"{abs(avg_after_p5 - avg_before_p5):.4f})")
            report_lines.append(f"  MRR 变化: {avg_before_mrr:.4f} → {avg_after_mrr:.4f} "
                              f"({'提升' if avg_after_mrr > avg_before_mrr else '下降'} "
                              f"{abs(avg_after_mrr - avg_before_mrr):.4f})")
            report_lines.append(f"  NDCG@5 变化: {avg_before_ndcg:.4f} → {avg_after_ndcg:.4f} "
                              f"({'提升' if avg_after_ndcg > avg_before_ndcg else '下降'} "
                              f"{abs(avg_after_ndcg - avg_before_ndcg):.4f})")

        print("\n".join(report_lines))

        # 断言: 重排序后 NDCG 不应显著下降
        if after_metrics:
            avg_after_ndcg_val = np.mean([m["ndcg5"] for m in after_metrics])
            assert avg_after_ndcg_val > 0.2, f"重排序后 NDCG@5 过低: {avg_after_ndcg_val:.4f}"

    def test_reranker_attribute_scoring(self, all_instances, reranker):
        """测试重排器的属性匹配评分准确性"""
        # 构造已知属性的候选
        candidates = []
        for i, (color, vtype) in enumerate([
            ("白色", "轿车"), ("黑色", "SUV"), ("白色", "SUV"),
            ("黑色", "轿车"), ("白色", "轿车"),
        ]):
            inst = TargetInstance(
                instance_id=f"RERANK_{i}",
                camera_id="c001",
                timestamp=datetime(2020, 1, 1),
                frame_id=i,
                target_type="vehicle",
                bbox=BoundingBox(0, 0, 100, 100, 0.9),
                attributes={"color": color, "vehicle_type": vtype},
                plate_number=None,
                plate_confidence=0.0,
                quality_score=0.8,
                reid_vector=None,
                clip_vector=np.random.randn(768).astype(np.float32),
                keyframe_path=None,
            )
            candidates.append(RetrievalCandidate(
                instance=inst,
                text_score=0.5,  # 相同的向量分数
                attribute_match_score=0.0,
                combined_score=0.5,
                rank=i + 1,
            ))

        # 查询 "白色轿车"
        query = QueryResult(
            query_text="白色轿车", query_type="vehicle",
            target_type="vehicle", color="白色", vehicle_type="轿车",
        )

        reranked = reranker.rerank(candidates, query, top_n=5)

        # 白色轿车 (index 0, 4) 应该排在前面
        top2_ids = [c.instance.instance_id for c in reranked[:2]]
        assert "RERANK_0" in top2_ids or "RERANK_4" in top2_ids, \
            "属性完全匹配的候选应排在前列"

        print(f"\n--- 重排器属性评分测试 ---")
        print(f"查询: 白色轿车")
        for c in reranked:
            attrs = c.instance.attributes
            print(f"  Rank {c.rank}: color={attrs.get('color')}, "
                  f"type={attrs.get('vehicle_type')}, "
                  f"attr_score={c.attribute_match_score:.2f}, "
                  f"combined={c.combined_score:.4f}")


# ============================================================
# D. 综合检索准确率报告
# ============================================================

class TestComprehensiveReport:
    """综合检索准确率报告 - 输出完整指标表格"""

    def test_full_retrieval_accuracy_report(self, all_instances, attr_filter, parser):
        """输出完整的检索准确率报告"""
        vec_matrix_list = []
        ids_list = []
        id_to_inst = {}
        for inst in all_instances:
            id_to_inst[inst.instance_id] = inst
            if inst.has_clip:
                vec_matrix_list.append(inst.clip_vector)
                ids_list.append(inst.instance_id)

        vec_matrix = np.stack(vec_matrix_list) if vec_matrix_list else None

        # 定义综合查询集
        query_groups = {
            "颜色+车型": [
                ("白色轿车", "白色", "轿车"),
                ("黑色SUV", "黑色", "SUV"),
                ("红色跑车", "红色", "跑车"),
                ("蓝色卡车", "蓝色", "卡车"),
                ("灰色面包车", "灰色", "面包车"),
            ],
            "纯车型": [
                ("轿车", None, "轿车"),
                ("SUV", None, "SUV"),
                ("卡车", None, "卡车"),
                ("面包车", None, "面包车"),
                ("公交车", None, "公交车"),
            ],
            "纯颜色": [
                ("白色车辆", "白色", None),
                ("黑色车辆", "黑色", None),
                ("红色车辆", "红色", None),
                ("蓝色车辆", "蓝色", None),
                ("灰色车辆", "灰色", None),
            ],
        }

        report_lines = ["\n" + "=" * 100]
        report_lines.append("=== 检索准确率综合报告 ===")
        report_lines.append(f"数据源: cityflow_results.json ({len(all_instances)} 检测记录)")
        report_lines.append(f"CLIP向量维度: 768, 有向量实例: {len(ids_list)}")
        report_lines.append("=" * 100)

        summary_table = []

        for group_name, queries in query_groups.items():
            group_results = []

            for query_text, color, vtype in queries:
                # Ground truth
                gt_ids = _get_ground_truth(all_instances, color=color, vehicle_type=vtype)
                if not gt_ids:
                    continue

                # 属性过滤结果
                query_result = parser.parse(query_text)
                if color:
                    query_result.color = color
                if vtype:
                    query_result.vehicle_type = vtype
                query_result.target_type = "vehicle"

                filtered = attr_filter.apply(all_instances, query_result)
                filtered_ids = [inst.instance_id for inst in filtered]

                # 向量检索结果 (如果有向量)
                vec_retrieved = []
                if vec_matrix is not None:
                    gt_vectors = []
                    for gid in gt_ids:
                        inst = id_to_inst.get(gid)
                        if inst and inst.has_clip:
                            gt_vectors.append(inst.clip_vector)
                    if gt_vectors:
                        query_vec = np.mean(np.stack(gt_vectors), axis=0)
                        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
                        gallery_norm = vec_matrix / (np.linalg.norm(vec_matrix, axis=1, keepdims=True) + 1e-8)
                        sims = gallery_norm @ query_norm
                        top_idx = np.argsort(sims)[::-1][:50]
                        vec_retrieved = [ids_list[i] for i in top_idx]

                group_results.append({
                    "query": query_text,
                    "gt_size": len(gt_ids),
                    "filter_count": len(filtered_ids),
                    # 属性过滤指标
                    "filter_p": len(set(filtered_ids) & gt_ids) / len(filtered_ids) if filtered_ids else 0,
                    "filter_r": len(set(filtered_ids) & gt_ids) / len(gt_ids) if gt_ids else 0,
                    # 向量检索指标
                    "vp5": precision_at_k(vec_retrieved, gt_ids, 5) if vec_retrieved else 0,
                    "vp10": precision_at_k(vec_retrieved, gt_ids, 10) if vec_retrieved else 0,
                    "vp20": precision_at_k(vec_retrieved, gt_ids, 20) if vec_retrieved else 0,
                    "vr10": recall_at_k(vec_retrieved, gt_ids, 10) if vec_retrieved else 0,
                    "vf1_10": f1_at_k(vec_retrieved, gt_ids, 10) if vec_retrieved else 0,
                    "vmrr": reciprocal_rank(vec_retrieved, gt_ids) if vec_retrieved else 0,
                    "vmap": average_precision(vec_retrieved, gt_ids) if vec_retrieved else 0,
                })

            if not group_results:
                continue

            # 输出分组表格
            report_lines.append(f"\n--- {group_name}查询 ---")
            report_lines.append(
                f"{'查询':<14} {'GT':<5} {'过滤P':<7} {'过滤R':<7} "
                f"{'P@5':<7} {'P@10':<7} {'P@20':<7} {'R@10':<7} "
                f"{'F1@10':<7} {'MRR':<7} {'mAP':<7}"
            )
            report_lines.append("-" * 100)

            for r in group_results:
                report_lines.append(
                    f"{r['query']:<14} {r['gt_size']:<5} "
                    f"{r['filter_p']:<7.4f} {r['filter_r']:<7.4f} "
                    f"{r['vp5']:<7.4f} {r['vp10']:<7.4f} {r['vp20']:<7.4f} "
                    f"{r['vr10']:<7.4f} {r['vf1_10']:<7.4f} "
                    f"{r['vmrr']:<7.4f} {r['vmap']:<7.4f}"
                )

            # 分组平均
            n = len(group_results)
            avg = {
                "filter_p": np.mean([r["filter_p"] for r in group_results]),
                "filter_r": np.mean([r["filter_r"] for r in group_results]),
                "vp5": np.mean([r["vp5"] for r in group_results]),
                "vp10": np.mean([r["vp10"] for r in group_results]),
                "vp20": np.mean([r["vp20"] for r in group_results]),
                "vr10": np.mean([r["vr10"] for r in group_results]),
                "vf1_10": np.mean([r["vf1_10"] for r in group_results]),
                "vmrr": np.mean([r["vmrr"] for r in group_results]),
                "vmap": np.mean([r["vmap"] for r in group_results]),
            }
            report_lines.append("-" * 100)
            report_lines.append(
                f"{'平均':<14} {'':<5} "
                f"{avg['filter_p']:<7.4f} {avg['filter_r']:<7.4f} "
                f"{avg['vp5']:<7.4f} {avg['vp10']:<7.4f} {avg['vp20']:<7.4f} "
                f"{avg['vr10']:<7.4f} {avg['vf1_10']:<7.4f} "
                f"{avg['vmrr']:<7.4f} {avg['vmap']:<7.4f}"
            )

            summary_table.append({
                "group": group_name,
                "count": n,
                **avg,
            })

        # 汇总表
        report_lines.append("\n" + "=" * 100)
        report_lines.append("=== 汇总表 ===")
        report_lines.append(
            f"{'查询类型':<12} {'查询数':<6} {'过滤P':<8} {'过滤R':<8} "
            f"{'P@5':<7} {'P@10':<7} {'P@20':<7} {'R@10':<7} "
            f"{'F1@10':<7} {'MRR':<7} {'mAP':<7}"
        )
        report_lines.append("-" * 100)
        for row in summary_table:
            report_lines.append(
                f"{row['group']:<12} {row['count']:<6} "
                f"{row['filter_p']:<8.4f} {row['filter_r']:<8.4f} "
                f"{row['vp5']:<7.4f} {row['vp10']:<7.4f} {row['vp20']:<7.4f} "
                f"{row['vr10']:<7.4f} {row['vf1_10']:<7.4f} "
                f"{row['vmrr']:<7.4f} {row['vmap']:<7.4f}"
            )

        # 全局平均
        if summary_table:
            all_filter_p = np.mean([r["filter_p"] for r in summary_table])
            all_filter_r = np.mean([r["filter_r"] for r in summary_table])
            all_vp5 = np.mean([r["vp5"] for r in summary_table])
            all_vp10 = np.mean([r["vp10"] for r in summary_table])
            all_vp20 = np.mean([r["vp20"] for r in summary_table])
            all_vr10 = np.mean([r["vr10"] for r in summary_table])
            all_vf1 = np.mean([r["vf1_10"] for r in summary_table])
            all_vmrr = np.mean([r["vmrr"] for r in summary_table])
            all_vmap = np.mean([r["vmap"] for r in summary_table])
            total_queries = sum(r["count"] for r in summary_table)

            report_lines.append("-" * 100)
            report_lines.append(
                f"{'总计':<12} {total_queries:<6} "
                f"{all_filter_p:<8.4f} {all_filter_r:<8.4f} "
                f"{all_vp5:<7.4f} {all_vp10:<7.4f} {all_vp20:<7.4f} "
                f"{all_vr10:<7.4f} {all_vf1:<7.4f} "
                f"{all_vmrr:<7.4f} {all_vmap:<7.4f}"
            )

            report_lines.append(f"\n分析说明:")
            report_lines.append(f"  - 属性过滤 Precision 受模糊匹配影响, 可能包含近似颜色/类型的结果")
            report_lines.append(f"  - 向量检索使用 CLIP 图像特征余弦相似度, 因无模型权重无法编码文本")
            report_lines.append(f"  - 查询向量由 ground truth 平均图像向量近似, 代表该类车辆的视觉中心")
            report_lines.append(f"  - 总检测数: {len(all_instances)}, 有CLIP向量: {len(ids_list)}")

        report_lines.append("=" * 100)
        print("\n".join(report_lines))

        # 基本断言
        assert len(all_instances) > 100, "实例数过少"
        assert summary_table, "未生成任何结果"
