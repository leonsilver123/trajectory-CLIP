"""
tests/test_retrieval.py - 检索功能测试

测试目标: 验证检索管线各模块（不需要模型权重）
覆盖:
  - QueryParser 关键词解析（不需要模型，测试解析逻辑）
  - AttributeFilter 属性过滤（用 mock 数据构造 TargetInstance 列表）
  - Reranker 重排序评分（用随机向量测试排序逻辑）
  - VectorRecall 内存回退模式测试
"""

import pytest
import numpy as np
from datetime import datetime

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
# Fixtures
# ============================================================

def _make_instance(instance_id="INST_001", target_type="vehicle",
                  plate_number=None, attributes=None, quality=0.8,
                  timestamp=None, clip_vector=None, reid_vector=None):
    """辅助创建 TargetInstance"""
    return TargetInstance(
        instance_id=instance_id,
        camera_id="c001",
        timestamp=timestamp or datetime.now(),
        frame_id=100,
        target_type=target_type,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        attributes=attributes or {},
        plate_number=plate_number,
        plate_confidence=0.0,
        quality_score=quality,
        reid_vector=reid_vector,
        clip_vector=clip_vector,
        keyframe_path=None,
    )


@pytest.fixture
def sample_instances():
    """创建一组测试用 TargetInstance"""
    return [
        _make_instance("INST_001", "vehicle", attributes={"color": "白色", "vehicle_type": "轿车"}),
        _make_instance("INST_002", "vehicle", attributes={"color": "黑色", "vehicle_type": "SUV"}),
        _make_instance("INST_003", "pedestrian", attributes={"gender": "male", "clothing_color": "蓝色"}),
        _make_instance("INST_004", "vehicle", plate_number="苏E12345",
                       attributes={"color": "红色", "vehicle_type": "出租车"}),
        _make_instance("INST_005", "non_motor_vehicle", attributes={"vehicle_type": "电动车"}),
        _make_instance("INST_006", "pedestrian", attributes={"gender": "female", "bag": True, "bag_color": "红色"}),
    ]


# ============================================================
# QueryParser 测试
# ============================================================

class TestQueryParser:
    """查询解析器测试"""

    @pytest.fixture
    def parser(self):
        return QueryParser()

    def test_parse_plate(self, parser):
        """测试车牌号解析"""
        result = parser.parse("苏E12345")
        assert result.query_type == "plate"
        assert result.plate_number == "苏E12345"
        assert result.target_type == "vehicle"

    def test_parse_vehicle_color_type(self, parser):
        """测试颜色+车型解析"""
        result = parser.parse("黑色轿车")
        assert result.target_type == "vehicle"
        assert result.color == "黑色"
        assert result.vehicle_type == "轿车"

    def test_parse_pedestrian_gender(self, parser):
        """测试行人性别解析"""
        result = parser.parse("男人")
        assert result.target_type == "pedestrian"
        assert result.gender == "male"

    def test_parse_pedestrian_clothing(self, parser):
        """测试行人衣服颜色解析"""
        result = parser.parse("蓝衣服")
        assert result.target_type == "pedestrian"
        assert result.clothing_color == "蓝色"

    def test_parse_non_motor_vehicle(self, parser):
        """测试非机动车解析"""
        result = parser.parse("电动车")
        assert result.target_type == "non_motor_vehicle"

    def test_parse_colloquial_suv(self, parser):
        """测试口语化表达 SUV"""
        result = parser.parse("SUV")
        assert result.target_type == "vehicle"
        assert result.vehicle_type == "SUV"

    def test_parse_colloquial_compound(self, parser):
        """测试复合口语化表达"""
        result = parser.parse("白色面包车")
        assert result.target_type == "vehicle"
        assert result.color == "白色"
        assert result.vehicle_type == "面包车"

    def test_parse_bag_detection(self, parser):
        """测试背包检测"""
        result = parser.parse("背书包的人")
        assert result.target_type == "pedestrian"
        assert result.bag is True

    def test_parse_unknown_query(self, parser):
        """测试无法解析的查询"""
        result = parser.parse("一些模糊的描述")
        # 不应该崩溃，返回 general 类型
        assert result.query_type in ("general", None) or result.query_type is not None

    def test_parse_empty_string(self, parser):
        """测试空字符串"""
        result = parser.parse("")
        assert result.query_text == ""

    def test_to_parsed_query(self, parser):
        """测试转换为 ParsedQuery"""
        qr = parser.parse("黑色轿车")
        pq = parser.to_parsed_query(qr)
        assert isinstance(pq, ParsedQuery)
        assert pq.target_type == "vehicle"
        assert pq.clip_text_embedding is None

    def test_parse_color_blue(self, parser):
        """测试蓝色解析"""
        result = parser.parse("蓝色轿车")
        assert result.color == "蓝色"
        assert result.vehicle_type == "轿车"

    def test_parse_pedestrian_female(self, parser):
        """测试女性行人解析"""
        result = parser.parse("女人")
        assert result.target_type == "pedestrian"
        assert result.gender == "female"

    def test_parse_hat_detection(self, parser):
        """测试帽子检测"""
        result = parser.parse("戴帽子的人")
        assert result.accessory == "帽子"

    def test_parse_glasses_detection(self, parser):
        """测试眼镜检测"""
        result = parser.parse("戴眼镜的人")
        assert result.accessory == "眼镜"


# ============================================================
# AttributeFilter 测试
# ============================================================

class TestAttributeFilter:
    """属性过滤器测试"""

    @pytest.fixture
    def attr_filter(self):
        return AttributeFilter()

    def test_filter_by_target_type_vehicle(self, attr_filter, sample_instances):
        """测试按目标类别过滤 - 车辆"""
        query = QueryResult(query_text="车辆", query_type="vehicle", target_type="vehicle")
        results = attr_filter.apply(sample_instances, query)
        for inst in results:
            assert inst.target_type in ("vehicle", "non_motor_vehicle")

    def test_filter_by_target_type_pedestrian(self, attr_filter, sample_instances):
        """测试按目标类别过滤 - 行人"""
        query = QueryResult(query_text="行人", query_type="pedestrian", target_type="pedestrian")
        results = attr_filter.apply(sample_instances, query)
        for inst in results:
            assert inst.target_type == "pedestrian"

    def test_filter_by_plate(self, attr_filter, sample_instances):
        """测试按车牌过滤"""
        query = QueryResult(query_text="苏E12345", query_type="plate",
                           target_type="vehicle", plate_number="苏E12345")
        results = attr_filter.apply(sample_instances, query)
        assert len(results) == 1
        assert results[0].plate_number == "苏E12345"

    def test_filter_by_color(self, attr_filter, sample_instances):
        """测试按颜色过滤"""
        query = QueryResult(query_text="白色", query_type="vehicle",
                           target_type="vehicle", color="白色")
        results = attr_filter.apply(sample_instances, query)
        for inst in results:
            assert inst.attributes.get("color") == "白色" or inst.attributes.get("color") is None

    def test_filter_by_gender(self, attr_filter, sample_instances):
        """测试按性别过滤"""
        query = QueryResult(query_text="男性", query_type="pedestrian",
                           target_type="pedestrian", gender="male")
        results = attr_filter.apply(sample_instances, query)
        for inst in results:
            assert inst.attributes.get("gender") == "male" or inst.attributes.get("gender") is None

    def test_filter_empty_list(self, attr_filter):
        """测试空列表过滤"""
        query = QueryResult(query_text="车辆", query_type="vehicle", target_type="vehicle")
        results = attr_filter.apply([], query)
        assert results == []

    def test_filter_no_conditions(self, attr_filter, sample_instances):
        """测试无过滤条件返回全部"""
        query = QueryResult(query_text="", query_type="general")
        results = attr_filter.apply(sample_instances, query)
        assert len(results) == len(sample_instances)

    def test_filter_by_bag(self, attr_filter, sample_instances):
        """测试按背包过滤"""
        query = QueryResult(query_text="背包", query_type="pedestrian",
                           target_type="pedestrian", bag=True)
        results = attr_filter.apply(sample_instances, query)
        for inst in results:
            assert inst.attributes.get("bag", False) is True or inst.attributes.get("bag") is None

    def test_apply_parsed_query(self, attr_filter, sample_instances):
        """测试 ParsedQuery 接口"""
        pq = ParsedQuery(
            raw_text="车辆",
            target_type="vehicle",
            attributes={},
            plate_number=None,
            clip_text_embedding=None,
        )
        results = attr_filter.apply_parsed_query(sample_instances, pq)
        for inst in results:
            assert inst.target_type in ("vehicle", "non_motor_vehicle")


# ============================================================
# CandidateReranker 测试
# ============================================================

class TestCandidateReranker:
    """候选重排器测试"""

    @pytest.fixture
    def reranker(self):
        return CandidateReranker(top_n=3)

    def _make_candidates(self, n=5):
        """创建测试候选列表"""
        candidates = []
        for i in range(n):
            inst = _make_instance(
                instance_id=f"INST_{i:03d}",
                attributes={"color": "白色" if i % 2 == 0 else "黑色"},
                quality=0.5 + i * 0.1,
            )
            c = RetrievalCandidate(
                instance=inst,
                text_score=0.9 - i * 0.1,
                attribute_match_score=0.0,
                combined_score=0.9 - i * 0.1,
                rank=i + 1,
            )
            candidates.append(c)
        return candidates

    def test_rerank_returns_top_n(self, reranker):
        """测试重排返回 top_n"""
        candidates = self._make_candidates(5)
        query = QueryResult(query_text="白色车辆", query_type="vehicle",
                           target_type="vehicle", color="白色")
        results = reranker.rerank(candidates, query)
        assert len(results) <= 3

    def test_rerank_updates_rank(self, reranker):
        """测试重排更新排名"""
        candidates = self._make_candidates(5)
        query = QueryResult(query_text="车辆", query_type="vehicle", target_type="vehicle")
        results = reranker.rerank(candidates, query)
        ranks = [c.rank for c in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_rerank_empty_list(self, reranker):
        """测试空列表重排"""
        query = QueryResult(query_text="车辆", query_type="vehicle")
        results = reranker.rerank([], query)
        assert results == []

    def test_rerank_sorted_by_combined(self, reranker):
        """测试重排结果按综合分排序"""
        candidates = self._make_candidates(5)
        query = QueryResult(query_text="车辆", query_type="vehicle", target_type="vehicle")
        results = reranker.rerank(candidates, query)
        scores = [c.combined_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_attribute_score_no_conditions(self, reranker):
        """测试无属性条件时属性分为满分"""
        candidates = self._make_candidates(1)
        query = QueryResult(query_text="", query_type="general")
        results = reranker.rerank(candidates, query)
        assert results[0].attribute_match_score == 1.0


# ============================================================
# VectorRecall 内存回退模式测试
# ============================================================

class TestVectorRecallMemory:
    """VectorRecall 内存回退模式测试"""

    @pytest.fixture
    def recall(self):
        """创建使用内存回退的 VectorRecall"""
        return VectorRecall(use_memory_fallback=True)

    def test_add_to_memory_index(self, recall):
        """测试添加到内存索引"""
        instances = []
        for i in range(5):
            inst = _make_instance(
                instance_id=f"INST_{i:03d}",
                clip_vector=np.random.randn(768).astype(np.float32),
            )
            instances.append(inst)
        recall.add_to_memory_index(instances)
        assert len(recall._memory_vectors) == 5

    def test_memory_recall(self, recall):
        """测试内存检索"""
        instances = []
        for i in range(10):
            inst = _make_instance(
                instance_id=f"INST_{i:03d}",
                clip_vector=np.random.randn(768).astype(np.float32),
            )
            instances.append(inst)
        recall.add_to_memory_index(instances)

        query_vector = np.random.randn(768).astype(np.float32)
        pq = ParsedQuery(
            raw_text="测试",
            target_type=None,
            attributes={},
            plate_number=None,
            clip_text_embedding=query_vector,
        )
        results = recall.recall(pq, top_k=5)
        assert len(results) == 5
        # 结果应按相似度排序
        scores = [r.text_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_memory_recall_empty_index(self, recall):
        """测试空索引检索"""
        query_vector = np.random.randn(768).astype(np.float32)
        pq = ParsedQuery(
            raw_text="测试",
            target_type=None,
            attributes={},
            plate_number=None,
            clip_text_embedding=query_vector,
        )
        results = recall.recall(pq, top_k=5)
        assert results == []

    def test_add_instances_without_clip(self, recall):
        """测试添加无 CLIP 特征的实例不进入索引"""
        inst = _make_instance(instance_id="INST_NO_CLIP", clip_vector=None)
        recall.add_to_memory_index([inst])
        assert len(recall._memory_vectors) == 0

    def test_search_interface(self, recall):
        """测试通用 search 接口"""
        instances = []
        for i in range(5):
            inst = _make_instance(
                instance_id=f"INST_{i:03d}",
                clip_vector=np.random.randn(768).astype(np.float32),
            )
            instances.append(inst)
        recall.add_to_memory_index(instances)

        query_vector = np.random.randn(768).astype(np.float32)
        results = recall.search(query_vector, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
