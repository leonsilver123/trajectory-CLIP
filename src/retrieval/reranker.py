"""
src.retrieval.reranker - 候选图片重排模块

对向量召回的候选结果进行精排:
- 属性匹配度重排
- 图文相似度加权
- 质量分加权
- 时间新鲜度加权

输入: 粗召回候选列表
输出: 精排后的 Top-N 候选
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.common.data_models import ParsedQuery, RetrievalCandidate
from src.common.logger import get_logger
from src.retrieval.query_parser import QueryResult

logger = get_logger("retrieval.reranker")


class CandidateReranker:
    """
    候选重排器

    对粗召回结果进行精细化重排。

    综合评分公式:
        score = w1*vector_score + w2*attribute_match + w3*quality_score + w4*freshness_score

    使用方式:
        reranker = CandidateReranker()
        top_results = reranker.rerank(candidates, query, top_n=5)
    """

    def __init__(
        self,
        top_n: int = 5,
        vector_weight: float = 0.4,
        attribute_weight: float = 0.3,
        quality_weight: float = 0.2,
        freshness_weight: float = 0.1,
    ) -> None:
        """
        初始化重排器

        Args:
            top_n: 重排后返回数量
            vector_weight: 向量相似度权重
            attribute_weight: 属性匹配权重
            quality_weight: 质量分权重
            freshness_weight: 时间新鲜度权重
        """
        self.top_n = top_n
        self.vector_weight = vector_weight
        self.attribute_weight = attribute_weight
        self.quality_weight = quality_weight
        self.freshness_weight = freshness_weight

        logger.info(f"候选重排器初始化: top_n={top_n}")

    def rerank(
        self,
        candidates: List[RetrievalCandidate],
        query: QueryResult,
        top_n: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        对候选结果进行重排

        Args:
            candidates: 粗召回候选列表
            query: 解析后的查询结果
            top_n: 返回数量(覆盖默认值)

        Returns:
            重排后的候选列表
        """
        if not candidates:
            return []

        n = top_n or self.top_n
        logger.info(f"开始重排 {len(candidates)} 个候选, 返回 top {n}")

        # 计算每个候选的综合分数
        for candidate in candidates:
            # 1. 向量相似度分数 (已在召回阶段计算)
            vector_score = candidate.text_score

            # 2. 属性匹配分数
            attr_score = self._compute_attribute_score(candidate, query)

            # 3. 质量分数
            quality_score = candidate.instance.quality_score

            # 4. 时间新鲜度分数
            freshness_score = self._compute_freshness_score(candidate)

            # 综合评分
            combined_score = (
                self.vector_weight * vector_score +
                self.attribute_weight * attr_score +
                self.quality_weight * quality_score +
                self.freshness_weight * freshness_score
            )

            candidate.attribute_match_score = attr_score
            candidate.combined_score = combined_score

        # 按综合分数降序排序
        candidates.sort(key=lambda c: c.combined_score, reverse=True)

        # 更新排名
        for rank, candidate in enumerate(candidates[:n]):
            candidate.rank = rank + 1

        result = candidates[:n]
        logger.info(f"重排完成, 返回 {len(result)} 个候选")
        return result

    def rerank_parsed_query(
        self,
        candidates: List[RetrievalCandidate],
        query: ParsedQuery,
        top_n: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        对候选结果进行重排(兼容 ParsedQuery 接口)

        Args:
            candidates: 粗召回候选列表
            query: 解析后的查询
            top_n: 返回数量

        Returns:
            重排后的候选列表
        """
        if not candidates:
            return []

        n = top_n or self.top_n

        for candidate in candidates:
            vector_score = candidate.text_score
            attr_score = self._compute_attribute_score_parsed(candidate, query)
            quality_score = candidate.instance.quality_score
            freshness_score = self._compute_freshness_score(candidate)

            combined_score = (
                self.vector_weight * vector_score +
                self.attribute_weight * attr_score +
                self.quality_weight * quality_score +
                self.freshness_weight * freshness_score
            )

            candidate.attribute_match_score = attr_score
            candidate.combined_score = combined_score

        candidates.sort(key=lambda c: c.combined_score, reverse=True)

        for rank, candidate in enumerate(candidates[:n]):
            candidate.rank = rank + 1

        return candidates[:n]

    def _compute_attribute_score(
        self,
        candidate: RetrievalCandidate,
        query: QueryResult,
    ) -> float:
        """
        计算属性匹配分数

        查询中的属性在候选目标的属性中出现则加分。

        Args:
            candidate: 候选结果
            query: 查询条件

        Returns:
            属性匹配分数 [0, 1]
        """
        instance = candidate.instance
        attrs = instance.attributes
        matches = 0
        total = 0

        # 颜色匹配
        if query.color:
            total += 1
            inst_color = attrs.get("color")
            if inst_color:
                if inst_color == query.color:
                    matches += 1
                elif query.color in inst_color or inst_color in query.color:
                    matches += 0.5  # 部分匹配

        # 车型匹配
        if query.vehicle_type:
            total += 1
            inst_type = attrs.get("vehicle_type")
            if inst_type == query.vehicle_type:
                matches += 1

        # 性别匹配
        if query.gender:
            total += 1
            inst_gender = attrs.get("gender")
            if inst_gender == query.gender:
                matches += 1

        # 背包匹配
        if query.bag is not None:
            total += 1
            inst_bag = attrs.get("bag", False)
            if inst_bag == query.bag:
                matches += 1

        # 背包颜色匹配
        if query.bag_color:
            total += 1
            inst_bag_color = attrs.get("bag_color")
            if inst_bag_color == query.bag_color:
                matches += 1

        # 衣服颜色匹配
        if query.clothing_color:
            total += 1
            inst_clothing = attrs.get("clothing_color")
            if inst_clothing == query.clothing_color:
                matches += 1

        if total == 0:
            return 1.0  # 没有属性条件时返回满分

        return matches / total

    def _compute_attribute_score_parsed(
        self,
        candidate: RetrievalCandidate,
        query: ParsedQuery,
    ) -> float:
        """
        计算属性匹配分数(兼容 ParsedQuery)

        Args:
            candidate: 候选结果
            query: 查询条件

        Returns:
            属性匹配分数 [0, 1]
        """
        instance = candidate.instance
        attrs = instance.attributes
        matches = 0
        total = 0

        for key, value in query.attributes.items():
            total += 1
            inst_value = attrs.get(key)
            if inst_value is not None:
                if inst_value == value:
                    matches += 1
                elif isinstance(value, str) and isinstance(inst_value, str):
                    # 部分字符串匹配
                    if value in inst_value or inst_value in value:
                        matches += 0.5

        if total == 0:
            return 1.0

        return matches / total

    def _compute_freshness_score(
        self,
        candidate: RetrievalCandidate,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        计算时间新鲜度分数

        最近的观测获得更高的分数。

        Args:
            candidate: 候选结果
            reference_time: 参考时间(默认为当前时间)

        Returns:
            新鲜度分数 [0, 1]
        """
        if reference_time is None:
            reference_time = datetime.now()

        instance_time = candidate.instance.timestamp
        time_diff = abs((reference_time - instance_time).total_seconds())

        # 使用指数衰减函数: 1小时内分数接近1, 超过1天分数接近0
        # 时间常数: 3600秒 (1小时)
        import math
        freshness = math.exp(-time_diff / 3600.0)

        return min(1.0, max(0.0, freshness))
