"""
src.retrieval.vector_recall - 图文向量召回模块

使用 Chinese-CLIP 进行图文跨模态向量检索:
- 将查询文本编码为 CLIP 文本向量
- 在 Qdrant 向量数据库中检索最相似的目标图像向量
- 返回 Top-K 候选结果
- 支持内存回退方案(当 Qdrant 不可用时)

数据流向: ParsedQuery → Qdrant 检索 → RetrievalCandidate 列表
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.data_models import ParsedQuery, RetrievalCandidate, TargetInstance
from src.common.logger import get_logger
from src.common.utils import batch_cosine_similarity, cosine_similarity

logger = get_logger("retrieval.vector_recall")


class VectorRecall:
    """
    图文向量召回器

    基于 Chinese-CLIP 和 Qdrant 向量数据库进行跨模态检索。
    当 Qdrant 服务不可用时，自动回退到内存暴力检索。

    使用方式:
        recall = VectorRecall(host="localhost", port=6333)
        candidates = recall.recall(parsed_query, top_k=20)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "traffic_targets",
        clip_model=None,
        use_memory_fallback: bool = True,
    ) -> None:
        """
        初始化向量召回器

        Args:
            host: Qdrant 服务地址
            port: Qdrant 服务端口
            collection_name: Qdrant 集合名称
            clip_model: Chinese-CLIP 模型实例
            use_memory_fallback: 是否启用内存回退方案
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._clip_model = clip_model
        self._qdrant_client = None
        self._use_memory_fallback = use_memory_fallback

        # 内存索引(回退方案)
        self._memory_vectors: List[np.ndarray] = []  # 存储向量
        self._memory_instances: List[TargetInstance] = []  # 存储实例

        # 延迟初始化标志
        self._initialized = False

        logger.info(f"向量召回器初始化: qdrant={host}:{port}, collection={collection_name}")

    def _ensure_initialized(self) -> None:
        """
        确保召回器已初始化

        延迟初始化 Qdrant 连接和 CLIP 模型。
        """
        if self._initialized:
            return

        # 尝试连接 Qdrant
        try:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(host=self.host, port=self.port)
            # 测试连接
            self._qdrant_client.get_collections()
            logger.info(f"成功连接 Qdrant: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"无法连接 Qdrant: {e}")
            if self._use_memory_fallback:
                logger.info("启用内存回退方案")
                self._qdrant_client = None
            else:
                raise

        self._initialized = True

    def encode_text(self, query: str) -> np.ndarray:
        """
        使用 Chinese-CLIP 编码查询文本

        Args:
            query: 查询文本

        Returns:
            文本向量 (dim=768)

        Raises:
            RuntimeError: 如果 CLIP 模型未加载
        """
        if self._clip_model is None:
            raise RuntimeError("Chinese-CLIP 模型未加载")

        # 假设 clip_model 有 encode_text 方法
        try:
            # 尝试使用 chinese_clip 风格
            vector = self._clip_model.encode_text(query)
            if isinstance(vector, np.ndarray):
                return vector
            return vector.cpu().numpy()
        except AttributeError:
            # 尝试使用 transformers 风格
            try:
                inputs = self._clip_model.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                import torch
                device = next(self._clip_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_features = self._clip_model.get_text_features(**inputs)
                return text_features.cpu().numpy().flatten()
            except Exception as e:
                raise RuntimeError(f"文本编码失败: {e}")

    def recall(
        self,
        query: ParsedQuery,
        top_k: int = 20,
    ) -> List[RetrievalCandidate]:
        """
        执行向量召回

        Args:
            query: 解析后的查询(需包含 CLIP 文本向量)
            top_k: 召回数量

        Returns:
            候选结果列表(按相似度排序)
        """
        self._ensure_initialized()

        # 获取查询向量
        query_vector = query.clip_text_embedding
        if query_vector is None:
            logger.warning("查询向量未提供，尝试实时编码")
            try:
                query_vector = self.encode_text(query.raw_text)
            except RuntimeError as e:
                logger.error(f"无法获取查询向量: {e}")
                return []

        # 执行检索
        if self._qdrant_client is not None:
            return self._recall_qdrant(query_vector, top_k)
        elif self._use_memory_fallback:
            return self._recall_memory(query_vector, top_k)
        else:
            logger.error("无可用检索后端")
            return []

    def _recall_qdrant(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """
        使用 Qdrant 进行向量检索

        Args:
            query_vector: 查询向量
            top_k: 召回数量

        Returns:
            候选结果列表
        """
        try:
            results = self._qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
            )

            candidates = []
            for rank, hit in enumerate(results):
                # 从 payload 中恢复实例信息
                payload = hit.payload
                instance = self._payload_to_instance(payload)
                if instance:
                    candidates.append(RetrievalCandidate(
                        instance=instance,
                        text_score=hit.score,
                        attribute_match_score=0.0,  # 稍后由 reranker 计算
                        combined_score=hit.score,
                        rank=rank + 1,
                    ))

            logger.info(f"Qdrant 召回 {len(candidates)} 个候选")
            return candidates

        except Exception as e:
            logger.error(f"Qdrant 检索失败: {e}")
            if self._use_memory_fallback:
                return self._recall_memory(query_vector, top_k)
            return []

    def _recall_memory(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """
        内存暴力检索(回退方案)

        使用 numpy 计算查询向量与所有存储向量的余弦相似度。

        Args:
            query_vector: 查询向量
            top_k: 召回数量

        Returns:
            候选结果列表
        """
        if not self._memory_vectors:
            logger.warning("内存索引为空，无法检索")
            return []

        # 批量计算余弦相似度
        gallery = np.stack(self._memory_vectors)
        similarities = batch_cosine_similarity(query_vector, gallery)

        # 获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = []
        for rank, idx in enumerate(top_indices):
            instance = self._memory_instances[idx]
            score = float(similarities[idx])
            candidates.append(RetrievalCandidate(
                instance=instance,
                text_score=score,
                attribute_match_score=0.0,
                combined_score=score,
                rank=rank + 1,
            ))

        logger.info(f"内存召回 {len(candidates)} 个候选")
        return candidates

    def recall_by_reid(
        self,
        reid_vector: np.ndarray,
        top_k: int = 20,
    ) -> List[RetrievalCandidate]:
        """
        基于 ReID 向量进行召回(用于跨镜匹配)

        Args:
            reid_vector: ReID 查询向量
            top_k: 召回数量

        Returns:
            候选结果列表
        """
        self._ensure_initialized()

        # ReID 向量使用内存检索
        if not self._memory_vectors:
            logger.warning("内存索引为空，无法进行 ReID 检索")
            return []

        # 过滤有 ReID 向量的实例
        reid_instances = []
        reid_vecs = []
        for inst, vec in zip(self._memory_instances, self._memory_vectors):
            if inst.has_reid:
                reid_instances.append(inst)
                reid_vecs.append(inst.reid_vector)

        if not reid_vecs:
            return []

        gallery = np.stack(reid_vecs)
        similarities = batch_cosine_similarity(reid_vector, gallery)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = []
        for rank, idx in enumerate(top_indices):
            instance = reid_instances[idx]
            score = float(similarities[idx])
            candidates.append(RetrievalCandidate(
                instance=instance,
                text_score=score,
                attribute_match_score=0.0,
                combined_score=score,
                rank=rank + 1,
            ))

        logger.info(f"ReID 召回 {len(candidates)} 个候选")
        return candidates

    def add_to_memory_index(self, instances: List[TargetInstance]) -> None:
        """
        添加实例到内存索引

        用于内存回退方案。

        Args:
            instances: 目标实例列表
        """
        for inst in instances:
            if inst.has_clip:
                self._memory_vectors.append(inst.clip_vector)
                self._memory_instances.append(inst)

        logger.info(f"内存索引已更新, 当前大小: {len(self._memory_vectors)}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        通用向量检索接口

        Args:
            query_vector: 查询向量
            top_k: 召回数量
            filters: 过滤条件

        Returns:
            (instance_id, score) 元组列表
        """
        self._ensure_initialized()

        if self._qdrant_client is not None:
            try:
                results = self._qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k,
                )
                return [(hit.payload.get("instance_id", ""), hit.score) for hit in results]
            except Exception as e:
                logger.error(f"Qdrant 检索失败: {e}")

        # 内存回退
        if self._memory_vectors:
            gallery = np.stack(self._memory_vectors)
            similarities = batch_cosine_similarity(query_vector, gallery)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(self._memory_instances[idx].instance_id, float(similarities[idx])) for idx in top_indices]

        return []

    def _payload_to_instance(self, payload: Dict[str, Any]) -> Optional[TargetInstance]:
        """
        从 Qdrant payload 恢复 TargetInstance

        Args:
            payload: Qdrant 存储的 payload

        Returns:
            TargetInstance 或 None
        """
        try:
            from datetime import datetime
            from src.common.data_models import BoundingBox

            # 解析时间戳
            timestamp_str = payload.get("timestamp", "")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

            # 解析 bbox
            bbox_data = payload.get("bbox", {})
            bbox = BoundingBox(
                x1=bbox_data.get("x1", 0),
                y1=bbox_data.get("y1", 0),
                x2=bbox_data.get("x2", 0),
                y2=bbox_data.get("y2", 0),
                confidence=bbox_data.get("confidence", 0.5),
            )

            return TargetInstance(
                instance_id=payload.get("instance_id", ""),
                camera_id=payload.get("camera_id", ""),
                timestamp=timestamp,
                frame_id=payload.get("frame_id", 0),
                target_type=payload.get("target_type", "vehicle"),
                bbox=bbox,
                attributes=payload.get("attributes", {}),
                plate_number=payload.get("plate_number"),
                plate_confidence=payload.get("plate_confidence", 0.0),
                quality_score=payload.get("quality_score", 0.5),
                reid_vector=None,  # 向量不存储在 payload 中
                clip_vector=None,
                keyframe_path=payload.get("keyframe_path"),
            )
        except Exception as e:
            logger.error(f"解析 payload 失败: {e}")
            return None

    def _ensure_collection(self, vector_dim: int) -> None:
        """
        确保 Qdrant 集合存在

        Args:
            vector_dim: 向量维度
        """
        if self._qdrant_client is None:
            return

        try:
            collections = self._qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                from qdrant_client.models import VectorParams, Distance
                self._qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                )
                logger.info(f"创建 Qdrant 集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"确保集合存在时出错: {e}")
