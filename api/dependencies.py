"""
api.dependencies - 依赖注入模块

提供 FastAPI 路由所需的依赖注入工厂函数。
每个组件延迟初始化并缓存到全局 _services 字典中。
"""

from __future__ import annotations

from typing import Any, Dict

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("api.dependencies")

# 全局服务实例缓存
_services: Dict[str, Any] = {}


def _get_or_create(key: str, factory):
    """通用延迟初始化+缓存"""
    if key not in _services:
        try:
            _services[key] = factory()
            logger.info(f"组件 [{key}] 初始化成功")
        except Exception as e:
            logger.warning(f"组件 [{key}] 初始化失败: {e}")
            return None
    return _services[key]


def get_camera_manager():
    """获取摄像头管理器实例"""
    def _factory():
        from src.data_governance.camera_manager import CameraManager
        return CameraManager()
    return _get_or_create("camera_manager", _factory)


def get_road_topology():
    """获取道路拓扑实例"""
    def _factory():
        from src.data_governance.road_topology import RoadTopology
        return RoadTopology()
    return _get_or_create("road_topology", _factory)


def get_track_manager():
    """获取轨迹管理器实例"""
    def _factory():
        from src.tracking.track_manager import TrackManager
        return TrackManager()
    return _get_or_create("track_manager", _factory)


def get_feature_extractor():
    """获取特征提取器实例"""
    def _factory():
        from src.perception.feature_extractor import FeatureExtractor
        return FeatureExtractor()
    return _get_or_create("feature_extractor", _factory)


def get_query_parser():
    """获取查询解析器实例"""
    def _factory():
        from src.retrieval.query_parser import QueryParser
        return QueryParser()
    return _get_or_create("query_parser", _factory)


def get_vector_recall():
    """获取向量召回器实例"""
    def _factory():
        from src.retrieval.vector_recall import VectorRecall
        return VectorRecall()
    return _get_or_create("vector_recall", _factory)


def get_observation_chain_builder():
    """获取观测链构建器实例"""
    def _factory():
        from src.stitching.observation_chain import ObservationChainBuilder
        return ObservationChainBuilder()
    return _get_or_create("observation_chain_builder", _factory)
