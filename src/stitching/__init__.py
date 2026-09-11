"""
src.stitching - 跨镜轨迹拼接模块
负责候选边生成、跨镜连接评分和观测链构建。
"""

from src.stitching.candidate_edge import CandidateEdgeGenerator
from src.stitching.scoring import CrossCameraScorer
from src.stitching.observation_chain import ObservationChainBuilder

__all__ = ["CandidateEdgeGenerator", "CrossCameraScorer", "ObservationChainBuilder"]
