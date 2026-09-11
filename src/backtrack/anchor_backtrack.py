"""
src.backtrack.anchor_backtrack - 锚点回溯模块

用户确认目标后，以该目标实例为锚点在跨镜候选图中进行回溯:
1. 查找锚点所属 Tracklet
2. 在跨镜候选图中查找与该 Tracklet 相连的所有边
3. 向上游和下游扩展观测链

数据流向: 用户确认 → AnchorBacktracker → 观测链列表
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.common.data_models import ObservationChain, TargetInstance, Tracklet
from src.common.logger import get_logger
from src.stitching.observation_chain import ObservationChainBuilder

logger = get_logger("backtrack.anchor_backtrack")


class AnchorBacktracker:
    """
    锚点回溯器

    以用户确认的目标为锚点，在跨镜候选图中回溯观测链。

    使用方式:
        backtracker = AnchorBacktracker(chain_builder)
        chains = backtracker.backtrack(confirmed_instance)
    """

    def __init__(
        self,
        chain_builder: ObservationChainBuilder,
        track_manager=None,
        min_confidence: float = 0.5,
    ) -> None:
        """
        初始化锚点回溯器

        Args:
            chain_builder: 观测链构建器
            track_manager: 轨迹管理器(用于查找 Tracklet)
            min_confidence: 最低置信度阈值
        """
        self.chain_builder = chain_builder
        self.track_manager = track_manager
        self.min_confidence = min_confidence

        # 实例到 Tracklet 的映射缓存
        self._instance_to_tracklet: Dict[str, Tracklet] = {}

        logger.info(f"锚点回溯器初始化, min_confidence={min_confidence}")

    def backtrack(
        self,
        anchor_instance: TargetInstance,
        max_upstream: int = 10,
        max_downstream: int = 10,
    ) -> List[ObservationChain]:
        """
        以锚点实例进行轨迹回溯

        Args:
            anchor_instance: 用户确认的目标实例
            max_upstream: 最大上游回溯深度
            max_downstream: 最大下游回溯深度

        Returns:
            观测链列表(按置信度排序)
        """
        logger.info(f"开始锚点回溯: instance_id={anchor_instance.instance_id}")

        # 1. 查找锚点实例所属的 Tracklet
        anchor_tracklet = self._find_anchor_tracklet(anchor_instance)
        if anchor_tracklet is None:
            logger.warning(f"未找到锚点实例所属的 Tracklet: {anchor_instance.instance_id}")
            return []

        logger.info(f"找到锚点 Tracklet: {anchor_tracklet.tracklet_id}")

        # 2. 以该 Tracklet 为锚点调用 chain_builder
        chains = self.chain_builder.build_chains(
            anchor_tracklet_id=anchor_tracklet.tracklet_id,
            max_upstream_depth=max_upstream,
            max_downstream_depth=max_downstream,
        )

        # 3. 过滤低置信度链
        filtered_chains = [
            chain for chain in chains
            if chain.total_confidence >= self.min_confidence
        ]

        # 4. 按置信度排序
        filtered_chains.sort(key=lambda c: c.total_confidence, reverse=True)

        logger.info(
            f"回溯完成: 生成 {len(chains)} 条观测链, "
            f"过滤后保留 {len(filtered_chains)} 条"
        )

        return filtered_chains

    def backtrack_multiple(
        self,
        anchor_instances: List[TargetInstance],
        max_upstream: int = 10,
        max_downstream: int = 10,
    ) -> List[ObservationChain]:
        """
        对多个锚点实例进行回溯

        用户可能确认多个目标，需要分别回溯后合并结果。

        Args:
            anchor_instances: 用户确认的目标实例列表
            max_upstream: 最大上游回溯深度
            max_downstream: 最大下游回溯深度

        Returns:
            所有观测链列表(按置信度排序)
        """
        all_chains = []

        for instance in anchor_instances:
            chains = self.backtrack(instance, max_upstream, max_downstream)
            all_chains.extend(chains)

        # 去重并排序
        seen_chain_ids = set()
        unique_chains = []
        for chain in all_chains:
            if chain.chain_id not in seen_chain_ids:
                seen_chain_ids.add(chain.chain_id)
                unique_chains.append(chain)

        unique_chains.sort(key=lambda c: c.total_confidence, reverse=True)

        logger.info(f"多锚点回溯完成: 共 {len(unique_chains)} 条唯一观测链")
        return unique_chains

    def _find_anchor_tracklet(
        self,
        instance: TargetInstance,
    ) -> Optional[Tracklet]:
        """
        查找目标实例所属的 Tracklet

        搜索策略:
        1. 先查缓存
        2. 通过 track_manager 查找
        3. 遍历所有 Tracklet 查找

        Args:
            instance: 目标实例

        Returns:
            所属 Tracklet 或 None
        """
        # 1. 查缓存
        if instance.instance_id in self._instance_to_tracklet:
            return self._instance_to_tracklet[instance.instance_id]

        # 2. 通过 track_manager 查找
        if self.track_manager is not None:
            # 尝试通过 camera_id 查找该摄像头下的 Tracklets
            tracklets = self.track_manager.get_tracklets_by_camera(
                camera_id=instance.camera_id
            )
            for tracklet in tracklets:
                # 检查实例是否在该 Tracklet 中
                for inst in tracklet.instances:
                    if inst.instance_id == instance.instance_id:
                        self._instance_to_tracklet[instance.instance_id] = tracklet
                        return tracklet
                # 也可以通过时间戳判断
                if (tracklet.start_time <= instance.timestamp <= tracklet.end_time and
                        tracklet.camera_id == instance.camera_id):
                    self._instance_to_tracklet[instance.instance_id] = tracklet
                    return tracklet

        # 3. 遍历 chain_builder 中的 Tracklets
        for tracklet_id, tracklet in self.chain_builder._tracklets.items():
            for inst in tracklet.instances:
                if inst.instance_id == instance.instance_id:
                    self._instance_to_tracklet[instance.instance_id] = tracklet
                    return tracklet

        logger.warning(f"未找到实例 {instance.instance_id} 所属的 Tracklet")
        return None

    def register_instance_tracklet(self, instance_id: str, tracklet: Tracklet) -> None:
        """
        手动注册实例到 Tracklet 的映射

        用于外部已知道映射关系的场景。

        Args:
            instance_id: 实例 ID
            tracklet: 对应的 Tracklet
        """
        self._instance_to_tracklet[instance_id] = tracklet
        logger.debug(f"注册实例-Tracklet映射: {instance_id} -> {tracklet.tracklet_id}")
