"""
src.stitching.observation_chain - 观测链构建模块

基于跨镜候选边构建观测链:
- 以 Tracklet 为节点、候选边为边构建有向图
- 搜索高置信度的观测链路径
- 生成完整的跨摄像头观测序列
- 输出 TrajectoryResult 供回溯/可视化模块使用

数据流向: CrossCameraEdge → ObservationChain → 回溯/输出模块

核心算法:
    1. 以锚点 tracklet 为起点
    2. 将所有候选边组织为有向图 (邻接表)
    3. 向上游 (时间更早) 和下游 (时间更晚) 分别贪心扩展
    4. 每次选择得分最高的候选连接
    5. 支持最大深度限制
    6. 构建观测节点、观测段、推断段、候选路径
    7. 计算整体置信度并返回 TrajectoryResult
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.common.data_models import (
    CandidatePath,
    CrossCameraEdge,
    InferenceSegment,
    ObservationChain,
    ObservationNode,
    ObservationSegment,
    TargetInstance,
    Tracklet,
    TrajectoryResult,
)
from src.common.logger import get_logger
from src.common.utils import generate_id, time_diff_seconds

logger = get_logger("stitching.observation_chain")


class ObservationChainBuilder:
    """
    观测链构建器

    基于跨镜候选边构建完整的观测链。
    支持以锚点 Tracklet 为中心，向上游和下游贪心扩展，
    生成高置信度的跨镜轨迹。

    使用方式:
        builder = ObservationChainBuilder(tracklets, edges, road_topology)
        result = builder.build_chain(anchor_tracklet)

    也可分步使用:
        builder = ObservationChainBuilder()
        builder.add_tracklets(all_tracklets)
        builder.add_edges(all_edges)
        chains = builder.build_chains(anchor_tracklet_id="TRK_001")
    """

    def __init__(
        self,
        tracklets: Optional[List[Tracklet]] = None,
        edges: Optional[List[CrossCameraEdge]] = None,
        road_topology=None,
        min_chain_confidence: float = 0.4,
        max_upstream_depth: int = 15,
        max_downstream_depth: int = 15,
    ) -> None:
        """
        初始化观测链构建器

        Args:
            tracklets: Tracklet 列表 (可选)
            edges: 候选边列表 (可选)
            road_topology: 道路拓扑 (可选, 用于生成候选路径)
            min_chain_confidence: 最低链置信度
            max_upstream_depth: 最大上游扩展深度
            max_downstream_depth: 最大下游扩展深度
        """
        self.road_topology = road_topology
        self.min_chain_confidence = min_chain_confidence
        self.max_upstream_depth = max_upstream_depth
        self.max_downstream_depth = max_downstream_depth

        # Tracklet 存储 {tracklet_id: Tracklet}
        self._tracklets: Dict[str, Tracklet] = {}

        # 下游邻接表 {source_tracklet_id: [(target_tracklet_id, CrossCameraEdge), ...]}
        self._downstream_adj: Dict[str, List[Tuple[str, CrossCameraEdge]]] = {}
        # 上游邻接表 {target_tracklet_id: [(source_tracklet_id, CrossCameraEdge), ...]}
        self._upstream_adj: Dict[str, List[Tuple[str, CrossCameraEdge]]] = {}

        # 所有有效边
        self._all_edges: List[CrossCameraEdge] = []

        # 如果初始化时提供了数据, 直接加载
        if tracklets:
            self.add_tracklets(tracklets)
        if edges:
            self.add_edges(edges)

        logger.info("观测链构建器初始化 | min_confidence=%.2f", min_chain_confidence)

    # ================================================================
    # 数据加载
    # ================================================================

    def add_tracklets(self, tracklets: List[Tracklet]) -> None:
        """
        添加 Tracklet 节点

        Args:
            tracklets: Tracklet 列表
        """
        for t in tracklets:
            self._tracklets[t.tracklet_id] = t
        logger.debug("添加 %d 个 Tracklet, 总计 %d", len(tracklets), len(self._tracklets))

    def add_edges(self, edges: List[CrossCameraEdge]) -> None:
        """
        添加候选连接边并构建双向邻接表

        下游邻接表: source → [(target, edge), ...]
        上游邻接表: target → [(source, edge), ...]

        邻接表内按得分降序排列, 方便贪心选取。

        Args:
            edges: 有效的候选边列表
        """
        for edge in edges:
            if not edge.is_valid:
                continue

            self._all_edges.append(edge)

            # 下游邻接
            src = edge.source_tracklet_id
            if src not in self._downstream_adj:
                self._downstream_adj[src] = []
            self._downstream_adj[src].append((edge.target_tracklet_id, edge))

            # 上游邻接
            tgt = edge.target_tracklet_id
            if tgt not in self._upstream_adj:
                self._upstream_adj[tgt] = []
            self._upstream_adj[tgt].append((edge.source_tracklet_id, edge))

        # 按得分降序排序, 方便贪心选取最高分边
        for src_id in self._downstream_adj:
            self._downstream_adj[src_id].sort(key=lambda x: x[1].score, reverse=True)
        for tgt_id in self._upstream_adj:
            self._upstream_adj[tgt_id].sort(key=lambda x: x[1].score, reverse=True)

        logger.debug(
            "添加 %d 条有效边 | 下游邻接节点=%d | 上游邻接节点=%d",
            len(self._all_edges),
            len(self._downstream_adj),
            len(self._upstream_adj),
        )

    # ================================================================
    # 核心方法: 构建完整轨迹
    # ================================================================

    def build_chain(
        self,
        anchor_tracklet: Tracklet,
        candidate_edges: Optional[List[CrossCameraEdge]] = None,
        road_topology=None,
        query_id: Optional[str] = None,
        target_instance: Optional[TargetInstance] = None,
    ) -> TrajectoryResult:
        """
        以锚点 Tracklet 为中心构建完整轨迹

        流程:
            1. 加载候选边 (如果提供)
            2. 向上游扩展 (时间更早的 Tracklet)
            3. 向下游扩展 (时间更晚的 Tracklet)
            4. 组合为有序 Tracklet 序列
            5. 构建观测节点、观测段、推断段
            6. 生成候选路径
            7. 计算整体置信度
            8. 返回 TrajectoryResult

        Args:
            anchor_tracklet: 锚点 Tracklet
            candidate_edges: 候选边列表 (可选, 如果未在初始化时提供)
            road_topology: 道路拓扑 (可选)
            query_id: 查询 ID (可选, 自动生成)
            target_instance: 确认的目标实例 (可选, 从锚点推导)

        Returns:
            完整的 TrajectoryResult
        """
        # 加载数据
        if anchor_tracklet.tracklet_id not in self._tracklets:
            self._tracklets[anchor_tracklet.tracklet_id] = anchor_tracklet

        if candidate_edges:
            self.add_edges(candidate_edges)

        topology = road_topology or self.road_topology
        anchor_id = anchor_tracklet.tracklet_id

        logger.info("开始构建观测链 | 锚点=%s", anchor_id)

        # ---- 向上游扩展 ----
        upstream_nodes = self.expand_upstream(
            anchor_tracklet, self._upstream_adj, self.max_upstream_depth
        )

        # ---- 向下游扩展 ----
        downstream_nodes = self.expand_downstream(
            anchor_tracklet, self._downstream_adj, self.max_downstream_depth
        )

        # ---- 组合完整序列 (上游逆序 + 锚点 + 下游正序) ----
        # upstream_nodes 是从锚点往上游走得到的, 需要反转
        upstream_nodes_reversed = list(reversed(upstream_nodes))

        # 构建锚点观测节点
        anchor_node = self._tracklet_to_observation_node(anchor_tracklet, confidence=1.0)

        # 完整有序观测节点列表
        all_nodes = upstream_nodes_reversed + [anchor_node] + downstream_nodes

        # ---- 构建 Tracklet 序列和边映射 ----
        ordered_tracklet_ids = [n.tracklet_id for n in all_nodes]
        edge_map = self._build_edge_map()

        # ---- 构建观测段 ----
        observation_segments = self._build_observation_segments(all_nodes)

        # ---- 构建推断段 ----
        inference_segments = self._build_inference_segments(
            all_nodes, edge_map
        )

        # ---- 生成候选路径 ----
        candidate_paths = self._generate_candidate_paths_for_chain(
            all_nodes, topology
        )

        # ---- 计算整体置信度 ----
        overall_confidence = self._calculate_overall_confidence(
            all_nodes, edge_map
        )

        # ---- 收集拼接证据 ----
        evidence = self._collect_evidence(all_nodes, edge_map)

        # ---- 确定目标实例 ----
        if target_instance is None and anchor_tracklet.instances:
            target_instance = anchor_tracklet.instances[0]
        elif target_instance is None:
            # 构造一个最小实例
            target_instance = TargetInstance(
                instance_id="",
                camera_id=anchor_tracklet.camera_id,
                timestamp=anchor_tracklet.start_time,
                frame_id=0,
                target_type=anchor_tracklet.target_type,
                bbox=None,
                attributes=anchor_tracklet.attributes,
                plate_number=anchor_tracklet.plate_number,
                plate_confidence=0.0,
                quality_score=0.0,
                reid_vector=anchor_tracklet.avg_reid_vector,
                clip_vector=anchor_tracklet.avg_clip_vector,
                keyframe_path=anchor_tracklet.keyframe_paths[0] if anchor_tracklet.keyframe_paths else None,
            )

        result = TrajectoryResult(
            query_id=query_id or generate_id("QUERY"),
            target_instance=target_instance,
            observation_nodes=all_nodes,
            observation_segments=observation_segments,
            inference_segments=inference_segments,
            candidate_paths=candidate_paths,
            evidence=evidence,
            overall_confidence=overall_confidence,
        )

        logger.info(
            "观测链构建完成 | 锚点=%s | 节点数=%d | 置信度=%.4f | 摄像头序列=%s",
            anchor_id, len(all_nodes), overall_confidence,
            result.camera_sequence,
        )

        return result

    # ================================================================
    # 上游/下游扩展
    # ================================================================

    def expand_upstream(
        self,
        tracklet: Tracklet,
        edges: Optional[Dict] = None,
        max_depth: int = 10,
    ) -> List[ObservationNode]:
        """
        向上游 (时间更早) 贪心扩展

        从锚点出发, 每次选择得分最高的上游连接, 直到:
            - 达到最大深度
            - 没有更多上游连接
            - 遇到已访问的摄像头 (避免环路)

        Args:
            tracklet: 起始 Tracklet
            edges: 上游邻接表 (可选, 默认使用内部)
            max_depth: 最大扩展深度

        Returns:
            上游观测节点列表 (从锚点向外, 时间递减顺序)
        """
        upstream_adj = edges or self._upstream_adj
        return self._greedy_expand(
            start_tracklet_id=tracklet.tracklet_id,
            adjacency=upstream_adj,
            max_depth=max_depth,
            direction="upstream",
        )

    def expand_downstream(
        self,
        tracklet: Tracklet,
        edges: Optional[Dict] = None,
        max_depth: int = 10,
    ) -> List[ObservationNode]:
        """
        向下游 (时间更晚) 贪心扩展

        从锚点出发, 每次选择得分最高的下游连接。

        Args:
            tracklet: 起始 Tracklet
            edges: 下游邻接表 (可选, 默认使用内部)
            max_depth: 最大扩展深度

        Returns:
            下游观测节点列表 (从锚点向外, 时间递增顺序)
        """
        downstream_adj = edges or self._downstream_adj
        return self._greedy_expand(
            start_tracklet_id=tracklet.tracklet_id,
            adjacency=downstream_adj,
            max_depth=max_depth,
            direction="downstream",
        )

    def _greedy_expand(
        self,
        start_tracklet_id: str,
        adjacency: Dict[str, List[Tuple[str, CrossCameraEdge]]],
        max_depth: int,
        direction: str,
    ) -> List[ObservationNode]:
        """
        贪心扩展算法

        从起始 Tracklet 出发, 沿邻接表每次选择得分最高的边,
        避免重复访问同一摄像头。

        Args:
            start_tracklet_id: 起始 Tracklet ID
            adjacency: 邻接表
            max_depth: 最大深度
            direction: "upstream" 或 "downstream"

        Returns:
            扩展得到的观测节点列表
        """
        nodes: List[ObservationNode] = []
        visited_cameras: Set[str] = set()
        visited_tracklets: Set[str] = {start_tracklet_id}
        current_id = start_tracklet_id

        # 记录起始摄像头
        start_tracklet = self._tracklets.get(start_tracklet_id)
        if start_tracklet:
            visited_cameras.add(start_tracklet.camera_id)

        for depth in range(max_depth):
            neighbors = adjacency.get(current_id, [])
            if not neighbors:
                break

            # 选择得分最高且未访问过摄像头的邻居
            best_next_id = None
            best_edge = None

            for neighbor_id, edge in neighbors:
                neighbor_tracklet = self._tracklets.get(neighbor_id)
                if neighbor_tracklet is None:
                    continue
                if neighbor_id in visited_tracklets:
                    continue
                # 允许同摄像头不同 Tracklet (但避免环路)
                if neighbor_tracklet.camera_id in visited_cameras:
                    # 如果方向是向时间前进/后退, 同摄像头也可能合理
                    # 但为简化, 我们跳过已访问摄像头
                    continue

                best_next_id = neighbor_id
                best_edge = edge
                break  # 邻接表已按得分排序, 第一个通过的就是最优

            if best_next_id is None or best_edge is None:
                break

            # 记录节点
            next_tracklet = self._tracklets[best_next_id]
            node = self._tracklet_to_observation_node(
                next_tracklet, confidence=best_edge.score
            )
            nodes.append(node)

            # 更新已访问集合
            visited_tracklets.add(best_next_id)
            visited_cameras.add(next_tracklet.camera_id)
            current_id = best_next_id

        logger.debug(
            "贪心扩展 (%s) | 起点=%s | 深度=%d/%d | 扩展节点=%d",
            direction, start_tracklet_id, len(nodes), max_depth, len(nodes),
        )
        return nodes

    # ================================================================
    # 兼容原有接口: build_chains 返回 ObservationChain 列表
    # ================================================================

    def build_chains(
        self,
        anchor_tracklet_id: str,
        max_upstream_depth: int = 15,
        max_downstream_depth: int = 15,
    ) -> List[ObservationChain]:
        """
        以锚点 Tracklet 为中心构建观测链 (返回 ObservationChain 格式)

        分别向上游和下游搜索，生成候选观测链。

        Args:
            anchor_tracklet_id: 锚点 Tracklet ID
            max_upstream_depth: 最大上游搜索深度
            max_downstream_depth: 最大下游搜索深度

        Returns:
            观测链列表(按置信度排序)
        """
        anchor = self._tracklets.get(anchor_tracklet_id)
        if anchor is None:
            logger.warning("锚点 Tracklet 不存在: %s", anchor_tracklet_id)
            return []

        # 向上游搜索
        upstream_paths = self._search_upstream(anchor_tracklet_id, max_upstream_depth)
        # 向下游搜索
        downstream_paths = self._search_downstream(anchor_tracklet_id, max_downstream_depth)

        chains: List[ObservationChain] = []

        # 组合上下游路径
        for up_path in (upstream_paths if upstream_paths else [[]]):
            for down_path in (downstream_paths if downstream_paths else [[]]):
                # 完整路径: 上游逆序 + 锚点 + 下游
                ordered_ids = list(reversed(up_path)) + [anchor_tracklet_id] + down_path

                # 收集边
                chain_edges = self._collect_edges_for_path(ordered_ids)

                # 计算置信度
                if chain_edges:
                    total_conf = sum(e.score for e in chain_edges) / len(chain_edges)
                else:
                    total_conf = 1.0

                if total_conf < self.min_chain_confidence:
                    continue

                # 摄像头序列
                cam_seq = []
                for tid in ordered_ids:
                    t = self._tracklets.get(tid)
                    if t:
                        cam_seq.append(t.camera_id)

                # 计算总时长
                first_t = self._tracklets.get(ordered_ids[0])
                last_t = self._tracklets.get(ordered_ids[-1])
                if first_t and last_t:
                    total_duration = time_diff_seconds(first_t.start_time, last_t.end_time)
                else:
                    total_duration = 0.0

                # 获取锚点实例 ID
                anchor_inst_id = ""
                if anchor.instances:
                    anchor_inst_id = anchor.instances[0].instance_id

                chain = ObservationChain(
                    chain_id=generate_id("CHAIN"),
                    anchor_tracklet_id=anchor_tracklet_id,
                    anchor_instance_id=anchor_inst_id,
                    ordered_tracklets=ordered_ids,
                    edges=chain_edges,
                    total_confidence=total_conf,
                    total_duration_seconds=total_duration,
                    camera_sequence=cam_seq,
                )
                chains.append(chain)

        # 按置信度降序排序
        chains.sort(key=lambda c: c.total_confidence, reverse=True)

        logger.info(
            "构建 %d 条观测链 | 锚点=%s | 上游路径=%d | 下游路径=%d",
            len(chains), anchor_tracklet_id,
            len(upstream_paths), len(downstream_paths),
        )
        return chains

    # ================================================================
    # 图搜索 (DFS)
    # ================================================================

    def _search_upstream(
        self,
        tracklet_id: str,
        max_depth: int,
    ) -> List[List[str]]:
        """
        向上游 DFS 搜索所有候选路径

        Args:
            tracklet_id: 起始 Tracklet ID
            max_depth: 最大搜索深度

        Returns:
            路径列表，每条路径为 Tracklet ID 序列 (不含起始点)
        """
        paths: List[List[str]] = []
        visited: Set[str] = {tracklet_id}
        self._dfs_upstream(tracklet_id, max_depth, visited, [], paths)
        return paths

    def _dfs_upstream(
        self,
        current_id: str,
        remaining_depth: int,
        visited: Set[str],
        current_path: List[str],
        all_paths: List[List[str]],
    ) -> None:
        """DFS 上游搜索递归"""
        if remaining_depth <= 0:
            if current_path:
                all_paths.append(list(current_path))
            return

        neighbors = self._upstream_adj.get(current_id, [])
        if not neighbors:
            if current_path:
                all_paths.append(list(current_path))
            return

        has_valid_extension = False
        for neighbor_id, edge in neighbors:
            if neighbor_id in visited:
                continue
            neighbor_tracklet = self._tracklets.get(neighbor_id)
            if neighbor_tracklet is None:
                continue

            has_valid_extension = True
            visited.add(neighbor_id)
            current_path.append(neighbor_id)

            self._dfs_upstream(
                neighbor_id, remaining_depth - 1, visited, current_path, all_paths
            )

            current_path.pop()
            visited.remove(neighbor_id)

        if not has_valid_extension and current_path:
            all_paths.append(list(current_path))

    def _search_downstream(
        self,
        tracklet_id: str,
        max_depth: int,
    ) -> List[List[str]]:
        """
        向下游 DFS 搜索所有候选路径

        Args:
            tracklet_id: 起始 Tracklet ID
            max_depth: 最大搜索深度

        Returns:
            路径列表，每条路径为 Tracklet ID 序列 (不含起始点)
        """
        paths: List[List[str]] = []
        visited: Set[str] = {tracklet_id}
        self._dfs_downstream(tracklet_id, max_depth, visited, [], paths)
        return paths

    def _dfs_downstream(
        self,
        current_id: str,
        remaining_depth: int,
        visited: Set[str],
        current_path: List[str],
        all_paths: List[List[str]],
    ) -> None:
        """DFS 下游搜索递归"""
        if remaining_depth <= 0:
            if current_path:
                all_paths.append(list(current_path))
            return

        neighbors = self._downstream_adj.get(current_id, [])
        if not neighbors:
            if current_path:
                all_paths.append(list(current_path))
            return

        has_valid_extension = False
        for neighbor_id, edge in neighbors:
            if neighbor_id in visited:
                continue
            neighbor_tracklet = self._tracklets.get(neighbor_id)
            if neighbor_tracklet is None:
                continue

            has_valid_extension = True
            visited.add(neighbor_id)
            current_path.append(neighbor_id)

            self._dfs_downstream(
                neighbor_id, remaining_depth - 1, visited, current_path, all_paths
            )

            current_path.pop()
            visited.remove(neighbor_id)

        if not has_valid_extension and current_path:
            all_paths.append(list(current_path))

    # ================================================================
    # 辅助构建方法
    # ================================================================

    def _build_edge_graph(
        self, edges: List[CrossCameraEdge]
    ) -> Dict[str, List[Tuple[str, CrossCameraEdge]]]:
        """
        从边列表构建下游邻接表

        Args:
            edges: 候选边列表

        Returns:
            下游邻接表 {source_id: [(target_id, edge), ...]}
        """
        graph: Dict[str, List[Tuple[str, CrossCameraEdge]]] = {}
        for edge in edges:
            src = edge.source_tracklet_id
            if src not in graph:
                graph[src] = []
            graph[src].append((edge.target_tracklet_id, edge))
        # 按得分排序
        for src_id in graph:
            graph[src_id].sort(key=lambda x: x[1].score, reverse=True)
        return graph

    def _find_best_path(
        self,
        graph: Dict[str, List[Tuple[str, CrossCameraEdge]]],
        start: str,
        end: str,
    ) -> Optional[List[str]]:
        """
        在候选图中找最优路径 (BFS + 最高得分优先)

        Args:
            graph: 邻接表
            start: 起始 Tracklet ID
            end: 目标 Tracklet ID

        Returns:
            最优路径的 Tracklet ID 序列, 不可达返回 None
        """
        if start == end:
            return [start]

        # BFS, 记录路径和累积得分
        queue: List[Tuple[str, List[str], float]] = [(start, [start], 0.0)]
        visited: Set[str] = {start}
        best_path: Optional[List[str]] = None
        best_score: float = -1.0

        while queue:
            current, path, score = queue.pop(0)
            if current == end:
                if score > best_score:
                    best_score = score
                    best_path = list(path)
                continue

            for neighbor_id, edge in graph.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((
                        neighbor_id,
                        path + [neighbor_id],
                        score + edge.score,
                    ))

        return best_path

    def _build_edge_map(self) -> Dict[Tuple[str, str], CrossCameraEdge]:
        """
        构建 (source_id, target_id) → CrossCameraEdge 的映射

        Returns:
            边映射字典
        """
        edge_map: Dict[Tuple[str, str], CrossCameraEdge] = {}
        for edge in self._all_edges:
            key = (edge.source_tracklet_id, edge.target_tracklet_id)
            # 保留得分最高的边
            if key not in edge_map or edge.score > edge_map[key].score:
                edge_map[key] = edge
        return edge_map

    def _tracklet_to_observation_node(
        self, tracklet: Tracklet, confidence: float = 1.0
    ) -> ObservationNode:
        """
        将 Tracklet 转换为观测节点

        Args:
            tracklet: 源 Tracklet
            confidence: 观测置信度

        Returns:
            ObservationNode
        """
        # 选择关键帧: 优先使用第一个
        keyframe = tracklet.keyframe_paths[0] if tracklet.keyframe_paths else ""

        return ObservationNode(
            camera_id=tracklet.camera_id,
            tracklet_id=tracklet.tracklet_id,
            timestamp=tracklet.start_time,
            keyframe_path=keyframe,
            confidence=confidence,
        )

    def _build_observation_segments(
        self, nodes: List[ObservationNode]
    ) -> List[ObservationSegment]:
        """
        从有序观测节点列表构建观测段

        每个观测节点对应一个观测段 (即该 Tracklet 在摄像头内的运动片段)。

        Args:
            nodes: 有序观测节点列表

        Returns:
            观测段列表
        """
        segments: List[ObservationSegment] = []

        for node in nodes:
            tracklet = self._tracklets.get(node.tracklet_id)
            if tracklet is None:
                continue

            # 生成进入/离开描述
            entry_desc = self._describe_entry(tracklet)
            exit_desc = self._describe_exit(tracklet)

            seg = ObservationSegment(
                tracklet_id=tracklet.tracklet_id,
                camera_id=tracklet.camera_id,
                start_time=tracklet.start_time,
                end_time=tracklet.end_time,
                entry_description=entry_desc,
                exit_description=exit_desc,
                direction=tracklet.direction,
            )
            segments.append(seg)

        return segments

    def _build_inference_segments(
        self,
        nodes: List[ObservationNode],
        edge_map: Dict[Tuple[str, str], CrossCameraEdge],
    ) -> List[InferenceSegment]:
        """
        从有序观测节点列表构建推断段

        每对相邻观测节点之间如果存在边, 则生成一个推断段。

        Args:
            nodes: 有序观测节点列表
            edge_map: 边映射

        Returns:
            推断段列表
        """
        segments: List[InferenceSegment] = []

        for i in range(len(nodes) - 1):
            src_node = nodes[i]
            tgt_node = nodes[i + 1]

            edge = edge_map.get((src_node.tracklet_id, tgt_node.tracklet_id))
            if edge is None:
                # 尝试反向 (上游边)
                edge = edge_map.get((tgt_node.tracklet_id, src_node.tracklet_id))

            src_tracklet = self._tracklets.get(src_node.tracklet_id)
            tgt_tracklet = self._tracklets.get(tgt_node.tracklet_id)

            if src_tracklet is None or tgt_tracklet is None:
                continue

            # 实际旅行时间
            actual_time = time_diff_seconds(
                src_tracklet.end_time, tgt_tracklet.start_time
            )

            # 估算旅行时间
            estimated_time = self._estimate_travel_time_between(
                src_tracklet.camera_id, tgt_tracklet.camera_id
            )

            # 路径描述
            route_desc = f"{src_tracklet.camera_id} → {tgt_tracklet.camera_id}"
            if edge:
                route_desc += f" (score={edge.score:.3f})"

            seg = InferenceSegment(
                source_camera_id=src_tracklet.camera_id,
                target_camera_id=tgt_tracklet.camera_id,
                source_tracklet_id=src_tracklet.tracklet_id,
                target_tracklet_id=tgt_tracklet.tracklet_id,
                confidence=edge.score if edge else 0.5,
                estimated_travel_time=estimated_time,
                actual_travel_time=actual_time,
                route_description=route_desc,
            )
            segments.append(seg)

        return segments

    def _generate_candidate_paths_for_chain(
        self,
        nodes: List[ObservationNode],
        topology,
    ) -> List[CandidatePath]:
        """
        为整条链生成候选路网路径

        对每对相邻摄像头, 尝试从拓扑中获取候选路径。

        Args:
            nodes: 有序观测节点列表
            topology: 道路拓扑

        Returns:
            候选路径列表
        """
        paths: List[CandidatePath] = []

        for i in range(len(nodes) - 1):
            src_cam = nodes[i].camera_id
            tgt_cam = nodes[i + 1].camera_id

            sub_paths = self._generate_candidate_paths(src_cam, tgt_cam, topology)
            paths.extend(sub_paths)

        return paths

    def _generate_candidate_paths(
        self,
        source_cam: str,
        target_cam: str,
        topology,
    ) -> List[CandidatePath]:
        """
        生成两个摄像头之间的候选路网路径

        如果拓扑有多条路径, 返回多条候选。

        Args:
            source_cam: 源摄像头 ID
            target_cam: 目标摄像头 ID
            topology: 道路拓扑

        Returns:
            候选路径列表
        """
        if source_cam == target_cam:
            return []

        candidate_paths: List[CandidatePath] = []

        # 尝试从拓扑获取最短路径
        try:
            path = topology.shortest_path(source_cam, target_cam)
            if path and len(path) >= 2:
                # 计算距离和时间
                distance = self._get_path_distance(path, topology)
                estimated_time = self._estimate_time_from_distance(distance)

                candidate_paths.append(CandidatePath(
                    path_id=generate_id("PATH"),
                    road_segments=path,
                    confidence=0.8 if len(path) == 2 else 0.6,
                    distance_meters=distance,
                    estimated_time=estimated_time,
                ))
        except (NotImplementedError, AttributeError):
            # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
            # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
            logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
            pass

        # 如果没有拓扑信息, 基于直线距离估算
        if not candidate_paths:
            distance = self._get_camera_distance_direct(source_cam, target_cam)
            if distance and distance > 0:
                estimated_time = self._estimate_time_from_distance(distance)
                candidate_paths.append(CandidatePath(
                    path_id=generate_id("PATH"),
                    road_segments=[source_cam, target_cam],
                    confidence=0.4,  # 无拓扑信息, 置信度较低
                    distance_meters=distance,
                    estimated_time=estimated_time,
                ))

        return candidate_paths

    def _calculate_overall_confidence(
        self,
        nodes: List[ObservationNode],
        edge_map: Dict[Tuple[str, str], CrossCameraEdge],
    ) -> float:
        """
        计算整体轨迹置信度

        使用所有连接边得分的加权几何平均:
            confidence = exp(Σ(log(score_i)) / n)
        如果只有 1 个节点 (无边), 置信度为 1.0。

        Args:
            nodes: 有序观测节点列表
            edge_map: 边映射

        Returns:
            整体置信度 [0, 1]
        """
        if len(nodes) <= 1:
            return 1.0

        scores: List[float] = []
        for i in range(len(nodes) - 1):
            src_id = nodes[i].tracklet_id
            tgt_id = nodes[i + 1].tracklet_id

            edge = edge_map.get((src_id, tgt_id))
            if edge is None:
                edge = edge_map.get((tgt_id, src_id))

            if edge is not None:
                scores.append(max(edge.score, 0.01))  # 避免 log(0)
            else:
                scores.append(0.3)  # 无边信息时给较低分

        if not scores:
            return 0.5

        # 几何平均
        log_sum = sum(math.log(s) for s in scores)
        confidence = math.exp(log_sum / len(scores))

        return float(max(0.0, min(1.0, confidence)))

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _collect_edges_for_path(
        self, ordered_ids: List[str]
    ) -> List[CrossCameraEdge]:
        """
        收集一条路径上所有连接边

        Args:
            ordered_ids: 有序 Tracklet ID 序列

        Returns:
            边列表
        """
        edges: List[CrossCameraEdge] = []
        edge_map = self._build_edge_map()

        for i in range(len(ordered_ids) - 1):
            src_id = ordered_ids[i]
            tgt_id = ordered_ids[i + 1]

            edge = edge_map.get((src_id, tgt_id))
            if edge is None:
                edge = edge_map.get((tgt_id, src_id))
            if edge is not None:
                edges.append(edge)

        return edges

    def _describe_entry(self, tracklet: Tracklet) -> str:
        """生成进入画面描述"""
        if tracklet.entry_point:
            x, y = tracklet.entry_point
            if x < 0.33:
                return "从画面左侧进入"
            elif x < 0.66:
                return "从画面中部进入"
            else:
                return "从画面右侧进入"
        return f"进入 {tracklet.camera_id} 画面"

    def _describe_exit(self, tracklet: Tracklet) -> str:
        """生成离开画面描述"""
        if tracklet.exit_point:
            x, y = tracklet.exit_point
            if x < 0.33:
                return "从画面左侧离开"
            elif x < 0.66:
                return "从画面中部离开"
            else:
                return "从画面右侧离开"
        return f"离开 {tracklet.camera_id} 画面"

    def _estimate_travel_time_between(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> float:
        """
        估算两个摄像头之间的旅行时间 (秒)

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            估算旅行时间 (秒)
        """
        distance = self._get_camera_distance_direct(src_camera_id, tgt_camera_id)
        if distance is None or distance <= 0:
            return 60.0  # 默认 60 秒
        return self._estimate_time_from_distance(distance)

    def _estimate_time_from_distance(
        self, distance_meters: float, avg_speed_kmh: float = 40.0
    ) -> float:
        """
        基于距离和平均速度估算旅行时间

        Args:
            distance_meters: 距离 (米)
            avg_speed_kmh: 平均速度 (km/h)

        Returns:
            估算时间 (秒)
        """
        if distance_meters <= 0:
            return 0.0
        speed_m_per_s = avg_speed_kmh * 1000.0 / 3600.0
        return distance_meters / speed_m_per_s

    def _get_camera_distance_direct(
        self, src_camera_id: str, tgt_camera_id: str
    ) -> Optional[float]:
        """
        获取两个摄像头之间的直线距离

        Args:
            src_camera_id: 源摄像头 ID
            tgt_camera_id: 目标摄像头 ID

        Returns:
            距离 (米)
        """
        try:
            from src.common.utils import haversine_distance
            src_cam = None
            tgt_cam = None

            # 从 tracklets 中找摄像头信息 (通过 CameraManager)
            # 这里尝试从 road_topology 获取
            if self.road_topology:
                try:
                    dist = self.road_topology.get_segment_distance(
                        src_camera_id, tgt_camera_id
                    )
                    if dist is not None:
                        return dist
                except (NotImplementedError, AttributeError):
                    # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
                    # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
                    logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
                    pass

        except Exception:
            pass

        return None

    def _get_path_distance(
        self, path: List[str], topology
    ) -> float:
        """
        计算路径总距离

        Args:
            path: 摄像头 ID 序列
            topology: 道路拓扑

        Returns:
            总距离 (米)
        """
        total = 0.0
        for i in range(len(path) - 1):
            try:
                d = topology.get_segment_distance(path[i], path[i + 1])
                if d is not None:
                    total += d
            except (NotImplementedError, AttributeError):
                # 不再静默：原来的 pass/宽松兜底会掩盖"上游方法被改名"这类问题，
                # exc_info 直接把调用栈写进日志，不依赖每处手写消息。
                logger.debug("上游方法不可用（未实现或不存在），走宽松兜底", exc_info=True)
                # 回退: 使用 Haversine
                d = self._get_camera_distance_direct(path[i], path[i + 1])
                if d is not None:
                    total += d
        return total

    def _collect_evidence(
        self,
        nodes: List[ObservationNode],
        edge_map: Dict[Tuple[str, str], CrossCameraEdge],
    ) -> Dict[str, any]:
        """
        收集拼接证据

        包括车牌、属性、ReID 相似度、旅行时间等。

        Args:
            nodes: 有序观测节点列表
            edge_map: 边映射

        Returns:
            证据字典
        """
        evidence: Dict[str, any] = {
            "plate_matches": [],
            "attribute_consistency": [],
            "reid_scores": [],
            "travel_times": [],
            "camera_transitions": [],
        }

        for i in range(len(nodes) - 1):
            src_id = nodes[i].tracklet_id
            tgt_id = nodes[i + 1].tracklet_id

            edge = edge_map.get((src_id, tgt_id))
            if edge is None:
                edge = edge_map.get((tgt_id, src_id))

            if edge is not None:
                evidence["reid_scores"].append(edge.appearance_score)
                evidence["attribute_consistency"].append(edge.attribute_score)
                evidence["plate_matches"].append(edge.plate_score)

                src_t = self._tracklets.get(src_id)
                tgt_t = self._tracklets.get(tgt_id)
                if src_t and tgt_t:
                    travel_time = time_diff_seconds(src_t.end_time, tgt_t.start_time)
                    evidence["travel_times"].append(travel_time)
                    evidence["camera_transitions"].append({
                        "from": src_t.camera_id,
                        "to": tgt_t.camera_id,
                        "score": edge.score,
                        "travel_time": travel_time,
                    })

        return evidence
