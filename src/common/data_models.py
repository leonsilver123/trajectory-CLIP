"""
src.common.data_models - 核心数据模型

定义系统中所有模块共享的数据结构。
数据流向: 视频感知 → 单摄轨迹 → 跨镜拼接 → 文本检索 → 轨迹回溯 → 输出

所有模块通过本文件中的数据类进行解耦通信。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 枚举类型
# ============================================================

class TargetType(str, Enum):
    """目标类别枚举"""
    VEHICLE = "vehicle"                 # 机动车
    PEDESTRIAN = "pedestrian"           # 行人
    NON_MOTOR_VEHICLE = "non_motor_vehicle"  # 非机动车


class LaneDirection(str, Enum):
    """车道方向枚举"""
    EASTBOUND = "eastbound"             # 由西向东
    WESTBOUND = "westbound"             # 由东向西
    NORTHBOUND = "northbound"           # 由南向北
    SOUTHBOUND = "southbound"           # 由北向南


class RoadDirection(str, Enum):
    """路段方向枚举"""
    EAST_WEST = "east-west"
    NORTH_SOUTH = "north-south"
    NORTHEAST_SOUTHWEST = "northeast-southwest"
    NORTHWEST_SOUTHEAST = "northwest-southeast"


# ============================================================
# 摄像头与道路元数据
# ============================================================

@dataclass
class CameraMetadata:
    """
    摄像头元数据

    描述一个道路摄像头的基本信息，包括空间位置、朝向和覆盖范围。
    摄像头元数据是跨镜轨迹拼接的基础数据。

    AICity22 数据集说明:
        - latitude/longitude 为可选（数据集使用场景级 PNG 地图而非 per-camera GPS）
        - scene_id 标识摄像头所属场景（如 'S01'）
    """
    camera_id: str                      # 摄像头唯一编号（如 'c001'）
    name: str                           # 摄像头名称(描述性)
    scene_id: Optional[str] = None      # 所属场景 ID（如 'S01'）
    latitude: Optional[float] = None    # 纬度（AICity22 中可为空）
    longitude: Optional[float] = None   # 经度（AICity22 中可为空）
    direction: float = 0.0              # 朝向角度(正北为0°, 顺时针)
    covered_road_segment: str = ""      # 覆盖的路段ID
    lane_direction: str = "eastbound"   # 车道方向(eastbound/westbound/northbound/southbound)

    def distance_to(self, other: CameraMetadata) -> float:
        """
        计算与另一个摄像头之间的直线距离(米)

        使用 Haversine 公式计算球面距离。
        如果任一摄像头缺少 GPS 坐标则抛出异常。

        Args:
            other: 另一个摄像头元数据

        Returns:
            两点之间的直线距离(米)
        """
        if self.latitude is None or self.longitude is None:
            raise ValueError(f"摄像头 {self.camera_id} 缺少 GPS 坐标，无法计算距离")
        if other.latitude is None or other.longitude is None:
            raise ValueError(f"摄像头 {other.camera_id} 缺少 GPS 坐标，无法计算距离")
        from src.common.utils import haversine_distance
        return haversine_distance(self.latitude, self.longitude, other.latitude, other.longitude)

    def is_opposite_direction(self, other: CameraMetadata) -> bool:
        """判断两个摄像头是否方向相反"""
        opposite_map = {
            "eastbound": "westbound",
            "westbound": "eastbound",
            "northbound": "southbound",
            "southbound": "northbound",
        }
        return opposite_map.get(self.lane_direction) == other.lane_direction


@dataclass
class RoadSegment:
    """
    路段

    表示两个摄像头之间的道路段落，用于拓扑可达性判断和旅行时间估算。
    """
    segment_id: str                     # 路段唯一编号
    name: str                           # 路段名称
    start_camera_ids: List[str]         # 起始摄像头ID列表
    end_camera_ids: List[str]           # 终止摄像头ID列表
    distance_meters: float              # 路段长度(米)
    direction: str                      # 路段方向


# ============================================================
# 检测与识别结果
# ============================================================

@dataclass
class BoundingBox:
    """
    检测框

    表示目标在图像中的位置，坐标为像素坐标。
    """
    x1: float                           # 左上角 x
    y1: float                           # 左上角 y
    x2: float                           # 右下角 x
    y2: float                           # 右下角 y
    confidence: float                   # 检测置信度 [0, 1]

    @property
    def width(self) -> float:
        """检测框宽度"""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """检测框高度"""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """检测框面积"""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """检测框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BoundingBox) -> float:
        """
        计算与另一个检测框的 IoU

        Args:
            other: 另一个检测框

        Returns:
            IoU 值 [0, 1]
        """
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h

        union_area = self.area + other.area - intersection_area
        if union_area <= 0:
            return 0.0
        return intersection_area / union_area


@dataclass
class TargetInstance:
    """
    目标实例 - 一次检测到的目标

    这是系统中最基本的观测单元。每次在某个摄像头的某帧中检测到一个目标，
    就生成一个 TargetInstance。它包含了该次观测的所有结构化信息。

    数据流向: 感知层输出 → 存入目标实例库 → 被检索/回溯模块引用
    """
    instance_id: str                    # 实例唯一ID (自动生成)
    camera_id: str                      # 来源摄像头ID（如 'c001'）
    timestamp: datetime                 # 检测时间戳
    frame_id: int                       # 帧编号
    target_type: str                    # 目标类别: vehicle/pedestrian/non_motor_vehicle
    bbox: BoundingBox                   # 检测框
    attributes: Dict[str, Any]          # 属性字典: color, type, gender, clothing, bag 等
    plate_number: Optional[str]         # 车牌号(仅车辆, 可为空)
    plate_confidence: float             # 车牌识别置信度
    quality_score: float                # 质量评分 [0, 1]
    reid_vector: Optional[np.ndarray]   # ReID 特征向量 (dim=512)
    clip_vector: Optional[np.ndarray]   # Chinese-CLIP 图像特征向量 (dim=768)
    keyframe_path: Optional[str]        # 关键帧图片保存路径
    scene_id: Optional[str] = None      # 所属场景 ID（如 'S01'）

    def __post_init__(self) -> None:
        """自动生成 instance_id (如果未提供)"""
        if not self.instance_id:
            self.instance_id = f"INST_{uuid.uuid4().hex[:12]}"

    @property
    def has_plate(self) -> bool:
        """是否有有效车牌信息"""
        return self.plate_number is not None and self.plate_number.strip() != ""

    @property
    def has_reid(self) -> bool:
        """是否有有效 ReID 特征"""
        return self.reid_vector is not None

    @property
    def has_clip(self) -> bool:
        """是否有有效 CLIP 特征"""
        return self.clip_vector is not None


# ============================================================
# 单摄轨迹
# ============================================================

@dataclass
class Tracklet:
    """
    单摄轨迹片段 (Tracklet)

    表示一个目标在单个摄像头视野内的连续观测序列。
    Tracklet 是跨镜轨迹拼接的基本节点。

    示例:
        c001 中某辆黑色轿车:
          08:31:12 进入画面
          08:31:18 离开画面
          方向: 由西向东
          车牌: 苏E12345（示例，支持全国各省市车牌）
    """
    tracklet_id: str                    # Tracklet 唯一ID
    camera_id: str                      # 所属摄像头ID（如 'c001'）
    target_type: str                    # 目标类别
    start_time: datetime                # 进入画面时间
    end_time: datetime                  # 离开画面时间
    instances: List[TargetInstance]     # 组成该 Tracklet 的目标实例序列
    direction: str                      # 运动方向描述
    plate_number: Optional[str]         # 车牌号(聚合)
    attributes: Dict[str, Any]          # 聚合属性(取众数或最高置信度)
    avg_reid_vector: Optional[np.ndarray]   # 平均 ReID 向量
    avg_clip_vector: Optional[np.ndarray]   # 平均 CLIP 向量
    keyframe_paths: List[str]           # 关键帧路径列表
    scene_id: Optional[str] = None      # 所属场景 ID（如 'S01'）
    entry_point: Optional[Tuple[float, float]] = None   # 进入画面位置 (x, y)
    exit_point: Optional[Tuple[float, float]] = None    # 离开画面位置 (x, y)

    @property
    def duration_seconds(self) -> float:
        """Tracklet 持续时间(秒)"""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def instance_count(self) -> int:
        """包含的目标实例数量"""
        return len(self.instances)

    @property
    def has_plate(self) -> bool:
        """是否有有效车牌"""
        return self.plate_number is not None and self.plate_number.strip() != ""


# ============================================================
# 跨镜轨迹拼接
# ============================================================

@dataclass
class CrossCameraEdge:
    """
    跨镜候选连接边

    表示两个 Tracklet 之间可能属于同一目标的跨摄像头连接。
    边是跨镜轨迹候选图的基本单元。

    评分公式:
        跨镜连接得分 = 外观相似度 + 属性一致性 + 车牌一致性
                     + 时间可达性 + 空间可达性 + 方向一致性
                     - 路径分叉惩罚 - 观测缺失惩罚
    """
    source_tracklet_id: str             # 源 Tracklet ID
    target_tracklet_id: str             # 目标 Tracklet ID
    score: float                        # 综合得分 [0, 1]
    appearance_score: float             # 外观相似度 (ReID 余弦相似度)
    attribute_score: float              # 属性一致性
    plate_score: float                  # 车牌一致性
    temporal_score: float               # 时间可达性
    spatial_score: float                # 空间可达性
    direction_score: float              # 方向一致性
    penalty: float                      # 总惩罚值
    is_valid: bool                      # 是否为有效连接
    reasoning: str                      # 连接推理说明(用于可解释性)


# ============================================================
# 轨迹输出结构
# ============================================================

@dataclass
class ObservationNode:
    """
    观测节点

    表示目标在某个摄像头中被实际看到的一次记录。
    是轨迹输出中最基本的真实观测单元。

    示例:
        c001: 08:31:12
    """
    camera_id: str                      # 摄像头ID（如 'c001'）
    tracklet_id: str                    # 对应的 Tracklet ID
    timestamp: datetime                 # 观测时间
    keyframe_path: str                  # 关键帧路径
    confidence: float                   # 观测置信度


@dataclass
class ObservationSegment:
    """
    观测段 - 摄像头视野内真实轨迹

    表示目标在单个摄像头视野内的真实运动片段。
    观测段是可信度最高的轨迹组成部分，可视化时用实线表示。

    示例:
        c001 内，从画面左侧进入，从右侧离开
        c005 内，沿东向西通过路口
    """
    tracklet_id: str                    # 对应的 Tracklet ID
    camera_id: str                      # 摄像头ID
    start_time: datetime                # 进入画面时间
    end_time: datetime                  # 离开画面时间
    entry_description: str              # 进入描述(如"从画面左侧进入")
    exit_description: str               # 离开描述(如"从画面右侧离开")
    direction: str                      # 运动方向


@dataclass
class InferenceSegment:
    """
    推断段 - 摄像头之间推断连接

    表示两个摄像头之间基于拓扑和时间推断的可能路径。
    推断段没有连续画面支撑，可视化时用虚线表示。

    示例:
        c001 → c005: 推断连接, 置信度 0.82
    """
    source_camera_id: str               # 源摄像头ID
    target_camera_id: str               # 目标摄像头ID
    source_tracklet_id: str             # 源 Tracklet ID
    target_tracklet_id: str             # 目标 Tracklet ID
    confidence: float                   # 推断置信度 [0, 1]
    estimated_travel_time: float        # 理论行驶时间(秒)
    actual_travel_time: float           # 实际时间差(秒)
    route_description: str              # 路径描述


@dataclass
class CandidatePath:
    """
    候选路径

    两个摄像头之间有多条可能道路时，返回多条候选路径及其置信度。

    示例:
        候选路径 1: 置信度 0.72
        候选路径 2: 置信度 0.41
    """
    path_id: str                        # 路径唯一ID
    road_segments: List[str]            # 经过的路段ID序列
    confidence: float                   # 路径置信度 [0, 1]
    distance_meters: float              # 路径总距离(米)
    estimated_time: float               # 预估行驶时间(秒)


@dataclass
class TrajectoryResult:
    """
    最终轨迹输出

    整合一次完整轨迹回溯的所有结果，包括观测节点、观测段、
    推断段和候选路径。这是系统最终返回给用户的数据结构。

    系统输出十类信息:
    1. 候选目标图片
    2. 用户确认后的目标实例
    3. 目标经过摄像头序列
    4. 每个摄像头的关键帧
    5. 每个节点的出现时间
    6. 摄像头视野内观测段
    7. 摄像头之间推断段
    8. 候选路网路径
    9. 每段连接置信度
    10. 车牌、属性、ReID、旅行时间等拼接证据
    """
    query_id: str                                   # 查询唯一ID
    target_instance: TargetInstance                 # 确认的目标实例(锚点)
    observation_nodes: List[ObservationNode]        # 观测节点列表
    observation_segments: List[ObservationSegment]  # 观测段列表
    inference_segments: List[InferenceSegment]      # 推断段列表
    candidate_paths: List[CandidatePath]            # 候选路径列表
    evidence: Dict[str, Any]                        # 拼接证据(车牌/属性/ReID/旅行时间等)
    overall_confidence: float                       # 整体轨迹置信度 [0, 1]

    @property
    def camera_sequence(self) -> List[str]:
        """获取目标经过的摄像头序列"""
        return [node.camera_id for node in self.observation_nodes]

    @property
    def time_sequence(self) -> List[datetime]:
        """获取每个观测节点的时间序列"""
        return [node.timestamp for node in self.observation_nodes]


# ============================================================
# 检索相关数据模型
# ============================================================

@dataclass
class ParsedQuery:
    """
    解析后的查询结构

    将用户自然语言输入解析为结构化查询条件。

    示例:
        用户输入 "蓝色背包的男人" →
        ParsedQuery(
            target_type="pedestrian",
            attributes={"gender": "male", "bag": true, "bag_color": "blue"}
        )
    """
    raw_text: str                           # 原始查询文本
    target_type: Optional[str]              # 推断的目标类别
    attributes: Dict[str, Any]              # 解析出的属性条件
    plate_number: Optional[str]             # 车牌号(如果查询包含车牌)
    clip_text_embedding: Optional[np.ndarray]  # 查询文本的 CLIP 向量


@dataclass
class RetrievalCandidate:
    """
    检索候选结果

    表示一个通过文本检索召回的候选目标实例。
    """
    instance: TargetInstance                # 目标实例
    text_score: float                       # 图文匹配分数
    attribute_match_score: float            # 属性匹配分数
    combined_score: float                   # 综合排序分数
    rank: int                               # 排名


# ============================================================
# 观测链数据模型
# ============================================================

@dataclass
class ObservationChain:
    """
    观测链

    表示以某个确认目标为锚点，在跨镜候选图中回溯出的完整观测链。
    包含有序的观测节点和连接它们的推断段。
    """
    chain_id: str                           # 观测链唯一ID
    anchor_tracklet_id: str                 # 锚点 Tracklet ID
    anchor_instance_id: str                 # 锚点实例 ID
    ordered_tracklets: List[str]            # 有序的 Tracklet ID 序列
    edges: List[CrossCameraEdge]            # 连接边列表
    total_confidence: float                 # 链总置信度
    total_duration_seconds: float           # 链总时长(秒)
    camera_sequence: List[str]              # 经过的摄像头序列
