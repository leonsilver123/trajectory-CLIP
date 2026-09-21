"""
src.trajectory.builder - 跨镜轨迹构建器

本模块把原先散落在 api/routes/backtrack.py 里的「按 vehicle_id 聚合真实检测」
逻辑收拢到一处，并在同一条入口上补上弱身份路径：

    确定锚点实例
      ├─ 强身份（车牌 / 真实 vehicle_id）→ 直接强匹配          ← 主路径
      └─ 弱身份（无车牌 / 无真值）      → src/stitching 六维评分 → 概率推断

两条路径的输出结构完全一致，且**每一段都带 `basis` 字段标注依据**：
`strong_identity`（强身份匹配）或 `probabilistic_inference`（概率推断）。

弱身份路径是 src/stitching 首次被线上引用的地方，用到的组件：
    - `CandidateEdgeGenerator`  交通约束粗筛 + 外观精筛
    - `CrossCameraScorer`       六维评分（外观/属性/车牌/时间/空间/方向）+ 惩罚项
    - `ObservationChainBuilder` 观测链构建（上下游贪心扩展、证据收集）

不使用任何随机数：所有 ID 由 (instance_id, target_id) 确定性派生，
同一锚点重复调用结果完全一致。

T5 跨摄时间对齐
---------------
`output/cityflow_results.json` 里每辆车的 `timestamp` 是**各摄像头自己的本地
时间**：每路都从 2020-01-01 00:00:00 附近起跳，因此不同摄像头的时间戳直接
相减是错位的（V0034 在 c004 的 [0, 4.6s] 与 c005 的 [1.3, 6.6s] 看似"重叠"，
其实是两台相机各自从 0 起跳）。

数据集在 `cam_timestamp/{scene}.txt` 中给出了每个摄像头的**全局起始偏移**
（秒，如 S04 的 c016=0、c017=14.318、c018=29.955）。本模块据此建立
`摄像头 → 偏移秒` 映射：

    全局时间 = 本地时间 + 该摄像头偏移

只用于**跨摄像头的时间差**（`actual_travel_time`、跨镜总时长、摄像头先后
排序），每个检测自身展示用的 `timestamp` 字段保持本机时间不变——若需要
全局时间，一律作为新增字段（`global_timestamp` 等）给出，不覆盖原字段。

使用方式:
    from src.trajectory.builder import get_trajectory_builder

    builder = get_trajectory_builder()
    result = builder.build("CF3_c001_V0034_000001")            # 自动选路
    result = builder.build("CF3_c001_V0034_000001", mode="stitch")  # 强制弱身份
    timeline = builder.build_camera_timeline(instance_id=...)   # /trajectory 视图
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.common.config import get_config
from src.common.data_models import (
    BoundingBox,
    CrossCameraEdge,
    TargetInstance,
    Tracklet,
    TrajectoryResult,
)
from src.common.ids import extract_vehicle_id, parse_track_id
from src.common.logger import get_logger
from src.common.utils import haversine_distance
from src.storage.datastore import load_results
from src.stitching.candidate_edge import CandidateEdgeGenerator
from src.stitching.observation_chain import ObservationChainBuilder
from src.stitching.scoring import CrossCameraScorer

logger = get_logger("trajectory.builder")

# 项目根目录（与 src/storage/datastore.py 同一约定）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# 常量
# ============================================================

# 依据标注（每段输出都会带上）
BASIS_STRONG = "strong_identity"            # 强身份匹配
BASIS_WEAK = "probabilistic_inference"      # 概率推断
BASIS_TEXT = {
    BASIS_STRONG: "强身份匹配",
    BASIS_WEAK: "概率推断",
}

# 支持的时间戳格式（cityflow_results.json 用毫秒精度）
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

# 中英文属性映射（原先在 api/routes/backtrack.py，随聚合逻辑一并搬来）
_COLOR_CN_MAP = {0: "黄色", 1: "橙色", 2: "绿色", 3: "灰色", 4: "红色", 5: "蓝色", 6: "白色",
                 7: "金色", 8: "棕色", 9: "黑色", 10: "紫色", 11: "粉色"}
_TYPE_CN_MAP = {0: "轿车", 1: "SUV", 2: "面包车", 3: "两厢车", 4: "MPV", 5: "皮卡",
                6: "公交车", 7: "卡车", 8: "旅行车", 9: "跑车", 10: "房车"}
_EN_COLOR_MAP = {"yellow": "黄色", "orange": "橙色", "green": "绿色", "gray": "灰色", "red": "红色",
                 "blue": "蓝色", "white": "白色", "golden": "金色", "brown": "棕色", "black": "黑色",
                 "purple": "紫色", "pink": "粉色"}
_EN_TYPE_MAP = {"sedan": "轿车", "suv": "SUV", "van": "面包车", "hatchback": "两厢车", "mpv": "MPV",
                "pickup": "皮卡", "bus": "公交车", "truck": "卡车", "estate": "旅行车",
                "sportscar": "跑车", "rv": "房车"}

# 车道方向 → 中文（摄像头元数据 lane_direction → 展示文案）
_LANE_DIR_CN = {
    "northbound": "由南向北",
    "southbound": "由北向南",
    "eastbound": "由西向东",
    "westbound": "由东向西",
}


# ============================================================
# 异常
# ============================================================

class TrajectoryNotFoundError(Exception):
    """锚点目标在真实数据中不存在（调用方转 404）"""


class TrajectoryDataUnavailableError(Exception):
    """真实数据文件不可用（调用方转 500 / 演示数据）"""


# ============================================================
# 评分器：绕开 src/stitching/scoring.py 的路径枚举性能缺陷
# ============================================================

class BoundedCrossCameraScorer(CrossCameraScorer):
    """
    六维评分器（保留的历史类名，现已无行为差异）

    ## 它曾经解决的问题

    上游 `_count_possible_paths` 用无深度上限的 DFS 枚举简单路径，在本项目
    46 摄像头 / 91 条邻接边的拓扑上会组合爆炸：**实测单对摄像头耗时 4–23 秒**
    （c001→c002 23.2s、c003→c004 12.1s、c004→c005 4.2s），线上每对摄像头都
    调用一次，接口会直接超时。

    当时的处理是**在子类里覆盖这一个方法**绕开它 —— 但这只救了走 builder 的
    那条路：任何直接用 `CrossCameraScorer` 的代码依然会撞上爆炸。

    ## 现状（2026-09-21 修复）

    有界实现已**下沉到基类** `src/stitching/scoring.CrossCameraScorer`，
    基类自己就带 `MAX_PATH_HOPS` / `PATH_EXPANSION_BUDGET` 两个上限
    （实测另一对 c029→c030 单次 3,486,156 次递归、1.84 秒，同样被修掉）。
    因此本类不再覆盖任何方法，保留类名只为兼容既有引用与类型标注。

    其余全部评分维度（外观/属性/车牌/时间/空间/方向、加权求和、惩罚项）
    一直由 `src.stitching.scoring.CrossCameraScorer` 提供，未做改动。
    """


# ============================================================
# 时间戳工具
# ============================================================

def _parse_timestamp(value: Any) -> Optional[datetime]:
    """解析时间戳字符串，失败返回 None（不抛异常）"""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _format_ts_full(value: Any) -> str:
    """格式化为 YYYY-MM-DD HH:MM:SS（沿用既有接口的展示口径）"""
    dt = _parse_timestamp(value)
    if dt is None:
        return "--" if not value else str(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_ts(value: Any) -> str:
    """格式化为 HH:MM:SS（沿用既有接口的展示口径）"""
    dt = _parse_timestamp(value)
    if dt is None:
        return "--" if not value else str(value)
    return dt.strftime("%H:%M:%S")


def _duration_seconds(ts1: Any, ts2: Any) -> int:
    """两个时间戳之间的秒数差（负值截为 0）"""
    dt1, dt2 = _parse_timestamp(ts1), _parse_timestamp(ts2)
    if dt1 is None or dt2 is None:
        return 0
    return max(0, int((dt2 - dt1).total_seconds()))


def _geometric_mean(scores: List[float]) -> float:
    """几何平均（用于把多条边的得分合成整体置信度）"""
    if not scores:
        return 0.0
    log_sum = sum(math.log(max(s, 0.01)) for s in scores)
    return float(max(0.0, min(1.0, math.exp(log_sum / len(scores)))))


def _mean_or_none(values: List[float]) -> Optional[float]:
    """列表均值；空列表返回 None（用于"填不出来就留空"）"""
    return round(float(np.mean(values)), 4) if values else None


# ============================================================
# 构建器
# ============================================================

class TrajectoryBuilder:
    """
    跨镜轨迹构建器

    同时承担数据索引（cityflow_results.json 的加载与分组）与两条拼接路径，
    是 backtrack 接口唯一的数据来源。
    """

    def __init__(
        self,
        results_path: Optional[str] = None,
        camera_metadata_path: Optional[str] = None,
        config=None,
    ) -> None:
        """
        初始化构建器（此时不加载数据，首次调用时才惰性加载）

        Args:
            results_path: cityflow_results.json 路径，缺省用项目默认位置
            camera_metadata_path: 摄像头元数据 YAML 路径，缺省读 configs/default.yaml
            config: Config 实例，缺省用全局单例
        """
        self._config = config or get_config()
        self._root = Path(__file__).resolve().parent.parent.parent

        self._results_path = Path(results_path) if results_path else self._root / "output" / "cityflow_results.json"
        metadata_file = camera_metadata_path or self._config.get(
            "camera.metadata_file", "configs/cityflow_camera_metadata.yaml"
        )
        self._camera_metadata_path = Path(metadata_file)
        if not self._camera_metadata_path.is_absolute():
            self._camera_metadata_path = self._root / metadata_file

        # ---- 拼接配置（全部来自 configs/default.yaml，不硬编码）----
        stitching_cfg = self._config.get("stitching", {}) or {}
        self._max_time_gap = float(stitching_cfg.get("max_time_gap_seconds", 600.0))
        self._min_appearance_score = float(stitching_cfg.get("min_appearance_score", 0.6))
        self._spatial_radius_km = float(stitching_cfg.get("spatial_search_radius_km", 10.0))
        weights_cfg = stitching_cfg.get("weights", {}) or {}
        self._vehicle_weights = weights_cfg.get("vehicle")
        self._pedestrian_weights = weights_cfg.get("pedestrian")
        self._penalties = stitching_cfg.get("penalties")
        self._min_chain_confidence = float(
            self._config.get("backtrack.min_confidence_threshold", 0.4)
        )
        self._default_upstream = int(self._config.get("backtrack.max_upstream_depth", 10))
        self._default_downstream = int(self._config.get("backtrack.max_downstream_depth", 10))

        # ---- 跨摄时间对齐（T5，路径全部来自 configs/default.yaml）----
        # cam_timestamp 文件给出每个摄像头的全局起始偏移；缺失时保持本机时间，
        # 绝不用估算值替代（对齐前本机时间相减会得到无意义的负跨镜间隔）。
        align_cfg = self._config.get("time_alignment", {}) or {}
        self._time_alignment_enabled = bool(align_cfg.get("enabled", True))
        self._dataset_dir = str(
            self._config.get("system.data_dir", "cityflow/AICity22_Track1_MTMC_Tracking")
        )
        self._cam_timestamp_template = str(
            self._config.get("data_format.cam_timestamp_file", "cam_timestamp/{scene}.txt")
        )
        self._camera_offsets_by_scene: Dict[Tuple[str, str], float] = {}
        # 跨场景兜底表: camera_id -> (来源场景, 偏移秒)，来源场景仅用于冲突告警
        self._camera_offsets_by_camera: Dict[str, Tuple[str, float]] = {}
        self._offsets_loaded = False

        # ---- 数据索引（惰性加载）----
        self._loaded = False
        self._load_failed = False
        self._target_tracks: Dict[str, List[Dict[str, Any]]] = {}
        self._vehicle_detections: Dict[str, List[Dict[str, Any]]] = {}
        self._tracklet_detections: Dict[str, List[Dict[str, Any]]] = {}
        self._det_to_track: Dict[str, str] = {}
        self._camera_meta: Dict[str, Dict[str, Any]] = {}
        self._camera_direction_map: Dict[str, str] = {}
        self._all_detections: List[Dict[str, Any]] = []

        # ---- 惰性组件 ----
        self._camera_manager = None
        self._road_topology = None
        self._scorer: Optional[BoundedCrossCameraScorer] = None
        self._track_reid_map: Optional[Dict[str, np.ndarray]] = None

        # ---- 缓存（同一锚点重复调用直接命中，保证结果稳定且低延迟）----
        self._tracklet_index: Dict[str, Tracklet] = {}
        self._scene_tracklets: Dict[str, List[Tracklet]] = {}
        self._scene_edges: Dict[str, List[CrossCameraEdge]] = {}
        self._result_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # 摄像头 → 推断出的画面尺寸 (宽, 高)；用于时间线帧的 bbox 归一化（PLAN4-T1）
        self._camera_frame_size_cache: Dict[str, Optional[Tuple[int, int]]] = {}

    # ================================================================
    # 数据加载
    # ================================================================

    @property
    def data_available(self) -> bool:
        """真实数据文件是否已成功加载（决定查不到目标时返回 404 还是演示数据）"""
        self.ensure_loaded()
        return self._loaded

    def ensure_loaded(self) -> bool:
        """
        确保真实数据已加载（仅首次真正读盘）

        Returns:
            数据是否可用
        """
        if self._loaded or self._load_failed:
            return self._loaded
        self._load()
        return self._loaded

    def _load(self) -> None:
        """
        加载检测记录并建立各类索引

        数据统一由 src.storage.datastore 提供：优先读 output/datastore/ 的
        Parquet + SQLite，datastore 缺失时自动回退 cityflow_results.json 直读，
        两条路径返回的结构完全一致（该模块的 verify 会逐字段比对）。
        """
        try:
            data = load_results(str(self._results_path))
        except Exception as e:  # 文件损坏 / 编码异常都不应让接口 500
            logger.error("加载 CityFlow 结果数据失败: %s", e)
            self._load_failed = True
            return

        if data is None:
            logger.warning("CityFlow 结果数据不可用（datastore 与 JSON 均缺失）: %s",
                           self._results_path)
            self._load_failed = True
            return

        detections = data.get("detections", []) or []
        tracks_list = data.get("tracks", []) or []
        self._det_to_track = data.get("det_to_track_map", {}) or {}
        self._all_detections = detections

        # 按 vehicle_id 分组（跨镜轨迹主路径的基础）
        for det in detections:
            vehicle_id = extract_vehicle_id(det.get("target_id", ""))
            if vehicle_id:
                self._vehicle_detections.setdefault(vehicle_id, []).append(det)
        for vehicle_id in self._vehicle_detections:
            self._vehicle_detections[vehicle_id].sort(key=lambda d: d.get("timestamp", ""))

        # 按 target_id 分组（单 target 轨迹）+ 摄像头元数据
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for det in detections:
            target_id = det.get("target_id", "")
            if target_id:
                grouped[target_id].append(det)

        for target_id, dets in grouped.items():
            dets_sorted = sorted(dets, key=lambda d: d.get("timestamp", ""))
            self._target_tracks[target_id] = dets_sorted
            for det in dets_sorted:
                camera_id = det.get("camera_id", "")
                if camera_id and camera_id not in self._camera_meta:
                    self._camera_meta[camera_id] = {
                        "camera_id": camera_id,
                        "name": f"{det.get('scene_id', 'Unknown')}-{camera_id}",
                        "scene": det.get("scene_id", ""),
                        "latitude": det.get("latitude"),
                        "longitude": det.get("longitude"),
                    }

        # 按 track_id 分组（单摄轨迹，弱身份拼接的节点）
        for target_id, dets in grouped.items():
            track_id = self._det_to_track.get(target_id)
            if track_id:
                self._tracklet_detections.setdefault(track_id, []).extend(dets)
        for track_id in self._tracklet_detections:
            self._tracklet_detections[track_id].sort(key=lambda d: d.get("timestamp", ""))

        self._load_camera_metadata()
        self._load_camera_time_offsets()
        self._loaded = True
        logger.info(
            "CityFlow 数据加载完成 | 检测=%d | 目标=%d | 车辆=%d | 单摄轨迹=%d | 摄像头=%d",
            len(detections), len(self._target_tracks), len(self._vehicle_detections),
            len(self._tracklet_detections), len(self._camera_meta),
        )

    def _load_camera_metadata(self) -> None:
        """从摄像头元数据 YAML 补充摄像头名称与车道方向"""
        if not self._camera_metadata_path.exists():
            logger.warning("摄像头元数据文件不存在: %s", self._camera_metadata_path)
            return
        try:
            with open(self._camera_metadata_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("加载摄像头元数据失败: %s", e)
            return

        # 场景近似中心（数据集唯一真实的 GPS，见 ReadMe §12；逐摄像头 GPS 不存在）
        scene_gps = data.get("scene_gps_center", {}) or {}

        for cam in data.get("cameras", []) or []:
            camera_id = cam.get("camera_id", "")
            if not camera_id:
                continue
            scene = cam.get("scene", cam.get("scene_id", ""))
            entry = self._camera_meta.setdefault(camera_id, {
                "camera_id": camera_id,
                "name": cam.get("name", camera_id),
                "scene": scene,
                "latitude": cam.get("latitude"),
                "longitude": cam.get("longitude"),
                "scene_gps_center": scene_gps.get(scene),
            })
            entry["name"] = cam.get("name", entry.get("name", camera_id))
            if scene:
                entry["scene"] = scene
            if entry.get("latitude") is None:
                entry["latitude"] = cam.get("latitude")
            if entry.get("longitude") is None:
                entry["longitude"] = cam.get("longitude")
            if entry.get("scene_gps_center") is None:
                entry["scene_gps_center"] = scene_gps.get(scene)
            self._camera_direction_map[camera_id] = _LANE_DIR_CN.get(cam.get("lane_direction", ""), "")

        logger.info("加载了 %d 个摄像头的方向信息", len(self._camera_direction_map))

    # ================================================================
    # 跨摄时间对齐（T5）
    # ================================================================

    def _scene_ids_from_config(self) -> List[str]:
        """场景 ID 列表（优先取 configs/default.yaml 的 camera.scenes）"""
        scenes = self._config.get("camera.scenes", []) or []
        scene_ids = [str(s.get("scene_id", "")) for s in scenes if isinstance(s, dict)]
        scene_ids = [s for s in scene_ids if s]
        if scene_ids:
            return sorted(scene_ids)
        # 配置里没有场景定义时，退回扫描目录（仍按文件名排序，保证确定性）
        return sorted(p.stem for p in self._cam_timestamp_path("*").parent.glob("*.txt"))

    def _cam_timestamp_path(self, scene_id: str) -> Path:
        """拼接某场景的 cam_timestamp 文件绝对路径（相对项目根）"""
        relative = self._cam_timestamp_template.format(scene=scene_id)
        path = Path(relative)
        return path if path.is_absolute() else self._root / self._dataset_dir / relative

    def _load_camera_time_offsets(self) -> None:
        """
        加载各场景的摄像头时间偏移（本地时间 → 全局时间的秒数偏移）

        文件格式（cam_timestamp/{scene}.txt，每行一个摄像头）::

            c001 0
            c002 1.640
            c003 2.049

        同一个摄像头可能出现在多个场景文件中（如 c016 同时在 S04 与 S05）。
        按场景精确查表优先；跨场景兜底表按**场景名排序后先到先得**，保证
        结果与加载顺序无关（确定性），冲突只记日志不静默覆盖。

        幂等：只读一次盘，重复调用直接返回（`_get_scorer` 会独立触发一次，
        保证评分器无论何时被构造都能拿到偏移表）。
        """
        if self._offsets_loaded:
            return
        if not self._time_alignment_enabled:
            self._offsets_loaded = True
            logger.info("跨摄时间对齐已关闭（configs/default.yaml time_alignment.enabled=false）")
            return

        loaded_scenes = 0
        for scene_id in self._scene_ids_from_config():
            path = self._cam_timestamp_path(scene_id)
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    lines = f.readlines()
            except OSError as e:
                # 缺文件不是致命错误：该场景保持本机时间（offset 视为 0）
                logger.warning("摄像头时间偏移文件不可用: %s (%s)", path, e)
                continue

            count = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                camera_id = parts[0]
                try:
                    offset = float(parts[1])
                except ValueError:
                    continue
                self._camera_offsets_by_scene[(scene_id, camera_id)] = offset
                count += 1

                previous = self._camera_offsets_by_camera.get(camera_id)
                if previous is None:
                    self._camera_offsets_by_camera[camera_id] = (scene_id, offset)
                elif previous[1] != offset:
                    logger.warning(
                        "摄像头 %s 在不同场景的偏移不一致（%s=%.3f vs %s=%.3f），"
                        "跨场景兜底取先加载的 %s=%.3f；按场景精确查表不受影响",
                        camera_id, previous[0], previous[1], scene_id, offset,
                        previous[0], previous[1],
                    )
            if count:
                loaded_scenes += 1

        self._offsets_loaded = True
        logger.info(
            "跨摄时间偏移加载完成 | 场景=%d | 摄像头(场景维度)=%d | 摄像头(兜底)=%d",
            loaded_scenes, len(self._camera_offsets_by_scene), len(self._camera_offsets_by_camera),
        )

    def camera_time_offset(
        self, camera_id: str, scene_id: Optional[str] = None
    ) -> Optional[float]:
        """
        取摄像头的全局时间偏移（秒）

        按场景精确查表优先，查不到再退回跨场景兜底表；两条路都查不到时返回
        None——调用方据此保持本机时间，**不要**拿 0.0 冒充"已对齐"。
        """
        if not self._time_alignment_enabled:
            return None
        if scene_id:
            offset = self._camera_offsets_by_scene.get((scene_id, camera_id))
            if offset is not None:
                return offset
        fallback = self._camera_offsets_by_camera.get(camera_id)
        return fallback[1] if fallback is not None else None

    def to_global_time(
        self,
        camera_id: str,
        local_time: Optional[datetime],
        scene_id: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        本机时间 → 全局时间（`本地时间 + 摄像头偏移`）

        偏移未知时**原样返回本机时间**（等价于偏移 0），这样调用方无需分支；
        是否真的对齐过由 `is_time_aligned()` 单独回答，避免"看起来对齐了"。
        """
        if local_time is None:
            return None
        offset = self.camera_time_offset(camera_id, scene_id)
        if offset is None:
            return local_time
        return local_time + timedelta(seconds=offset)

    def camera_offset_map(self) -> Dict[str, float]:
        """
        摄像头 → 全局偏移(秒) 的扁平映射（供评分器做跨镜时间换算）

        同一摄像头出现在多个场景时取先加载的场景（与兜底表口径一致）；评分器
        按 camera_id 查表，因此这里只保留一份。
        """
        return {cam: offset for cam, (_, offset) in self._camera_offsets_by_camera.items()}

    def is_time_aligned(
        self, src_camera_id: str, tgt_camera_id: str,
        src_scene_id: Optional[str] = None, tgt_scene_id: Optional[str] = None,
    ) -> bool:
        """两侧摄像头都有已知偏移，且确实发生了对齐（用于输出里的可解释标注）"""
        return (
            self.camera_time_offset(src_camera_id, src_scene_id) is not None
            and self.camera_time_offset(tgt_camera_id, tgt_scene_id) is not None
        )

    def _global_sort_key(self, camera_id: str, first_det: Dict[str, Any]):
        """
        摄像头先后排序键：场景 ID + 全局首现时间

        先按场景分组（不同场景是不同视频，时间不可比），场景内按**全局**首现
        时间排序。时间戳解析失败时排在最后，保证顺序确定。
        """
        scene_id = first_det.get("scene_id", "") or ""
        local = _parse_timestamp(first_det.get("timestamp"))
        if local is None:
            return (scene_id, 1, datetime.max)
        global_time = self.to_global_time(camera_id, local, scene_id) or local
        return (scene_id, 0, global_time)

    # ================================================================
    # 惰性组件
    # ================================================================

    def _get_camera_manager(self):
        """摄像头管理器（惰性 + 缓存）"""
        if self._camera_manager is None:
            from src.data_governance.camera_manager import CameraManager
            self._camera_manager = CameraManager(str(self._camera_metadata_path))
        return self._camera_manager

    def _get_road_topology(self):
        """道路拓扑（惰性 + 缓存）"""
        if self._road_topology is None:
            from src.data_governance.road_topology import RoadTopology
            self._road_topology = RoadTopology(str(self._camera_metadata_path))
        return self._road_topology

    def _get_track_reid_map(self) -> Dict[str, np.ndarray]:
        """
        轨迹级 ReID 向量表（惰性 + 缓存），track_id -> (2048,) 已 L2 归一化。

        数据由二期 T4 的 `scripts/build_reid_tracks.py` 产出，以**旁挂文件**形式
        落盘（不改动 cityflow_results.json，避免 107MB JSON 再膨胀 ~50MB）：

            output/reid/track_reid_vectors.npy  +  output/reid/track_ids.json

        文件不存在（未跑 T4）时返回空表 —— 此时 avg_reid_vector 保持 None，
        scoring 的 reid 维度按"无证据"处理并重分摊权重，与二期之前的行为一致，
        不会因为缺文件而报错。
        """
        if self._track_reid_map is None:
            self._track_reid_map = {}
            try:
                base = _PROJECT_ROOT / "output" / "reid"
                vec_path = base / "track_reid_vectors.npy"
                ids_path = base / "track_ids.json"
                if vec_path.is_file() and ids_path.is_file():
                    vectors = np.load(vec_path)
                    with open(ids_path, encoding="utf-8") as f:
                        ids = json.load(f)
                    if len(ids) != len(vectors):
                        logger.warning(
                            "track ReID 表行数不一致: ids=%d vectors=%d，忽略该文件",
                            len(ids), len(vectors),
                        )
                    else:
                        self._track_reid_map = {
                            tid: vectors[i] for i, tid in enumerate(ids)
                        }
                        logger.info("轨迹级 ReID 向量已加载: %d 条", len(ids))
            except Exception as e:  # 缺文件/格式异常都不该影响主流程
                logger.warning("加载轨迹级 ReID 向量失败，按无 ReID 处理: %s", e)
                self._track_reid_map = {}
        return self._track_reid_map

    def _get_scorer(self) -> BoundedCrossCameraScorer:
        """
        六维评分器（惰性 + 缓存）

        权重与惩罚项全部取自 configs/default.yaml 的 stitching.weights / stitching.penalties，
        车辆与行人各一套；速度区间沿用 src/stitching 的城区默认值（配置中无该项）。

        同时传入摄像头全局时间偏移（T5），让时间/空间可达性维度按全局时间而非
        各摄像头自己的本机时间计算跨镜间隔。
        """
        if self._scorer is None:
            # 偏移表只依赖 6 个小文本文件，与检测数据无关：这里主动加载一次，
            # 避免"评分器先于数据加载被构造"时缓存下一份空的偏移表
            self._load_camera_time_offsets()
            self._scorer = BoundedCrossCameraScorer(
                camera_manager=self._get_camera_manager(),
                road_topology=self._get_road_topology(),
                vehicle_weights=self._vehicle_weights,
                pedestrian_weights=self._pedestrian_weights,
                penalties=self._penalties,
                camera_time_offsets=self.camera_offset_map(),
                config=self._config,
            )
        return self._scorer

    # ================================================================
    # 锚点定位
    # ================================================================

    def find_target_id(self, instance_id: str) -> Optional[str]:
        """
        根据确认的 instance_id 定位真实的 target_id，只做直接匹配

        instance_id 可能是:
            1. 直接的 target_id（如 CF3_c001_V0034_000001）→ 命中则返回
            2. 轨迹 ID（如 CF3_TRACK_c001_V0034）→ 解析 vehicle_id 后定位真实检测

        找不到返回 None（调用方转 404），绝不随机抽取任何目标。
        """
        if not instance_id:
            return None

        if instance_id in self._target_tracks:
            return instance_id

        vehicle_id = extract_vehicle_id(instance_id)
        if not vehicle_id:
            return None

        vehicle_dets = self._vehicle_detections.get(vehicle_id, [])
        if not vehicle_dets:
            return None

        # track_id 带了摄像头信息时，优先返回该摄像头下的检测
        camera_id = parse_track_id(instance_id).get("camera_id")
        if camera_id:
            for det in vehicle_dets:
                if det.get("camera_id") == camera_id:
                    return det.get("target_id") or None

        return vehicle_dets[0].get("target_id") or None

    # ================================================================
    # 主入口
    # ================================================================

    def build(
        self,
        instance_id: str,
        mode: str = "auto",
        max_upstream: Optional[int] = None,
        max_downstream: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        构建跨镜轨迹（统一入口）

        Args:
            instance_id: 用户确认的锚点实例 ID
            mode: "auto" 自动选路（默认）/ "strong" 强制强身份 / "stitch" 强制概率拼接
            max_upstream: 弱身份路径的最大上游扩展深度
            max_downstream: 弱身份路径的最大下游扩展深度

        Returns:
            回溯结果字典（含 identity.basis / 每段的 basis 标注）

        Raises:
            TrajectoryDataUnavailableError: 真实数据文件不可用
            TrajectoryNotFoundError: 数据可用但锚点目标不存在
        """
        self.ensure_loaded()
        if not self._loaded:
            raise TrajectoryDataUnavailableError("真实数据文件不可用")

        cache_key = (instance_id, mode)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached

        target_id = self.find_target_id(instance_id)
        if target_id is None:
            raise TrajectoryNotFoundError(f"未找到目标 {instance_id} 的真实轨迹记录")

        vehicle_id = extract_vehicle_id(target_id)
        # 真实 vehicle_id 是数据集自带的强身份真值；拿不到就必须走概率拼接
        use_strong = bool(vehicle_id) and mode != "stitch"
        if mode == "strong" and not vehicle_id:
            raise TrajectoryNotFoundError(f"目标 {instance_id} 无真实 vehicle_id，无法强身份匹配")

        if use_strong:
            result = self._build_strong(instance_id, target_id, vehicle_id)
        else:
            result = self._build_weak(
                instance_id, target_id,
                max_upstream or self._default_upstream,
                max_downstream or self._default_downstream,
            )

        self._result_cache[cache_key] = result
        return result

    # ================================================================
    # 强身份路径：真实 vehicle_id 直接聚合
    # ================================================================

    def _build_strong(self, instance_id: str, target_id: str, vehicle_id: str) -> Dict[str, Any]:
        """
        强身份路径

        同一 vehicle_id 在所有摄像头的检测直接聚合为一个目标——不做外观/属性
        判别（身份已由真值确定），但段与段之间的时空可达性仍用六维评分器真实计算，
        用于填充置信度与证据分项。
        """
        # 强身份聚合口径与 /trajectory 一致：取该 vehicle_id 在所有摄像头下的真实检测，
        # 而不是只看被点选的那一条 target_id（单条 target_id 只属于一个摄像头）
        dets = self._vehicle_detections.get(vehicle_id, [])
        if not dets:
            raise TrajectoryNotFoundError(f"目标 {instance_id} 没有可用的检测记录")

        query_id = self._deterministic_query_id(instance_id, f"strong|{vehicle_id}")

        # 按摄像头分组（dets 已按时间排序），按各摄像头**全局**首次出现时间排序
        # （本机时间跨摄像头不可比，见模块 docstring 的 T5 说明）
        camera_groups = self._group_by_camera(dets)
        sorted_cameras = sorted(
            camera_groups, key=lambda c: self._global_sort_key(c, camera_groups[c][0])
        )

        # 每个摄像头构造一个 Tracklet（单摄轨迹），供评分器使用
        tracklets: List[Tracklet] = []
        for camera_id in sorted_cameras:
            tracklet_id = self._det_to_track.get(
                camera_groups[camera_id][0].get("target_id", "")
            ) or f"TRK_{uuid.uuid5(uuid.NAMESPACE_URL, f'{target_id}|{camera_id}').hex[:8]}"
            tracklets.append(self._tracklet_from_detections(tracklet_id, camera_groups[camera_id]))

        # 相邻摄像头两两评分（真实六维打分，用于推断段置信度与证据）
        edges = self._score_consecutive(tracklets)

        observation_nodes = []
        observation_segments = []
        for camera_id, tracklet in zip(sorted_cameras, tracklets):
            camera_dets = camera_groups[camera_id]
            cam_info = self._camera_meta.get(camera_id, {})
            tracklet_id = tracklet.tracklet_id
            # timestamp 保持该摄像头的本机时间（前端已依赖该格式）；
            # 全局时间只作为新增字段 global_timestamp 给出
            local_start = _parse_timestamp(camera_dets[0].get("timestamp"))
            global_start = self.to_global_time(
                camera_id, local_start, camera_dets[0].get("scene_id")
            )
            observation_nodes.append({
                "camera_id": camera_id,
                "camera_name": cam_info.get("name", camera_id),
                "tracklet_id": tracklet_id,
                "timestamp": _format_ts_full(camera_dets[0].get("timestamp", "")),
                # 新增字段：本机时间 + 该摄像头全局偏移（偏移未知时等于本机时间）
                "global_timestamp": _format_ts_full(global_start) if global_start else None,
                "time_offset_seconds": self.camera_time_offset(
                    camera_id, camera_dets[0].get("scene_id")
                ),
                "latitude": cam_info.get("latitude"),
                "longitude": cam_info.get("longitude"),
                "scene_gps_center": cam_info.get("scene_gps_center"),
                "keyframe_path": camera_dets[0].get("keyframe_path"),
                "confidence": round(camera_dets[0].get("confidence", 0.0), 3),
                # 依据标注：该观测来自真实 vehicle_id 的检测
                "basis": BASIS_STRONG,
                "basis_text": f"{BASIS_TEXT[BASIS_STRONG]}（真实 vehicle_id={vehicle_id}）",
            })
            observation_segments.append({
                "tracklet_id": tracklet_id,
                "camera_id": camera_id,
                "camera_name": cam_info.get("name", camera_id),
                "start_time": _format_ts(camera_dets[0].get("timestamp", "")),
                "end_time": _format_ts(camera_dets[-1].get("timestamp", "")),
                "direction": self._camera_direction_map.get(camera_id, ""),
                # 进出画面的方位无真实标注，留空（不造数）
                "entry_description": "",
                "exit_description": "",
                "basis": BASIS_STRONG,
                "basis_text": f"{BASIS_TEXT[BASIS_STRONG]}（真实 vehicle_id={vehicle_id}）",
            })

        inference_segments = self._inference_segments_from_edges(edges, BASIS_STRONG)
        candidate_paths = [self._build_candidate_path(sorted_cameras, edges)]

        # 整体置信度：相邻摄像头连接得分的几何平均；无跨镜连接时为 0.0
        overall = _geometric_mean([e.score for e in edges])
        evidence = self._evidence_from_edges(edges, source="cityflow_results")
        evidence.update({
            "target_id": target_id,
            "vehicle_id": vehicle_id,
            "detection_count": len(dets),
            "camera_count": len(sorted_cameras),
            "identity_basis": BASIS_STRONG,
            # 强身份路径的身份来自数据集真值，确定无疑；overall_confidence 度量的是
            # 相邻摄像头之间的时空链接合理性，两者不是一回事，这里分开给出
            "identity_certainty": 1.0,
            "linkage_confidence": round(overall, 4),
        })

        logger.info(
            "强身份回溯完成 | instance=%s | vehicle=%s | 摄像头=%d | 跨镜连接=%d | 置信度=%.4f",
            instance_id, vehicle_id, len(sorted_cameras), len(edges), overall,
        )

        return {
            "query_id": query_id,
            "identity": {
                "mode": "strong",
                "basis": BASIS_STRONG,
                "basis_text": f"{BASIS_TEXT[BASIS_STRONG]}（真实 vehicle_id={vehicle_id}，跨镜检测直接聚合）",
                "vehicle_id": vehicle_id,
                "target_id": target_id,
            },
            "target_instance": self._target_instance_payload(instance_id, target_id, dets[0]),
            "camera_sequence": sorted_cameras,
            "observation_nodes": observation_nodes,
            "observation_segments": observation_segments,
            "inference_segments": inference_segments,
            "candidate_paths": candidate_paths,
            "overall_confidence": round(overall, 4),
            "evidence": evidence,
        }

    # ================================================================
    # 弱身份路径：src/stitching 六维评分拼接
    # ================================================================

    def _build_weak(
        self,
        instance_id: str,
        target_id: str,
        max_upstream: int,
        max_downstream: int,
    ) -> Dict[str, Any]:
        """
        弱身份路径（无车牌 / 无真实 vehicle_id）

        流程:
            1. 锚点检测 → 锚点 Tracklet（单摄轨迹）
            2. 同场景内的所有 Tracklet 作为候选节点
            3. `CandidateEdgeGenerator` 生成候选边（交通约束粗筛 + 外观精筛 + 六维评分）
            4. `ObservationChainBuilder.build_chain` 上下游扩展成链
            5. 链结果转成与强身份一致的输出结构，每段标注为「概率推断」
        """
        anchor_dets = self._target_tracks.get(target_id, [])
        if not anchor_dets:
            raise TrajectoryNotFoundError(f"目标 {instance_id} 没有可用的检测记录")

        query_id = self._deterministic_query_id(instance_id, f"weak|{target_id}")
        anchor_tracklet_id = self._det_to_track.get(target_id) or (
            f"TRK_{uuid.uuid5(uuid.NAMESPACE_URL, f'{target_id}|anchor').hex[:8]}"
        )
        anchor_tracklet = self._tracklet_from_detections(anchor_tracklet_id, anchor_dets)

        scene_id = anchor_tracklet.scene_id or self._camera_meta.get(
            anchor_tracklet.camera_id, {}
        ).get("scene", "")
        candidates = self._scene_tracklet_pool(scene_id, anchor_tracklet)
        edge_pool = self._scene_edge_pool(scene_id, candidates)

        # ---- 观测链构建（src/stitching）----
        chain_builder = ObservationChainBuilder(
            tracklets=candidates,
            edges=edge_pool,
            road_topology=self._get_road_topology(),
            min_chain_confidence=self._min_chain_confidence,
            max_upstream_depth=max_upstream,
            max_downstream_depth=max_downstream,
        )
        trajectory: TrajectoryResult = chain_builder.build_chain(
            anchor_tracklet,
            edge_pool,
            self._get_road_topology(),
            query_id,
            anchor_tracklet.instances[0],
        )

        # 链中每条连接边 → 推断段（含真实六维评分）
        edge_by_pair: Dict[Tuple[str, str], CrossCameraEdge] = {}
        for edge in edge_pool:
            edge_by_pair[(edge.source_tracklet_id, edge.target_tracklet_id)] = edge

        node_ids = [n.tracklet_id for n in trajectory.observation_nodes]
        used_edges: List[CrossCameraEdge] = []
        for src_id, tgt_id in zip(node_ids, node_ids[1:]):
            edge = edge_by_pair.get((src_id, tgt_id)) or edge_by_pair.get((tgt_id, src_id))
            if edge is not None:
                used_edges.append(edge)

        observation_nodes = []
        observation_segments = []
        tracklet_by_id = {t.tracklet_id: t for t in candidates}
        for index, node in enumerate(trajectory.observation_nodes):
            tracklet = tracklet_by_id.get(node.tracklet_id)
            camera_id = node.camera_id
            cam_info = self._camera_meta.get(camera_id, {})
            # 锚点节点是用户确认过的实例，不是拼接出来的，单独标注
            is_anchor = node.tracklet_id == anchor_tracklet.tracklet_id
            basis = BASIS_STRONG if is_anchor else BASIS_WEAK
            basis_text = (
                f"{BASIS_TEXT[BASIS_STRONG]}（用户确认的锚点实例）"
                if is_anchor
                else f"{BASIS_TEXT[BASIS_WEAK]}（六维评分拼接，无车牌/无真值）"
            )

            observation_nodes.append({
                "camera_id": camera_id,
                "camera_name": cam_info.get("name", camera_id),
                "tracklet_id": node.tracklet_id,
                "timestamp": _format_ts_full(node.timestamp),
                "latitude": cam_info.get("latitude"),
                "longitude": cam_info.get("longitude"),
                "scene_gps_center": cam_info.get("scene_gps_center"),
                "keyframe_path": node.keyframe_path,
                # 非锚点节点的"置信度"是拼接边得分，不是检测置信度
                "confidence": None if is_anchor else round(node.confidence, 4),
                "basis": basis,
                "basis_text": basis_text,
            })
            observation_segments.append({
                "tracklet_id": node.tracklet_id,
                "camera_id": camera_id,
                "camera_name": cam_info.get("name", camera_id),
                "start_time": _format_ts(tracklet.start_time) if tracklet else "--",
                "end_time": _format_ts(tracklet.end_time) if tracklet else "--",
                "direction": self._camera_direction_map.get(camera_id, ""),
                "entry_description": next(
                    (s.entry_description for s in trajectory.observation_segments
                     if s.tracklet_id == node.tracklet_id), ""
                ),
                "exit_description": next(
                    (s.exit_description for s in trajectory.observation_segments
                     if s.tracklet_id == node.tracklet_id), ""
                ),
                "basis": basis,
                "basis_text": basis_text,
            })

        inference_segments = self._inference_segments_from_edges(used_edges, BASIS_WEAK)
        candidate_paths = self._candidate_paths_from_chain(trajectory, node_ids)
        if not candidate_paths:
            # 链构建器没给出候选路径时，前端仍需至少一条可展示的路径（数值型字段）
            candidate_paths = [
                self._build_candidate_path(list(trajectory.camera_sequence), [], BASIS_WEAK)
            ]

        # 无跨镜连接时不给"整体置信度"——没有跨镜证据，报 1.0 会误导前端
        overall = _geometric_mean([e.score for e in used_edges]) if used_edges else 0.0

        evidence = self._evidence_from_edges(used_edges, source="src.stitching")
        evidence.update({
            "target_id": target_id,
            "vehicle_id": None,
            "detection_count": len(anchor_dets),
            "camera_count": len(trajectory.camera_sequence),
            "identity_basis": BASIS_WEAK,
            "candidate_tracklets": len(candidates),
            "candidate_edges": len(edge_pool),
            "chain_confidence": round(float(trajectory.overall_confidence), 4),
        })

        logger.info(
            "弱身份拼接完成 | instance=%s | 候选轨迹=%d | 候选边=%d | 链节点=%d | 摄像头序列=%s | 置信度=%.4f",
            instance_id, len(candidates), len(edge_pool),
            len(trajectory.observation_nodes), trajectory.camera_sequence, overall,
        )

        return {
            "query_id": query_id,
            "identity": {
                "mode": "weak",
                "basis": BASIS_WEAK,
                "basis_text": f"{BASIS_TEXT[BASIS_WEAK]}（无车牌/无真实 vehicle_id，src.stitching 六维评分拼接）",
                "vehicle_id": None,
                "target_id": target_id,
            },
            "target_instance": self._target_instance_payload(instance_id, target_id, anchor_dets[0]),
            "camera_sequence": trajectory.camera_sequence,
            "observation_nodes": observation_nodes,
            "observation_segments": observation_segments,
            "inference_segments": inference_segments,
            "candidate_paths": candidate_paths,
            "overall_confidence": round(overall, 4),
            "evidence": evidence,
        }

    # ================================================================
    # 弱身份路径的候选池（同场景 + 缓存）
    # ================================================================

    def _scene_tracklet_pool(self, scene_id: str, anchor: Tracklet) -> List[Tracklet]:
        """
        取锚点所在场景的全部单摄轨迹作为候选节点（含锚点自身）

        只按场景圈定范围，不做身份过滤——配对是否成立交给评分器判断。
        """
        tracklets = self._scene_tracklets.get(scene_id)
        if tracklets is None:
            tracklets = []
            for tracklet_id, dets in self._tracklet_detections.items():
                if not dets:
                    continue
                if dets[0].get("scene_id") != scene_id:
                    continue
                tracklets.append(self._tracklet_from_detections(tracklet_id, dets))
            self._scene_tracklets[scene_id] = tracklets
            logger.info("场景 %s 构建了 %d 条候选单摄轨迹", scene_id, len(tracklets))

        if all(t.tracklet_id != anchor.tracklet_id for t in tracklets):
            tracklets = tracklets + [anchor]
        return tracklets

    def _scene_edge_pool(self, scene_id: str, tracklets: List[Tracklet]) -> List[CrossCameraEdge]:
        """生成场景内候选边（缓存，阈值全部来自 configs/default.yaml 的 stitching 段）"""
        edges = self._scene_edges.get(scene_id)
        if edges is None:
            generator = CandidateEdgeGenerator(
                camera_manager=self._get_camera_manager(),
                road_topology=self._get_road_topology(),
                max_time_gap_seconds=self._max_time_gap,
                min_appearance_score=self._min_appearance_score,
                spatial_search_radius_km=self._spatial_radius_km,
                scorer=self._get_scorer(),
            )
            edges = generator.generate_candidates(tracklets)
            self._scene_edges[scene_id] = edges
            logger.info("场景 %s 生成 %d 条有效候选边（stitching 配置驱动）", scene_id, len(edges))
        return edges

    # ================================================================
    # Tracklet / 检测构造
    # ================================================================

    @staticmethod
    def _group_by_camera(dets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按摄像头分组（保持原有的时间顺序）"""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for det in dets:
            camera_id = det.get("camera_id", "")
            if camera_id:
                groups[camera_id].append(det)
        return groups

    def _tracklet_from_detections(
        self,
        tracklet_id: str,
        dets: List[Dict[str, Any]],
    ) -> Tracklet:
        """
        把 JSON 检测记录适配成 src/common/data_models.py 的 Tracklet

        - `crop_path` / `keyframe_path` 有，直接映射为 keyframe_path
        - `clip_image_vector` 仅部分检测带（本数据集 949/68349），有则求均值作为
          avg_clip_vector，没有就留 None（评分器会回退到中性 0.5）
        - ReID 向量取**轨迹级**旁挂表（二期 T4 产出，见 _get_track_reid_map）。
          表不存在时留 None —— 与二期之前的行为一致，评分器按"无证据"重分摊权重。
        - 车牌本数据集完全没有 → plate_number=None（评分器按"双方都无"给中性 0.5）
        - direction 用摄像头元数据的车道方向（英文），与既有 /trajectory 的口径一致
        """
        dets_sorted = sorted(dets, key=lambda d: d.get("timestamp", ""))
        instances: List[TargetInstance] = []
        keyframe_paths: List[str] = []
        clip_vectors: List[np.ndarray] = []

        for det in dets_sorted:
            timestamp = _parse_timestamp(det.get("timestamp"))
            if timestamp is None:
                continue
            bbox_raw = det.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox_raw) < 4:
                bbox_raw = [0.0, 0.0, 0.0, 0.0]
            clip_vector = None
            raw_clip = det.get("clip_image_vector")
            if raw_clip:
                try:
                    clip_vector = np.asarray(raw_clip, dtype=np.float32)
                    clip_vectors.append(clip_vector)
                except (TypeError, ValueError):
                    clip_vector = None

            keyframe = det.get("keyframe_path") or det.get("crop_path")
            if keyframe:
                keyframe_paths.append(keyframe)

            instances.append(TargetInstance(
                instance_id=det.get("target_id", ""),
                camera_id=det.get("camera_id", ""),
                timestamp=timestamp,
                frame_id=int(det.get("frame_id", 0) or 0),
                target_type=det.get("target_type", "vehicle"),
                bbox=BoundingBox(
                    x1=float(bbox_raw[0]), y1=float(bbox_raw[1]),
                    x2=float(bbox_raw[2]), y2=float(bbox_raw[3]),
                    confidence=float(det.get("confidence", 0.0) or 0.0),
                ),
                attributes=det.get("attributes", {}) or {},
                plate_number=None,
                plate_confidence=0.0,
                quality_score=float(det.get("confidence", 0.0) or 0.0),
                reid_vector=None,
                clip_vector=clip_vector,
                keyframe_path=keyframe,
                scene_id=det.get("scene_id"),
            ))

        avg_clip = None
        if clip_vectors:
            avg_clip = np.mean(np.stack(clip_vectors), axis=0).astype(np.float32)

        first = dets_sorted[0]
        camera_id = first.get("camera_id", "")
        tracklet = Tracklet(
            tracklet_id=tracklet_id,
            camera_id=camera_id,
            target_type=first.get("target_type", "vehicle"),
            start_time=instances[0].timestamp,
            end_time=instances[-1].timestamp,
            instances=instances,
            # 方向用摄像头车道方向（本数据集没有目标级运动方向标注）
            direction=self._lane_direction(camera_id),
            plate_number=None,
            attributes=first.get("attributes", {}) or {},
            avg_reid_vector=self._get_track_reid_map().get(tracklet_id),
            avg_clip_vector=avg_clip,
            keyframe_paths=keyframe_paths,
            scene_id=first.get("scene_id"),
        )
        self._tracklet_index[tracklet_id] = tracklet
        return tracklet

    def _lane_direction(self, camera_id: str) -> str:
        """取摄像头车道方向（英文，供评分器做方向一致性判断）"""
        try:
            camera = self._get_camera_manager().get_camera(camera_id)
            if camera is not None:
                return camera.lane_direction or ""
        except Exception:  # 元数据缺失不影响主流程
            pass
        return ""

    # ================================================================
    # 评分与证据
    # ================================================================

    def _score_consecutive(self, tracklets: List[Tracklet]) -> List[CrossCameraEdge]:
        """对时间上相邻的两个单摄轨迹调用六维评分器（强身份路径用）"""
        scorer = self._get_scorer()
        edges: List[CrossCameraEdge] = []
        for source, target in zip(tracklets, tracklets[1:]):
            try:
                edges.append(scorer.score(source, target))
            except Exception as e:  # 单对评分失败不应让整个回溯失败
                logger.warning("评分失败 %s → %s: %s", source.tracklet_id, target.tracklet_id, e)
        return edges

    def _inference_segments_from_edges(
        self,
        edges: List[CrossCameraEdge],
        basis: str,
    ) -> List[Dict[str, Any]]:
        """
        由连接边生成推断段

        - `confidence`   ← 六维评分器的真实得分
        - `actual_travel_time`  ← 两个单摄轨迹**全局**时间的差值（秒）：
          目标首现于 tgt − 末现于 src。本数据集的 timestamp 是各摄像头自己的
          视频内时间（每路都从 0 起跳），直接相减必然错位甚至恒为负，因此先
          用 cam_timestamp 的偏移换算成全局时间再相减（T5）。换算后仍为负
          （数据异常 / 两侧观测确实重叠）就留 None，不取绝对值凑数。
        - `local_travel_time`   ← 对齐前的本机时间差值，仅作诊断对照
        - `estimated_travel_time` ← 仅在道路拓扑给出真实距离时才有值
          （距离 ÷ 评分器城区速度区间中值）；距离未知则留 None，不造数
        """
        topology = self._get_road_topology()
        scorer = self._get_scorer()
        mid_speed_mps = (scorer.min_speed_kmh + scorer.max_speed_kmh) / 2.0 * 1000.0 / 3600.0

        segments: List[Dict[str, Any]] = []
        for edge in edges:
            src_tracklet = self._tracklet_lookup(edge.source_tracklet_id)
            tgt_tracklet = self._tracklet_lookup(edge.target_tracklet_id)
            if src_tracklet is None or tgt_tracklet is None:
                continue
            src_cam, tgt_cam = src_tracklet.camera_id, tgt_tracklet.camera_id
            src_info = self._camera_meta.get(src_cam, {})
            tgt_info = self._camera_meta.get(tgt_cam, {})

            try:
                distance = topology.get_segment_distance(src_cam, tgt_cam)
            except Exception:
                distance = None
            estimated = round(distance / mid_speed_mps, 1) if distance else None

            # ---- 跨摄时间对齐（T5）----
            src_offset = self.camera_time_offset(src_cam, src_tracklet.scene_id)
            tgt_offset = self.camera_time_offset(tgt_cam, tgt_tracklet.scene_id)
            src_end_global = self.to_global_time(
                src_cam, src_tracklet.end_time, src_tracklet.scene_id
            )
            tgt_start_global = self.to_global_time(
                tgt_cam, tgt_tracklet.start_time, tgt_tracklet.scene_id
            )
            local_gap = (tgt_tracklet.start_time - src_tracklet.end_time).total_seconds()
            gap = (tgt_start_global - src_end_global).total_seconds() if (
                src_end_global is not None and tgt_start_global is not None
            ) else local_gap
            actual = round(gap, 3) if gap >= 0 else None

            segments.append({
                "source_camera_id": src_cam,
                "target_camera_id": tgt_cam,
                "source_camera_name": src_info.get("name", src_cam),
                "target_camera_name": tgt_info.get("name", tgt_cam),
                "source_latitude": src_info.get("latitude"),
                "source_longitude": src_info.get("longitude"),
                "target_latitude": tgt_info.get("latitude"),
                "target_longitude": tgt_info.get("longitude"),
                "source_tracklet_id": edge.source_tracklet_id,
                "target_tracklet_id": edge.target_tracklet_id,
                "confidence": round(float(edge.score), 4),
                "estimated_travel_time": estimated,
                "actual_travel_time": actual,
                # 诊断对照：对齐前的本机时间差值（负值即"错位"的直接证据）
                "local_travel_time": round(local_gap, 3),
                # 时间对齐的可解释标注：两侧摄像头偏移都已知才算真的对齐过
                "time_aligned": src_offset is not None and tgt_offset is not None,
                "source_time_offset_seconds": src_offset,
                "target_time_offset_seconds": tgt_offset,
                "route_description": (
                    f"{src_info.get('name', src_cam)} → {tgt_info.get('name', tgt_cam)}"
                    f"（六维得分 {edge.score:.3f}，惩罚 {edge.penalty:.3f}"
                    + ("；两侧观测全局时间重叠，实际行程时间无法确定" if actual is None else "")
                    + "）"
                ),
                # 评分明细（可解释性）
                "score_detail": {
                    "appearance": round(float(edge.appearance_score), 4),
                    "attribute": round(float(edge.attribute_score), 4),
                    "plate": round(float(edge.plate_score), 4),
                    "temporal": round(float(edge.temporal_score), 4),
                    "spatial": round(float(edge.spatial_score), 4),
                    "direction": round(float(edge.direction_score), 4),
                    "penalty": round(float(edge.penalty), 4),
                },
                "basis": basis,
                "basis_text": (
                    f"{BASIS_TEXT[basis]}"
                    + ("（真实 vehicle_id 下相邻摄像头的时空可达性评分）"
                       if basis == BASIS_STRONG else "（六维评分拼接）")
                ),
            })
        return segments

    def _evidence_from_edges(
        self,
        edges: List[CrossCameraEdge],
        source: str,
    ) -> Dict[str, Any]:
        """
        汇总证据分项

        全部取自六维评分器的真实分项均值，取不到就留 None（不写 0.0 假装算过）。
        """
        return {
            "plate_consistency": _mean_or_none([e.plate_score for e in edges]),
            "appearance_similarity": _mean_or_none([e.appearance_score for e in edges]),
            "attribute_consistency": _mean_or_none([e.attribute_score for e in edges]),
            "temporal_feasibility": _mean_or_none([e.temporal_score for e in edges]),
            "spatial_feasibility": _mean_or_none([e.spatial_score for e in edges]),
            "direction_consistency": _mean_or_none([e.direction_score for e in edges]),
            "source": source,
            "link_count": len(edges),
        }

    def _tracklet_lookup(self, tracklet_id: str) -> Optional[Tracklet]:
        """按 tracklet_id 找回 Tracklet（索引里查，找不到返回 None）"""
        tracklet = self._tracklet_index.get(tracklet_id)
        if tracklet is not None:
            return tracklet
        dets = self._tracklet_detections.get(tracklet_id)
        if dets:
            return self._tracklet_from_detections(tracklet_id, dets)
        return None

    # ================================================================
    # 候选路径
    # ================================================================

    def _path_distance_meters(self, camera_sequence: List[str]) -> float:
        """
        按摄像头序列累加相邻摄像头的真实球面距离（米）

        经纬度取自摄像头元数据；缺坐标的段按 0 计，不用任何估算值。
        """
        total = 0.0
        for src_cid, tgt_cid in zip(camera_sequence, camera_sequence[1:]):
            src_info = self._camera_meta.get(src_cid, {})
            tgt_info = self._camera_meta.get(tgt_cid, {})
            src_lat, src_lon = src_info.get("latitude"), src_info.get("longitude")
            tgt_lat, tgt_lon = tgt_info.get("latitude"), tgt_info.get("longitude")
            if None in (src_lat, src_lon, tgt_lat, tgt_lon):
                continue
            total += haversine_distance(src_lat, src_lon, tgt_lat, tgt_lon)
        return round(total, 1)

    def _build_candidate_path(
        self,
        camera_sequence: List[str],
        edges: List[CrossCameraEdge],
        basis: str = BASIS_STRONG,
    ) -> Dict[str, Any]:
        """
        强身份路径的候选路径（单条，标识为目标经过的真实摄像头序列）

        - `distance_meters` ← 相邻摄像头真实球面距离累加
        - `estimated_time`  ← 距离 ÷ 评分器城区速度区间中值（无距离则为 0.0）
        - `confidence`      ← 相邻摄像头六维连接得分的几何平均（无跨镜连接时为 0.0）

        前端按 {:.0%} / {:.0f} 直接格式化这三个字段，因此必须保持数值型，不能给 null。
        """
        path_id = "PATH_" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"path|{'|'.join(camera_sequence)}"
        ).hex[:6]
        distance = self._path_distance_meters(camera_sequence)
        scorer = self._get_scorer()
        mid_speed_mps = (scorer.min_speed_kmh + scorer.max_speed_kmh) / 2.0 * 1000.0 / 3600.0
        return {
            "path_id": path_id,
            "road_segments": [f"SEG_{cid}" for cid in camera_sequence],
            "confidence": round(_geometric_mean([e.score for e in edges]), 4),
            "distance_meters": distance,
            "estimated_time": round(distance / mid_speed_mps, 1) if distance else 0.0,
            "description": " → ".join(
                self._camera_meta.get(cid, {}).get("name", cid) for cid in camera_sequence
            ),
            "basis": basis,
            "basis_text": f"{BASIS_TEXT[basis]}（"
            + ("目标真实经过的摄像头序列" if basis == BASIS_STRONG else "拼接链的摄像头序列")
            + "）",
        }

    def _candidate_paths_from_chain(
        self,
        trajectory: TrajectoryResult,
        node_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        弱身份路径的候选路径

        链构建器已按相邻摄像头对生成了候选路径（含拓扑距离与按 40 km/h 估算的时间），
        这里保留其真实距离与置信度，并把路径 ID 换成确定性 ID。
        """
        paths: List[Dict[str, Any]] = []
        for index, path in enumerate(trajectory.candidate_paths):
            segments = list(path.road_segments)
            paths.append({
                "path_id": "PATH_" + uuid.uuid5(
                    uuid.NAMESPACE_URL, f"weak|{'|'.join(segments)}|{index}"
                ).hex[:6],
                "road_segments": segments,
                "confidence": round(float(path.confidence), 4),
                "distance_meters": round(float(path.distance_meters), 1),
                # src/stitching 按城区 40 km/h 估算的行程时间（其自身实现口径）
                "estimated_time": round(float(path.estimated_time), 1),
                "description": " → ".join(segments),
                "basis": BASIS_WEAK,
                "basis_text": f"{BASIS_TEXT[BASIS_WEAK]}（拓扑候选路径）",
            })
        return paths

    # ================================================================
    # 锚点实例载荷
    # ================================================================

    def _target_instance_payload(
        self,
        instance_id: str,
        target_id: str,
        first_det: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造 target_instance 展示载荷（字段与既有接口保持一致）"""
        attrs = first_det.get("attributes", {}) or {}
        color = attrs.get("颜色") or attrs.get("color", "未知")
        vehicle_type = attrs.get("车型") or attrs.get("vehicle_type", "未知")
        return {
            "instance_id": instance_id,
            "target_id": target_id,
            "camera_id": first_det.get("camera_id", ""),
            "timestamp": _format_ts_full(first_det.get("timestamp", "")),
            "target_type": first_det.get("target_type", "vehicle"),
            "attributes": {"颜色": color, "类型": vehicle_type},
            # 本数据集没有车牌识别结果，留空不造数
            "plate_number": None,
            "quality_score": round(float(first_det.get("confidence", 0.0) or 0.0), 3),
            "keyframe_path": first_det.get("keyframe_path"),
        }

    @staticmethod
    def _deterministic_query_id(instance_id: str, salt: str) -> str:
        """确定性 query_id：同一锚点重复回溯得到完全一样的 ID"""
        return f"BT_{uuid.uuid5(uuid.NAMESPACE_URL, f'{instance_id}|{salt}').hex[:8]}"

    # ================================================================
    # /trajectory 视图（按 vehicle_id 的完整摄像头时间线）
    # ================================================================

    # ---- 时间线帧构造（轨迹还原动画的数据基础，PLAN4-T1） ----

    def _camera_frame_size(self, camera_id: str) -> Optional[Tuple[int, int]]:
        """
        推断某摄像头的画面尺寸 (宽, 高)

        数据里没有直接存画面尺寸，但检测框是**原始视频像素坐标**，因此可以用
        该摄像头全部检测框的最大范围作为画面尺寸的**下界估计**：
        画面至少装得下所有框，所以真实尺寸 >= 这个值。

        - 同一摄像头的所有帧共用同一个分母 ⇒ **相对运动（归一化后的位移）是准确的**，
          这对轨迹还原动画来说才是关键。
        - 返回值会随 `frame_size_source` 一起给出，调用方据此决定是否标注"比例未知"。

        Returns:
            (宽, 高)；该摄像头没有任何检测时返回 None
        """
        if camera_id in self._camera_frame_size_cache:
            return self._camera_frame_size_cache[camera_id]

        size: Optional[Tuple[int, int]] = None
        max_w = max_h = 0
        for det in self._all_detections:
            if det.get("camera_id") != camera_id:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                max_w = max(max_w, int(float(bbox[2])))
                max_h = max(max_h, int(float(bbox[3])))
            except (TypeError, ValueError):
                continue
        if max_w > 0 and max_h > 0:
            size = (max_w, max_h)

        self._camera_frame_size_cache[camera_id] = size
        return size

    def _timeline_frame(self, camera_id: str, det: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造时间线里的单帧记录

        ## 为什么加 bbox / bbox_norm

        轨迹还原动画要展示"车在摄像头画面里怎么动"。最初的 frames[] 只有
        frame_id/timestamp/crop_path/confidence —— 能显示实拍裁剪图，但**画不出位置**。
        检测本身是带 bbox 的，只是没往上传，这里补上。

        ## 归一化口径

        `bbox_norm` = [cx/W, cy/H, w/W, h/H]，全部落在 [0,1]（除非框超出估计的画面范围）。
        分母 W/H 来自 `_camera_frame_size()` 的**下界估计**，因此当 `frame_size_source`
        为 `"inferred_from_detections"` 时，绝对值可能偏大 —— 但**相对运动不受影响**。
        """
        bbox = det.get("bbox")
        bbox_out: Optional[List[float]] = None
        bbox_norm: Optional[List[float]] = None

        if bbox and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                bbox_out = [x1, y1, x2, y2]

                size = self._camera_frame_size(camera_id)
                if size is not None and size[0] > 0 and size[1] > 0:
                    fw, fh = size
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    bbox_norm = [
                        round(cx / fw, 6),
                        round(cy / fh, 6),
                        round((x2 - x1) / fw, 6),
                        round((y2 - y1) / fh, 6),
                    ]
            except (TypeError, ValueError):
                bbox_out = None
                bbox_norm = None

        # 全局时间：跨摄像头比较必须用它（本机时间各摄像头独立起跳，见 T5）
        global_ts = None
        ts = det.get("timestamp", "")
        parsed = _parse_timestamp(ts)
        if parsed is not None:
            converted = self.to_global_time(camera_id, parsed, det.get("scene_id"))
            if converted is not None:
                global_ts = _format_ts_full(converted)

        frame_size = self._camera_frame_size(camera_id)

        return {
            "frame_id": det.get("frame_id", 0),
            "timestamp": _format_ts_full(ts),
            "crop_path": det.get("crop_path", det.get("keyframe_path", "")),
            "confidence": det.get("confidence", 0.5),
            # ---- PLAN4-T1 新增（纯追加，不删旧字段，向后兼容）----
            "bbox": bbox_out,
            "bbox_norm": bbox_norm,
            "global_timestamp": global_ts,
            "frame_size": list(frame_size) if frame_size else None,
            # 画面尺寸是**推断**出来的下界，不是实测值 —— 让前端能如实标注
            "frame_size_source": (
                "inferred_from_detections" if frame_size else "unavailable"
            ),
        }

    def build_camera_timeline(
        self,
        track_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        构建目标的完整跨镜时间线（/trajectory 接口的数据源）

        按 vehicle_id 找到其在所有摄像头中的真实检测，按时间排序构建
        带逐帧明细的摄像头序列。

        Args:
            track_id: 轨迹 ID（CF3_TRACK_c001_V0034）
            instance_id: 实例 ID（target_id 或 track_id 均可）

        Returns:
            {"track_id", "vehicle_id", "camera_sequence", "attributes", ...} 的轨迹载荷

        Raises:
            TrajectoryDataUnavailableError: 真实数据不可用
            TrajectoryNotFoundError: 无法解析 vehicle_id 或该车无检测记录
        """
        self.ensure_loaded()
        if not self._loaded or not self._all_detections:
            raise TrajectoryDataUnavailableError("数据未加载")

        source_track_id = track_id or ""
        vehicle_id = extract_vehicle_id(instance_id or "") or extract_vehicle_id(source_track_id)
        if not vehicle_id:
            raise TrajectoryNotFoundError("无法解析 vehicle_id")

        vehicle_dets = self._vehicle_detections.get(vehicle_id, [])
        if not vehicle_dets:
            raise TrajectoryNotFoundError(f"未找到 vehicle {vehicle_id} 的检测记录")

        camera_groups = self._group_by_camera(vehicle_dets)
        for camera_id in camera_groups:
            camera_groups[camera_id].sort(key=lambda d: d.get("timestamp", ""))

        # 摄像头先后按**全局**首现时间排序（本机时间跨摄像头不可比，见 T5 说明）
        camera_sequence = []
        for camera_id, cam_dets in sorted(
            camera_groups.items(),
            key=lambda item: self._global_sort_key(item[0], item[1][0]),
        ):
            arrival = cam_dets[0].get("timestamp", "")
            departure = cam_dets[-1].get("timestamp", "")
            scene_id = cam_dets[0].get("scene_id")
            global_arrival = self.to_global_time(
                camera_id, _parse_timestamp(arrival), scene_id
            )
            global_departure = self.to_global_time(
                camera_id, _parse_timestamp(departure), scene_id
            )
            cam_info = self._camera_meta.get(camera_id, {})
            camera_sequence.append({
                "camera_id": camera_id,
                "camera_name": cam_info.get("name", camera_id),
                "arrival_time": _format_ts_full(arrival),
                "departure_time": _format_ts_full(departure),
                # 新增字段：全局到达/离开时间与偏移（本机时间字段保持原样）
                "global_arrival_time": _format_ts_full(global_arrival) if global_arrival else None,
                "global_departure_time": (
                    _format_ts_full(global_departure) if global_departure else None
                ),
                "time_offset_seconds": self.camera_time_offset(camera_id, scene_id),
                # 单摄持续时长只看同一摄像头内的时间，本机时间即可
                "duration_seconds": _duration_seconds(arrival, departure),
                "detection_count": len(cam_dets),
                "direction": self._camera_direction_map.get(camera_id, ""),
                "frames": [
                    self._timeline_frame(camera_id, det)
                    for det in cam_dets
                ],
                # 依据标注：整条时间线都来自真实 vehicle_id 的检测
                "basis": BASIS_STRONG,
                "basis_text": f"{BASIS_TEXT[BASIS_STRONG]}（真实 vehicle_id={vehicle_id}）",
            })

        # 跨镜总时长必须用全局时间：本机时间跨摄像头错位，差值没有意义
        first_arrival = camera_sequence[0]["arrival_time"] if camera_sequence else ""
        last_departure = camera_sequence[-1]["departure_time"] if camera_sequence else ""
        total_duration = _duration_seconds(
            first_arrival.replace("--", ""), last_departure.replace("--", "")
        )
        if camera_sequence:
            first_global = camera_sequence[0].get("global_arrival_time")
            last_global = camera_sequence[-1].get("global_departure_time")
            if first_global and last_global:
                total_duration = _duration_seconds(first_global, last_global)

        attrs = vehicle_dets[0].get("attributes", {}) or {}
        color_id = attrs.get("color_id", -1)
        type_id = attrs.get("type_id", -1)
        color_cn = (
            attrs.get("颜色", "")
            or _COLOR_CN_MAP.get(color_id, "")
            or _EN_COLOR_MAP.get(str(attrs.get("color", "")).lower(), "")
            or attrs.get("color", "未知")
        )
        type_cn = (
            attrs.get("车型", "")
            or _TYPE_CN_MAP.get(type_id, "")
            or _EN_TYPE_MAP.get(str(attrs.get("vehicle_type", "")).lower(), "")
            or attrs.get("vehicle_type", "未知")
        )

        return {
            "track_id": source_track_id,
            "vehicle_id": vehicle_id,
            "first_appearance": first_arrival,
            "last_appearance": last_departure,
            # 新增字段：跨镜首末出现的全局时间（first/last_appearance 保持本机时间）
            "global_first_appearance": (
                camera_sequence[0].get("global_arrival_time") if camera_sequence else None
            ),
            "global_last_appearance": (
                camera_sequence[-1].get("global_departure_time") if camera_sequence else None
            ),
            "total_duration_seconds": total_duration,
            "total_cameras": len(camera_sequence),
            "total_detections": sum(len(c["frames"]) for c in camera_sequence),
            "camera_sequence": camera_sequence,
            "attributes": {"颜色": color_cn, "车型": type_cn},
        }


# ============================================================
# 全局单例（沿用项目里"模块级惰性缓存"的惯例）
#
# 双检锁：FastAPI 的同步端点跑在线程池里，两个 worker 并发首次回溯时
# 可能同时看到 `_builder is None`，各自构造一个 Builder —— 白付一次数据加载，
# 且两份结果缓存互相不可见。写法与 `src/common/session_store.py` 的单例对齐。
# ============================================================

_builder: Optional[TrajectoryBuilder] = None
_builder_lock = threading.Lock()


def get_trajectory_builder() -> TrajectoryBuilder:
    """
    获取全局 TrajectoryBuilder 单例

    Returns:
        进程内共享的构建器（数据只在首次调用时加载）
    """
    global _builder
    if _builder is None:
        with _builder_lock:
            if _builder is None:
                _builder = TrajectoryBuilder()
    return _builder


def reset_trajectory_builder() -> None:
    """重置全局构建器（主要用于测试）"""
    global _builder
    with _builder_lock:
        _builder = None
