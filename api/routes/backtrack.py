"""
api.routes.backtrack - 回溯 API

提供轨迹回溯接口:
- POST /api/v1/backtrack/trace  以确认目标为锚点进行轨迹回溯
- GET  /api/v1/backtrack/result/{query_id}  获取回溯结果
- POST /api/v1/backtrack/trajectory  获取完整跨镜轨迹（多摄像头时间线拼接）

本模块只负责 HTTP 层：参数校验、异常→状态码映射、会话状态写回。
所有轨迹构建逻辑（强身份直接匹配 / 弱身份 src.stitching 概率拼接）都在
`src.trajectory.builder.TrajectoryBuilder` 中，两条路径的输出每段都带
`basis` / `basis_text` 标注依据，本模块不再自行聚合任何轨迹。
"""

from __future__ import annotations

import random  # 仅 _generate_fallback_mock() 使用；主回溯路径严禁随机数
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.logger import get_logger
from src.common.session_store import SessionNotFoundError, get_session_store
from src.trajectory.builder import (
    TrajectoryBuilder,
    TrajectoryDataUnavailableError,
    TrajectoryNotFoundError,
    get_trajectory_builder,
)

logger = get_logger("api.routes.backtrack")

router = APIRouter()

# 内存存储最近的回溯结果
_backtrack_results: Dict[str, Dict[str, Any]] = {}


# ============================================================
# Fallback mock（仅在数据文件不存在时使用）
# ============================================================

_FALLBACK_CAMERAS = [
    {"camera_id": "c001", "name": "Scene01-Camera01", "scene": "S01", "latitude": 42.526, "longitude": -90.7236},
    {"camera_id": "c010", "name": "Scene03-Camera01", "scene": "S03", "latitude": 42.4988, "longitude": -90.6874},
    {"camera_id": "c016", "name": "Scene04-Camera01", "scene": "S04", "latitude": 42.4978, "longitude": -90.6874},
]


def _generate_fallback_mock(instance_id: str) -> Dict[str, Any]:
    """
    演示用回溯结果（**非真实回溯**）

    仅在 output/cityflow_results.json 完全不存在、系统无任何真实数据可用时调用，
    内容为硬编码摄像头 + 随机数生成的演示轨迹，evidence.source 标记为 "fallback_mock"。
    数据文件存在但目标查不到时，接口返回 404，不落到这里。
    """
    base_time = datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)
    query_id = f"BT_{uuid.uuid4().hex[:8]}"

    observation_nodes = []
    observation_segments = []
    inference_segments = []

    for i, cam in enumerate(_FALLBACK_CAMERAS):
        enter_t = base_time + timedelta(seconds=i * 180)
        exit_t = enter_t + timedelta(seconds=random.randint(4, 12))
        node_ts = enter_t + timedelta(seconds=2)

        observation_nodes.append({
            "camera_id": cam["camera_id"],
            "camera_name": cam["name"],
            "tracklet_id": f"TRK_{uuid.uuid4().hex[:8]}",
            "timestamp": node_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": cam["latitude"],
            "longitude": cam["longitude"],
            "keyframe_path": None,
            "confidence": round(random.uniform(0.85, 0.99), 3),
        })
        observation_segments.append({
            "tracklet_id": f"TRK_{uuid.uuid4().hex[:8]}",
            "camera_id": cam["camera_id"],
            "camera_name": cam["name"],
            "start_time": enter_t.strftime("%H:%M:%S"),
            "end_time": exit_t.strftime("%H:%M:%S"),
            "entry_description": "从画面左侧进入",
            "exit_description": "从画面右侧离开",
            "direction": "由西向东" if cam.get("scene") == "S01" else "由北向南",
        })

    for i in range(len(_FALLBACK_CAMERAS) - 1):
        src = _FALLBACK_CAMERAS[i]
        tgt = _FALLBACK_CAMERAS[i + 1]
        inference_segments.append({
            "source_camera_id": src["camera_id"],
            "target_camera_id": tgt["camera_id"],
            "source_camera_name": src["name"],
            "target_camera_name": tgt["name"],
            "source_latitude": src["latitude"],
            "source_longitude": src["longitude"],
            "target_latitude": tgt["latitude"],
            "target_longitude": tgt["longitude"],
            "confidence": round(random.uniform(0.65, 0.92), 3),
            "estimated_travel_time": round(random.uniform(60, 200), 1),
            "actual_travel_time": round(random.uniform(80, 220), 1),
            "route_description": f"{src['name']} → {tgt['name']}",
        })

    candidate_paths = [{
        "path_id": f"PATH_{uuid.uuid4().hex[:6]}",
        "road_segments": ["SEG_001", "SEG_002"],
        "confidence": 0.72,
        "distance_meters": 770.0,
        "estimated_time": 95.0,
        "description": "Scene01-Camera01 → Scene03-Camera01 → Scene04-Camera01",
    }]

    result = {
        "query_id": query_id,
        "target_instance": {
            "instance_id": instance_id,
            "camera_id": _FALLBACK_CAMERAS[0]["camera_id"],
            "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_type": "vehicle",
            "attributes": {"颜色": "未知", "类型": "车辆"},
            "plate_number": None,
            "quality_score": 0.85,
        },
        "camera_sequence": [c["camera_id"] for c in _FALLBACK_CAMERAS],
        "observation_nodes": observation_nodes,
        "observation_segments": observation_segments,
        "inference_segments": inference_segments,
        "candidate_paths": candidate_paths,
        "overall_confidence": 0.78,
        "evidence": {
            "plate_consistency": 0.0,
            "appearance_similarity": 0.75,
            "attribute_consistency": 0.80,
            "temporal_feasibility": 0.79,
            "spatial_feasibility": 0.85,
            "source": "fallback_mock",
        },
    }
    _backtrack_results[query_id] = result
    return result


class BacktrackRequest(BaseModel):
    """回溯请求"""
    instance_id: Optional[str] = None   # 确认的目标实例 ID（缺省时从 session 的确认结果读取）
    query_id: Optional[str] = None      # 检索会话 ID，用于串联「检索→确认→回溯」
    max_upstream: int = 10              # 最大上游回溯深度
    max_downstream: int = 10            # 最大下游回溯深度
    # 拼接模式: auto 自动选路 / strong 强制强身份 / stitch 强制概率拼接
    # 注：当前数据集每条检测都带真实 vehicle_id，auto 恒走强身份路径，
    # 弱身份分支需显式传 stitch 驱动（用于验收与人工核查）
    mode: str = "auto"


class BacktrackResponse(BaseModel):
    """回溯响应"""
    query_id: str
    camera_sequence: List[str]          # 摄像头序列
    observation_nodes: List[Dict[str, Any]]
    observation_segments: List[Dict[str, Any]]
    inference_segments: List[Dict[str, Any]]
    candidate_paths: List[Dict[str, Any]]
    overall_confidence: float
    # 以下为新增字段，便于前端区分「强身份匹配」与「概率推断」
    identity: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    target_instance: Optional[Dict[str, Any]] = None


def _run_builder(
    builder: TrajectoryBuilder,
    instance_id: str,
    mode: str,
    max_upstream: int,
    max_downstream: int,
) -> Dict[str, Any]:
    """
    调用 TrajectoryBuilder，并把构建器异常映射成 HTTP 状态码

    - TrajectoryNotFoundError      → 404（数据可用但目标不存在）
    - TrajectoryDataUnavailableError → 500（数据文件不可用，调用方决定是否回退演示数据）
    """
    try:
        return builder.build(
            instance_id,
            mode=mode,
            max_upstream=max_upstream,
            max_downstream=max_downstream,
        )
    except TrajectoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TrajectoryDataUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trace", response_model=BacktrackResponse)
async def backtrack_trajectory(request: BacktrackRequest):
    """
    轨迹回溯

    以确认目标为锚点构建跨镜轨迹：强身份（有真实 vehicle_id）直接匹配，
    弱身份（无车牌 / 无真值）走 src.stitching 六维评分概率拼接。
    观测段来自真实检测，推断段的置信度与行程时间来自真实评分，
    每个节点/段/推断段都带 `basis` 标注其依据。

    instance_id 缺省时从会话（query_id）中读取用户已确认的目标；
    数据文件存在但目标查不到时返回 404，不再回落到随机目标或演示数据。
    """
    builder = get_trajectory_builder()
    store = get_session_store()

    # 1. 确定要回溯的目标：请求体优先，其次取会话中已确认的实例
    instance_id = request.instance_id
    if not instance_id and request.query_id:
        session = store.get(request.query_id)
        if session:
            instance_id = session.get("confirmed_instance_id")
    if not instance_id:
        raise HTTPException(
            status_code=404,
            detail="未指定 instance_id，且会话中没有已确认的目标",
        )

    # 2. 数据文件完全不存在时，退化为明确标注的演示数据
    if not builder.data_available:
        logger.warning(f"真实数据不可用，返回演示数据: instance_id={instance_id}")
        result = _generate_fallback_mock(instance_id)
    else:
        # 3. 交给 TrajectoryBuilder 构建（强身份 / 弱身份由 mode 决定）
        result = _run_builder(
            builder, instance_id, request.mode,
            request.max_upstream, request.max_downstream,
        )
        # 缓存供 GET /result/{query_id} 取回（沿用原有行为）
        _backtrack_results[result["query_id"]] = result
        logger.info(
            f"回溯成功: instance_id={instance_id}, "
            f"mode={result.get('identity', {}).get('mode')}, "
            f"cameras={len(result.get('camera_sequence', []))}, "
            f"confidence={result.get('overall_confidence')}"
        )

    # 4. 写回会话状态（confirmed → backtracked）
    if request.query_id:
        try:
            store.set_backtracked(request.query_id, result)
        except SessionNotFoundError:
            logger.warning(f"回溯完成但会话不存在，跳过状态写回: {request.query_id}")

    return BacktrackResponse(
        query_id=result["query_id"],
        camera_sequence=result["camera_sequence"],
        observation_nodes=result["observation_nodes"],
        observation_segments=result["observation_segments"],
        inference_segments=result["inference_segments"],
        candidate_paths=result["candidate_paths"],
        overall_confidence=result["overall_confidence"],
        identity=result.get("identity"),
        evidence=result.get("evidence"),
        target_instance=result.get("target_instance"),
    )


@router.get("/result/{query_id}")
async def get_backtrack_result(query_id: str):
    """
    获取回溯结果

    根据查询 ID 获取已完成的回溯结果。
    """
    result = _backtrack_results.get(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"回溯结果 {query_id} 不存在")
    return result


# ============================================================
# 跨镜轨迹回溯 API
# ============================================================

class TrajectoryRequest(BaseModel):
    """跨镜轨迹请求"""
    track_id: Optional[str] = None
    instance_id: Optional[str] = None


@router.post("/trajectory")
async def get_full_trajectory(request: TrajectoryRequest):
    """
    获取目标的完整跨镜轨迹。

    根据 track_id 或 instance_id，找到同一 vehicle_id 在所有摄像头中的检测记录，
    按时间排序构建跨摄像头时间线（聚合逻辑在 TrajectoryBuilder 中，
    与 /trace 共用同一份数据索引与同一套依据标注）。
    """
    builder = get_trajectory_builder()
    try:
        trajectory = builder.build_camera_timeline(
            track_id=request.track_id,
            instance_id=request.instance_id,
        )
    except TrajectoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TrajectoryDataUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "trajectory": trajectory}
