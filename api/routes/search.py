"""
api.routes.search - 检索 API v2（支持CLIP向量二次召回）

提供文本检索接口:
- POST /api/v1/search/query  文本查询检索候选目标（属性粗筛 + CLIP精排）
- POST /api/v1/search/plate  车牌精确查询

数据来源: output/cityflow_results.json（包含detections、tracks、det_to_track_map）
"""

from __future__ import annotations

import json
import os
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.logger import get_logger
from src.common.session_store import get_session_store

logger = get_logger("api.routes.search")

router = APIRouter()

# 项目根目录与结果文件路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_RESULTS_JSON = _OUTPUT_DIR / "cityflow_results.json"
_CLIP_INDEX_FILE = _OUTPUT_DIR / "clip_vectors.faiss"
_TRACK_CLIP_INDEX_FILE = _OUTPUT_DIR / "track_clip_vectors.faiss"
_CAMERA_METADATA_PATH = _PROJECT_ROOT / "configs" / "cityflow_camera_metadata.yaml"

# 加载摄像头名称映射
def _load_camera_name_map() -> Dict[str, str]:
    """从 cityflow_camera_metadata.yaml 加载摄像头ID到名称的映射"""
    name_map: Dict[str, str] = {}
    if _CAMERA_METADATA_PATH.exists():
        try:
            with open(_CAMERA_METADATA_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for cam in data.get("cameras", []):
                cid = cam.get("camera_id", "")
                cname = cam.get("name", cid)
                name_map[cid] = cname
        except Exception as e:
            logger.warning(f"加载摄像头元数据失败: {e}")
    return name_map

CAMERA_NAME_MAP = _load_camera_name_map()


class SearchRequest(BaseModel):
    """检索请求"""
    query_text: str                     # 用户查询文本
    top_k: int = 20                     # 召回数量
    target_type: Optional[str] = None   # 目标类别过滤


class PlateSearchRequest(BaseModel):
    """车牌查询请求"""
    plate_number: str                   # 车牌号


class SearchResponse(BaseModel):
    """检索响应"""
    query_id: str
    candidates: List[Dict[str, Any]]    # 候选目标列表
    total_count: int


def _load_results() -> Optional[Dict[str, Any]]:
    """加载 output/cityflow_results.json"""
    if not _RESULTS_JSON.exists():
        return None
    try:
        with open(_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 cityflow_results.json 失败: {e}")
        return None


def _load_clip_index() -> Optional[Any]:
    """加载Detection-level CLIP FAISS索引（带缓存）"""
    return _get_clip_index()


def _load_track_clip_index() -> Optional[Any]:
    """加载Track-level CLIP FAISS索引（带缓存）"""
    return _get_track_clip_index()


# --------------- CLIP 模型 & FAISS 索引缓存 ---------------
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_INDEX = None
_TRACK_CLIP_INDEX = None


def _load_clip_model():
    """加载 Chinese-CLIP 模型（模块级缓存，只加载一次）"""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    try:
        import torch
        from cn_clip.clip import load_from_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载 Chinese-CLIP ViT-B-16 模型, device={device}")
        _CLIP_MODEL, _CLIP_PREPROCESS = load_from_name(
            "ViT-B-16",
            device=device,
            download_root=str(_PROJECT_ROOT / "models")
        )
        _CLIP_MODEL.eval()
        logger.info("Chinese-CLIP 模型加载成功")
    except Exception as e:
        logger.error(f"Chinese-CLIP 模型加载失败: {e}")
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _encode_text(query_text: str) -> Optional[np.ndarray]:
    """使用 Chinese-CLIP 编码查询文本为 512 维归一化向量"""
    model, _ = _load_clip_model()
    if model is None:
        return None
    try:
        import torch
        import cn_clip.clip as clip
        tokens = clip.tokenize([query_text])
        tokens = tokens.to(next(model.parameters()).device)
        with torch.no_grad():
            text_embedding = model.encode_text(tokens)
            # L2 归一化（与图像特征构建时一致）
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"CLIP 文本编码失败: {e}")
        return None


def _get_clip_index():
    """获取 Detection-level FAISS 索引（缓存）"""
    global _CLIP_INDEX
    if _CLIP_INDEX is not None:
        return _CLIP_INDEX
    if not _CLIP_INDEX_FILE.exists():
        return None
    try:
        import faiss
        _CLIP_INDEX = faiss.read_index(str(_CLIP_INDEX_FILE))
        logger.info(f"加载 Detection CLIP 索引: {_CLIP_INDEX.ntotal} 个向量")
        return _CLIP_INDEX
    except Exception as e:
        logger.warning(f"加载 CLIP 索引失败: {e}")
        return None


def _get_track_clip_index():
    """获取 Track-level FAISS 索引（缓存）"""
    global _TRACK_CLIP_INDEX
    if _TRACK_CLIP_INDEX is not None:
        return _TRACK_CLIP_INDEX
    if not _TRACK_CLIP_INDEX_FILE.exists():
        return None
    try:
        import faiss
        _TRACK_CLIP_INDEX = faiss.read_index(str(_TRACK_CLIP_INDEX_FILE))
        logger.info(f"加载 Track CLIP 索引: {_TRACK_CLIP_INDEX.ntotal} 个向量")
        return _TRACK_CLIP_INDEX
    except Exception as e:
        logger.warning(f"加载 Track CLIP 索引失败: {e}")
        return None


def _clip_vector_search(query_vector: np.ndarray, index, top_k: int = 200) -> List[Tuple[int, float]]:
    """在 FAISS 索引中搜索最相似的向量（IndexFlatIP，向量已归一化）"""
    if index is None:
        return []
    try:
        import faiss
        # 确保查询向量形状正确 (1, dim)
        qv = query_vector.reshape(1, -1).astype(np.float32)
        # 归一化（确保与索引中的向量一致）
        faiss.normalize_L2(qv)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(qv, k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))
    except Exception as e:
        logger.error(f"CLIP 向量检索失败: {e}")
        return []


def _extract_query_features(query_text: str) -> Dict[str, Any]:
    """
    从查询文本中提取特征
    
    返回: {
        'color': 颜色关键词或None,
        'vehicle_type': 车型关键词或None,
        'keywords': 其他关键词列表
    }
    """
    query_lower = query_text.lower()
    
    # 颜色映射
    COLOR_MAP = {
        '白色': ['白', 'white'],
        '黑色': ['黑', 'black'],
        '红色': ['红', 'red'],
        '蓝色': ['蓝', 'blue'],
        '绿色': ['绿', 'green'],
        '黄色': ['黄', 'yellow'],
        '灰色': ['灰', 'gray', 'grey'],
        '棕色': ['棕', 'brown'],
        '银色': ['银', 'silver'],
    }
    
    # 车型映射
    TYPE_MAP = {
        '轿车': ['轿', 'sedan', 'car'],
        'SUV': ['suv', '越野'],
        '卡车': ['卡', 'truck', '货'],
        '面包车': ['面包', 'van'],
        '跑车': ['跑', 'sports'],
        '两厢车': ['两厢', 'hatchback'],
        'MPV': ['mpv', '商务'],
        '皮卡': ['皮卡', 'pickup'],
    }
    
    extracted = {
        'color': None,
        'vehicle_type': None,
        'keywords': []
    }
    
    # 提取颜色
    for color_cn, keywords in COLOR_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                extracted['color'] = color_cn
                break
        if extracted['color']:
            break
    
    # 提取车型
    for type_cn, keywords in TYPE_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                extracted['vehicle_type'] = type_cn
                break
        if extracted['vehicle_type']:
            break
    
    # 剩余关键词
    words = query_text.split()
    for word in words:
        word_lower = word.lower()
        is_color = any(kw in word_lower for colors in COLOR_MAP.values() for kw in colors)
        is_type = any(kw in word_lower for types in TYPE_MAP.values() for kw in types)
        if not is_color and not is_type:
            extracted['keywords'].append(word)
    
    return extracted


def _attribute_filter(detections: List[Dict], query_features: Dict) -> List[Dict]:
    """
    第一层：基于属性的粗筛
    
    根据提取的颜色和车型过滤检测结果
    """
    filtered = []
    
    for det in detections:
        attrs = det.get('attributes', {})
        det_color = attrs.get('color', '')
        det_type = attrs.get('vehicle_type', '')
        
        # 颜色过滤
        if query_features['color']:
            if det_color != query_features['color'] and det_color != 'unknown':
                continue
        
        # 车型过滤
        if query_features['vehicle_type']:
            if det_type != query_features['vehicle_type'] and det_type != 'unknown':
                continue
        
        filtered.append(det)
    
    logger.info(f"属性粗筛: {len(detections)} → {len(filtered)}")
    return filtered


def _filter_business_attributes(raw_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """过滤属性字段，只保留业务相关的中文键，隐藏技术字段。"""
    BUSINESS_KEYS = {"颜色", "车型", "车牌", "场景"}
    # 英文键到中文键的映射
    EN_TO_CN = {
        "color": "颜色",
        "vehicle_type": "车型",
        "plate": "车牌",
    }
    result = {}
    # 先保留中文业务键
    for key in BUSINESS_KEYS:
        if key in raw_attrs:
            result[key] = raw_attrs[key]
    # 如果中文键不存在，尝试从英文键映射
    for en_key, cn_key in EN_TO_CN.items():
        if cn_key not in result and en_key in raw_attrs:
            result[cn_key] = raw_attrs[en_key]
    return result


def _build_candidates_with_clip(data: Dict[str, Any], query: str, 
                                 query_features: Dict, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    构建候选列表（双层检索）
    
    流程：
    1. 属性粗筛
    2. CLIP向量精排（如果可用）
    3. 按轨迹聚合
    """
    detections = data.get("detections", [])
    tracks = data.get("tracks", [])
    det_to_track_map = data.get("det_to_track_map", {})
    
    # 预构建索引：track_id -> track 对象
    track_map: Dict[str, Dict] = {}
    for track in tracks:
        tid = track.get('track_id', '')
        if tid:
            track_map[tid] = track
    
    # 预构建反向索引：track_id -> 该轨迹的所有 detection 列表
    track_to_det_map: Dict[str, List[Dict]] = {}
    for det in detections:
        target_id = det.get('target_id', '')
        track_id = det_to_track_map.get(target_id, '')
        if track_id:
            track_to_det_map.setdefault(track_id, []).append(det)
    
    # Step 1: 属性粗筛
    filtered_dets = _attribute_filter(detections, query_features)
    
    if not filtered_dets:
        logger.warning("属性粗筛后无结果，返回空列表")
        return []
    
    # Step 2: CLIP 向量精排
    # 2a. 对查询文本做 CLIP 文本编码
    clip_scores: Dict[int, float] = {}  # det_index -> clip similarity
    query_vector = _encode_text(query)
    if query_vector is not None:
        clip_index = _get_clip_index()
        if clip_index is not None:
            clip_results = _clip_vector_search(query_vector, clip_index, top_k=200)
            for det_idx, score in clip_results:
                clip_scores[det_idx] = float(score)
            logger.info(f"CLIP 向量精排: top-{len(clip_results)} 结果, "
                        f"score range [{min(s for _, s in clip_results):.4f}, {max(s for _, s in clip_results):.4f}]")
        else:
            logger.info("CLIP 索引不可用，跳过向量精排")
    else:
        logger.info("CLIP 文本编码失败，跳过向量精排")

    # Step 3: 计算融合分数并排序
    # 为每个 detection 记录其在 detections 数组中的原始索引
    det_index_map: Dict[str, int] = {}
    for i, det in enumerate(detections):
        tid = det.get('target_id', '')
        if tid:
            det_index_map[tid] = i

    query_lower = query.lower()
    keywords = [k for k in query_lower.split() if k] if query_lower else []

    # 融合权重
    ALPHA = 0.6  # CLIP 向量相似度权重

    scored_dets = []
    for det in filtered_dets:
        t_type = det.get("target_type", "unknown")
        conf = det.get("confidence", 0.5)
        attrs = det.get("attributes", {})

        # 属性/文本匹配分数（作为 attribute_score）
        attr_score = conf
        if keywords:
            match_count = 0
            searchable = f"{t_type} {json.dumps(attrs, ensure_ascii=False)}".lower()
            for kw in keywords:
                if kw in searchable:
                    match_count += 1
            if match_count > 0:
                attr_score = min(0.5 + match_count * 0.15 + conf * 0.3, 0.99)

        # CLIP 向量相似度分数
        target_id = det.get('target_id', '')
        det_idx = det_index_map.get(target_id, -1)
        clip_score = clip_scores.get(det_idx, 0.0)

        # 融合排序
        if clip_scores:
            # 有 CLIP 结果时做融合
            final_score = ALPHA * clip_score + (1 - ALPHA) * attr_score
        else:
            # 无 CLIP 时退化为属性分数
            final_score = attr_score

        scored_dets.append((det, final_score, clip_score, attr_score))

    # 按融合分数降序
    scored_dets.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: 按轨迹聚合（去重）
    seen_tracks = set()
    candidates = []
    rank = 0

    for det, final_score, clip_score, attr_score in scored_dets:
        target_id = det.get('target_id', '')
        track_id = det_to_track_map.get(target_id, '')
        
        # 如果该轨迹已经出现过，跳过
        if track_id and track_id in seen_tracks:
            continue
        
        if track_id:
            seen_tracks.add(track_id)
        
        # 获取轨迹信息（从预构建索引查找）
        track_info = track_map.get(track_id) if track_id else None
        
        # 获取该轨迹的所有检测记录
        track_dets = track_to_det_map.get(track_id, []) if track_id else []
        
        # 构建候选项
        instance_id = det.get("target_id", f"DET_{rank+1:04d}")
        image_path = det.get("image_path", "")
        keyframe_path = det.get("keyframe_path", "")
        crop_path = det.get("crop_path", "")
        frame_id = det.get("frame_id", 0)
        
        # 优先使用crop图片
        final_keyframe = crop_path if crop_path and (Path(crop_path).exists() or (_OUTPUT_DIR / crop_path).exists()) else keyframe_path
        
        rank += 1
        candidate = {
            "instance_id": instance_id,
            "track_id": track_id,
            "camera_id": det.get("camera_id", "unknown"),
            "camera_name": CAMERA_NAME_MAP.get(det.get("camera_id", ""), det.get("camera_id", "unknown")),
            "timestamp": det.get("timestamp", f"frame_{frame_id:04d}"),
            "target_type": det.get("target_type", "unknown"),
            "attributes": _filter_business_attributes(det.get("attributes", {})),
            "plate_number": det.get("attributes", {}).get("plate"),
            "quality_score": det.get("confidence", 0.5),
            "clip_score": round(clip_score, 4),
            "attribute_score": round(attr_score, 4),
            "text_score": round(final_score, 4),
            "attribute_match_score": round(attr_score, 4),
            "combined_score": round(final_score, 4),
            "final_score": round(final_score, 4),
            "rank": rank,
            "keyframe_path": str(final_keyframe) if final_keyframe else None,
            "detection_bbox": det.get("bbox"),
            # 轨迹信息
            "has_trajectory": bool(track_info) or bool(track_dets),
            "trajectory_frame_range": track_info.get("frame_range") if track_info else None,
            "trajectory_camera_ids": track_info.get("camera_ids") if track_info else None,
            "trajectory_detection_count": track_info.get("detection_count") if track_info else len(track_dets),
            # 完整轨迹回溯信息
            "track_frame_count": len(track_dets),
            "track_cameras": list(set(d.get('camera_id', '') for d in track_dets)) if track_dets else (track_info.get('camera_ids', []) if track_info else []),
            "track_frames": [
                {
                    'frame_id': d.get('frame_id'),
                    'camera_id': d.get('camera_id', ''),
                    'camera_name': CAMERA_NAME_MAP.get(d.get('camera_id', ''), d.get('camera_id', '')),
                    'crop_path': d.get('crop_path', ''),
                    'timestamp': d.get('timestamp', ''),
                    'bbox': d.get('bbox'),
                    'confidence': d.get('confidence', 0.5),
                }
                for d in sorted(track_dets, key=lambda x: (x.get('frame_id', 0), x.get('camera_id', '')))[:20]
            ] if track_dets else [],
        }
        
        candidates.append(candidate)
        
        # 达到top_k后停止
        if len(candidates) >= top_k:
            break
    
    logger.info(f"最终候选数: {len(candidates)}")
    return candidates


@router.post("/query", response_model=SearchResponse)
async def search_by_text(request: SearchRequest):
    """
    文本查询检索（双层检索：属性粗筛 + CLIP精排）
    
    用户输入目标描述，系统：
    1. 解析查询文本，提取颜色、车型等属性
    2. 基于属性进行粗筛
    3. 使用CLIP向量相似度进行精排
    4. 按轨迹聚合，返回完整时空轨迹
    """
    data = _load_results()
    if data is None:
        raise HTTPException(
            status_code=500,
            detail="无法加载检测结果数据，请确保已运行预处理管线"
        )
    
    # 解析查询特征
    query_features = _extract_query_features(request.query_text)
    logger.info(f"查询特征: {query_features}")
    
    # 构建候选列表
    candidates = _build_candidates_with_clip(
        data, 
        request.query_text, 
        query_features,
        top_k=request.top_k
    )
    
    # 创建会话（后端状态真源：state=searched），后续 confirm/backtrack 共用该 query_id
    query_id = str(uuid.uuid4())
    get_session_store().create(
        query_id,
        request.query_text,
        candidate_count=len(candidates),
        query_type="text",
        target_type=request.target_type,
    )

    # 构建响应
    response = SearchResponse(
        query_id=query_id,
        candidates=candidates,
        total_count=len(candidates)
    )

    logger.info(f"检索完成: query='{request.query_text}', results={len(candidates)}")
    return response


@router.post("/plate", response_model=SearchResponse)
async def search_by_plate(request: PlateSearchRequest):
    """
    车牌精确查询
    
    根据车牌号查找对应车辆的所有检测记录
    """
    data = _load_results()
    if data is None:
        raise HTTPException(
            status_code=500,
            detail="无法加载检测结果数据"
        )
    
    # 直接使用_build_candidates，传入plate_number参数
    # （需要修改_build_candidates支持plate参数，这里简化处理）
    detections = data.get("detections", [])
    candidates = []
    rank = 0
    
    for det in detections:
        attrs = det.get("attributes", {})
        plate = attrs.get("plate", attrs.get("车牌", ""))
        
        if request.plate_number.lower() in plate.lower():
            rank += 1
            candidates.append({
                "instance_id": det.get("target_id", f"DET_{rank:04d}"),
                "camera_id": det.get("camera_id", "unknown"),
                "timestamp": det.get("timestamp", ""),
                "target_type": det.get("target_type", "unknown"),
                "attributes": attrs,
                "plate_number": plate,
                "quality_score": det.get("confidence", 0.5),
                "rank": rank,
                "keyframe_path": det.get("keyframe_path"),
            })
    
    top_candidates = candidates[:20]

    # 创建会话（后端状态真源：state=searched）
    query_id = str(uuid.uuid4())
    get_session_store().create(
        query_id,
        request.plate_number,
        candidate_count=len(top_candidates),
        query_type="plate",
    )

    response = SearchResponse(
        query_id=query_id,
        candidates=top_candidates,
        total_count=len(candidates)
    )

    return response
