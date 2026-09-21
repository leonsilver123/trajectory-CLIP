"""
修复轨迹数据并建立检测-轨迹关联

解决批量推理后轨迹属性丢失的问题，确保公安业务平台的轨迹回溯功能完整可用。

流程：
1. 加载cityflow_results.json（包含detections和tracks）
2. 构建target_id → track_id的映射表
3. 对每个track，聚合其所有detections的属性（投票机制）
4. 为每个track提取代表性帧的CLIP向量
5. 更新tracks数组并保存
6. 构建Track-level FAISS索引
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import faiss
from collections import Counter

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "output" / "cityflow_results.json"
CROPS_DIR = PROJECT_ROOT / "output" / "aicity22_crops"
TRACK_CLIP_INDEX_FILE = PROJECT_ROOT / "output" / "track_clip_vectors.faiss"

print("=" * 70)
print(" 修复轨迹数据 & 建立检测-轨迹关联")
print("=" * 70)

# 加载结果JSON
print("\n[1/6] 加载检测结果...")
with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

detections = data['detections']
tracks = data.get('tracks', [])

print(f"   检测数: {len(detections)}")
print(f"   轨迹数: {len(tracks)}")

# 构建target_id → detection的映射
print("\n[2/6] 构建 target_id → detection 映射...")
target_to_det = {}
for det in detections:
    target_id = det.get('target_id', '')
    if target_id:
        target_to_det[target_id] = det

print(f"   映射表大小: {len(target_to_det)}")

# 构建target_id → track_id的映射（通过解析target_id）
print("\n[3/6] 构建 target_id → track_id 映射...")
# AICity22的target_id格式: CF3_c001_V0034_000001
# track_id格式: CF3_TRACK_c001_V0034
# 需要从target_id中提取vehicle_id来匹配track

det_track_map = {}  # target_id → track_id
for det in detections:
    target_id = det.get('target_id', '')
    camera_id = det.get('camera_id', '')
    
    # 从target_id提取vehicle_id (CF3_c001_V0034_000001 → V0034)
    parts = target_id.split('_')
    if len(parts) >= 3:
        vehicle_id = parts[2]  # V0034
        # 构造track_id
        track_id = f"CF3_TRACK_{camera_id}_{vehicle_id}"
        det_track_map[target_id] = track_id

print(f"   检测-轨迹映射数: {len(det_track_map)}")

# 预加载CLIP模型（避免在循环中重复加载）
print("\n   预加载 CLIP 模型...")
try:
    from cn_clip.clip import load_from_name
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = load_from_name(
        "ViT-B-16", 
        device=device, 
        download_root=str(PROJECT_ROOT / "models")
    )
    clip_model.eval()
    print(f"   OK CLIP模型已加载到 {device}")
except Exception as e:
    print(f"   ERROR CLIP模型加载失败: {e}")
    clip_model = None
    preprocess = None

# 对每个track，聚合其所有detections的属性
print("\n[4/6] 聚合轨迹属性...")
updated_tracks = []
track_clip_vectors = []

def extract_clip_vector(image_path):
    """提取单张图片的CLIP向量"""
    if clip_model is None or preprocess is None:
        return None
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        return None

for track in tqdm(tracks, desc="处理轨迹"):
    track_id = track.get('track_id', '')
    
    # 找到该轨迹对应的所有detections
    track_dets = []
    for target_id, mapped_track_id in det_track_map.items():
        if mapped_track_id == track_id and target_id in target_to_det:
            track_dets.append(target_to_det[target_id])
    
    if not track_dets:
        # 没有找到对应的detection，保留原样
        updated_tracks.append(track)
        continue
    
    # 投票机制确定颜色
    colors = [d['attributes']['color'] for d in track_dets if d['attributes']['color'] != 'unknown']
    if colors:
        color_counter = Counter(colors)
        most_common_color = color_counter.most_common(1)[0][0]
    else:
        most_common_color = 'unknown'
    
    # 投票机制确定车型
    types = [d['attributes']['vehicle_type'] for d in track_dets if d['attributes']['vehicle_type'] != 'unknown']
    if types:
        type_counter = Counter(types)
        most_common_type = type_counter.most_common(1)[0][0]
    else:
        most_common_type = 'unknown'
    
    # 更新track的属性
    track['attributes']['color'] = most_common_color
    track['attributes']['vehicle_type'] = most_common_type
    
    # 提取代表性帧的CLIP向量（取中间帧）
    frame_range = track.get('frame_range', [0, 0])
    mid_frame = (frame_range[0] + frame_range[1]) // 2
    
    # 找到最接近中间帧的detection
    best_det = min(track_dets, key=lambda d: abs(d.get('frame_id', 0) - mid_frame))
    crop_path = best_det.get('crop_path', '')
    
    clip_vec = None
    if crop_path:
        full_crop_path = None
        if Path(crop_path).exists():
            full_crop_path = crop_path
        elif (PROJECT_ROOT / crop_path).exists():
            full_crop_path = str(PROJECT_ROOT / crop_path)
        
        if full_crop_path and Path(full_crop_path).exists():
            clip_vec = extract_clip_vector(full_crop_path)
    
    updated_tracks.append(track)
    if clip_vec is not None:
        track_clip_vectors.append(clip_vec)

print(f"\n   更新轨迹数: {len(updated_tracks)}")
print(f"   有效CLIP向量数: {len(track_clip_vectors)}")

# 统计更新后的属性分布
print("\n[5/6] 统计轨迹属性分布...")
track_colors = [t.get('attributes', {}).get('color', 'unknown') for t in updated_tracks]
track_types = [t.get('attributes', {}).get('vehicle_type', 'unknown') for t in updated_tracks]

color_dist = dict(sorted({c: track_colors.count(c) for c in set(track_colors)}.items(), 
                         key=lambda x: x[1], reverse=True)[:10])
type_dist = dict(sorted({t: track_types.count(t) for t in set(track_types)}.items(), 
                        key=lambda x: x[1], reverse=True)[:10])

print(f"   颜色分布TOP10: {color_dist}")
print(f"   车型分布TOP10: {type_dist}")

unknown_tracks = sum(1 for t in updated_tracks if t.get('attributes', {}).get('color', 'unknown') == 'unknown')
print(f"   剩余unknown轨迹: {unknown_tracks} ({unknown_tracks/len(updated_tracks)*100:.1f}%)")

# 更新JSON中的tracks
data['tracks'] = updated_tracks

# 添加detection→track映射表（用于快速查询）
data['det_to_track_map'] = det_track_map

# 保存更新后的JSON
print("\n[6/6] 保存结果...")
with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"   OK 已保存到: {RESULTS_JSON}")

# 构建Track-level FAISS索引
if len(track_clip_vectors) > 0:
    print("\n[7/7] 构建 Track-level CLIP 向量索引 (FAISS)...")
    
    vectors = np.array(track_clip_vectors, dtype=np.float32)
    dimension = vectors.shape[1]
    
    print(f"   向量数量: {len(vectors)}")
    print(f"   向量维度: {dimension}")
    
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    faiss.write_index(index, str(TRACK_CLIP_INDEX_FILE))
    print(f"   OK Track-level FAISS索引已保存到: {TRACK_CLIP_INDEX_FILE}")
    
    # 测试检索
    print("\n   测试轨迹检索示例:")
    test_query = vectors[0:1]
    distances, indices = index.search(test_query, 3)
    print(f"   查询轨迹 {indices[0]} 的最近邻距离: {distances[0]}")
else:
    print("\n   WARNING 没有有效的轨迹CLIP向量，跳过索引构建")

print("\n" + "=" * 70)
print("轨迹数据修复完成！")
print("=" * 70)
print(f"\n改进效果:")
print(f"  - 轨迹总数: {len(tracks)} → {len(updated_tracks)}")
print(f"  - unknown轨迹: {unknown_tracks} ({unknown_tracks/len(updated_tracks)*100:.1f}%)")
print(f"  - Track-level CLIP向量: {len(track_clip_vectors)}")
print(f"\n下一步:")
print(f"  1. 修改API检索接口，实现双层检索（Detection + Track）")
print(f"  2. 在搜索结果中展示完整轨迹（多摄像头视角）")
print(f"  3. 重启API服务并测试")
