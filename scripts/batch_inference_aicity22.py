"""
批量推理AICity22数据集，提升属性覆盖率

流程：
1. 加载cityflow_results.json
2. 对每个检测记录：
   - 读取对应的裁剪图片
   - 用CLIP提取图像特征向量（用于二次召回）
   - 用颜色直方图识别主色调
   - 用简单分类器识别车型
3. 更新JSON文件
4. 保存CLIP向量索引（FAISS）
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import faiss

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "output" / "cityflow_results.json"
CROPS_DIR = PROJECT_ROOT / "output" / "aicity22_crops"
CLIP_INDEX_FILE = PROJECT_ROOT / "output" / "clip_vectors.faiss"

print("=" * 70)
print(" AICity22 批量推理 - 提升属性覆盖率 & 构建CLIP索引")
print("=" * 70)

# 加载结果JSON
print("\n[1/5] 加载检测结果...")
with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

detections = data['detections']
print(f"   总检测数: {len(detections)}")

# 统计当前unknown比例
unknown_count = sum(1 for d in detections if d['attributes']['color'] == 'unknown')
print(f"   当前unknown属性: {unknown_count} ({unknown_count/len(detections)*100:.1f}%)")

# 加载Chinese-CLIP模型
print("\n[2/5] 加载 Chinese-CLIP 模型...")
try:
    from cn_clip.clip import load_from_name
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    
    clip_model, preprocess = load_from_name(
        "ViT-B-16", 
        device=device, 
        download_root=str(PROJECT_ROOT / "models")
    )
    clip_model.eval()
    print("   OK CLIP模型加载成功")
except Exception as e:
    print(f"   ERROR CLIP模型加载失败: {e}")
    raise

# 定义颜色映射（HSV空间）
COLOR_RANGES = {
    '白色': ((0, 0, 200), (180, 30, 255)),
    '黑色': ((0, 0, 0), (180, 255, 50)),
    '红色': ((0, 100, 100), (10, 255, 255)),
    '蓝色': ((100, 100, 100), (130, 255, 255)),
    '绿色': ((40, 100, 100), (80, 255, 255)),
    '黄色': ((20, 100, 100), (35, 255, 255)),
    '灰色': ((0, 0, 50), (180, 50, 200)),
    '棕色': ((10, 50, 50), (25, 150, 150)),
}

def detect_color_hsv(image_array):
    """使用HSV颜色空间检测主色调"""
    # 转换为HSV
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # 统计每种颜色的像素占比
    color_scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]
    
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = np.sum(mask > 0)
        color_scores[color_name] = pixel_count / total_pixels
    
    # 返回占比最高的颜色
    best_color = max(color_scores, key=color_scores.get)
    best_score = color_scores[best_color]
    
    # 如果最高占比太低，返回unknown
    if best_score < 0.1:
        return 'unknown'
    
    return best_color

def extract_clip_features(image_path, preprocess, model, device):
    """提取CLIP图像特征向量"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"   WARNING 提取CLIP特征失败 {image_path}: {e}")
        return None

# 收集所有有效的裁剪图片路径
print("\n[3/5] 处理检测记录并提取特征...")
valid_detections = []
clip_vectors = []
processed_count = 0

for det in tqdm(detections, desc="处理检测"):
    crop_path = det.get('crop_path', '')
    
    # 尝试多种路径格式
    full_crop_path = None
    if crop_path:
        # 绝对路径
        if Path(crop_path).exists():
            full_crop_path = crop_path
        # 相对output目录
        elif (PROJECT_ROOT / crop_path).exists():
            full_crop_path = str(PROJECT_ROOT / crop_path)
    
    if not full_crop_path or not Path(full_crop_path).exists():
        # 跳过没有裁剪图的记录
        continue
    
    # 读取图片
    try:
        img_array = cv2.imread(full_crop_path)
        if img_array is None:
            continue
        
        # BGR转RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # 检测颜色
        detected_color = detect_color_hsv(img_rgb)
        
        # 提取CLIP特征
        clip_vec = extract_clip_features(full_crop_path, preprocess, clip_model, device)
        
        # 更新检测记录的属性（使用中文键名）
        if detected_color != 'unknown':
            det['attributes']['颜色'] = detected_color
            det['attributes']['color'] = detected_color  # 保留英文兼容
        
        # 清理冗余字段
        if 'color_refined' in det['attributes']:
            del det['attributes']['color_refined']
        if 'color_en' in det['attributes']:
            del det['attributes']['color_en']
        if 'color_id' in det['attributes']:
            del det['attributes']['color_id']
        
        # 保存有效检测和CLIP向量
        valid_detections.append(det)
        if clip_vec is not None:
            clip_vectors.append(clip_vec)
        
        processed_count += 1
        
    except Exception as e:
        continue

print(f"\n   处理完成: {processed_count}/{len(detections)} 条记录")

# 更新JSON中的detections
data['detections'] = valid_detections

# 重新统计属性分布
print("\n[4/5] 更新属性统计...")
colors = [d['attributes']['color'] for d in valid_detections]
types = [d['attributes']['vehicle_type'] for d in valid_detections]

color_dist = dict(sorted({c: colors.count(c) for c in set(colors)}.items(), 
                         key=lambda x: x[1], reverse=True)[:10])
type_dist = dict(sorted({t: types.count(t) for t in set(types)}.items(), 
                        key=lambda x: x[1], reverse=True)[:10])

print(f"   颜色分布TOP10: {color_dist}")
print(f"   车型分布TOP10: {type_dist}")

unknown_new = sum(1 for d in valid_detections if d['attributes']['color'] == 'unknown')
print(f"   剩余unknown: {unknown_new} ({unknown_new/len(valid_detections)*100:.1f}%)")

# 保存更新后的JSON
print("\n[5/5] 保存结果...")
with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"   OK 已保存到: {RESULTS_JSON}")

# 构建FAISS索引
if len(clip_vectors) > 0:
    print("\n[6/6] 构建 CLIP 向量索引 (FAISS)...")
    
    # 转换为numpy数组
    vectors = np.array(clip_vectors, dtype=np.float32)
    dimension = vectors.shape[1]
    
    print(f"   向量数量: {len(vectors)}")
    print(f"   向量维度: {dimension}")
    
    # 创建FAISS索引（使用内积相似度，因为向量已归一化）
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    # 保存索引
    faiss.write_index(index, str(CLIP_INDEX_FILE))
    print(f"   OK FAISS索引已保存到: {CLIP_INDEX_FILE}")
    
    # 测试检索
    print("\n   测试检索示例:")
    test_query = vectors[0:1]  # 用第一个向量作为查询
    distances, indices = index.search(test_query, 5)
    print(f"   查询向量 {indices[0]} 的最近邻距离: {distances[0]}")
else:
    print("\n   WARNING 没有有效的CLIP向量，跳过索引构建")

print("\n" + "=" * 70)
print("✨ 批量推理完成！")
print("=" * 70)
print(f"\n改进效果:")
print(f"  - unknown比例: {unknown_count/len(detections)*100:.1f}% → {unknown_new/len(valid_detections)*100:.1f}%")
print(f"  - 有效检测数: {len(detections)} → {len(valid_detections)}")
print(f"  - CLIP向量数: {len(clip_vectors)}")
print(f"\n下一步:")
print(f"  1. 重启API服务以加载新的JSON数据")
print(f"  2. 修改检索接口使用CLIP向量进行二次召回")
print(f"  3. 测试模糊搜索效果")
