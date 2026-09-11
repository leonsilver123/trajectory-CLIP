"""
从AICity22 vdo.avi视频中提取检测帧并生成裁剪图，更新cityflow_results.json

流程:
1. 解析 cityflow_results.json，收集每个摄像头需要提取的帧号
2. 从 vdo.avi 中提取指定帧
3. 从帧中裁剪检测框
4. 更新 JSON 中的 image_path 和 crop_path
"""
import cv2
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# === 路径配置 ===
PROJECT_ROOT = Path("h:/trajectory CLIP")
RESULTS_PATH = PROJECT_ROOT / "output" / "cityflow_results.json"
FRAMES_ROOT = PROJECT_ROOT / "output" / "aicity22_frames"
CROPS_ROOT = PROJECT_ROOT / "output" / "aicity22_crops"
DATASET_ROOT = PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking"

# === 摄像头到 split/scene 的映射 ===
# train: S01(c001-c005), S03(c010-c015), S04(c016-c040)
# validation: S02(c006-c009), S05(c010,c016-c029,c033-c036)
# test: S06(c041-c046)
CAMERA_MAP = {}

def build_camera_map():
    """构建 camera_id -> (split, scene) 映射"""
    mapping = {}
    for split in ['train', 'validation', 'test']:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            for cam_dir in scene_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam = cam_dir.name
                vdo = cam_dir / "vdo.avi"
                if vdo.exists():
                    mapping[cam] = (split, scene)
    return mapping

def load_results():
    """加载 cityflow_results.json"""
    print(f"加载 {RESULTS_PATH} ...")
    t0 = time.time()
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  加载完成, 耗时 {time.time()-t0:.1f}s")
    print(f"  检测记录数: {len(data['detections'])}")
    return data

def collect_needed_frames(data):
    """收集每个摄像头需要提取的帧号及对应的检测索引"""
    # cam_id -> {frame_id: [det_indices]}
    cam_frames = defaultdict(lambda: defaultdict(list))
    
    for i, det in enumerate(data['detections']):
        cam = det.get('camera_id', '')
        fid = det.get('frame_id')
        if cam and fid is not None:
            cam_frames[cam][fid].append(i)
    
    total_needed = sum(len(fids) for fids in cam_frames.values())
    print(f"\n需要处理的摄像头: {len(cam_frames)} 个")
    print(f"需要提取的 (camera, frame) 组合: {total_needed} 个")
    
    return cam_frames

def find_video_path(cam_id, split, scene):
    """查找视频文件路径"""
    vdo = DATASET_ROOT / split / scene / cam_id / "vdo.avi"
    if vdo.exists():
        return vdo
    return None

def extract_frames_and_crops(cam_id, video_path, frame_det_map):
    """
    从视频中提取指定帧并裁剪检测框
    
    Args:
        cam_id: 摄像头ID (如 c001)
        video_path: vdo.avi 路径
        frame_det_map: {frame_id: [det_indices]} 每帧对应的检测索引
    
    Returns:
        extracted: 成功提取的帧数
        cropped: 成功裁剪的检测数
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return 0, 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed_frames = sorted(frame_det_map.keys())
    max_frame = max(needed_frames) if needed_frames else 0
    
    # 输出目录
    split, scene = CAMERA_MAP[cam_id]
    frame_dir = FRAMES_ROOT / split / scene / cam_id / "img1"
    crop_dir = CROPS_ROOT / cam_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    cropped = 0
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        if frame_id in frame_det_map:
            # 保存完整帧
            frame_path = frame_dir / f"{frame_id:06d}.jpg"
            if not frame_path.exists():
                cv2.imwrite(str(frame_path), frame)
            extracted += 1
            
            # 裁剪每个检测框
            for det_idx in frame_det_map[frame_id]:
                det = data_global['detections'][det_idx]
                bbox = det.get('bbox', [])
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # 确保坐标有效
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop_path = crop_dir / f"{cam_id}_f{frame_id:05d}_idx{det_idx:06d}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        
                        # 更新检测记录的路径
                        rel_frame = str(frame_path.relative_to(PROJECT_ROOT)).replace('/', '\\')
                        rel_crop = str(crop_path.relative_to(PROJECT_ROOT)).replace('/', '\\')
                        det['image_path'] = rel_frame
                        det['crop_path'] = rel_crop
                        cropped += 1
        
        # 如果已经超过需要的最大帧，提前退出
        if frame_id >= max_frame:
            break
        
        if frame_id % 1000 == 0:
            pct = frame_id / min(total_frames, max_frame) * 100
            print(f"    已处理 {frame_id}/{min(total_frames, max_frame)} 帧 ({pct:.0f}%)")
    
    cap.release()
    return extracted, cropped

def main():
    global data_global
    
    # 构建摄像头映射
    print("=" * 60)
    print("步骤1: 构建摄像头映射")
    print("=" * 60)
    global CAMERA_MAP
    CAMERA_MAP = build_camera_map()
    print(f"  找到 {len(CAMERA_MAP)} 个有效摄像头")
    for cam in sorted(CAMERA_MAP):
        split, scene = CAMERA_MAP[cam]
        print(f"    {cam} -> {split}/{scene}")
    
    # 加载检测结果
    print("\n" + "=" * 60)
    print("步骤2: 加载检测结果")
    print("=" * 60)
    data_global = load_results()
    
    # 收集需要的帧
    print("\n" + "=" * 60)
    print("步骤3: 收集需要提取的帧")
    print("=" * 60)
    cam_frames = collect_needed_frames(data_global)
    
    # 提取帧和裁剪
    print("\n" + "=" * 60)
    print("步骤4: 提取视频帧并生成裁剪图")
    print("=" * 60)
    
    total_extracted = 0
    total_cropped = 0
    skipped_cams = []
    
    for cam_id in sorted(cam_frames.keys()):
        frame_det_map = cam_frames[cam_id]
        n_frames = len(frame_det_map)
        n_dets = sum(len(v) for v in frame_det_map.values())
        
        if cam_id not in CAMERA_MAP:
            print(f"\n  [{cam_id}] 跳过 - 不在摄像头映射中 (非AICity22数据)")
            skipped_cams.append(cam_id)
            continue
        
        split, scene = CAMERA_MAP[cam_id]
        video_path = find_video_path(cam_id, split, scene)
        
        if video_path is None:
            print(f"\n  [{cam_id}] 跳过 - 视频文件不存在")
            skipped_cams.append(cam_id)
            continue
        
        print(f"\n  [{cam_id}] {split}/{scene} - 需提取 {n_frames} 帧, {n_dets} 个检测框")
        print(f"    视频: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        t0 = time.time()
        extracted, cropped = extract_frames_and_crops(cam_id, video_path, frame_det_map)
        elapsed = time.time() - t0
        
        total_extracted += extracted
        total_cropped += cropped
        print(f"    完成: 提取 {extracted} 帧, 裁剪 {cropped} 个检测框, 耗时 {elapsed:.1f}s")
    
    # 保存更新后的 JSON
    print("\n" + "=" * 60)
    print("步骤5: 保存更新后的 JSON")
    print("=" * 60)
    
    # 统计更新结果
    has_image = sum(1 for d in data_global['detections'] if d.get('image_path'))
    has_crop = sum(1 for d in data_global['detections'] if d.get('crop_path'))
    print(f"  image_path 已更新: {has_image}/{len(data_global['detections'])}")
    print(f"  crop_path 已更新: {has_crop}/{len(data_global['detections'])}")
    
    if skipped_cams:
        print(f"\n  跳过的摄像头 ({len(skipped_cams)}): {skipped_cams}")
    
    print(f"\n  总计: 提取 {total_extracted} 帧, 生成 {total_cropped} 裁剪图")
    
    # 保存
    print(f"\n  正在保存 JSON...")
    t0 = time.time()
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data_global, f, indent=2, ensure_ascii=False)
    print(f"  保存完成, 耗时 {time.time()-t0:.1f}s")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
