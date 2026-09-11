"""诊断轨迹数据完整性"""
import json
import sys

# 1. 检查 cityflow_results.json 中 tracks 结构
print("=" * 60)
print("1. 检查 cityflow_results.json 数据结构")
print("=" * 60)

with open("output/cityflow_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"detections 数量: {len(data.get('detections', []))}")
print(f"tracks 数量: {len(data.get('tracks', []))}")
print(f"det_to_track_map 条目数: {len(data.get('det_to_track_map', {}))}")

# 检查 tracks 结构
tracks = data.get("tracks", [])
if tracks:
    print("\n--- 第1条 track 完整结构 ---")
    print(json.dumps(tracks[0], ensure_ascii=False, indent=2))
    if len(tracks) > 10:
        print("\n--- 第10条 track 完整结构 ---")
        print(json.dumps(tracks[10], ensure_ascii=False, indent=2))
    if len(tracks) > 100:
        print("\n--- 第100条 track 完整结构 ---")
        print(json.dumps(tracks[100], ensure_ascii=False, indent=2))
    
    # 统计 tracks 中是否有 detections 字段
    with_dets = sum(1 for t in tracks if 'detections' in t and t['detections'])
    print(f"\ntracks 中有 detections 字段的数量: {with_dets}/{len(tracks)}")
    
    # 检查 track 的字段
    print(f"\ntrack 的 keys: {list(tracks[0].keys())}")

# 2. 检查 det_to_track_map
det_map = data.get("det_to_track_map", {})
print(f"\n{'=' * 60}")
print("2. det_to_track_map 检查")
print(f"{'=' * 60}")
# 取前5个映射
for i, (k, v) in enumerate(det_map.items()):
    if i >= 5:
        break
    print(f"  {k} -> {v}")

# 反向构建 track_to_detections
track_to_dets = {}
for det_id, track_id in det_map.items():
    if track_id not in track_to_dets:
        track_to_dets[track_id] = []
    track_to_dets[track_id].append(det_id)

print(f"\n反向映射 track_to_detections 条目数: {len(track_to_dets)}")
# 取3条看看
for i, (tid, dets) in enumerate(track_to_dets.items()):
    if i >= 3:
        break
    print(f"  track {tid}: {len(dets)} 个检测")

# 3. 检查 detections 结构
dets = data.get("detections", [])
if dets:
    print(f"\n{'=' * 60}")
    print("3. detection 结构示例")
    print(f"{'=' * 60}")
    print(f"detection keys: {list(dets[0].keys())}")
    print(json.dumps(dets[0], ensure_ascii=False, indent=2))
