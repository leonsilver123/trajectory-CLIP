import requests
import json

BASE = "http://localhost:8000"

def test_search(query, expected_color=None):
    """测试搜索并验证结果"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    r = requests.post(f"{BASE}/api/v1/search/query", json={"query_text": query, "top_k": 10})
    print(f"Status: {r.status_code}")
    data = r.json()
    results = data.get("candidates", data.get("results", data.get("data", {}).get("results", [])))
    print(f"Results count: {len(results)}")
    
    if not results:
        print("WARNING: No results!")
        return False
    
    all_ok = True
    for i, item in enumerate(results[:5]):
        attrs = item.get("attributes", {})
        color = attrs.get("颜色", "MISSING")
        color_en = attrs.get("color", "MISSING")
        vehicle_type = attrs.get("车型", attrs.get("vehicle_type", "MISSING"))
        camera_name = item.get("camera_name", "MISSING")
        has_traj = item.get("has_trajectory", False)
        track_frames = item.get("track_frames", [])
        track_cameras = item.get("track_cameras", [])
        track_frame_count = item.get("track_frame_count", 0)
        crop_path = item.get("keyframe_path", item.get("crop_path", ""))
        
        print(f"\n  Result {i+1}:")
        print(f"    颜色: {color} (en: {color_en})")
        print(f"    车型: {vehicle_type}")
        print(f"    Camera: {camera_name}")
        print(f"    Has trajectory: {has_traj}")
        print(f"    Track frame count: {track_frame_count}")
        print(f"    Track cameras: {track_cameras}")
        print(f"    Track frames sample: {len(track_frames)} frames")
        print(f"    Crop path: {crop_path[:60]}...")
        
        # 验证
        if color == "MISSING":
            print(f"    ERROR: 颜色字段缺失!")
            all_ok = False
        if expected_color and color != expected_color and color != "unknown":
            print(f"    WARNING: 期望颜色={expected_color}, 实际={color}")
        if has_traj and track_frame_count == 0:
            print(f"    WARNING: has_trajectory=True but track_frame_count=0")
        if track_frames:
            frame = track_frames[0]
            print(f"    First frame: camera={frame.get('camera_name','?')}, frame_id={frame.get('frame_id','?')}")
    
    return all_ok

# 测试用例
print("=" * 60)
print("端到端验证测试")
print("=" * 60)

# 测试1: 白色车
ok1 = test_search("白色车", expected_color="白色")

# 测试2: 黑色车
ok2 = test_search("黑色车", expected_color="黑色")

# 测试3: 灰色车
ok3 = test_search("灰色车", expected_color="灰色")

# 测试4: 蓝色车
ok4 = test_search("蓝色车", expected_color="蓝色")

# 总结
print(f"\n{'='*60}")
print("测试总结")
print(f"{'='*60}")
print(f"白色车: {'PASS' if ok1 else 'FAIL'}")
print(f"黑色车: {'PASS' if ok2 else 'FAIL'}")
print(f"灰色车: {'PASS' if ok3 else 'FAIL'}")
print(f"蓝色车: {'PASS' if ok4 else 'FAIL'}")
