import requests
import json

BASE = "http://localhost:8000"

def test_query(query, top_k=10):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    r = requests.post(f"{BASE}/api/v1/search/query", json={"query_text": query, "top_k": top_k})
    print(f"Status: {r.status_code}")
    data = r.json()
    candidates = data.get("candidates", data.get("results", []))
    print(f"Results: {len(candidates)}")
    
    if not candidates:
        print("WARNING: No results!")
        return
    
    for i, c in enumerate(candidates[:5]):
        attrs = c.get("attributes", {})
        color = attrs.get("颜色", "MISSING")
        vtype = attrs.get("车型", "MISSING")
        camera = c.get("camera_name", "?")
        clip_score = c.get("clip_score", "N/A")
        final_score = c.get("final_score", "N/A")
        has_traj = c.get("has_trajectory", False)
        track_count = c.get("track_frame_count", 0)
        
        clip_str = f"{clip_score:.4f}" if isinstance(clip_score, (int, float)) else str(clip_score)
        final_str = f"{final_score:.4f}" if isinstance(final_score, (int, float)) else str(final_score)
        
        print(f"  {i+1}. color={color}, type={vtype}, camera={camera}, clip={clip_str}, final={final_str}, traj={has_traj}({track_count}frames)")

# 测试各种查询
test_query("白色轿车")
test_query("黑色SUV")
test_query("灰色面包车")
test_query("蓝色卡车")
test_query("红色车")

print(f"\n{'='*60}")
print("Verification complete")
