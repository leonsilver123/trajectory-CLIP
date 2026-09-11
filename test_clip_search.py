"""
test_clip_search.py - 测试 CLIP 跨模态向量检索
"""
import requests
import json

API_URL = "http://localhost:8000/api/v1/search/query"

queries = ["白色轿车", "黑色SUV", "灰色面包车", "蓝色卡车", "红色跑车", "白色巴士"]

print("=" * 70)
print("CLIP 跨模态向量检索测试")
print("=" * 70)

for q in queries:
    print(f"\nQuery: {q}")
    try:
        r = requests.post(API_URL, json={"query_text": q, "top_k": 5}, timeout=30)
        data = r.json()
        candidates = data.get("candidates", data.get("results", []))
        print(f"  Results: {len(candidates)}")
        for i, c in enumerate(candidates[:3]):
            attrs = c.get("attributes", {})
            clip_score = c.get("clip_score", "N/A")
            final_score = c.get("final_score", c.get("combined_score", "N/A"))
            color = attrs.get("颜色", attrs.get("color", "?"))
            vtype = attrs.get("车型", attrs.get("vehicle_type", "?"))
            cam = c.get("camera_id", "?")
            ts = c.get("timestamp", "?")
            clip_str = f"{clip_score:.4f}" if isinstance(clip_score, float) else str(clip_score)
            final_str = f"{final_score:.4f}" if isinstance(final_score, float) else str(final_score)
            print(f"  {i+1}. color={color}, type={vtype}, cam={cam}, ts={ts}, clip={clip_str}, final={final_str}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("测试完成")
