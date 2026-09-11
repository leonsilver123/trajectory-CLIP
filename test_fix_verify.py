"""Verify the fix for colors and vehicle types."""
import requests
import json

queries = ["白色轿车", "黑色SUV", "灰色面包车", "蓝色卡车", "红色车"]
for q in queries:
    try:
        r = requests.post("http://localhost:8000/api/v1/search/query", json={"query_text": q, "top_k": 5}, timeout=15)
        data = r.json()
        candidates = data.get("candidates", data.get("results", []))
        print(f"\nQuery: {q} -> {len(candidates)} results")
        for i, c in enumerate(candidates[:3]):
            attrs = c.get("attributes", {})
            print(f"  {i+1}. color={attrs.get('颜色','?')}, type={attrs.get('车型','?')}, clip={c.get('clip_score','N/A')}")
    except Exception as e:
        print(f"\nQuery: {q} -> ERROR: {e}")
