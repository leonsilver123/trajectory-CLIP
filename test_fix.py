import requests
import time
import json

time.sleep(3)

r = requests.post('http://localhost:8000/api/v1/search/query',
                  json={'query_text': '白色车', 'top_k': 5},
                  timeout=30)
print(f"Status: {r.status_code}")
data = r.json()
candidates = data.get('candidates', [])
print(f"Candidates count: {len(candidates)}")
for item in candidates[:5]:
    attrs = item.get('attributes', {})
    print(f"  color_cn={attrs.get('颜色', 'MISSING')}, color_en={attrs.get('color', 'MISSING')}, camera={item.get('camera_id', '?')}")
