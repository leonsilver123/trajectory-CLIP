import requests
import json

# 测试1: 搜索"白色车"
print("=" * 60)
print("测试1: 搜索'白色车'")
print("=" * 60)
r = requests.post('http://localhost:8000/api/v1/search/query', 
                  json={'query_text': '白色车', 'top_k': 5})
print(f'状态码: {r.status_code}')
data = r.json()
print(f'结果数: {data["total_count"]}')
if data['candidates']:
    print('\n第一条结果:')
    print(json.dumps(data['candidates'][0], indent=2, ensure_ascii=False))

# 测试2: 搜索"卡车"
print("\n" + "=" * 60)
print("测试2: 搜索'卡车'")
print("=" * 60)
r = requests.post('http://localhost:8000/api/v1/search/query', 
                  json={'query_text': '卡车', 'top_k': 5})
print(f'状态码: {r.status_code}')
data = r.json()
print(f'结果数: {data["total_count"]}')
if data['candidates']:
    print('\n第一条结果:')
    print(json.dumps(data['candidates'][0], indent=2, ensure_ascii=False))

# 测试3: 检查摄像头名称
print("\n" + "=" * 60)
print("测试3: 检查摄像头名称映射")
print("=" * 60)
if data['candidates']:
    for i, cand in enumerate(data['candidates'][:3]):
        print(f"结果{i+1}: camera_id={cand.get('camera_id')}, camera_name={cand.get('camera_name')}")
