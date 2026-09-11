"""测试跨镜轨迹回溯 API"""
import requests
import json

API_BASE = "http://localhost:8000"

# 1. 先测试 API 健康检查
print("=" * 60)
print("1. 健康检查")
try:
    r = requests.get(f"{API_BASE}/health", timeout=5)
    print(f"   状态: {r.status_code} - {r.json()}")
except Exception as e:
    print(f"   API 不可用: {e}")
    print("   请先启动 API: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
    exit(1)

# 2. 用已知 instance_id 测试跨镜轨迹 API
print("\n" + "=" * 60)
print("2. 测试跨镜轨迹 API (instance_id=CF3_c001_V0034_000001)")
r2 = requests.post(
    f"{API_BASE}/api/v1/backtrack/trajectory",
    json={"instance_id": "CF3_c001_V0034_000001"},
    timeout=30,
)
print(f"   状态码: {r2.status_code}")
if r2.status_code == 200:
    data = r2.json()
    traj = data.get("trajectory", {})
    print(f"   success: {data.get('success')}")
    print(f"   vehicle_id: {traj.get('vehicle_id')}")
    print(f"   first_appearance: {traj.get('first_appearance')}")
    print(f"   last_appearance: {traj.get('last_appearance')}")
    print(f"   total_cameras: {traj.get('total_cameras')}")
    print(f"   total_detections: {traj.get('total_detections')}")
    print(f"   total_duration_seconds: {traj.get('total_duration_seconds')}")
    print(f"   attributes: {traj.get('attributes')}")
    print(f"\n   Camera Sequence:")
    for cam in traj.get("camera_sequence", []):
        print(f"     - {cam['camera_name']} ({cam['camera_id']}): "
              f"{cam['arrival_time']} -> {cam['departure_time']} "
              f"({cam['detection_count']} frames, {cam['direction']})")
else:
    print(f"   错误: {r2.text}")

# 3. 用 track_id 测试
print("\n" + "=" * 60)
print("3. 测试跨镜轨迹 API (track_id=CF3_TRACK_c001_V0034)")
r3 = requests.post(
    f"{API_BASE}/api/v1/backtrack/trajectory",
    json={"track_id": "CF3_TRACK_c001_V0034"},
    timeout=30,
)
print(f"   状态码: {r3.status_code}")
if r3.status_code == 200:
    data3 = r3.json()
    traj3 = data3.get("trajectory", {})
    print(f"   vehicle_id: {traj3.get('vehicle_id')}")
    print(f"   total_cameras: {traj3.get('total_cameras')}")
    print(f"   total_detections: {traj3.get('total_detections')}")
else:
    print(f"   错误: {r3.text}")

# 4. 测试另一个 vehicle
print("\n" + "=" * 60)
print("4. 测试另一个 vehicle (instance_id=CF3_c001_V0001_000001)")
r4 = requests.post(
    f"{API_BASE}/api/v1/backtrack/trajectory",
    json={"instance_id": "CF3_c001_V0001_000001"},
    timeout=30,
)
print(f"   状态码: {r4.status_code}")
if r4.status_code == 200:
    data4 = r4.json()
    traj4 = data4.get("trajectory", {})
    print(f"   vehicle_id: {traj4.get('vehicle_id')}")
    print(f"   total_cameras: {traj4.get('total_cameras')}")
    print(f"   total_detections: {traj4.get('total_detections')}")
    for cam in traj4.get("camera_sequence", []):
        print(f"     - {cam['camera_name']} ({cam['camera_id']}): "
              f"{cam['arrival_time']} -> {cam['departure_time']} "
              f"({cam['detection_count']} frames)")
else:
    print(f"   错误: {r4.text}")

print("\n" + "=" * 60)
print("测试完成!")
