"""
端到端检索流程校验脚本
验证从搜索→确认→轨迹回溯的完整链路
"""
import requests
import json
import sys

def test_complete_workflow():
    print("=" * 70)
    print("端到端检索流程校验")
    print("=" * 70)
    
    base_url = "http://localhost:8000"
    
    # Step 1: 搜索
    print("\n[Step 1] 搜索 '蓝色车'...")
    try:
        r1 = requests.post(f"{base_url}/api/v1/search/query", 
                           json={"query_text": "蓝色车", "top_k": 3},
                           timeout=10)
    except Exception as e:
        print(f"  ✗ API连接失败: {e}")
        print("  提示: 请确保 API 服务在端口 8000 运行")
        return False
    
    if r1.status_code != 200:
        print(f"  ✗ 搜索失败: HTTP {r1.status_code}")
        print(f"  响应: {r1.text[:500]}")
        return False
    
    data1 = r1.json()
    candidates = data1.get("candidates", [])
    
    if not candidates:
        print("  ✗ 无搜索结果")
        return False
    
    print(f"  ✓ 找到 {len(candidates)} 个结果")
    
    candidate = candidates[0]
    instance_id = candidate.get("instance_id", "")
    track_id = candidate.get("track_id", "")
    
    print(f"  ✓ 第一个结果:")
    print(f"    - instance_id: {instance_id[:50]}...")
    print(f"    - track_id: {track_id[:50] if track_id else '(空)'}...")
    print(f"    - 颜色: {candidate.get('attributes', {}).get('颜色', 'N/A')}")
    print(f"    - 车型: {candidate.get('attributes', {}).get('车型', 'N/A')}")
    print(f"    - 摄像头: {candidate.get('camera_name', 'N/A')}")
    print(f"    - clip_score: {candidate.get('clip_score', 0):.4f}")
    print(f"    - final_score: {candidate.get('final_score', 0):.4f}")
    print(f"    - has_trajectory: {candidate.get('has_trajectory', False)}")
    print(f"    - track_frame_count: {candidate.get('track_frame_count', 0)}")
    print(f"    - track_cameras: {candidate.get('track_cameras', [])}")
    
    # 检查关键字段是否存在
    if not instance_id:
        print("  ✗ 缺少 instance_id 字段")
        return False
    
    if not track_id:
        print("  ⚠️ 警告: 缺少 track_id 字段（可能影响跨镜轨迹）")
    
    # Step 2: 跨镜轨迹 API
    print("\n[Step 2] 调用跨镜轨迹 API /api/v1/backtrack/trajectory...")
    try:
        r2 = requests.post(f"{base_url}/api/v1/backtrack/trajectory",
                           json={"instance_id": instance_id, "track_id": track_id},
                           timeout=10)
    except Exception as e:
        print(f"  ✗ API连接失败: {e}")
        return False
    
    if r2.status_code != 200:
        print(f"  ✗ 轨迹API失败: HTTP {r2.status_code}")
        print(f"  响应: {r2.text[:500]}")
        return False
    
    data2 = r2.json()
    
    if not data2.get("success"):
        print(f"  ✗ 轨迹API返回失败: {data2.get('message', '未知错误')}")
        return False
    
    traj = data2.get("trajectory", {})
    total_cameras = traj.get("total_cameras", 0)
    total_detections = traj.get("total_detections", 0)
    duration = traj.get("total_duration_seconds", 0)
    camera_seq = traj.get("camera_sequence", [])
    
    print(f"  ✓ 轨迹数据:")
    print(f"    - track_id: {traj.get('track_id', 'N/A')[:50]}...")
    print(f"    - 首次出现: {traj.get('first_appearance', 'N/A')}")
    print(f"    - 最后出现: {traj.get('last_appearance', 'N/A')}")
    print(f"    - 经过摄像头数: {total_cameras}")
    print(f"    - 检测帧数: {total_detections}")
    print(f"    - 总时长: {duration}秒")
    
    # 关键验证点
    checks_passed = True
    
    if total_cameras < 2:
        print(f"  ✗ 严重: 摄像头数 {total_cameras} < 2，跨镜拼接失败")
        checks_passed = False
    else:
        print(f"  ✓ 跨镜拼接成功: {total_cameras}个摄像头")
    
    if total_detections == 0:
        print(f"  ✗ 严重: 检测帧数为 0")
        checks_passed = False
    else:
        print(f"  ✓ 检测帧数正常: {total_detections}帧")
    
    if duration == 0:
        print(f"  ✗ 严重: 总时长为 0 秒")
        checks_passed = False
    else:
        print(f"  ✓ 总时长正常: {duration}秒")
    
    # 检查摄像头序列
    print(f"\n  ✓ 摄像头序列 ({len(camera_seq)}个):")
    for idx, cam in enumerate(camera_seq[:5], 1):
        cam_name = cam.get('camera_name', 'N/A')
        arrival = cam.get('arrival_time', 'N/A')
        departure = cam.get('departure_time', 'N/A')
        det_count = cam.get('detection_count', 0)
        frames = cam.get('frames', [])
        
        print(f"    {idx}. {cam_name}")
        print(f"       时间: {arrival} -> {departure}")
        print(f"       帧数: {det_count} (实际: {len(frames)})")
        
        if det_count == 0 or len(frames) == 0:
            print(f"        警告: 该摄像头无帧数据")
            checks_passed = False
    
    if len(camera_seq) > 5:
        print(f"    ... 还有 {len(camera_seq) - 5} 个摄像头未显示")
    
    # Step 3: 总结
    print("\n" + "=" * 70)
    if checks_passed:
        print("✓✓✓ 全流程校验通过！系统工作正常")
        print("=" * 70)
        return True
    else:
        print("✗✗ 全流程校验失败！发现以下问题:")
        print("  1. 跨镜轨迹拼接不完整（摄像头数 < 2）")
        print("  2. 或检测帧数为 0")
        print("  3. 或总时长为 0 秒")
        print("  4. 或部分摄像头无帧数据")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
