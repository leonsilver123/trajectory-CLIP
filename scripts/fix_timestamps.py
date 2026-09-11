"""
修复 cityflow_results.json 中的时间戳
根据帧号和FPS计算真实的视频时间戳
AICity22 Track1: 大部分摄像头 10 FPS, c015 为 8 FPS
"""
import json
from pathlib import Path
from datetime import datetime, timedelta

# AICity22 摄像头 FPS 配置
CAMERA_FPS = {
    "c015": 8,  # S03/c015 例外
}
DEFAULT_FPS = 10

# 假设所有视频从 2020-01-01 00:00:00 开始（实际需要根据数据集文档调整）
BASE_TIME = datetime(2020, 1, 1, 0, 0, 0)

def get_fps(camera_id: str) -> int:
    """获取摄像头的FPS"""
    return CAMERA_FPS.get(camera_id, DEFAULT_FPS)

def frame_to_timestamp(frame_id: int, camera_id: str) -> str:
    """将帧号转换为时间戳字符串"""
    fps = get_fps(camera_id)
    # 帧号从1开始，所以减1
    seconds = (frame_id - 1) / fps
    timestamp = BASE_TIME + timedelta(seconds=seconds)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{int((seconds % 1) * 1000):03d}"

def main():
    results_file = Path("output/cityflow_results.json")
    
    if not results_file.exists():
        print(f"❌ 文件不存在: {results_file}")
        return
    
    print(f" 读取 {results_file}...")
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections = data.get('detections', [])
    print(f"📊 共 {len(detections)} 条检测记录")
    
    # 统计需要更新的记录数
    updated_count = 0
    for det in detections:
        frame_id = det.get('frame_id')
        camera_id = det.get('camera_id')
        
        if frame_id and camera_id:
            old_timestamp = det.get('timestamp')
            new_timestamp = frame_to_timestamp(frame_id, camera_id)
            
            if old_timestamp != new_timestamp:
                det['timestamp'] = new_timestamp
                updated_count += 1
    
    print(f"✅ 更新了 {updated_count}/{len(detections)} 条记录的时间戳")
    
    # 保存回文件
    print(f"💾 保存回 {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("✨ 完成！请重启 API 服务以加载新数据")
    print("\n提示:")
    print("1. 停止当前运行的 API 服务 (Ctrl+C)")
    print("2. 重新启动: python scripts/run_server.py")
    print("3. 刷新浏览器页面 (Ctrl+F5 强制刷新)")

if __name__ == "__main__":
    main()
