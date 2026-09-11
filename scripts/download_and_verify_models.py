"""
下载并验证模型权重
- YOLOv8x: 车辆检测
- Chinese-CLIP: 图像-文本特征提取
- OSNet: ReID特征提取
"""
import os
from pathlib import Path
import subprocess
import sys

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print(" 模型权重下载与验证")
print("=" * 60)

# 1. 下载 YOLOv8x
yolo_model_path = MODELS_DIR / "yolov8x.pt"
if not yolo_model_path.exists():
    print("\n[1/3] 下载 YOLOv8x 模型...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8x.pt')
        # YOLO会自动下载到 ~/.ultralytics/models/
        # 我们需要复制到项目的models目录
        import shutil
        default_path = Path.home() / ".ultralytics" / "models" / "yolov8x.pt"
        if default_path.exists():
            shutil.copy(default_path, yolo_model_path)
            print(f"OK YOLOv8x 已保存到: {yolo_model_path}")
        else:
            print(f"⚠️  YOLOv8x 默认路径不存在: {default_path}")
            print("   请手动下载: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt")
    except Exception as e:
        print(f"❌ 下载 YOLOv8x 失败: {e}")
        print("   请手动下载并放到 models/yolov8x.pt")
else:
    print(f"\n[1/3] OK YOLOv8x 已存在: {yolo_model_path}")

# 2. 检查 Chinese-CLIP
print("\n[2/3] 检查 Chinese-CLIP...")
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
    print("OK Chinese-CLIP 已安装")
    
    # 尝试加载模型
    clip_model_path = MODELS_DIR / "chinese-clip-vit-base-patch16.pt"
    if not clip_model_path.exists():
        print("   ️  首次使用将自动下载 Chinese-CLIP 模型...")
        print("   (这可能需要几分钟，取决于网络速度)")
        try:
            model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=str(MODELS_DIR))
            print(f"OK Chinese-CLIP 已保存到: {MODELS_DIR}")
        except Exception as e:
            print(f"⚠️  自动下载失败: {e}")
            print("   请手动下载 Chinese-CLIP 模型")
    else:
        print(f"OK Chinese-CLIP 模型已存在")
        
except ImportError:
    print(" Chinese-CLIP 未安装")
    print("   请运行: pip install cn_clip")

# 3. 检查 FAISS（用于向量检索）
print("\n[3/3] 检查 FAISS...")
try:
    import faiss
    print("OK FAISS 已安装")
except ImportError:
    print("❌ FAISS 未安装")
    print("   请运行: pip install faiss-cpu  (CPU版本)")
    print("   或: pip install faiss-gpu  (GPU版本)")

# 4. 测试 YOLO 推理
print("\n" + "=" * 60)
print(" 测试 YOLOv8x 推理...")
print("=" * 60)

if yolo_model_path.exists():
    try:
        from ultralytics import YOLO
        model = YOLO(str(yolo_model_path))
        
        # 找一张测试图片
        test_image = None
        crops_dir = PROJECT_ROOT / "output" / "aicity22_crops"
        if crops_dir.exists():
            for root, dirs, files in os.walk(crops_dir):
                if files:
                    test_image = Path(root) / files[0]
                    break
        
        if test_image and test_image.exists():
            print(f"\n测试图片: {test_image}")
            results = model.predict(str(test_image), verbose=False)
            
            # 解析结果
            boxes = results[0].boxes
            if len(boxes) > 0:
                print(f"OK 检测到 {len(boxes)} 个目标")
                for i, box in enumerate(boxes[:3]):  # 只显示前3个
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    print(f"   [{i+1}] 类别={cls_id}, 置信度={conf:.3f}, 框=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            else:
                print("⚠️  未检测到任何目标")
        else:
            print("⚠️  未找到测试图片，跳过推理测试")
            
    except Exception as e:
        print(f"❌ YOLO 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  YOLOv8x 模型不存在，跳过推理测试")

print("\n" + "=" * 60)
print("✨ 模型检查完成！")
print("=" * 60)
print("\n下一步:")
print("1. 如果模型缺失，请手动下载或使用上述脚本自动下载")
print("2. 运行 scripts/batch_inference.py 进行批量推理")
print("3. 运行 scripts/build_clip_index.py 构建CLIP向量索引")
