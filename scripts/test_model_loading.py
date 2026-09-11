"""
scripts/test_model_loading.py - AI 模型权重加载与特征提取验证

逐一验证项目中使用的 AI 模型:
1. YOLO 目标检测 (ultralytics)
2. Chinese-CLIP 图文特征 (transformers)
3. ReID 外观特征 (torchvision ResNet50)
4. 车牌 OCR (可选)

禁止使用模拟数据，全部使用真实图片进行 forward pass。
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

# ============================================================
# 全局配置
# ============================================================
REPORT_LINES: list[str] = []
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = DATA_DIR / "model_validation_report.txt"

# 测试图片路径：优先使用 ultralytics 自带图片
ULTRA_ASSETS = Path(__file__).resolve().parent.parent / "ultralytics_assets"


def log(msg: str = "") -> None:
    """打印并记录日志"""
    print(msg)
    REPORT_LINES.append(msg)


def get_test_images() -> list[tuple[str, np.ndarray]]:
    """
    获取测试用真实交通图片。
    优先级：
    1. data/ 目录下的图片
    2. ultralytics 自带的 bus.jpg / zidane.jpg
    3. 从网上下载交通场景图片
    """
    images: list[tuple[str, np.ndarray]] = []

    # 1. 尝试从 data/ 读取
    if DATA_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in DATA_DIR.glob(ext):
                try:
                    img = np.array(Image.open(p).convert("RGB"))
                    if img.shape[0] > 50 and img.shape[1] > 50:
                        images.append((p.name, img))
                except Exception:
                    pass
        if len(images) >= 2:
            return images[:4]  # 最多取 4 张

    # 2. 使用 ultralytics 自带图片
    try:
        import ultralytics
        assets_dir = Path(ultralytics.__file__).parent / "assets"
        for p in assets_dir.glob("*.jpg"):
            img = np.array(Image.open(p).convert("RGB"))
            images.append((p.name, img))
        if images:
            print(f"[INFO] 使用 ultralytics 自带测试图片: {[n for n, _ in images]}")
            return images
    except Exception:
        pass

    # 3. 从网上下载
    print("[INFO] 尝试从网络下载测试图片...")
    urls = [
        ("https://ultralytics.com/images/bus.jpg", "bus_download.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "zidane_download.jpg"),
    ]
    try:
        import requests
        for url, fname in urls:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                from io import BytesIO
                pil = Image.open(BytesIO(resp.content)).convert("RGB")
                img = np.array(pil)
                images.append((fname, img))
    except Exception as e:
        print(f"[WARN] 下载测试图片失败: {e}")

    if not images:
        raise RuntimeError("无法获取任何测试图片！请检查网络连接或手动放置图片到 data/ 目录。")

    return images


# ============================================================
# 1. YOLO 目标检测
# ============================================================
def test_yolo(images: list[tuple[str, np.ndarray]]) -> dict:
    """验证 YOLO 模型加载与推理"""
    result = {
        "name": "YOLO 目标检测",
        "model": "yolov8x",
        "status": "✗ 失败",
        "weight_file": "N/A",
        "weight_size_mb": 0,
        "load_time": 0,
        "num_detections": 0,
        "infer_time_ms": 0,
        "detected_classes": [],
        "error": "",
    }

    try:
        from ultralytics import YOLO

        model_name = "yolov8x.pt"
        log(f"  加载模型: {model_name} ...")

        t0 = time.time()
        model = YOLO(model_name)
        load_time = time.time() - t0
        result["load_time"] = load_time

        # 获取权重文件信息
        weight_path = model.ckpt_path if hasattr(model, "ckpt_path") else model_name
        if os.path.exists(weight_path):
            size_mb = os.path.getsize(weight_path) / (1024 * 1024)
            result["weight_file"] = f"{Path(weight_path).name} ({size_mb:.1f} MB)"
            result["weight_size_mb"] = size_mb
        else:
            result["weight_file"] = model_name

        log(f"  加载时间: {load_time:.2f}s")
        log(f"  权重文件: {result['weight_file']}")

        # 推理测试 - 使用第一张图片
        test_img = images[0][1]
        log(f"  测试图片: {images[0][0]} ({test_img.shape[1]}x{test_img.shape[0]})")

        t0 = time.time()
        results = model.predict(
            source=test_img,
            conf=0.25,
            iou=0.45,
            imgsz=[640, 640],
            device="cuda",
            verbose=False,
        )
        infer_time = (time.time() - t0) * 1000
        result["infer_time_ms"] = infer_time

        # 解析结果
        all_classes = []
        total_dets = 0
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                n = len(boxes)
                total_dets += n
                for i in range(n):
                    cls_id = int(boxes.cls[i].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    all_classes.append(cls_name)

        result["num_detections"] = total_dets
        result["detected_classes"] = list(set(all_classes))
        result["status"] = "✓ 加载成功"

        log(f"  推理时间: {infer_time:.1f}ms")
        log(f"  检测到 {total_dets} 个目标")
        log(f"  检测类别: {result['detected_classes']}")

        # 清理 GPU 显存
        del model
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        result["status"] = f"✗ 失败: {e}"
        log(f"  [ERROR] {e}")
        traceback.print_exc()

    return result


# ============================================================
# 2. Chinese-CLIP
# ============================================================
def test_chinese_clip(images: list[tuple[str, np.ndarray]]) -> dict:
    """验证 Chinese-CLIP 模型加载与特征提取"""
    result = {
        "name": "Chinese-CLIP",
        "model": "N/A",
        "status": "✗ 失败",
        "text_dim": 0,
        "image_dim": 0,
        "text_norm": 0,
        "image_norm": 0,
        "cosine_sim": 0,
        "error": "",
    }

    # 尝试多个模型，从大到小
    model_candidates = [
        ("OFA-Sys/chinese-clip-vit-large-patch14", "ViT-L-14"),
        ("OFA-Sys/chinese-clip-vit-base-patch16", "ViT-B-16"),
    ]

    import torch

    for hf_name, short_name in model_candidates:
        log(f"  尝试加载: {hf_name} ...")
        try:
            from transformers import CLIPModel, CLIPProcessor

            t0 = time.time()
            clip_model = CLIPModel.from_pretrained(hf_name)
            clip_processor = CLIPProcessor.from_pretrained(hf_name)
            load_time = time.time() - t0

            clip_model = clip_model.to("cuda")
            clip_model.eval()

            result["model"] = hf_name
            log(f"  加载成功 ({load_time:.2f}s)")

            # --- 文本编码 ---
            test_text = "蓝色轿车"
            inputs = clip_processor(text=[test_text], return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            with torch.no_grad():
                text_features = clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                text_features_norm = torch.nn.functional.normalize(text_features, p=2, dim=1)

            text_vec = text_features_norm.cpu().numpy().flatten()
            result["text_dim"] = len(text_vec)
            result["text_norm"] = float(np.linalg.norm(text_vec))

            log(f"  文本 \"{test_text}\" 向量维度: {result['text_dim']}, L2范数: {result['text_norm']:.4f}")

            # --- 图像编码 ---
            pil_img = Image.fromarray(images[0][1])  # RGB
            img_inputs = clip_processor(images=pil_img, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to("cuda")

            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=pixel_values)
                image_features_norm = torch.nn.functional.normalize(image_features, p=2, dim=1)

            image_vec = image_features_norm.cpu().numpy().flatten()
            result["image_dim"] = len(image_vec)
            result["image_norm"] = float(np.linalg.norm(image_vec))

            log(f"  图像向量维度: {result['image_dim']}, L2范数: {result['image_norm']:.4f}")

            # --- 余弦相似度 ---
            cosine_sim = float(np.dot(text_vec, image_vec))
            result["cosine_sim"] = cosine_sim
            log(f"  文本-图片余弦相似度: {cosine_sim:.4f}")

            result["status"] = "✓ 加载成功"

            # 清理
            del clip_model, clip_processor
            torch.cuda.empty_cache()
            break  # 成功则跳出

        except Exception as e:
            log(f"  [WARN] {hf_name} 加载失败: {e}")
            result["error"] = str(e)
            import torch
            torch.cuda.empty_cache()
            continue

    if result["status"] != "✓ 加载成功":
        log(f"  [ERROR] 所有 Chinese-CLIP 模型均加载失败")
        log(f"  最后错误: {result['error']}")
        # 尝试 ModelScope 镜像
        log("  尝试 ModelScope 镜像下载...")
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download("OFA-Sys/chinese-clip-vit-large-patch14")
            log(f"  ModelScope 下载到: {model_dir}")
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained(model_dir)
            clip_processor = CLIPProcessor.from_pretrained(model_dir)
            clip_model = clip_model.to("cuda").eval()
            result["model"] = "OFA-Sys/chinese-clip-vit-large-patch14 (ModelScope)"
            result["status"] = "✓ 加载成功 (ModelScope)"
            log("  ModelScope 镜像加载成功!")
            del clip_model, clip_processor
            torch.cuda.empty_cache()
        except ImportError:
            log("  [INFO] modelscope 未安装，跳过镜像方案")
        except Exception as e2:
            log(f"  [ERROR] ModelScope 方案也失败: {e2}")
            result["error"] = str(e2)

    return result


# ============================================================
# 3. ReID 模型
# ============================================================
def test_reid(images: list[tuple[str, np.ndarray]]) -> dict:
    """验证 ReID 特征提取（使用 torchvision ResNet50 backbone）"""
    result = {
        "name": "ReID 模型",
        "model": "ResNet50 (torchvision pretrained)",
        "status": "✗ 失败",
        "feature_dim": 0,
        "vec_norm": 0,
        "cosine_sim": 0,
        "error": "",
    }

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        log(f"  加载 ResNet50 预训练权重 ...")

        t0 = time.time()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        load_time = time.time() - t0

        # 构建 backbone（与 feature_extractor.py 一致）
        backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        flatten = nn.Flatten()
        embedding_layer = nn.Linear(2048, 512)

        backbone = backbone.to("cuda")
        embedding_layer = embedding_layer.to("cuda")
        backbone.eval()
        embedding_layer.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        log(f"  加载时间: {load_time:.2f}s")
        result["feature_dim"] = 512

        # 对两张不同图片提取特征
        def extract_reid_feat(img_np):
            pil = Image.fromarray(img_np)
            tensor = transform(pil).unsqueeze(0).to("cuda")
            with torch.no_grad():
                feat = backbone(tensor)
                feat = flatten(feat)
                emb = embedding_layer(feat)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().flatten()

        # 图片 A
        img_a = images[0][1]
        vec_a = extract_reid_feat(img_a)
        result["vec_norm"] = float(np.linalg.norm(vec_a))
        log(f"  图片A ({images[0][0]}): 特征维度={len(vec_a)}, L2范数={result['vec_norm']:.4f}")

        # 图片 B
        if len(images) > 1:
            img_b = images[1][1]
        else:
            img_b = img_a  # 同一张图（自比较应为 1.0）
        vec_b = extract_reid_feat(img_b)
        log(f"  图片B ({images[min(1, len(images)-1)][0]}): 特征维度={len(vec_b)}, L2范数={np.linalg.norm(vec_b):.4f}")

        cosine_sim = float(np.dot(vec_a, vec_b))
        result["cosine_sim"] = cosine_sim
        log(f"  图片A-图片B 余弦相似度: {cosine_sim:.4f}")

        result["status"] = "✓ 加载成功"

        # 清理
        del resnet, backbone, flatten, embedding_layer
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        result["status"] = f"✗ 失败: {e}"
        log(f"  [ERROR] {e}")
        traceback.print_exc()

    return result


# ============================================================
# 4. 车牌 OCR（可选）
# ============================================================
def test_ocr(images: list[tuple[str, np.ndarray]]) -> dict:
    """验证车牌 OCR 模型（可选）"""
    result = {
        "name": "车牌 OCR",
        "model": "N/A",
        "status": "✗ 未安装",
        "error": "",
    }

    # 检查 EasyOCR
    try:
        import easyocr
        log("  检测到 EasyOCR，尝试加载 ...")
        t0 = time.time()
        reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
        load_time = time.time() - t0
        result["model"] = "EasyOCR (ch_sim+en)"
        log(f"  EasyOCR 加载成功 ({load_time:.2f}s)")

        # 推理测试
        test_img = images[0][1]
        t0 = time.time()
        ocr_results = reader.readtext(test_img)
        infer_time = (time.time() - t0) * 1000
        result["status"] = "✓ 加载成功"
        log(f"  推理时间: {infer_time:.1f}ms")
        log(f"  识别到 {len(ocr_results)} 个文本区域")
        if ocr_results:
            texts = [r[1] for r in ocr_results[:5]]
            log(f"  前5个识别结果: {texts}")
        del reader
        import torch
        torch.cuda.empty_cache()
        return result
    except ImportError:
        log("  EasyOCR 未安装，跳过")
    except Exception as e:
        log(f"  EasyOCR 加载失败: {e}")
        result["error"] = str(e)

    # 检查 PaddleOCR
    try:
        from paddleocr import PaddleOCR
        log("  检测到 PaddleOCR，尝试加载 ...")
        t0 = time.time()
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
        load_time = time.time() - t0
        result["model"] = "PaddleOCR (ch)"
        log(f"  PaddleOCR 加载成功 ({load_time:.2f}s)")
        result["status"] = "✓ 加载成功"
        del ocr
        import torch
        torch.cuda.empty_cache()
        return result
    except ImportError:
        log("  PaddleOCR 未安装，跳过")
    except Exception as e:
        log(f"  PaddleOCR 加载失败: {e}")
        result["error"] = str(e)

    result["status"] = "✗ 未安装 (EasyOCR/PaddleOCR 均未安装)"
    return result


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 56)
    log("          模型权重验证报告")
    log("=" * 56)
    log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  工作目录: {PROJECT_ROOT}")

    # 环境信息
    import torch
    log(f"  PyTorch: {torch.__version__}")
    log(f"  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    log("")

    # 获取测试图片
    log("[准备] 获取测试图片 ...")
    try:
        images = get_test_images()
        log(f"  共获取 {len(images)} 张测试图片:")
        for name, img in images:
            log(f"    - {name}: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
    except Exception as e:
        log(f"  [FATAL] 无法获取测试图片: {e}")
        save_report()
        return
    log("")

    # 逐一测试
    results = []

    log("[1/4] YOLO 目标检测")
    r1 = test_yolo(images)
    results.append(r1)
    log("")

    log("[2/4] Chinese-CLIP")
    r2 = test_chinese_clip(images)
    results.append(r2)
    log("")

    log("[3/4] ReID 模型")
    r3 = test_reid(images)
    results.append(r3)
    log("")

    log("[4/4] 车牌 OCR")
    r4 = test_ocr(images)
    results.append(r4)
    log("")

    # 总结
    passed = sum(1 for r in results if r["status"].startswith("✓"))
    log("=" * 56)
    log(f"  总结: {passed}/{len(results)} 模型验证通过")
    log("=" * 56)
    for r in results:
        icon = "✓" if r["status"].startswith("✓") else "✗"
        log(f"  {icon} {r['name']}: {r['status']}")
    log("")

    # 保存报告
    save_report()


def save_report():
    """保存报告到文件"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(REPORT_LINES)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[INFO] 报告已保存至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
