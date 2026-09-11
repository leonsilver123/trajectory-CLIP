"""
src.perception.feature_extractor - 特征提取模块

提取目标的外观特征向量，用于跨镜匹配和图文检索:
- ReID 特征向量 (外观相似性匹配, dim=512)
- Chinese-CLIP 图像特征向量 (图文跨模态检索, dim=768)

输入: 目标裁剪图像
输出: 归一化特征向量 (numpy array)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.common.logger import get_logger

logger = get_logger("perception.feature_extractor")


class FeatureExtractor:
    """
    特征提取器

    同时提供 ReID 和 Chinese-CLIP 两种特征提取能力。
    ReID 用于跨镜外观匹配，CLIP 用于图文跨模态检索。

    ReID: 使用 torchvision 预训练 ResNet50 作为 backbone，
          去掉最后分类层，加一个 512 维 embedding 层。
    CLIP: 使用 transformers 加载 Chinese-CLIP 或 openai CLIP 模型。

    使用方式:
        extractor = FeatureExtractor(device="cuda")
        reid_vec = extractor.extract_reid(crop_image)       # shape: (512,)
        clip_vec = extractor.extract_clip(crop_image)       # shape: (768,)
        reid_vecs = extractor.extract_reid_batch(crops)     # shape: (N, 512)
    """

    def __init__(
        self,
        reid_model: str = "osnet_x1_0",
        reid_dim: int = 512,
        clip_model: str = "CN-CLIP-ViT-L-14",
        clip_dim: int = 768,
        reid_batch_size: int = 32,
        clip_batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        """
        初始化特征提取器

        Args:
            reid_model: ReID 模型名称
            reid_dim: ReID 向量维度
            clip_model: Chinese-CLIP 模型名称
            clip_dim: CLIP 向量维度
            reid_batch_size: ReID 批处理大小
            clip_batch_size: CLIP 批处理大小
            device: 推理设备
        """
        self.reid_model_name = reid_model
        self.reid_dim = reid_dim
        self.clip_model_name = clip_model
        self.clip_dim = clip_dim
        self.reid_batch_size = reid_batch_size
        self.clip_batch_size = clip_batch_size
        self.device = device

        # 延迟初始化
        self._reid_model = None
        self._reid_embedding_layer = None
        self._reid_transform = None
        self._clip_model = None
        self._clip_processor = None

        logger.info(
            f"特征提取器初始化: reid={reid_model}(dim={reid_dim}), "
            f"clip={clip_model}(dim={clip_dim}), device={device}"
        )

    def _lazy_init_reid(self) -> None:
        """
        延迟初始化 ReID 模型

        使用 torchvision 预训练 ResNet50 作为 backbone，
        去掉最后分类层，添加一个 512 维 embedding 层。
        """
        if self._reid_model is not None:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            # 加载预训练 ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # 去掉最后的全连接层，保留特征提取部分
            # ResNet50 的 fc 层输入是 2048 维
            self._reid_backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                resnet.avgpool,
            )
            # 展平层
            self._reid_flatten = nn.Flatten()
            # 512 维 embedding 层（从 2048 维映射到 512 维）
            self._reid_embedding_layer = nn.Linear(2048, self.reid_dim)

            # 移到设备
            self._reid_backbone = self._reid_backbone.to(self.device)
            self._reid_embedding_layer = self._reid_embedding_layer.to(self.device)
            self._reid_backbone.eval()
            self._reid_embedding_layer.eval()

            # 图像预处理
            self._reid_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._reid_model = True  # 标记已初始化
            logger.info("ReID 模型加载完成 (ResNet50 backbone)")

        except Exception as e:
            logger.error(f"ReID 模型加载失败: {e}")
            self._reid_model = None
            raise

    def _lazy_init_clip(self) -> None:
        """
        延迟初始化 CLIP 模型

        尝试加载 Chinese-CLIP 或 openai CLIP 模型。
        """
        if self._clip_model is not None:
            return

        try:
            import torch

            # 尝试加载 Chinese-CLIP
            clip_model_name = self.clip_model_name
            # 映射配置中的模型名到 HuggingFace 模型路径
            model_map = {
                "CN-CLIP-ViT-L-14": "OFA-Sys/chinese-clip-vit-large-patch14",
                "CN-CLIP-ViT-B-16": "OFA-Sys/chinese-clip-vit-base-patch16",
                "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14": "openai/clip-vit-large-patch14",
            }
            hf_model_name = model_map.get(clip_model_name, clip_model_name)

            logger.info(f"加载 CLIP 模型: {hf_model_name}")

            # Chinese-CLIP 需要使用专用模型类，否则文本编码器权重无法正确加载
            is_chinese_clip = "chinese-clip" in hf_model_name.lower()
            if is_chinese_clip:
                from transformers import ChineseCLIPModel, ChineseCLIPProcessor
                self._clip_model = ChineseCLIPModel.from_pretrained(hf_model_name)
                self._clip_processor = ChineseCLIPProcessor.from_pretrained(hf_model_name)
            else:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_model = CLIPModel.from_pretrained(hf_model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(hf_model_name)

            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()

            logger.info(f"CLIP 模型加载完成: {hf_model_name}")

        except Exception as e:
            logger.error(f"CLIP 模型加载失败: {e}")
            self._clip_model = None
            raise

    def extract_reid(self, crop_image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 ReID 特征向量

        Args:
            crop_image: 目标裁剪图像 (H, W, 3) BGR 格式

        Returns:
            L2 归一化的 ReID 向量 (dim,)，失败返回 None
        """
        self._lazy_init_reid()

        try:
            import torch

            # BGR → RGB
            rgb_image = crop_image[:, :, ::-1].copy()

            # 预处理
            tensor = self._reid_transform(rgb_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 提取特征
                features = self._reid_backbone(tensor)
                features = self._reid_flatten(features)
                # 通过 embedding 层
                embedding = self._reid_embedding_layer(features)
                # L2 归一化
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"ReID 特征提取失败: {e}")
            return None

    def extract_clip(self, crop_image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 Chinese-CLIP 图像特征向量

        Args:
            crop_image: 目标裁剪图像 (H, W, 3) BGR 格式

        Returns:
            L2 归一化的 CLIP 图像向量 (dim,)，失败返回 None
        """
        self._lazy_init_clip()

        try:
            import torch
            from PIL import Image

            # 确保裁剪图不为空且尺寸合理
            if crop_image.ndim != 3 or crop_image.shape[0] < 2 or crop_image.shape[1] < 2:
                return None

            # BGR → RGB，确保 uint8 类型
            rgb_image = crop_image[:, :, ::-1].copy()
            if rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(rgb_image).convert("RGB")

            # 预处理
            inputs = self._clip_processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                # 使用图像编码器提取特征
                image_features = self._clip_model.get_image_features(pixel_values=pixel_values)
                # L2 归一化
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

            return image_features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"CLIP 特征提取失败: {e}")
            return None

    def extract_reid_batch(
        self,
        crop_images: List[np.ndarray],
    ) -> np.ndarray:
        """
        批量提取 ReID 特征向量

        Args:
            crop_images: 目标裁剪图像列表

        Returns:
            归一化 ReID 向量矩阵 (N, dim)
        """
        self._lazy_init_reid()

        if not crop_images:
            return np.zeros((0, self.reid_dim), dtype=np.float32)

        try:
            import torch

            all_embeddings = []
            # 分批处理
            for i in range(0, len(crop_images), self.reid_batch_size):
                batch = crop_images[i:i + self.reid_batch_size]
                tensors = []
                for img in batch:
                    rgb = img[:, :, ::-1].copy()
                    tensors.append(self._reid_transform(rgb))
                batch_tensor = torch.stack(tensors).to(self.device)

                with torch.no_grad():
                    features = self._reid_backbone(batch_tensor)
                    features = self._reid_flatten(features)
                    embedding = self._reid_embedding_layer(features)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

                all_embeddings.append(embedding.cpu().numpy())

            return np.concatenate(all_embeddings, axis=0).astype(np.float32)

        except Exception as e:
            logger.error(f"批量 ReID 提取失败: {e}")
            return np.zeros((len(crop_images), self.reid_dim), dtype=np.float32)

    def extract_clip_batch(
        self,
        crop_images: List[np.ndarray],
    ) -> np.ndarray:
        """
        批量提取 CLIP 图像特征向量

        Args:
            crop_images: 目标裁剪图像列表

        Returns:
            归一化 CLIP 向量矩阵 (N, dim)
        """
        self._lazy_init_clip()

        if not crop_images:
            return np.zeros((0, self.clip_dim), dtype=np.float32)

        try:
            import torch
            from PIL import Image

            all_embeddings = []
            for i in range(0, len(crop_images), self.clip_batch_size):
                batch = crop_images[i:i + self.clip_batch_size]
                pil_images = [Image.fromarray(img[:, :, ::-1].copy()) for img in batch]

                inputs = self._clip_processor(images=pil_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.no_grad():
                    image_features = self._clip_model.get_image_features(pixel_values=pixel_values)
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

                all_embeddings.append(image_features.cpu().numpy())

            return np.concatenate(all_embeddings, axis=0).astype(np.float32)

        except Exception as e:
            logger.error(f"批量 CLIP 提取失败: {e}")
            return np.zeros((len(crop_images), self.clip_dim), dtype=np.float32)

    def extract_text_clip(self, text: str) -> np.ndarray:
        """
        提取文本的 Chinese-CLIP 特征向量

        用于图文跨模态检索。

        Args:
            text: 输入文本

        Returns:
            L2 归一化的 CLIP 文本向量 (dim,)
        """
        self._lazy_init_clip()

        try:
            import torch

            inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True)
            text_kwargs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # ChineseCLIPModel 的 get_text_features 在 transformers 4.57+ 中
                # 因 pooler 未初始化而报错，需手动计算文本特征
                model_class = type(self._clip_model).__name__
                if model_class == "ChineseCLIPModel":
                    # 手动获取 last_hidden_state，取 [CLS] token 后经 text_projection
                    text_outputs = self._clip_model.text_model(**text_kwargs)
                    cls_token = text_outputs.last_hidden_state[:, 0, :]
                    text_features = self._clip_model.text_projection(cls_token)
                else:
                    text_features = self._clip_model.get_text_features(**text_kwargs)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

            return text_features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"CLIP 文本特征提取失败: {e}")
            return np.zeros(self.clip_dim, dtype=np.float32)
