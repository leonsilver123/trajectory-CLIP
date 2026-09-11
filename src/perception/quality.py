"""
src.perception.quality - 质量评分模块

对检测到的目标实例进行质量评估，用于过滤低质量检测结果。

评分维度:
- 清晰度 (clarity): 拉普拉斯方差
- 完整度 (completeness): 检测框面积占图像面积的比例
- 遮挡率 (occlusion): 基于检测置信度与边缘信息的估计
- 检测置信度 (detection_conf): 检测器的 confidence

综合质量分数 = 各维度加权求和
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.common.data_models import BoundingBox
from src.common.logger import get_logger

logger = get_logger("perception.quality")


class QualityScorer:
    """
    质量评分器

    对目标检测结果进行多维度质量评估。

    使用方式:
        scorer = QualityScorer(weights={"clarity": 0.3, "completeness": 0.3, ...})
        score = scorer.score(frame, bbox, detection_confidence=0.85)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_score: float = 0.3,
    ) -> None:
        """
        初始化质量评分器

        Args:
            weights: 各维度权重字典
            min_score: 最低质量阈值
        """
        self.weights = weights or {
            "clarity": 0.3,
            "completeness": 0.3,
            "occlusion": 0.2,
            "detection_conf": 0.2,
        }
        self.min_score = min_score

        logger.info(f"质量评分器初始化: min_score={min_score}, weights={self.weights}")

    def score(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        detection_confidence: float = 1.0,
    ) -> float:
        """
        计算目标实例的综合质量分数

        Args:
            frame: 原始视频帧
            bbox: 目标检测框
            detection_confidence: 检测器置信度

        Returns:
            综合质量分数 [0, 1]
        """
        clarity = self._score_clarity(frame, bbox)
        completeness = self._score_completeness(frame, bbox)
        occlusion = self._score_occlusion(frame, bbox)
        det_conf = detection_confidence

        total_score = (
            self.weights.get("clarity", 0) * clarity
            + self.weights.get("completeness", 0) * completeness
            + self.weights.get("occlusion", 0) * occlusion
            + self.weights.get("detection_conf", 0) * det_conf
        )

        return min(max(total_score, 0.0), 1.0)

    def _score_clarity(self, frame: np.ndarray, bbox: BoundingBox) -> float:
        """
        评估清晰度

        基于拉普拉斯方差(Laplacian variance)评估图像模糊程度。
        方差越大表示图像越清晰。

        Args:
            frame: 原始帧
            bbox: 目标检测框

        Returns:
            清晰度分数 [0, 1]
        """
        try:
            import cv2

            # 裁剪目标区域
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(frame.shape[1], int(bbox.x2)), min(frame.shape[0], int(bbox.y2))
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return 0.0

            # 转灰度图
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # 计算拉普拉斯方差
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # 归一化到 [0, 1]
            # 经验值：方差 > 500 认为很清晰，< 50 认为很模糊
            # 使用 sigmoid 映射
            score = 1.0 / (1.0 + np.exp(-(variance - 100) / 80))
            return float(np.clip(score, 0.0, 1.0))

        except ImportError:
            # 无 cv2，使用 numpy 近似计算
            return self._score_clarity_numpy(frame, bbox)

    def _score_clarity_numpy(self, frame: np.ndarray, bbox: BoundingBox) -> float:
        """
        无 cv2 时的简化清晰度评估

        使用像素梯度近似拉普拉斯方差。

        Args:
            frame: 原始帧
            bbox: 目标检测框

        Returns:
            清晰度分数 [0, 1]
        """
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(frame.shape[1], int(bbox.x2)), min(frame.shape[0], int(bbox.y2))
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return 0.0

        # 转灰度（简单平均）
        gray = crop.mean(axis=2)

        # 使用二阶差分近似拉普拉斯
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.5

        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        variance = laplacian.var()

        score = 1.0 / (1.0 + np.exp(-(variance - 100) / 80))
        return float(np.clip(score, 0.0, 1.0))

    def _score_completeness(self, frame: np.ndarray, bbox: BoundingBox) -> float:
        """
        评估完整度

        判断目标是否完整在画面内。
        通过检测框在图像内的有效面积占比来评估。
        如果检测框超出图像边界，说明目标不完整。

        Args:
            frame: 原始帧
            bbox: 目标检测框

        Returns:
            完整度分数 [0, 1]
        """
        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w

        if img_area == 0:
            return 0.0

        # 检测框总面积
        total_area = bbox.area

        # 有效面积（在图像内的部分）
        x1 = max(0, bbox.x1)
        y1 = max(0, bbox.y1)
        x2 = min(img_w, bbox.x2)
        y2 = min(img_h, bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        valid_area = (x2 - x1) * (y2 - y1)

        # 完整度 = 有效面积 / 总面积
        completeness = valid_area / total_area if total_area > 0 else 0.0

        # 同时考虑检测框面积占图像面积的比例
        # 太小的目标（< 0.1%）可能不完整
        area_ratio = total_area / img_area
        if area_ratio < 0.001:
            completeness *= 0.5

        return float(np.clip(completeness, 0.0, 1.0))

    def _score_occlusion(self, frame: np.ndarray, bbox: BoundingBox) -> float:
        """
        评估遮挡率

        通过分析检测框边缘的梯度信息来判断目标是否被遮挡。
        边缘梯度不连续说明可能被遮挡。

        Args:
            frame: 原始帧
            bbox: 目标检测框

        Returns:
            无遮挡程度分数 [0, 1] (1 表示完全无遮挡)
        """
        try:
            import cv2

            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(frame.shape[1], int(bbox.x2)), min(frame.shape[0], int(bbox.y2))
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return 0.0

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # 使用 Canny 边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 计算边缘密度
            edge_density = edges.sum() / (edges.size * 255)

            # 分析内部区域的边缘一致性
            # 将检测框分为 3x3 网格，检查各区域的边缘密度是否均匀
            h, w = gray.shape
            if h < 9 or w < 9:
                return 0.7  # 太小无法分析，给中等分数

            block_h = h // 3
            block_w = w // 3
            densities = []
            for i in range(3):
                for j in range(3):
                    block = edges[
                        i * block_h:(i + 1) * block_h,
                        j * block_w:(j + 1) * block_w,
                    ]
                    densities.append(block.sum() / (block.size * 255 + 1e-6))

            # 密度方差越小，说明边缘分布越均匀（越可能无遮挡）
            density_var = np.var(densities)
            # 使用 sigmoid 映射
            occlusion_score = 1.0 / (1.0 + np.exp((density_var - 0.01) * 200))

            return float(np.clip(occlusion_score, 0.0, 1.0))

        except ImportError:
            # 无 cv2，返回中等分数
            return 0.5

    def is_qualified(self, quality_score: float) -> bool:
        """
        判断质量分数是否达标

        Args:
            quality_score: 质量分数

        Returns:
            是否达到最低质量要求
        """
        return quality_score >= self.min_score
