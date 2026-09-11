"""
src.perception - 视频结构化感知模块
负责目标检测、单摄跟踪、属性识别、车牌OCR、质量评分和特征提取。
"""

from src.perception.detector import VehicleDetector
from src.perception.tracker import SingleCameraTracker
from src.perception.attribute import AttributeRecognizer
from src.perception.plate_ocr import PlateOCR
from src.perception.quality import QualityScorer
from src.perception.feature_extractor import FeatureExtractor

__all__ = [
    "VehicleDetector",
    "SingleCameraTracker",
    "AttributeRecognizer",
    "PlateOCR",
    "QualityScorer",
    "FeatureExtractor",
]
