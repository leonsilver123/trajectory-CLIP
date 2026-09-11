"""
src.perception.plate_ocr - 车牌 OCR 模块

实现车牌检测和识别:
1. 车牌区域检测（基于颜色/边缘检测）
2. 车牌字符识别（简化版：基于模板匹配）

输入: 车辆裁剪图像
输出: 车牌号字符串 + 识别置信度
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from src.common.data_models import BoundingBox
from src.common.logger import get_logger

logger = get_logger("perception.plate_ocr")


class PlateOCR:
    """
    车牌 OCR 识别器

    对车辆图像进行车牌检测和字符识别。
    车牌检测使用颜色/边缘特征，OCR 使用简化版模板匹配。
    支持中文车牌格式（如"沪A12345"、"京B67890"等全国各省市车牌）。

    使用方式:
        ocr = PlateOCR(device="cuda")
        plate, confidence = ocr.recognize(vehicle_crop)
        # plate: "沪A12345", confidence: 0.92
    """

    # 中国省份简称
    PROVINCE_CHARS = [
        "京", "津", "沪", "渝", "冀", "豫", "云", "辽",
        "黑", "湘", "皖", "鲁", "新", "苏", "浙", "赣",
        "鄂", "桂", "甘", "晋", "蒙", "陕", "吉", "闽",
        "贵", "粤", "川", "青", "藏", "琼", "宁",
    ]

    # 字母表（不含 I, O 避免混淆）
    LETTER_CHARS = [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in ("I", "O")]

    # 数字
    DIGIT_CHARS = [str(d) for d in range(10)]

    # 所有有效车牌字符
    ALL_CHARS = PROVINCE_CHARS + LETTER_CHARS + DIGIT_CHARS

    # 中国蓝色车牌的主色调 (BGR)
    BLUE_PLATE_LOWER = np.array([90, 60, 0])
    BLUE_PLATE_UPPER = np.array([130, 255, 255])

    # 绿色新能源车牌色调
    GREEN_PLATE_LOWER = np.array([40, 50, 0])
    GREEN_PLATE_UPPER = np.array([90, 255, 255])

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        device: str = "cuda",
    ) -> None:
        """
        初始化车牌 OCR

        Args:
            model_path: 模型路径，None 表示使用预训练模型
            confidence_threshold: 识别置信度阈值
            device: 推理设备
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._plate_detector = None
        self._ocr_model = None

        # 尝试加载 EasyOCR 或 PaddleOCR
        self._ocr_backend = None
        self._init_ocr_backend()

        logger.info(f"车牌 OCR 初始化: model_path={model_path}, device={device}")

    def _init_ocr_backend(self) -> None:
        """
        初始化 OCR 后端

        优先尝试 EasyOCR，其次 PaddleOCR，最后使用内置简化版。
        """
        # 尝试 EasyOCR
        try:
            import easyocr
            self._ocr_backend = easyocr.Reader(["ch_sim", "en"], gpu=(self.device == "cuda"))
            logger.info("OCR 后端: EasyOCR")
            return
        except ImportError:
            pass

        # 尝试 PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr_backend = PaddleOCR(use_angle_cls=True, lang="ch")
            logger.info("OCR 后端: PaddleOCR")
            return
        except ImportError:
            pass

        # 使用内置简化版
        logger.info("OCR 后端: 内置简化版（模板匹配）")
        self._ocr_backend = "builtin"

    def detect_plates(self, vehicle_image: np.ndarray) -> List[BoundingBox]:
        """
        检测车辆图像中的车牌区域

        使用颜色分割和轮廓检测定位车牌区域。

        Args:
            vehicle_image: 车辆裁剪图像

        Returns:
            车牌区域检测框列表
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return []

        plates = []
        try:
            import cv2
            hsv = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2HSV)
            h_img, w_img = vehicle_image.shape[:2]

            # 蓝色车牌检测
            blue_mask = cv2.inRange(hsv, self.BLUE_PLATE_LOWER, self.BLUE_PLATE_UPPER)
            # 绿色新能源车牌检测
            green_mask = cv2.inRange(hsv, self.GREEN_PLATE_LOWER, self.GREEN_PLATE_UPPER)
            # 合并掩码
            plate_mask = cv2.bitwise_or(blue_mask, green_mask)

            # 形态学操作：闭运算填充间隙
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel)
            plate_mask = cv2.medianBlur(plate_mask, 5)

            # 查找轮廓
            contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # 车牌尺寸过滤
                # 中国标准车牌比例约 440:140 ≈ 3.14:1
                if area < (h_img * w_img * 0.005):  # 太小
                    continue
                if area > (h_img * w_img * 0.5):  # 太大
                    continue

                aspect_ratio = w / h if h > 0 else 0
                # 车牌宽高比通常在 2.0 ~ 5.0 之间
                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                    continue

                # 计算置信度（基于面积和比例）
                conf = min(0.5 + area / (h_img * w_img * 0.1), 0.95)

                bbox = BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    confidence=conf,
                )
                plates.append(bbox)

        except ImportError:
            logger.warning("cv2 未安装，使用简化车牌检测")
            plates = self._detect_plates_simple(vehicle_image)

        return plates

    def _detect_plates_simple(self, image: np.ndarray) -> List[BoundingBox]:
        """
        无 cv2 时的简化车牌检测

        基于颜色均值分析定位可能的车牌区域。

        Args:
            image: 车辆图像

        Returns:
            车牌区域列表
        """
        h, w = image.shape[:2]
        if h < 20 or w < 50:
            return []

        # 简化：假设车牌在图像下方 1/3 区域
        plate_region = image[2 * h // 3:, :, :]
        ph, pw = plate_region.shape[:2]

        # 检查该区域是否偏蓝色（中国蓝牌）
        mean_b = plate_region[:, :, 0].mean()
        mean_g = plate_region[:, :, 1].mean()
        mean_r = plate_region[:, :, 2].mean()

        # BGR 中蓝色车牌: B 高, G 中, R 低
        if mean_b > 100 and mean_b > mean_r * 1.3:
            bbox = BoundingBox(
                x1=float(w * 0.15),
                y1=float(2 * h // 3),
                x2=float(w * 0.85),
                y2=float(h),
                confidence=0.5,
            )
            return [bbox]

        return []

    def recognize(
        self,
        vehicle_image: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        识别车牌号

        Args:
            vehicle_image: 车辆裁剪图像

        Returns:
            (车牌号, 置信度)，未识别到返回 (None, 0.0)
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return (None, 0.0)

        # 1. 检测车牌区域
        plates = self.detect_plates(vehicle_image)
        if not plates:
            # 如果没有检测到车牌区域，尝试直接对整图 OCR
            plate_crop = vehicle_image
        else:
            # 取置信度最高的车牌区域
            best_plate = max(plates, key=lambda b: b.confidence)
            x1, y1 = int(best_plate.x1), int(best_plate.y1)
            x2, y2 = int(best_plate.x2), int(best_plate.y2)
            plate_crop = vehicle_image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            return (None, 0.0)

        # 2. OCR 识别
        plate_text, ocr_conf = self.recognize_plate_region(plate_crop)

        if plate_text is None or ocr_conf < self.confidence_threshold:
            return (None, 0.0)

        # 3. 后处理：校验车牌格式
        plate_text = self._validate_plate_format(plate_text)
        if plate_text is None:
            return (None, 0.0)

        return (plate_text, ocr_conf)

    def recognize_plate_region(
        self,
        plate_crop: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        对已裁剪的车牌图像进行 OCR 识别

        Args:
            plate_crop: 车牌裁剪图像

        Returns:
            (车牌号, 置信度)
        """
        if plate_crop is None or plate_crop.size == 0:
            return (None, 0.0)

        if self._ocr_backend == "builtin":
            return self._ocr_builtin(plate_crop)

        # 使用 EasyOCR
        try:
            import easyocr
            if isinstance(self._ocr_backend, easyocr.Reader):
                results = self._ocr_backend.readtext(plate_crop)
                if not results:
                    return (None, 0.0)
                # 拼接所有识别文本
                texts = []
                confs = []
                for _, text, conf in results:
                    texts.append(text.strip())
                    confs.append(conf)
                plate_text = "".join(texts)
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                return (plate_text if plate_text else None, avg_conf)
        except ImportError:
            pass

        # 使用 PaddleOCR
        try:
            from paddleocr import PaddleOCR
            if isinstance(self._ocr_backend, PaddleOCR):
                results = self._ocr_backend.ocr(plate_crop, cls=True)
                if not results or not results[0]:
                    return (None, 0.0)
                texts = []
                confs = []
                for line in results[0]:
                    text = line[1][0].strip()
                    conf = line[1][1]
                    texts.append(text)
                    confs.append(conf)
                plate_text = "".join(texts)
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                return (plate_text if plate_text else None, avg_conf)
        except ImportError:
            pass

        return (None, 0.0)

    def _ocr_builtin(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        内置简化版 OCR（基于颜色分割 + 字符分割）

        这是一个非常简化的实现，仅用于在没有 OCR 库时提供基本功能。

        Args:
            plate_crop: 车牌裁剪图像

        Returns:
            (车牌号, 置信度)
        """
        try:
            import cv2
            h, w = plate_crop.shape[:2]
            if h < 10 or w < 30:
                return (None, 0.0)

            # 转灰度
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            # 二值化（自适应阈值）
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2,
            )

            # 垂直投影法分割字符
            col_sum = binary.sum(axis=0)
            # 找到字符列（非零区域）
            threshold = col_sum.max() * 0.1
            char_regions = []
            in_char = False
            start = 0
            for i in range(w):
                if col_sum[i] > threshold:
                    if not in_char:
                        start = i
                        in_char = True
                else:
                    if in_char:
                        if i - start > 2:  # 最小字符宽度
                            char_regions.append((start, i))
                        in_char = False
            if in_char and w - start > 2:
                char_regions.append((start, w))

            # 中国车牌格式：1个汉字 + 1个字母 + 5个数字/字母 = 7个字符
            # 简化处理：返回识别到的字符数量作为置信度参考
            num_chars = len(char_regions)
            if num_chars < 5:
                return (None, 0.0)

            # 无法精确识别字符内容，返回占位符
            # 实际使用中应依赖 EasyOCR/PaddleOCR
            conf = min(0.3 + num_chars * 0.05, 0.6)
            return (None, conf)

        except ImportError:
            return (None, 0.0)

    def _validate_plate_format(self, plate_text: str) -> Optional[str]:
        """
        校验和清洗车牌号格式

        中国车牌格式：
        - 普通车牌：省份简称 + 字母 + 5位字母数字（如"沪A12345"）
        - 新能源车牌：省份简称 + 字母 + 6位字母数字（如"沪AD12345"）

        Args:
            plate_text: OCR 识别的原始文本

        Returns:
            清洗后的车牌号，格式不合法返回 None
        """
        if not plate_text:
            return None

        # 去除空格和特殊字符
        cleaned = re.sub(r"[\s·\.\-]", "", plate_text)

        # 检查是否以省份简称开头
        starts_with_province = False
        for prov in self.PROVINCE_CHARS:
            if cleaned.startswith(prov):
                starts_with_province = True
                break

        if not starts_with_province:
            # 尝试找到第一个省份简称
            for i, ch in enumerate(cleaned):
                if ch in self.PROVINCE_CHARS:
                    cleaned = cleaned[i:]
                    starts_with_province = True
                    break

        if not starts_with_province:
            return None

        # 检查长度（7 或 8 位）
        if len(cleaned) < 7 or len(cleaned) > 8:
            # 如果长度不对，截取前 7-8 位
            if len(cleaned) >= 7:
                cleaned = cleaned[:8] if len(cleaned) >= 8 else cleaned[:7]
            else:
                return None

        return cleaned
