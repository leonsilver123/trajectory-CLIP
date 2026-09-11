"""
src.perception.attribute - 属性识别模块

对检测到的目标进行细粒度属性识别:
- 车辆: 车身颜色（颜色直方图+预定义颜色词典）、车辆类型
- 行人: 性别外观、上衣颜色、是否背包、背包颜色

输入: 目标裁剪图像
输出: 属性字典
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger

logger = get_logger("perception.attribute")


class AttributeRecognizer:
    """
    属性识别器

    对裁剪后的目标图像进行多标签属性分类。
    使用颜色直方图 + 预定义颜色词典进行颜色识别，
    使用基于规则的启发式方法进行车型和行人属性判断。

    使用方式:
        recognizer = AttributeRecognizer(device="cuda")
        attrs = recognizer.recognize(crop_image, target_type="vehicle")
        # attrs: {"color": "黑色", "vehicle_type": "轿车"}
    """

    # 支持的车辆颜色（中文名称 → BGR 参考值）
    VEHICLE_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
        "白色": (240, 240, 240),
        "黑色": (30, 30, 30),
        "银色": (192, 192, 192),
        "灰色": (128, 128, 128),
        "红色": (0, 0, 200),
        "蓝色": (200, 80, 0),
        "绿色": (0, 160, 0),
        "黄色": (0, 220, 220),
        "橙色": (0, 140, 255),
        "棕色": (50, 80, 140),
    }

    # 支持的车辆类型
    VEHICLE_TYPES = [
        "轿车", "SUV", "MPV", "卡车", "公交车", "面包车", "出租车",
    ]

    # 行人服饰颜色
    CLOTHING_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
        "白色": (240, 240, 240),
        "黑色": (30, 30, 30),
        "红色": (0, 0, 200),
        "蓝色": (200, 80, 0),
        "绿色": (0, 160, 0),
        "黄色": (0, 220, 220),
        "灰色": (128, 128, 128),
        "棕色": (50, 80, 140),
        "橙色": (0, 140, 255),
        "粉色": (200, 180, 220),
    }

    BAG_COLORS = [
        "白色", "黑色", "红色", "蓝色", "绿色", "黄色",
        "灰色", "棕色",
    ]

    def __init__(self, device: str = "cuda") -> None:
        """
        初始化属性识别器

        Args:
            device: 推理设备 ("cuda" 或 "cpu")
        """
        self.device = device
        self._vehicle_model = None
        self._pedestrian_model = None

        logger.info(f"属性识别器初始化: device={device}")

    def recognize(
        self,
        crop_image: np.ndarray,
        target_type: str,
    ) -> Dict[str, Any]:
        """
        识别目标属性

        Args:
            crop_image: 目标裁剪图像 (H, W, 3) BGR 格式
            target_type: 目标类别 ("vehicle" / "pedestrian" / "non_motor_vehicle")

        Returns:
            属性字典，如:
            - 车辆: {"color": "黑色", "vehicle_type": "轿车", "confidence": 0.85}
            - 行人: {"gender": "男性", "clothing_color": "白色", "bag": True, "bag_color": "蓝色"}
        """
        if crop_image is None or crop_image.size == 0:
            return {}

        if target_type == "vehicle":
            return self._recognize_vehicle(crop_image)
        elif target_type == "pedestrian":
            return self._recognize_pedestrian(crop_image)
        elif target_type == "non_motor_vehicle":
            # 非机动车复用车辆颜色识别
            return self._recognize_vehicle(crop_image)
        else:
            logger.warning(f"不支持的目标类别: {target_type}")
            return {}

    def _recognize_vehicle(self, crop_image: np.ndarray) -> Dict[str, Any]:
        """
        识别车辆属性

        使用颜色直方图匹配预定义颜色词典识别车身颜色，
        使用宽高比和面积启发式判断车辆类型。

        Args:
            crop_image: 车辆裁剪图像

        Returns:
            车辆属性字典
        """
        color, color_conf = self._classify_color(crop_image, self.VEHICLE_COLOR_MAP)
        vehicle_type, type_conf = self._classify_vehicle_type(crop_image)

        return {
            "color": color,
            "vehicle_type": vehicle_type,
            "confidence": round((color_conf + type_conf) / 2, 3),
        }

    def _recognize_pedestrian(self, crop_image: np.ndarray) -> Dict[str, Any]:
        """
        识别行人属性

        使用图像上半部分（头部区域）的亮度判断性别外观，
        使用图像中间部分（上身区域）的颜色直方图识别服饰颜色，
        使用图像下半部分的特征判断是否背包及背包颜色。

        Args:
            crop_image: 行人裁剪图像

        Returns:
            行人属性字典
        """
        h, w = crop_image.shape[:2]
        if h < 10 or w < 10:
            return {}

        # 将图像分为上（头肩）、中（上身）、下（下身）三部分
        upper = crop_image[:h // 3, :, :]
        middle = crop_image[h // 3: 2 * h // 3, :, :]
        lower = crop_image[2 * h // 3:, :, :]

        # 1. 性别外观判断（基于头部区域亮度和颜色饱和度）
        gender = self._estimate_gender(upper)

        # 2. 上衣颜色（中间区域）
        clothing_color, clothing_conf = self._classify_color(middle, self.CLOTHING_COLOR_MAP)

        # 3. 是否背包（分析图像一侧是否有异常凸起区域）
        has_bag, bag_conf = self._detect_bag(crop_image)

        # 4. 背包颜色
        bag_color = "未知"
        if has_bag:
            bag_color, bag_color_conf = self._detect_bag_color(crop_image)
        else:
            bag_color_conf = 0.0

        return {
            "gender": gender,
            "clothing_color": clothing_color,
            "bag": has_bag,
            "bag_color": bag_color,
            "confidence": round((clothing_conf + bag_conf) / 2, 3),
        }

    def _classify_color(
        self,
        image: np.ndarray,
        color_map: Dict[str, Tuple[int, int, int]],
    ) -> Tuple[str, float]:
        """
        使用颜色直方图匹配预定义颜色词典，识别图像主色调

        将图像从 BGR 转换到 HSV 空间，计算色调直方图，
        然后与预定义颜色的 HSV 参考值计算相似度。

        Args:
            image: BGR 图像
            color_map: 颜色名称 → BGR 参考值映射

        Returns:
            (颜色名称, 置信度)
        """
        if image is None or image.size == 0:
            return ("未知", 0.0)

        try:
            import cv2
            # 转换到 HSV 色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 计算色调(H)和饱和度(S)直方图
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()

            # 归一化
            h_total = h_hist.sum()
            if h_total > 0:
                h_hist = h_hist / h_total

            # 平均饱和度和亮度
            mean_v = hsv[:, :, 2].mean()
            mean_s = hsv[:, :, 1].mean()

            # 特殊情况：高亮度低饱和度 → 白色
            if mean_v > 200 and mean_s < 40:
                return ("白色", 0.85)
            # 低亮度 → 黑色
            if mean_v < 50:
                return ("黑色", 0.80)
            # 低饱和度高亮度 → 灰色
            if mean_s < 30 and 50 <= mean_v <= 200:
                return ("灰色", 0.75)

            # 与预定义颜色比较（在 HSV 空间）
            best_color = "未知"
            best_score = -1.0

            for color_name, bgr_ref in color_map.items():
                # 将 BGR 参考值转为 HSV
                ref_img = np.array([[bgr_ref]], dtype=np.uint8)
                ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
                ref_h = ref_hsv[0, 0, 0]
                ref_s = ref_hsv[0, 0, 1]
                ref_v = ref_hsv[0, 0, 2]

                # 计算色调直方图与参考色调的相似度
                # 使用色调距离（环形距离）
                h_diff = min(abs(ref_h - np.argmax(h_hist)), 180 - abs(ref_h - np.argmax(h_hist)))
                h_score = max(0, 1.0 - h_diff / 30.0)

                # 饱和度和亮度匹配
                s_score = max(0, 1.0 - abs(mean_s - ref_s) / 128.0)
                v_score = max(0, 1.0 - abs(mean_v - ref_v) / 128.0)

                total_score = h_score * 0.5 + s_score * 0.25 + v_score * 0.25

                if total_score > best_score:
                    best_score = total_score
                    best_color = color_name

            return (best_color, round(min(best_score, 1.0), 3))

        except ImportError:
            # 没有 cv2，使用简单的 numpy 方法
            return self._classify_color_numpy(image, color_map)

    def _classify_color_numpy(
        self,
        image: np.ndarray,
        color_map: Dict[str, Tuple[int, int, int]],
    ) -> Tuple[str, float]:
        """
        无 cv2 时的简化颜色分类（使用 numpy）

        Args:
            image: BGR 图像
            color_map: 颜色词典

        Returns:
            (颜色名称, 置信度)
        """
        mean_bgr = image.mean(axis=(0, 1))
        mean_v = mean_bgr.mean()

        # 简单规则判断
        if mean_v > 200:
            return ("白色", 0.7)
        if mean_v < 50:
            return ("黑色", 0.7)

        best_color = "未知"
        best_dist = float("inf")
        for color_name, bgr_ref in color_map.items():
            ref = np.array(bgr_ref, dtype=np.float64)
            dist = np.linalg.norm(mean_bgr - ref)
            if dist < best_dist:
                best_dist = dist
                best_color = color_name

        conf = max(0, 1.0 - best_dist / 200.0)
        return (best_color, round(conf, 3))

    def _classify_vehicle_type(self, image: np.ndarray) -> Tuple[str, float]:
        """
        基于宽高比和面积启发式判断车辆类型

        规则:
        - 宽高比接近 1:1 且面积较大 → SUV/MPV
        - 宽高比 > 1.5 且面积大 → 卡车/公交车
        - 宽高比 1.2~1.8 且面积中等 → 轿车
        - 面积小 → 面包车

        Args:
            image: 车辆裁剪图像

        Returns:
            (车辆类型, 置信度)
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return ("轿车", 0.5)

        aspect_ratio = w / h
        area = w * h

        # 基于宽高比的启发式规则
        if area > 100000:  # 大面积
            if aspect_ratio > 2.0:
                return ("卡车", 0.65)
            elif aspect_ratio > 1.5:
                return ("公交车", 0.60)
            elif aspect_ratio > 1.0:
                return ("SUV", 0.65)
            else:
                return ("MPV", 0.60)
        elif area > 30000:  # 中等面积
            if aspect_ratio > 1.8:
                return ("面包车", 0.60)
            elif 1.0 <= aspect_ratio <= 1.8:
                return ("轿车", 0.70)
            else:
                return ("轿车", 0.55)
        else:  # 小面积
            if aspect_ratio > 2.0:
                return ("卡车", 0.50)
            return ("轿车", 0.50)

    def _estimate_gender(self, upper_image: np.ndarray) -> str:
        """
        基于头部区域特征估计性别外观

        使用简化规则：基于头部区域的亮度分布和颜色特征。
        这是一个非常粗糙的启发式方法。

        Args:
            upper_image: 头部/肩部区域图像

        Returns:
            "男性" 或 "女性"
        """
        if upper_image is None or upper_image.size == 0:
            return "男性"

        try:
            import cv2
            hsv = cv2.cvtColor(upper_image, cv2.COLOR_BGR2HSV)
            mean_s = hsv[:, :, 1].mean()
            mean_v = hsv[:, :, 2].mean()

            # 高饱和度 + 高亮度 → 可能是女性（服饰颜色更鲜艳）
            if mean_s > 80 and mean_v > 150:
                return "女性"
            return "男性"
        except ImportError:
            mean_vals = upper_image.mean(axis=(0, 1))
            if mean_vals.max() > 180 and mean_vals.min() < 100:
                return "女性"
            return "男性"

    def _detect_bag(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        检测行人是否背包

        分析图像上半部分两侧是否有颜色差异区域（背包通常在肩部/背部）。

        Args:
            image: 行人裁剪图像

        Returns:
            (是否背包, 置信度)
        """
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return (False, 0.0)

        # 分析图像上半部分的左右两侧颜色差异
        upper = image[:h // 2, :, :]
        uh, uw = upper.shape[:2]
        if uw < 10:
            return (False, 0.0)

        left = upper[:, :uw // 3, :]
        right = upper[:, 2 * uw // 3:, :]

        # 计算左右两侧颜色差异
        diff = np.abs(left.mean(axis=(0, 1)) - right.mean(axis=(0, 1))).mean()

        # 差异较大说明可能有背包
        if diff > 40:
            return (True, min(0.5 + diff / 200, 0.9))
        return (False, max(0.0, 0.3 - diff / 200))

    def _detect_bag_color(self, image: np.ndarray) -> Tuple[str, float]:
        """
        检测背包颜色

        分析图像边缘区域（可能是背包所在区域）的颜色。

        Args:
            image: 行人裁剪图像

        Returns:
            (背包颜色, 置信度)
        """
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return ("未知", 0.0)

        # 取图像中部偏上区域（背包通常在背部）
        bag_region = image[h // 4: h // 2, :, :]
        return self._classify_color(bag_region, self.CLOTHING_COLOR_MAP)
