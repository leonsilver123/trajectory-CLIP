"""
src.retrieval.query_parser - 查询解析模块

将用户自然语言输入解析为结构化查询条件:
- 目标类别推断(车辆/行人/非机动车)
- 属性提取(颜色、车型、性别、服饰等)
- 车牌号识别
- CLIP 文本编码

示例:
    "蓝色背包的男人" → ParsedQuery(target_type="pedestrian", attributes={gender: "男性", bag: True, bag_color: "蓝色"})
    "苏E12345" → ParsedQuery(plate_number="苏E12345")
    "黑色轿车" → ParsedQuery(target_type="vehicle", attributes={color: "黑色", vehicle_type: "轿车"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.common.data_models import ParsedQuery
from src.common.logger import get_logger

logger = get_logger("retrieval.query_parser")


@dataclass
class QueryResult:
    """
    查询解析结果

    包含用户输入文本解析后的所有结构化信息。
    """
    query_text: str                           # 原始查询文本
    query_type: str                           # 查询类型: "plate" / "vehicle" / "pedestrian" / "general"
    target_type: Optional[str] = None         # 目标类别: vehicle/pedestrian/non_motor_vehicle
    plate_number: Optional[str] = None        # 车牌号
    color: Optional[str] = None               # 颜色
    vehicle_type: Optional[str] = None        # 车型
    gender: Optional[str] = None              # 性别
    clothing_color: Optional[str] = None      # 衣服颜色
    bag: Optional[bool] = None                # 是否背包
    bag_color: Optional[str] = None           # 背包颜色
    pants_color: Optional[str] = None         # 裤子颜色
    accessory: Optional[str] = None           # 附属物（帽子/眼镜等）
    raw_attributes: Dict[str, str] = None     # 原始属性字典
    color_candidates: list = None             # 模糊颜色候选列表（如"深色"→["黑色","深灰色",...])
    vehicle_type_candidates: list = None      # 泛化车型候选列表（如"大车"→["重型货车",...])

    def __post_init__(self):
        if self.raw_attributes is None:
            self.raw_attributes = {}
        if self.color_candidates is None:
            self.color_candidates = []
        if self.vehicle_type_candidates is None:
            self.vehicle_type_candidates = []


class QueryParser:
    """
    查询解析器

    将用户文本输入解析为结构化查询。

    使用方式:
        parser = QueryParser()
        result = parser.parse("蓝色背包的男人")
    """

    # 车牌号正则: 苏E开头 + 5位字母数字
    PLATE_PATTERN = re.compile(r'苏[E-Z][A-Z0-9]{5}')

    # 车辆关键词
    VEHICLE_KEYWORDS = ["轿车", "SUV", "卡车", "面包车", "出租车", "公交车", "车辆", "车"]
    # 行人关键词
    PEDESTRIAN_KEYWORDS = ["行人", "男人", "女人", "男性", "女性", "男的", "女的"]
    # 非机动车关键词
    NON_MOTOR_KEYWORDS = ["电动车", "自行车", "摩托车", "非机动车"]

    # 颜色词典
    COLOR_MAP = {
        "白色": "白色", "白": "白色",
        "黑色": "黑色", "黑": "黑色",
        "银色": "银色", "银灰": "银色",
        "灰色": "灰色", "灰": "灰色",
        "红色": "红色", "红": "红色",
        "蓝色": "蓝色", "蓝": "蓝色", "深蓝": "蓝色", "浅蓝": "蓝色",
        "绿色": "绿色", "绿": "绿色",
        "黄色": "黄色", "黄": "黄色",
        "橙色": "橙色", "橙": "橙色",
        "棕色": "棕色", "棕": "棕色",
        "粉色": "粉色", "粉": "粉色",
    }

    # 车型词典
    VEHICLE_TYPE_MAP = {
        "轿车": "轿车", "SUV": "SUV", "suv": "SUV",
        "卡车": "卡车", "货车": "卡车",
        "面包车": "面包车",
        "出租车": "出租车",
        "公交车": "公交车",
        "跑车": "跑车",
    }

    # 模糊颜色映射: 模糊颜色词 → 具体颜色列表
    # 与测试用例的 COLOR_GROUPS 保持一致
    FUZZY_COLOR_MAP = {
        "深色": ["黑色", "灰色", "棕色", "蓝色"],
        "暗色": ["黑色", "灰色", "棕色"],
        "浅色": ["白色", "黄色", "粉色", "金色"],
        "亮色": ["白色", "红色", "黄色", "橙色", "粉色", "金色"],
        "金属色": ["银色", "灰色", "金色"],
        "暖色": ["红色", "橙色", "黄色", "棕色", "金色", "粉色"],
        "冷色": ["蓝色", "绿色", "灰色", "黑色", "紫色"],
        "暗": ["黑色", "灰色", "棕色"],
        "亮": ["白色", "红色", "黄色"],
    }

    # 泛化车型映射: 泛化词 → 具体车型列表
    GENERIC_TYPE_MAP = {
        "大车": ["卡车", "公交车"],
        "大家伙": ["卡车", "公交车"],  # 口语中“大家伙”指大车
        "小车": ["轿车", "SUV", "跑车"],
        "货车": ["卡车"],
        "客车": ["公交车"],
    }

    # 车辆类型近义词/口语别名映射
    VEHICLE_TYPE_ALIAS = {
        "微面": "面包车", "面包车": "面包车", "小面": "面包车",
        "大货": "卡车", "大卡车": "卡车", "重型货车": "卡车",
        "小货": "卡车", "小卡车": "卡车", "轻型货车": "卡车",
        "吉普": "SUV", "越野": "SUV", "越野车": "SUV",
        "私家车": "轿车", "家用轿车": "轿车", "轿车": "轿车",
        "面的": "面包车",
        "大巴": "公交车", "公交": "公交车", "中巴": "公交车",
        "拖头": "卡车", "集装箱": "卡车",
        "三轮": "三轮车", "摩的": "摩托车",
        "两厢车": "轿车", "摩的": "摩托车",
    }

    # 附属物关键词
    BAG_KEYWORDS = ["背包", "书包", "挎包", "手提包"]
    HAT_KEYWORDS = ["帽子", "头盔"]
    GLASSES_KEYWORDS = ["眼镜", "墨镜"]

    # 口语化表达映射表: 口语化表达 → (目标类别, {属性字典})
    COLLOQUIAL_MAP = {
        # 车辆类型口语化表达
        "小轿车": ("vehicle", {"vehicle_type": "轿车"}),
        "轿车": ("vehicle", {"vehicle_type": "轿车"}),
        "越野车": ("vehicle", {"vehicle_type": "SUV"}),
        "SUV": ("vehicle", {"vehicle_type": "SUV"}),
        "suv": ("vehicle", {"vehicle_type": "SUV"}),
        "吉普车": ("vehicle", {"vehicle_type": "SUV"}),
        "面包车": ("vehicle", {"vehicle_type": "面包车"}),
        "小面": ("vehicle", {"vehicle_type": "面包车"}),
        "皮卡": ("vehicle", {"vehicle_type": "皮卡"}),
        "房车": ("vehicle", {"vehicle_type": "RV"}),
        "两厢车": ("vehicle", {"vehicle_type": "轿车"}),
        "小货车": ("vehicle", {"vehicle_type": "卡车"}),
        "货车": ("vehicle", {"vehicle_type": "卡车"}),
        "大卡车": ("vehicle", {"vehicle_type": "卡车"}),
        "拖头": ("vehicle", {"vehicle_type": "卡车"}),
        "大巴": ("vehicle", {"vehicle_type": "公交车"}),
        "大巴车": ("vehicle", {"vehicle_type": "公交车"}),
        "公交车": ("vehicle", {"vehicle_type": "公交车"}),
        "的士": ("vehicle", {"vehicle_type": "出租车"}),
        "出租车": ("vehicle", {"vehicle_type": "出租车"}),
        "三轮": ("non_motor_vehicle", {"vehicle_type": "三轮车"}),
        "三轮车": ("non_motor_vehicle", {"vehicle_type": "三轮车"}),
        "三蹦子": ("non_motor_vehicle", {"vehicle_type": "三轮车"}),
        "拖拉机": ("non_motor_vehicle", {"vehicle_type": "三轮车"}),
        "电驴": ("non_motor_vehicle", {"vehicle_type": "电动车"}),
        "小电驴": ("non_motor_vehicle", {"vehicle_type": "电动车"}),
        "电动车": ("non_motor_vehicle", {"vehicle_type": "电动车"}),
        "自行车": ("non_motor_vehicle", {"vehicle_type": "自行车"}),
        "摩托": ("non_motor_vehicle", {"vehicle_type": "摩托车"}),
        "摩托车": ("non_motor_vehicle", {"vehicle_type": "摩托车"}),

        # 近义词/口语化表达
        "微面": ("vehicle", {"vehicle_type": "面包车"}),
        "小型客车": ("vehicle", {"vehicle_type": "面包车"}),
        "大货": ("vehicle", {"vehicle_type": "卡车"}),
        "吉普": ("vehicle", {"vehicle_type": "SUV"}),
        "吉普车": ("vehicle", {"vehicle_type": "SUV"}),
        "越野": ("vehicle", {"vehicle_type": "SUV"}),
        "越野车": ("vehicle", {"vehicle_type": "SUV"}),
        "私家车": ("vehicle", {"vehicle_type": "轿车"}),
        "家用轿车": ("vehicle", {"vehicle_type": "轿车"}),
        "面的": ("vehicle", {"vehicle_type": "面包车"}),
        "中巴": ("vehicle", {"vehicle_type": "公交车"}),
        "公交": ("vehicle", {"vehicle_type": "公交车"}),
        "集装箱": ("vehicle", {"vehicle_type": "卡车"}),
        "摩的": ("non_motor_vehicle", {"vehicle_type": "摩托车"}),

        # 英文车辆表达
        "car": ("vehicle", {}),
        "truck": ("vehicle", {"vehicle_type": "卡车"}),
        "bus": ("vehicle", {"vehicle_type": "公交车"}),

        # 组合口语化表达（颜色+类型）
        "蓝车": ("vehicle", {"color": "蓝色"}),
        "红车": ("vehicle", {"color": "红色"}),
        "白车": ("vehicle", {"color": "白色"}),
        "黑车": ("vehicle", {"color": "黑色"}),
        "蓝色的士": ("vehicle", {"color": "蓝色", "vehicle_type": "出租车"}),
        "白色面包车": ("vehicle", {"color": "白色", "vehicle_type": "面包车"}),
        "黑色轿车": ("vehicle", {"color": "黑色", "vehicle_type": "轿车"}),
        "红色大巴": ("vehicle", {"color": "红色", "vehicle_type": "公交车"}),
        "白色货车": ("vehicle", {"color": "白色", "vehicle_type": "卡车"}),
        "白色小轿车": ("vehicle", {"color": "白色", "vehicle_type": "轿车"}),
        "蓝色的SUV": ("vehicle", {"color": "蓝色", "vehicle_type": "SUV"}),
        "红色大车": ("vehicle", {"color": "红色", "vehicle_type": "卡车"}),

        # 行人描述
        "男的": ("pedestrian", {"gender": "male"}),
        "男士": ("pedestrian", {"gender": "male"}),
        "女的": ("pedestrian", {"gender": "female"}),
        "小姐": ("pedestrian", {"gender": "female"}),
        "男人": ("pedestrian", {"gender": "male"}),
        "女人": ("pedestrian", {"gender": "female"}),
        "蓝衣服": ("pedestrian", {"clothing_color": "蓝色"}),
        "红衣服": ("pedestrian", {"clothing_color": "红色"}),
        "白衣服": ("pedestrian", {"clothing_color": "白色"}),
        "黑衣服": ("pedestrian", {"clothing_color": "黑色"}),
        "穿蓝色衣服的人": ("pedestrian", {"clothing_color": "蓝色"}),
        "穿红色衣服的人": ("pedestrian", {"clothing_color": "红色"}),
        "背红色书包的女人": ("pedestrian", {"gender": "female", "bag": True, "bag_color": "红色"}),
        "背书包的人": ("pedestrian", {"bag": True}),
        "戴帽子的人": ("pedestrian", {"accessory": "帽子"}),
        "戴眼镜的人": ("pedestrian", {"accessory": "眼镜"}),
        "黑裤子男人": ("pedestrian", {"gender": "male", "pants_color": "黑色"}),
        "白衣服女人": ("pedestrian", {"gender": "female", "clothing_color": "白色"}),

        # 颜色俗称
        "白色的士": ("vehicle", {"color": "白色", "vehicle_type": "出租车"}),
        "黑色越野车": ("vehicle", {"color": "黑色", "vehicle_type": "SUV"}),
        "蓝色轿车": ("vehicle", {"color": "蓝色", "vehicle_type": "轿车"}),
    }

    def __init__(self, clip_model=None) -> None:
        """
        初始化查询解析器

        Args:
            clip_model: Chinese-CLIP 模型实例(用于文本编码)
        """
        self._clip_model = clip_model
        logger.info("查询解析器初始化")

    def parse(self, text: str) -> QueryResult:
        """
        解析用户查询文本

        处理流程:
        1. 检测车牌号（最高优先级）
        2. 口语化映射: 精确匹配 → 滑动窗口匹配（从长到短）
        3. 传统关键词匹配作为后备

        Args:
            text: 用户输入文本

        Returns:
            结构化查询结果
        """
        text = text.strip()
        logger.info(f"解析查询: {text}")

        # 1. 检测车牌号
        plate = self._detect_plate(text)
        if plate:
            logger.info(f"检测到车牌号: {plate}")
            return QueryResult(
                query_text=text,
                query_type="plate",
                target_type="vehicle",
                plate_number=plate,
            )

        # 2. 尝试口语化映射匹配
        colloquial_result = self._match_colloquial(text)
        if colloquial_result is not None:
            target_type, attributes = colloquial_result
            logger.info(f"口语化映射匹配: target_type={target_type}, attrs={attributes}")
            # 用传统方法补充提取口语化映射未覆盖的属性（如颜色、裤子颜色等）
            extra_attrs = self._extract_attributes(text)
            for key, value in extra_attrs.items():
                if key.startswith("_"):
                    continue  # 跳过内部键
                if key not in attributes or attributes[key] is None:
                    attributes[key] = value
            # 提取模糊候选列表
            color_candidates = extra_attrs.get("_color_candidates", [])
            vehicle_type_candidates = extra_attrs.get("_vehicle_type_candidates", [])
            result = QueryResult(
                query_text=text,
                query_type=target_type or "general",
                target_type=target_type,
                color=attributes.get("color"),
                vehicle_type=attributes.get("vehicle_type"),
                gender=attributes.get("gender"),
                clothing_color=attributes.get("clothing_color"),
                bag=attributes.get("bag"),
                bag_color=attributes.get("bag_color"),
                pants_color=attributes.get("pants_color"),
                accessory=attributes.get("accessory"),
                raw_attributes=attributes,
                color_candidates=color_candidates,
                vehicle_type_candidates=vehicle_type_candidates,
            )
            logger.info(f"解析结果: {result}")
            return result

        # 3. 传统关键词匹配（后备逻辑）
        target_type = self._infer_target_type(text)
        attributes = self._extract_attributes(text)

        # 提取模糊候选列表
        color_candidates = attributes.pop("_color_candidates", [])
        vehicle_type_candidates = attributes.pop("_vehicle_type_candidates", [])

        result = QueryResult(
            query_text=text,
            query_type=target_type or "general",
            target_type=target_type,
            color=attributes.get("color"),
            vehicle_type=attributes.get("vehicle_type"),
            gender=attributes.get("gender"),
            clothing_color=attributes.get("clothing_color"),
            bag=attributes.get("bag"),
            bag_color=attributes.get("bag_color"),
            pants_color=attributes.get("pants_color"),
            accessory=attributes.get("accessory"),
            raw_attributes=attributes,
            color_candidates=color_candidates,
            vehicle_type_candidates=vehicle_type_candidates,
        )

        logger.info(f"解析结果: {result}")
        return result

    def _match_colloquial(self, text: str) -> Optional[tuple]:
        """
        匹配口语化表达

        先精确匹配完整文本，再用滑动窗口从长到短匹配子串。
        多个子串匹配结果会合并属性。

        Args:
            text: 用户输入文本

        Returns:
            (target_type, attributes_dict) 或 None
        """
        # 1. 精确匹配完整文本
        if text in self.COLLOQUIAL_MAP:
            target_type, attrs = self.COLLOQUIAL_MAP[text]
            return target_type, dict(attrs)

        # 2. 滑动窗口匹配（从长到短）
        text_len = len(text)
        matched_parts = []  # 收集所有匹配的子串及其结果
        used_ranges = []    # 记录已匹配的字符范围，避免重叠

        # 按子串长度从长到短枚举
        max_key_len = max(len(k) for k in self.COLLOQUIAL_MAP)
        for length in range(min(max_key_len, text_len), 0, -1):
            for start in range(text_len - length + 1):
                end = start + length
                # 检查是否与已匹配范围重叠
                if any(not (end <= us or start >= ue) for us, ue in used_ranges):
                    continue
                sub = text[start:end]
                if sub in self.COLLOQUIAL_MAP:
                    target_type, attrs = self.COLLOQUIAL_MAP[sub]
                    matched_parts.append((target_type, dict(attrs), start, end))
                    used_ranges.append((start, end))

        if not matched_parts:
            return None

        # 3. 合并所有匹配结果的属性
        merged_attrs: Dict[str, Any] = {}
        # 优先使用第一个匹配到的 target_type（最长子串的）
        merged_target_type = matched_parts[0][0]

        for _, attrs, _, _ in matched_parts:
            for key, value in attrs.items():
                if key not in merged_attrs:
                    merged_attrs[key] = value

        return merged_target_type, merged_attrs

    def _detect_plate(self, text: str) -> Optional[str]:
        """
        检测文本中是否包含车牌号

        Args:
            text: 用户输入文本

        Returns:
            车牌号或 None
        """
        match = self.PLATE_PATTERN.search(text)
        return match.group(0) if match else None

    def _infer_target_type(self, text: str) -> Optional[str]:
        """
        推断目标类别

        Args:
            text: 用户输入文本

        Returns:
            目标类别或 None
        """
        # 优先匹配非机动车(因为"电动车"包含"车")
        for keyword in self.NON_MOTOR_KEYWORDS:
            if keyword in text:
                return "non_motor_vehicle"

        # 行人关键词
        for keyword in self.PEDESTRIAN_KEYWORDS:
            if keyword in text:
                return "pedestrian"

        # 车辆关键词
        for keyword in self.VEHICLE_KEYWORDS:
            if keyword in text:
                return "vehicle"

        return None

    def _extract_attributes(self, text: str) -> Dict[str, Any]:
        """
        提取属性条件

        Args:
            text: 用户输入文本

        Returns:
            属性字典（包含 _color_candidates 和 _vehicle_type_candidates 特殊键）
        """
        attributes = {}

        # 提取颜色(优先匹配长词)
        sorted_colors = sorted(self.COLOR_MAP.keys(), key=len, reverse=True)
        for color_key in sorted_colors:
            if color_key in text:
                attributes["color"] = self.COLOR_MAP[color_key]
                break

        # 提取模糊颜色（如果未匹配到精确颜色）
        if "color" not in attributes:
            sorted_fuzzy = sorted(self.FUZZY_COLOR_MAP.keys(), key=len, reverse=True)
            for fuzzy_key in sorted_fuzzy:
                if fuzzy_key in text:
                    attributes["color"] = fuzzy_key  # 存储模糊颜色词本身
                    attributes["_color_candidates"] = self.FUZZY_COLOR_MAP[fuzzy_key]
                    break

        # 提取车型（合并所有来源，按关键词长度从长到短匹配，避免短词先匹配）
        all_type_sources = {}
        all_type_sources.update(self.VEHICLE_TYPE_ALIAS)  # 近义词（优先级最高）
        all_type_sources.update(self.GENERIC_TYPE_MAP)     # 泛化车型（会被转为列表）
        all_type_sources.update(self.VEHICLE_TYPE_MAP)     # 精确车型

        # 按关键词长度从长到短排序
        sorted_type_keys = sorted(all_type_sources.keys(), key=len, reverse=True)
        for type_key in sorted_type_keys:
            if type_key in text:
                val = all_type_sources[type_key]
                if isinstance(val, list):
                    # 泛化车型：存储第一个作为主值，完整列表作为候选
                    attributes["vehicle_type"] = type_key  # 存储泛化词本身
                    attributes["_vehicle_type_candidates"] = val
                else:
                    attributes["vehicle_type"] = val
                break

        # 提取性别
        if "男" in text or "男性" in text:
            attributes["gender"] = "male"
        elif "女" in text or "女性" in text:
            attributes["gender"] = "female"

        # 提取附属物
        for bag_key in self.BAG_KEYWORDS:
            if bag_key in text:
                attributes["bag"] = True
                break

        # 提取背包颜色(在"背包"之前的颜色)
        if "bag" in attributes:
            # 查找"背包"前面的颜色词
            bag_pos = text.find("背包") if "背包" in text else text.find("包")
            if bag_pos > 0:
                prefix = text[:bag_pos]
                for color_key in sorted_colors:
                    if color_key in prefix:
                        attributes["bag_color"] = self.COLOR_MAP[color_key]
                        break

        # 提取衣服颜色(在"穿"或"衣"之前的颜色)
        if "穿" in text or "衣" in text:
            for color_key in sorted_colors:
                if color_key in text:
                    # 如果颜色不是背包颜色，则视为衣服颜色
                    if "bag_color" not in attributes or attributes.get("bag_color") != self.COLOR_MAP[color_key]:
                        attributes["clothing_color"] = self.COLOR_MAP[color_key]
                        break

        # 提取裤子颜色
        pants_color_pattern = re.compile(r'([\u4e00-\u9fa5]{1,2}色?)裤子')
        pants_match = pants_color_pattern.search(text)
        if pants_match:
            color_part = pants_match.group(1)
            for color_key in sorted_colors:
                if color_part == color_key or color_key in color_part:
                    attributes["pants_color"] = self.COLOR_MAP[color_key]
                    break

        # 提取附属物（帽子/眼镜）
        for hat_key in self.HAT_KEYWORDS:
            if hat_key in text:
                attributes["accessory"] = "帽子"
                break
        if "accessory" not in attributes:
            for glasses_key in self.GLASSES_KEYWORDS:
                if glasses_key in text:
                    attributes["accessory"] = "眼镜"
                    break

        return attributes

    def to_parsed_query(self, result: QueryResult) -> ParsedQuery:
        """
        将 QueryResult 转换为 ParsedQuery(兼容旧接口)

        Args:
            result: QueryResult 对象

        Returns:
            ParsedQuery 对象
        """
        return ParsedQuery(
            raw_text=result.query_text,
            target_type=result.target_type,
            attributes=result.raw_attributes,
            plate_number=result.plate_number,
            clip_text_embedding=None,  # 延迟编码
        )
