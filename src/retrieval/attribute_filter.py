"""
src.retrieval.attribute_filter - 结构化属性过滤模块

基于解析后的查询条件，对目标实例库进行结构化属性过滤。
这是检索流程的第一步(粗过滤)，用于缩小候选范围。

过滤维度:
- 目标类别(vehicle/pedestrian/non_motor_vehicle)
- 车牌号(精确匹配)
- 颜色、车型、性别、背包等属性
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common.data_models import ParsedQuery, TargetInstance
from src.common.logger import get_logger
from src.retrieval.query_parser import QueryResult

logger = get_logger("retrieval.attribute_filter")


# 颜色模糊匹配映射: 相近颜色可以互相匹配
# 优化: 收紧匹配范围，只包含极相近的颜色变体，避免跨色系误匹配
# 例如: "黑色" 不再匹配 "灰色"，"红色" 不再匹配 "橙色"
COLOR_FUZZY_MAP = {
    "白色": ["白色", "乳白"],
    "黑色": ["黑色"],
    "银色": ["银色", "银灰"],
    "灰色": ["灰色"],
    "红色": ["红色"],
    "蓝色": ["蓝色"],
    "绿色": ["绿色"],
    "黄色": ["黄色"],
    "橙色": ["橙色"],
    "棕色": ["棕色"],
    "粉色": ["粉色"],
}

# 颜色别名标准化: 各种俗称映射到标准颜色名
# 优化: 将更多别名归一化到标准色，提高精确匹配率
COLOR_ALIAS_MAP = {
    "米白": "白色",
    "乳白": "白色",
    "象牙白": "白色",
    "深灰": "灰色",
    "浅灰": "灰色",
    "银灰": "银色",
    "深红": "红色",
    "粉红": "粉色",
    "浅粉": "粉色",
    "枣红": "红色",
    "深蓝": "蓝色",
    "浅蓝": "蓝色",
    "天蓝": "蓝色",
    "藏蓝": "蓝色",
    "宝蓝": "蓝色",
    "浅绿": "绿色",
    "深绿": "绿色",
    "草绿": "绿色",
    "墨绿": "绿色",
    "金色": "黄色",
    "淡黄": "黄色",
    "浅黄": "黄色",
    "橘色": "橙色",
    "褐色": "棕色",
    "咖啡色": "棕色",
    "深棕": "棕色",
    "浅棕": "棕色",
}

# 车型扩展映射: 查询某车型时，也匹配相关车型（单向）
# 仅当语义上“查询A应该也匹配B”时才扩展，避免反向误匹配
VEHICLE_TYPE_EXPAND = {
    "轿车": {"轿车", "两厢车"},      # 查“轿车”也匹配“两厢车”
    "两厢车": {"轿车", "两厢车"},    # 查“两厢车”也匹配“轿车”
    "卡车": {"卡车"},
    "面包车": {"面包车"},
    "公交车": {"公交车"},
    "出租车": {"出租车"},
    "SUV": {"SUV"},
    "跑车": {"跑车"},
}

# 支持的车辆类型全集（兼容 AICity22 / CityFlow 数据集）
ALL_VEHICLE_TYPES = {
    "轿车", "SUV", "卡车", "面包车", "出租车", "公交车",
    "三轮车", "电动车", "自行车", "摩托车",
}

# 支持的目标类型全集
ALL_TARGET_TYPES = {"vehicle", "pedestrian", "non_motor_vehicle"}


class AttributeFilter:
    """
    结构化属性过滤器

    对目标实例进行基于属性的粗过滤。

    使用方式:
        filter = AttributeFilter()
        candidates = filter.apply(all_instances, query_result)
    """

    def __init__(self) -> None:
        """初始化属性过滤器"""
        logger.info("属性过滤器初始化")

    def apply(
        self,
        instances: List[TargetInstance],
        query: QueryResult,
    ) -> List[TargetInstance]:
        """
        应用属性过滤

        Args:
            instances: 全量目标实例列表
            query: 解析后的查询结果

        Returns:
            过滤后的候选实例列表（按匹配度排序，已知匹配属性越多的排在越前）
        """
        if not instances:
            return []

        logger.info(f"开始属性过滤, 初始实例数: {len(instances)}")
        candidates = instances

        # 1. 目标类别过滤
        if query.target_type:
            candidates = [inst for inst in candidates if self._match_target_type(inst, query)]
            logger.info(f"目标类别过滤后: {len(candidates)}")

        # 2. 车牌精确匹配
        if query.plate_number:
            candidates = [inst for inst in candidates if self._match_plate(inst, query)]
            logger.info(f"车牌匹配后: {len(candidates)}")

        # 3. 颜色过滤
        if query.color:
            candidates = [inst for inst in candidates if self._match_color(inst, query)]
            logger.info(f"颜色过滤后: {len(candidates)}")

        # 4. 车型过滤
        if query.vehicle_type:
            candidates = [inst for inst in candidates if self._match_vehicle_type(inst, query)]
            logger.info(f"车型过滤后: {len(candidates)}")

        # 5. 性别过滤
        if query.gender:
            candidates = [inst for inst in candidates if self._match_gender(inst, query)]
            logger.info(f"性别过滤后: {len(candidates)}")

        # 6. 衣服颜色过滤
        if query.clothing_color:
            candidates = [inst for inst in candidates if self._match_clothing_color(inst, query)]
            logger.info(f"衣服颜色过滤后: {len(candidates)}")

        # 7. 裤子颜色过滤
        if query.pants_color:
            candidates = [inst for inst in candidates if self._match_pants_color(inst, query)]
            logger.info(f"裤子颜色过滤后: {len(candidates)}")

        # 8. 附属物过滤
        if query.accessory:
            candidates = [inst for inst in candidates if self._match_accessory(inst, query)]
            logger.info(f"附属物过滤后: {len(candidates)}")

        # 9. 背包过滤
        if query.bag is not None:
            candidates = [inst for inst in candidates if self._match_bag(inst, query)]
            logger.info(f"背包过滤后: {len(candidates)}")

        # 10. 按匹配度排序：已知匹配属性越多的排在越前
        #     这样top-K结果中优先展示已知匹配的记录，unknown记录排在后面
        if query.color or query.vehicle_type:
            candidates = self._sort_by_match_quality(candidates, query)

        logger.info(f"属性过滤完成, 最终候选数: {len(candidates)}")
        return candidates

    def _sort_by_match_quality(
        self,
        candidates: List[TargetInstance],
        query: QueryResult,
    ) -> List[TargetInstance]:
        """
        按匹配质量排序候选实例。

        已知匹配属性越多的记录排在越前，unknown属性的记录排在后面。
        这确保了 top-K 结果优先展示已知匹配的高质量记录。
        """
        def match_score(inst: TargetInstance) -> int:
            """计算匹配得分，越高表示匹配质量越好"""
            score = 0
            attrs = inst.attributes
            inst_color = attrs.get("color")
            inst_type = attrs.get("vehicle_type")

            # 颜色匹配得分
            if query.color:
                if inst_color is not None and inst_color != "unknown":
                    # 已知颜色且通过过滤 = 完全匹配
                    score += 100
                # unknown颜色得0分（排在后面）

            # 车型匹配得分
            if query.vehicle_type:
                if inst_type is not None and inst_type != "unknown":
                    # 已知车型且通过过滤 = 完全匹配
                    score += 100
                # unknown车型得0分（排在后面）

            return score

        # 稳定排序：得分高的在前，同分保持原顺序
        return sorted(candidates, key=match_score, reverse=True)

    def apply_parsed_query(
        self,
        instances: List[TargetInstance],
        query: ParsedQuery,
    ) -> List[TargetInstance]:
        """
        应用属性过滤(兼容 ParsedQuery 接口)

        Args:
            instances: 全量目标实例列表
            query: 解析后的查询

        Returns:
            过滤后的候选实例列表
        """
        if not instances:
            return []

        candidates = instances

        # 1. 目标类别过滤
        if query.target_type:
            candidates = [inst for inst in candidates if inst.target_type == query.target_type]
            logger.info(f"目标类别过滤后: {len(candidates)}")

        # 2. 车牌精确匹配
        if query.plate_number:
            candidates = [inst for inst in candidates if self._match_plate_parsed(inst, query)]
            logger.info(f"车牌匹配后: {len(candidates)}")

        # 3. 属性过滤
        candidates = [inst for inst in candidates if self._match_attributes_parsed(inst, query)]
        logger.info(f"属性过滤后: {len(candidates)}")

        return candidates

    def _match_target_type(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """
        匹配目标类别

        支持精确匹配和模糊匹配:
        - vehicle 匹配所有机动车类型
        - non_motor_vehicle 匹配所有非机动车类型
        - 类型别名兼容（如 "车" 匹配所有车辆）
        """
        if query.target_type is None:
            return True
        # 精确匹配
        if instance.target_type == query.target_type:
            return True
        # 模糊兼容: vehicle 和 non_motor_vehicle 都属于广义车辆
        if query.target_type == "vehicle" and instance.target_type in ("vehicle", "non_motor_vehicle"):
            return True
        if query.target_type == "non_motor_vehicle" and instance.target_type == "vehicle":
            # 如果实例是车辆但实际是电动车/自行车等，也尝试匹配
            inst_type = instance.attributes.get("vehicle_type", "")
            non_motor_types = {"三轮车", "电动车", "自行车", "摩托车"}
            if inst_type in non_motor_types:
                return True
        return False

    def _match_plate(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配车牌号(精确匹配)"""
        if query.plate_number is None:
            return True
        if instance.plate_number is None:
            return False
        return instance.plate_number == query.plate_number

    def _match_plate_parsed(
        self,
        instance: TargetInstance,
        query: ParsedQuery,
    ) -> bool:
        """匹配车牌号(兼容 ParsedQuery)"""
        if query.plate_number is None:
            return True
        if instance.plate_number is None:
            return False
        return instance.plate_number == query.plate_number

    def _match_color(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """
        匹配颜色(支持精确匹配 + 受限模糊匹配 + 模糊颜色集合匹配)

        优化策略:
        1. 先标准化颜色别名到标准色名
        2. 精确匹配(标准化后完全相同)
        3. 受限模糊匹配(仅极相近变体)
        4. 模糊颜色集合匹配（如"深色"匹配["黑色","深灰色",...]）
        5. unknown/空颜色不过滤（保留为候选）

        支持颜色别名标准化，如"米白"→"白色"
        """
        if query.color is None:
            return True

        inst_color = instance.attributes.get("color")
        if inst_color is None or inst_color == "unknown":
            return False  # 颜色未知时不通过颜色过滤（无法确认匹配）

        # 标准化实例颜色
        inst_color_normalized = COLOR_ALIAS_MAP.get(inst_color, inst_color)
        # 标准化查询颜色
        query_color_normalized = COLOR_ALIAS_MAP.get(query.color, query.color)

        # 精确匹配（标准化后完全相同）
        if inst_color_normalized == query_color_normalized:
            return True

        # 原始精确匹配（处理标准化未覆盖的情况）
        if inst_color == query.color:
            return True

        # 受限模糊匹配（使用标准化后的查询颜色）
        # 只检查实颜色是否在查询颜色的模糊列表中
        fuzzy_colors = COLOR_FUZZY_MAP.get(query_color_normalized, [])
        if inst_color_normalized in fuzzy_colors:
            return True

        # 也检查原始实例颜色是否在模糊列表中
        if inst_color in fuzzy_colors:
            return True

        # 模糊颜色集合匹配：如果查询有候选颜色列表，检查实例颜色是否在其中
        if query.color_candidates:
            # 标准化候选颜色列表
            normalized_candidates = [
                COLOR_ALIAS_MAP.get(c, c) for c in query.color_candidates
            ]
            if inst_color_normalized in normalized_candidates:
                return True
            if inst_color in query.color_candidates:
                return True

        return False

    def _match_vehicle_type(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """
        匹配车型
    
        支持 AICity22 / CityFlow 数据集中的所有车辆类型，
        以及常见别名（如"吉普车"→"SUV"）。
        支持泛化车型集合匹配（如"大车"匹配["卡车","公交车"]）。
        unknown/空车型不过滤（保留为候选）。
        """
        if query.vehicle_type is None:
            return True
    
        inst_type = instance.attributes.get("vehicle_type")
        if inst_type is None or inst_type == "unknown":
            return False  # 车型未知时不通过车型过滤（无法确认匹配）
    
        # 精确匹配
        if inst_type == query.vehicle_type:
            return True
    
        # 别名映射
        vehicle_aliases = {
            "吉普车": "SUV", "越野车": "SUV",
            "货车": "卡车", "拖头": "卡车", "大卡车": "卡车", "小货车": "卡车",
            "大巴": "公交车", "公共汽车": "公交车",
            "的士": "出租车", "小轿车": "轿车",
            "三轮": "三轮车", "三蹦子": "三轮车", "拖拉机": "三轮车",
            "电驴": "电动车", "小电驴": "电动车",
            "摩托": "摩托车",
            "重型货车": "卡车", "轻型货车": "卡车", "微型货车": "卡车",
            "大型客车": "公交车", "中型客车": "公交车", "小型客车": "面包车",
            "微面": "面包车", "面包车": "面包车",
        }
        query_normalized = vehicle_aliases.get(query.vehicle_type, query.vehicle_type)
        inst_normalized = vehicle_aliases.get(inst_type, inst_type)
    
        if query_normalized == inst_normalized:
            return True
    
        # 子串匹配（如"SUV" 在 "suv" 中）
        if query.vehicle_type.lower() == inst_type.lower():
            return True
    
        # 泛化车型集合匹配：如果查询有候选车型列表，检查实例车型是否在其中
        if query.vehicle_type_candidates:
            # 标准化候选车型列表，并扩展相关车型
            expanded_candidates = set()
            for c in query.vehicle_type_candidates:
                normalized_c = vehicle_aliases.get(c, c)
                expanded_candidates.add(normalized_c)
                # 扩展相关车型
                if normalized_c in VEHICLE_TYPE_EXPAND:
                    expanded_candidates.update(VEHICLE_TYPE_EXPAND[normalized_c])
                expanded_candidates.add(c)
            if inst_normalized in expanded_candidates:
                return True
            if inst_type in expanded_candidates:
                return True

        # 车型扩展匹配：查询某车型时也匹配相关车型
        expanded = VEHICLE_TYPE_EXPAND.get(query_normalized, {query_normalized})
        if inst_normalized in expanded:
            return True
        if inst_type in expanded:
            return True
    
        return False

    def _match_gender(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配性别"""
        if query.gender is None:
            return True

        inst_gender = instance.attributes.get("gender")
        if inst_gender is None or inst_gender == "unknown":
            return True  # 实例没有性别信息时不过滤

        return inst_gender == query.gender

    def _match_bag(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配附属物(背包)"""
        if query.bag is None:
            return True

        inst_bag = instance.attributes.get("bag", False)
        return inst_bag == query.bag

    def _match_clothing_color(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配衣服颜色（精确匹配 + 受限模糊匹配）"""
        if query.clothing_color is None:
            return True

        inst_clothing = instance.attributes.get("clothing_color")
        if inst_clothing is None or inst_clothing == "unknown":
            return True  # 实例没有衣服颜色信息时不过滤

        # 标准化颜色
        inst_normalized = COLOR_ALIAS_MAP.get(inst_clothing, inst_clothing)
        query_normalized = COLOR_ALIAS_MAP.get(query.clothing_color, query.clothing_color)

        if inst_normalized == query_normalized:
            return True

        # 受限模糊匹配
        fuzzy_colors = COLOR_FUZZY_MAP.get(query_normalized, [])
        if inst_normalized in fuzzy_colors:
            return True

        return False

    def _match_pants_color(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配裤子颜色（精确匹配 + 受限模糊匹配）"""
        if query.pants_color is None:
            return True

        inst_pants = instance.attributes.get("pants_color")
        if inst_pants is None or inst_pants == "unknown":
            return True  # 实例没有裤子颜色信息时不过滤

        inst_normalized = COLOR_ALIAS_MAP.get(inst_pants, inst_pants)
        query_normalized = COLOR_ALIAS_MAP.get(query.pants_color, query.pants_color)

        if inst_normalized == query_normalized:
            return True

        fuzzy_colors = COLOR_FUZZY_MAP.get(query_normalized, [])
        if inst_normalized in fuzzy_colors:
            return True

        return False

    def _match_accessory(
        self,
        instance: TargetInstance,
        query: QueryResult,
    ) -> bool:
        """匹配附属物（帽子/眼镜等）"""
        if query.accessory is None:
            return True

        inst_accessory = instance.attributes.get("accessory")
        if inst_accessory is None or inst_accessory == "unknown":
            return True  # 实例没有附属物信息时不过滤

        # 精确匹配
        if inst_accessory == query.accessory:
            return True

        # 别名匹配
        accessory_aliases = {
            "帽子": ["帽子", "头盔", "安全帽"],
            "眼镜": ["眼镜", "墨镜", "太阳镜"],
        }
        aliases = accessory_aliases.get(query.accessory, [query.accessory])
        return inst_accessory in aliases

    def _match_attributes_parsed(
        self,
        instance: TargetInstance,
        query: ParsedQuery,
    ) -> bool:
        """匹配属性条件(兼容 ParsedQuery)"""
        for key, value in query.attributes.items():
            if key in ["color", "vehicle_type", "gender", "bag", "bag_color",
                       "clothing_color", "pants_color", "accessory"]:
                inst_value = instance.attributes.get(key)
                if inst_value is not None and inst_value != value:
                    # 颜色模糊匹配
                    if key in ["color", "clothing_color", "bag_color", "pants_color"]:
                        value_normalized = COLOR_ALIAS_MAP.get(value, value)
                        inst_normalized = COLOR_ALIAS_MAP.get(inst_value, inst_value)
                        if inst_normalized == value_normalized:
                            continue
                        fuzzy_colors = COLOR_FUZZY_MAP.get(value_normalized, [])
                        if inst_normalized in fuzzy_colors:
                            continue
                    # 附属物别名匹配
                    if key == "accessory":
                        accessory_aliases = {
                            "帽子": ["帽子", "头盔", "安全帽"],
                            "眼镜": ["眼镜", "墨镜", "太阳镜"],
                        }
                        aliases = accessory_aliases.get(value, [value])
                        if inst_value in aliases:
                            continue
                    # 车型别名匹配
                    if key == "vehicle_type":
                        vehicle_aliases = {
                            "吉普车": "SUV", "越野车": "SUV",
                            "货车": "卡车", "拖头": "卡车",
                            "大巴": "公交车", "的士": "出租车",
                            "小轿车": "轿车", "三轮": "三轮车",
                            "电驴": "电动车", "摩托": "摩托车",
                        }
                        if vehicle_aliases.get(value, value) == vehicle_aliases.get(inst_value, inst_value):
                            continue
                    return False
        return True
