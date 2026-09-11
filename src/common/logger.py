"""
src.common.logger - 日志系统

提供统一的日志记录功能，支持:
- 同时输出到控制台和文件
- 日志文件按日期轮转
- 不同模块可独立配置日志级别

使用方式:
    from src.common.logger import get_logger
    logger = get_logger("perception.detector")
    logger.info("检测器初始化完成")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from src.common.config import get_config


# 已创建的 logger 缓存，避免重复创建
_loggers: dict[str, logging.Logger] = {}

# 日志格式
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_log_dir(log_dir: str) -> Path:
    """确保日志目录存在"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    获取指定名称的 logger

    如果 logger 已创建则直接返回，否则创建新的 logger 并配置 handler。

    Args:
        name: logger 名称，通常使用 "模块.类名" 格式
        level: 日志级别，默认从配置文件读取
        log_to_file: 是否输出到文件

    Returns:
        配置好的 Logger 实例

    示例:
        logger = get_logger("perception.detector")
        logger.info("目标检测开始")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        _loggers[name] = logger
        return logger

    # 设置日志级别
    if level is None:
        try:
            level = get_config().get("system.log_level", "INFO")
        except Exception:
            level = "INFO"
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    # 文件 handler (按日期轮转)
    if log_to_file:
        try:
            log_dir = _ensure_log_dir(
                get_config().get("system.log_dir", "logs")
            )
            file_handler = TimedRotatingFileHandler(
                filename=str(log_dir / f"{name.replace('.', '_')}.log"),
                when="midnight",          # 每天午夜轮转
                interval=1,
                backupCount=30,           # 保留 30 天日志
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)
        except Exception:
            # 如果文件日志创建失败，仅使用控制台输出
            logger.warning(f"无法创建文件日志，仅使用控制台输出: {name}")

    # 阻止日志向上传播(避免重复输出)
    logger.propagate = False

    _loggers[name] = logger
    return logger


def set_module_level(name: str, level: str) -> None:
    """
    设置指定模块的日志级别

    Args:
        name: logger 名称
        level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logger = get_logger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def reset_loggers() -> None:
    """重置所有 logger(主要用于测试)"""
    _loggers.clear()
