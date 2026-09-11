"""
src.trajectory - 跨镜轨迹构建包

把「强身份直接匹配」与「弱身份概率拼接」收敛到同一个入口，
供 api/routes/backtrack.py 调用。
"""

from src.trajectory.builder import (
    TrajectoryBuilder,
    TrajectoryDataUnavailableError,
    TrajectoryNotFoundError,
    get_trajectory_builder,
)

__all__ = [
    "TrajectoryBuilder",
    "TrajectoryNotFoundError",
    "TrajectoryDataUnavailableError",
    "get_trajectory_builder",
]
