"""
src.common.session_store - 会话状态存储

维护「检索 → 确认 → 回溯」人机协同闭环的后端状态，query_id 是唯一串联键。
后端为状态真源，前端的 session_state 退化为 UI 缓存。

状态机:
    idle ──search──► searched
                      ├──confirm──► confirmed ──backtrack──► backtracked
                      └──exclude/suspect──► excluded / suspect

使用方式:
    from src.common.session_store import get_session_store
    store = get_session_store()
    store.create("Q_xxx", "黑色轿车", candidate_count=10)   # state=searched
    store.confirm("Q_xxx", "CF3_c001_V0034_000001")         # state=confirmed
    store.set_backtracked("Q_xxx", result)                  # state=backtracked

FastAPI 是协程/多线程环境，内部 dict 由 threading.Lock 保护；
读取接口返回深拷贝，调用方修改副本不会污染存储。
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("common.session_store")

# ============================================================
# 容量上限（防止长驻进程内存无限增长）
# ============================================================

# 每次 /search/query 或 /search/plate 都会 create 一条会话，
# 而会话一旦创建就再也不会被删除 —— 长驻服务下这是必然的内存泄漏。
# 这里给一个 LRU 上限：超限时淘汰**最久未被访问**的会话。
#
# 淘汰是可接受的语义：被淘汰的 query_id 后续访问会得到 404（SessionNotFoundError），
# 前端已能处理；而"永不淘汰"才是真正危险的 —— 它会让进程慢慢地吃掉所有内存。
DEFAULT_MAX_SESSIONS = 1000

# ============================================================
# 状态常量
# ============================================================

STATE_IDLE = "idle"
STATE_SEARCHED = "searched"
STATE_CONFIRMED = "confirmed"
STATE_BACKTRACKED = "backtracked"
STATE_EXCLUDED = "excluded"
STATE_SUSPECT = "suspect"

ALL_STATES = (
    STATE_IDLE,
    STATE_SEARCHED,
    STATE_CONFIRMED,
    STATE_BACKTRACKED,
    STATE_EXCLUDED,
    STATE_SUSPECT,
)

# 允许的状态转移（键为当前状态，值为可转移到的状态集合）
_ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    STATE_IDLE: frozenset({STATE_SEARCHED}),
    STATE_SEARCHED: frozenset({STATE_CONFIRMED, STATE_EXCLUDED, STATE_SUSPECT}),
    # 已确认后允许改选另一个候选（confirmed → confirmed）或再回溯
    STATE_CONFIRMED: frozenset({
        STATE_BACKTRACKED, STATE_CONFIRMED, STATE_EXCLUDED, STATE_SUSPECT,
    }),
    STATE_BACKTRACKED: frozenset({
        STATE_BACKTRACKED, STATE_CONFIRMED, STATE_EXCLUDED, STATE_SUSPECT,
    }),
    STATE_EXCLUDED: frozenset({STATE_SEARCHED, STATE_CONFIRMED, STATE_EXCLUDED}),
    STATE_SUSPECT: frozenset({STATE_SEARCHED, STATE_CONFIRMED, STATE_SUSPECT}),
}


class SessionNotFoundError(KeyError):
    """query_id 对应的会话不存在（路由层据此返回 404）"""


class InvalidStateTransition(ValueError):
    """非法的状态转移"""


class SessionStore:
    """
    会话状态存储（线程安全）

    每次检索创建一条会话，后续的确认/回溯都通过 query_id 更新同一条会话。
    """

    def __init__(self, max_sessions: Optional[int] = None) -> None:
        """
        Args:
            max_sessions: 会话数上限。None 时读配置 `session.max_sessions`，
                读不到则退回 `DEFAULT_MAX_SESSIONS`。传 0 或负数表示不限制
                （仅用于测试对照，生产不应这么配）。
        """
        self._lock = threading.Lock()
        # OrderedDict 提供 O(1) 的 LRU：最近访问的移到末尾，淘汰从头部取
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.max_sessions = (
            max_sessions if max_sessions is not None else self._max_sessions_from_config()
        )
        self.evicted_count = 0

    @staticmethod
    def _max_sessions_from_config() -> int:
        """从配置读取上限；任何异常都退回默认值（会话存储不应因配置缺失而起不来）"""
        try:
            from src.common.config import get_config

            value = get_config().get("session.max_sessions", DEFAULT_MAX_SESSIONS)
            return int(value)
        except Exception:
            return DEFAULT_MAX_SESSIONS

    def _evict_locked(self) -> None:
        """
        超限时淘汰最久未访问的会话（调用方必须持锁）

        只在 create 时触发：淘汰发生在"业务上正在新增"的时刻，
        不会在一次读取中途把别的会话抽走。
        """
        if not self.max_sessions or self.max_sessions <= 0:
            return
        while len(self._sessions) > self.max_sessions:
            query_id, _ = self._sessions.popitem(last=False)
            self.evicted_count += 1
            logger.debug("会话超出上限(%s)，淘汰最久未访问: %s", self.max_sessions, query_id)

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _now() -> str:
        """当前时间（会话时间戳统一用本地时间字符串）"""
        return datetime.now().isoformat(timespec="seconds")

    def _require_locked(self, query_id: str) -> Dict[str, Any]:
        """持锁状态下取会话，不存在则抛 SessionNotFoundError

        取到即标记为最近访问 —— 状态转移（确认/回溯/排除）都走这里，
        正在流转的会话必须比闲置会话更难被淘汰。
        """
        session = self._sessions.get(query_id)
        if session is None:
            raise SessionNotFoundError(query_id)
        self._sessions.move_to_end(query_id)
        return session

    def _transition(self, query_id: str, to_state: str, **fields: Any) -> Dict[str, Any]:
        """
        执行一次状态转移并写入附加字段

        Raises:
            SessionNotFoundError: query_id 不存在
            InvalidStateTransition: 当前状态不允许转移到 to_state
        """
        with self._lock:
            session = self._require_locked(query_id)
            from_state = session["state"]
            if to_state != from_state and to_state not in _ALLOWED_TRANSITIONS.get(
                from_state, frozenset()
            ):
                raise InvalidStateTransition(
                    f"会话 {query_id} 不能从 {from_state} 转移到 {to_state}"
                )
            session["state"] = to_state
            session.update(fields)
            session["updated_at"] = self._now()
            return copy.deepcopy(session)

    # ---------------- 读写接口 ----------------

    def create(
        self,
        query_id: str,
        query_text: str = "",
        candidate_count: int = 0,
        query_type: str = "text",
        target_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建会话（检索完成时调用），初始状态 searched

        Args:
            query_id: 检索返回的查询 ID
            query_text: 用户查询文本或车牌号
            candidate_count: 候选数量
            query_type: 查询类型，"text"（文本）或 "plate"（车牌）
            target_type: 目标类别过滤条件（可选）

        Returns:
            会话副本；query_id 已存在时覆盖旧会话，并按 LRU 淘汰超限的历史会话
        """
        now = self._now()
        session: Dict[str, Any] = {
            "query_id": query_id,
            "query_text": query_text,
            "query_type": query_type,
            "target_type": target_type,
            "state": STATE_SEARCHED,
            "confirmed_instance_id": None,
            "excluded_instance_id": None,
            "suspect_instance_id": None,
            "candidate_count": candidate_count,
            "backtrack_result": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._sessions[query_id] = session
            self._sessions.move_to_end(query_id)   # 覆盖已有键时也要标记为最近使用
            self._evict_locked()
            return copy.deepcopy(session)

    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话副本

        Args:
            query_id: 查询 ID

        Returns:
            会话副本；不存在时返回 None
        """
        with self._lock:
            session = self._sessions.get(query_id)
            if session is not None:
                self._sessions.move_to_end(query_id)   # 读也算访问，避免活跃会话被淘汰
            return copy.deepcopy(session) if session is not None else None

    def require(self, query_id: str) -> Dict[str, Any]:
        """
        获取会话副本，不存在则抛异常

        Raises:
            SessionNotFoundError: query_id 不存在（调用方转 404）
        """
        with self._lock:
            return copy.deepcopy(self._require_locked(query_id))

    def confirm(self, query_id: str, instance_id: str) -> Dict[str, Any]:
        """
        用户确认目标：searched → confirmed，并记录确认的实例 ID

        Raises:
            SessionNotFoundError: query_id 不存在
            InvalidStateTransition: 当前状态不允许确认
        """
        return self._transition(
            query_id, STATE_CONFIRMED, confirmed_instance_id=instance_id
        )

    def set_backtracked(self, query_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        回溯完成：confirmed → backtracked，并保存回溯结果

        Raises:
            SessionNotFoundError: query_id 不存在
            InvalidStateTransition: 当前状态不允许回溯
        """
        return self._transition(query_id, STATE_BACKTRACKED, backtrack_result=result)

    def exclude(self, query_id: str, instance_id: str, reason: str = "") -> Dict[str, Any]:
        """
        用户排除候选：→ excluded（人机协同分支）

        Raises:
            SessionNotFoundError: query_id 不存在
            InvalidStateTransition: 当前状态不允许排除
        """
        return self._transition(
            query_id,
            STATE_EXCLUDED,
            excluded_instance_id=instance_id,
            exclude_reason=reason,
        )

    def suspect(self, query_id: str, instance_id: str, reason: str = "") -> Dict[str, Any]:
        """
        用户标记可疑：→ suspect（人机协同分支）

        Raises:
            SessionNotFoundError: query_id 不存在
            InvalidStateTransition: 当前状态不允许标记可疑
        """
        return self._transition(
            query_id,
            STATE_SUSPECT,
            suspect_instance_id=instance_id,
            suspect_reason=reason,
        )

    def list_all(self) -> List[Dict[str, Any]]:
        """列出全部会话副本（按创建时间排序）"""
        with self._lock:
            sessions = [copy.deepcopy(s) for s in self._sessions.values()]
        return sorted(sessions, key=lambda s: s.get("created_at", ""))

    def count(self) -> int:
        """当前会话数量"""
        with self._lock:
            return len(self._sessions)

    def reset(self) -> None:
        """清空全部会话（主要用于测试）"""
        with self._lock:
            self._sessions.clear()


# ============================================================
# 全局单例
# ============================================================

_global_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """
    获取全局会话存储单例

    Returns:
        SessionStore 实例
    """
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = SessionStore()
    return _global_store


def reset_session_store() -> None:
    """重置全局会话存储（主要用于测试）"""
    global _global_store
    with _store_lock:
        _global_store = None
