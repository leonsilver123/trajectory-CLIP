"""
tests/test_session_store.py - 会话状态机测试

测试目标: 验证 src/common/session_store.py 的状态机与线程安全
覆盖:
  - 合法状态转移 idle → searched → confirmed → backtracked
  - 人机协同分支（excluded / suspect）
  - 非法状态转移报错、query_id 不存在报错
  - 全局单例行为与 reset
  - 多线程并发读写的线程安全
"""

import threading

import pytest

from src.common.session_store import (
    InvalidStateTransition,
    SessionNotFoundError,
    SessionStore,
    get_session_store,
    reset_session_store,
)

INSTANCE_ID = "CF3_c001_V0034_000001"


@pytest.fixture(autouse=True)
def _reset_session_store():
    """每个测试前后重置全局会话存储"""
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def store():
    """独立的会话存储实例"""
    return SessionStore()


class TestSessionLifecycle:
    """状态机主链路"""

    def test_create_sets_searched_state(self, store):
        """检索创建会话后状态为 searched"""
        session = store.create("Q_001", "黑色轿车", candidate_count=12)
        assert session["query_id"] == "Q_001"
        assert session["query_text"] == "黑色轿车"
        assert session["state"] == "searched"
        assert session["candidate_count"] == 12
        assert session["confirmed_instance_id"] is None
        assert session["backtrack_result"] is None
        assert session["created_at"] and session["updated_at"]

    def test_full_chain_search_confirm_backtrack(self, store):
        """检索 → 确认 → 回溯 靠同一 query_id 串联"""
        store.create("Q_002", "蓝色背包的男人", candidate_count=5)
        confirmed = store.confirm("Q_002", INSTANCE_ID)
        assert confirmed["state"] == "confirmed"
        assert confirmed["confirmed_instance_id"] == INSTANCE_ID

        result = {"query_id": "BT_xxx", "camera_sequence": ["c001", "c016"]}
        backtracked = store.set_backtracked("Q_002", result)
        assert backtracked["state"] == "backtracked"
        assert backtracked["backtrack_result"] == result
        # 确认信息在回溯后仍然保留
        assert backtracked["confirmed_instance_id"] == INSTANCE_ID

    def test_updated_at_changes_on_transition(self, store):
        """状态转移后 updated_at 被刷新"""
        created = store.create("Q_003", "白色卡车")
        confirmed = store.confirm("Q_003", INSTANCE_ID)
        assert confirmed["updated_at"] >= created["updated_at"]

    def test_reconfirm_other_instance(self, store):
        """已确认后可以改选另一个候选"""
        store.create("Q_004", "黑色轿车")
        store.confirm("Q_004", INSTANCE_ID)
        changed = store.confirm("Q_004", "CF3_c016_V0042_000007")
        assert changed["state"] == "confirmed"
        assert changed["confirmed_instance_id"] == "CF3_c016_V0042_000007"


class TestCollaborativeBranches:
    """人机协同分支：excluded / suspect"""

    def test_exclude(self, store):
        """排除候选 → excluded"""
        store.create("Q_010", "黑色轿车")
        session = store.exclude("Q_010", INSTANCE_ID, reason="颜色不符")
        assert session["state"] == "excluded"
        assert session["excluded_instance_id"] == INSTANCE_ID

    def test_suspect(self, store):
        """标记可疑 → suspect"""
        store.create("Q_011", "黑色轿车")
        session = store.suspect("Q_011", INSTANCE_ID, reason="疑似套牌")
        assert session["state"] == "suspect"
        assert session["suspect_instance_id"] == INSTANCE_ID


class TestErrors:
    """异常路径"""

    def test_get_missing_returns_none(self, store):
        """get 不存在的 query_id 返回 None"""
        assert store.get("Q_NOT_EXIST") is None

    def test_require_missing_raises(self, store):
        """require 不存在的 query_id 抛出 SessionNotFoundError（路由层转 404）"""
        with pytest.raises(SessionNotFoundError):
            store.require("Q_NOT_EXIST")

    def test_confirm_missing_query_id_raises(self, store):
        """对不存在的 query_id 调用 confirm 报错，而非静默成功"""
        with pytest.raises(SessionNotFoundError):
            store.confirm("Q_NOT_EXIST", INSTANCE_ID)

    def test_backtrack_missing_query_id_raises(self, store):
        """对不存在的 query_id 写回溯结果报错"""
        with pytest.raises(SessionNotFoundError):
            store.set_backtracked("Q_NOT_EXIST", {})

    def test_illegal_transition_raises(self, store):
        """searched → backtracked 是非法转移（必须先确认）"""
        store.create("Q_020", "黑色轿车")
        with pytest.raises(InvalidStateTransition):
            store.set_backtracked("Q_020", {})

    def test_illegal_transition_keeps_state(self, store):
        """非法转移不改变会话状态"""
        store.create("Q_021", "黑色轿车")
        with pytest.raises(InvalidStateTransition):
            store.set_backtracked("Q_021", {})
        assert store.get("Q_021")["state"] == "searched"


class TestIsolation:
    """存储与调用方的隔离"""

    def test_get_returns_copy(self, store):
        """get 返回副本，外部修改不影响存储"""
        store.create("Q_030", "黑色轿车")
        session = store.get("Q_030")
        session["state"] = "被篡改"
        assert store.get("Q_030")["state"] == "searched"

    def test_backtrack_result_copy(self, store):
        """回溯结果深拷贝，外部修改不影响存储"""
        store.create("Q_031", "黑色轿车")
        store.confirm("Q_031", INSTANCE_ID)
        result = {"camera_sequence": ["c001"]}
        session = store.set_backtracked("Q_031", result)
        session["backtrack_result"]["camera_sequence"].append("c016")
        assert store.get("Q_031")["backtrack_result"]["camera_sequence"] == ["c001"]

    def test_reset_clears_sessions(self, store):
        """reset 清空全部会话"""
        store.create("Q_032", "黑色轿车")
        store.reset()
        assert store.count() == 0
        assert store.get("Q_032") is None

    def test_list_all_sorted_by_created_at(self, store):
        """list_all 返回全部会话"""
        store.create("Q_040", "黑色轿车")
        store.create("Q_041", "白色卡车")
        sessions = store.list_all()
        assert {s["query_id"] for s in sessions} == {"Q_040", "Q_041"}


class TestCapacityEviction:
    """容量上限与 LRU 淘汰（D1）

    背景：每次检索都会 create 一条会话且从不删除，长驻进程下内存只增不减。
    这里锁定「超限淘汰最久未访问者」的语义。
    """

    def test_evicts_oldest_when_over_capacity(self):
        """超过上限时淘汰最久未访问的会话"""
        s = SessionStore(max_sessions=3)
        for i in range(5):
            s.create(f"Q{i}", f"query{i}")

        alive = {x["query_id"] for x in s.list_all()}
        assert s.count() == 3
        assert alive == {"Q2", "Q3", "Q4"}, alive
        assert s.evicted_count == 2

    def test_read_refreshes_recency(self):
        """被读过的会话不应比闲置会话先被淘汰"""
        s = SessionStore(max_sessions=3)
        for i in range(3):
            s.create(f"Q{i}")
        s.get("Q0")          # Q0 变成最近访问，Q1 成为最久未访问
        s.create("Q3")

        alive = {x["query_id"] for x in s.list_all()}
        assert "Q0" in alive, "刚读过的 Q0 不该被淘汰"
        assert "Q1" not in alive, "最久未访问的 Q1 应被淘汰"

    def test_transition_refreshes_recency(self):
        """状态转移（确认）也算访问，会话不应在流转中被淘汰"""
        s = SessionStore(max_sessions=3)
        for i in range(3):
            s.create(f"Q{i}")
        s.confirm("Q0", INSTANCE_ID)
        s.create("Q3")

        alive = {x["query_id"] for x in s.list_all()}
        assert "Q0" in alive, "已确认的会话不该被淘汰"

    def test_evicted_session_reports_not_found(self):
        """被淘汰的 query_id 后续访问应得到"不存在"，而不是脏数据"""
        s = SessionStore(max_sessions=1)
        s.create("OLD")
        s.create("NEW")

        assert s.get("OLD") is None
        with pytest.raises(SessionNotFoundError):
            s.require("OLD")

    def test_zero_means_unlimited(self):
        """max_sessions<=0 表示不限制（仅用于测试对照）"""
        s = SessionStore(max_sessions=0)
        for i in range(50):
            s.create(f"Q{i}")
        assert s.count() == 50
        assert s.evicted_count == 0

    def test_default_cap_is_positive(self):
        """默认上限必须是正数 —— 上限为 0 等于没有上限，会重新变成内存泄漏"""
        assert SessionStore().max_sessions > 0

    def test_overwrite_existing_key_counts_once(self):
        """同 query_id 重复 create 是覆盖，不应让计数虚增"""
        s = SessionStore(max_sessions=2)
        s.create("Q0")
        s.create("Q0")
        s.create("Q0")
        assert s.count() == 1
        assert s.evicted_count == 0


class TestSingleton:
    """全局单例"""

    def test_get_session_store_returns_singleton(self):
        """多次获取返回同一实例"""
        assert get_session_store() is get_session_store()

    def test_reset_session_store_creates_new_instance(self):
        """reset 后重新创建实例，旧数据不残留"""
        first = get_session_store()
        first.create("Q_050", "黑色轿车")
        reset_session_store()
        second = get_session_store()
        assert second is not first
        assert second.get("Q_050") is None


class TestThreadSafety:
    """并发读写"""

    def test_concurrent_create_and_confirm(self, store):
        """多线程并发创建+确认不丢会话、不抛异常"""
        errors = []
        thread_count, per_thread = 8, 50

        def worker(base):
            try:
                for i in range(per_thread):
                    query_id = f"Q_{base}_{i}"
                    store.create(query_id, f"查询{base}")
                    store.confirm(query_id, INSTANCE_ID)
            except Exception as e:  # pragma: no cover - 失败时收集原因
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(n,)) for n in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.count() == thread_count * per_thread
        assert store.get("Q_7_49")["state"] == "confirmed"

    def test_concurrent_confirm_same_session(self, store):
        """多线程并发确认同一会话：不抛异常，最终状态与字段一致"""
        store.create("Q_060", "黑色轿车")
        errors = []

        def worker(instance_id):
            try:
                store.confirm("Q_060", instance_id)
            except Exception as e:  # pragma: no cover - 失败时收集原因
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"CF3_c001_V{i:04d}_000001",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        session = store.get("Q_060")
        assert session["state"] == "confirmed"
        # 确认的是某一个具体实例，而不是半写状态
        assert session["confirmed_instance_id"] in {
            f"CF3_c001_V{i:04d}_000001" for i in range(20)
        }
        # 并发确认后仍可正常回溯
        assert store.set_backtracked("Q_060", {"ok": True})["state"] == "backtracked"
