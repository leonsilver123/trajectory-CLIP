"""
tests.test_backtrack_cache - 回溯结果缓存的上限与线程安全（PLAN3-D1）

背景：`api/routes/backtrack.py` 的 `_backtrack_results` 早先是普通 dict，
每次 /trace 往里塞一条完整轨迹且**永不删除**；而 FastAPI 的同步端点跑在线程池里，
多线程并发读写同一个 dict 本身也不安全。

现在是一个带锁的 OrderedDict + LRU 上限（`_MAX_BACKTRACK_RESULTS`）。
本文件锁定三件事：上限生效、读算访问、并发不炸。
"""

from __future__ import annotations

import threading

import pytest

from api.routes import backtrack as bt


@pytest.fixture(autouse=True)
def _clean_cache():
    """每个测试前后清空缓存，避免测试间互相污染"""
    with bt._backtrack_results_lock:
        bt._backtrack_results.clear()
    yield
    with bt._backtrack_results_lock:
        bt._backtrack_results.clear()


class TestBoundedCache:
    def test_evicts_oldest_when_over_capacity(self):
        """超过上限时淘汰最久未访问的结果"""
        cap = bt._MAX_BACKTRACK_RESULTS
        for i in range(cap + 5):
            bt._store_backtrack_result(f"Q{i}", {"query_id": f"Q{i}"})

        assert len(bt._backtrack_results) == cap
        assert bt._load_backtrack_result("Q0") is None, "最早的 5 条应被淘汰"
        assert bt._load_backtrack_result(f"Q{cap + 4}") is not None, "最新的必须还在"

    def test_read_refreshes_recency(self):
        """读过的结果不应比闲置的先被淘汰"""
        cap = bt._MAX_BACKTRACK_RESULTS
        bt._store_backtrack_result("KEEP", {"query_id": "KEEP"})
        for i in range(cap - 1):
            bt._store_backtrack_result(f"Q{i}", {"query_id": f"Q{i}"})

        bt._load_backtrack_result("KEEP")          # KEEP 变成最近访问
        bt._store_backtrack_result("NEW", {})      # 触发一次淘汰

        assert bt._load_backtrack_result("KEEP") is not None
        assert bt._load_backtrack_result("Q0") is None, "最久未访问的 Q0 应被淘汰"

    def test_missing_key_returns_none(self):
        """未命中的 query_id 返回 None（端点据此返回 404）"""
        assert bt._load_backtrack_result("NOT_EXIST") is None

    def test_capacity_is_positive(self):
        """上限必须是正数 —— 为 0 等于没有上限，就退回内存泄漏"""
        assert bt._MAX_BACKTRACK_RESULTS > 0

    def test_overwrite_does_not_grow(self):
        """同 query_id 重复写入是覆盖，不应撑大缓存"""
        for _ in range(10):
            bt._store_backtrack_result("SAME", {"query_id": "SAME"})
        assert len(bt._backtrack_results) == 1


class TestThreadSafety:
    def test_concurrent_store_and_load(self):
        """多线程并发读写不抛异常，且最终不超上限"""
        errors: list = []

        def writer(base: int) -> None:
            try:
                for i in range(200):
                    bt._store_backtrack_result(f"W{base}_{i}", {"query_id": f"W{base}_{i}"})
            except Exception as e:      # noqa: BLE001 - 测试需要捕获一切
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(200):
                    bt._load_backtrack_result(f"W0_{i}")
            except Exception as e:      # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(b,)) for b in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发读写抛异常: {errors[:3]}"
        assert len(bt._backtrack_results) <= bt._MAX_BACKTRACK_RESULTS
