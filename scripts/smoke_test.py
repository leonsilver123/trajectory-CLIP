"""
scripts.smoke_test - 前后端冒烟测试

在 API 服务已启动的前提下，按「检索 → 确认 → 回溯」的真实闭环打一遍接口，
并对前端模块做可导入性检查。逐项打印通过/失败，最后给出汇总。

启动服务（另开一个终端）:
    .venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

运行本脚本:
    .venv/Scripts/python.exe scripts/smoke_test.py [--base-url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 直接运行本脚本时 sys.path[0] 是 scripts/ 而非项目根，导致 frontend/src 无法导入。
# 项目内其他脚本同样是显式把项目根插进 sys.path（见 CLAUDE.md 约定）。
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 逐项结果：(名称, 是否通过, 说明)
_RESULTS: list[Tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    """记录一条冒烟结果并即时打印"""
    _RESULTS.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}" + (f" — {detail}" if detail else ""))


def _post(base: str, path: str, payload: Dict[str, Any], timeout: int = 120) -> Tuple[Optional[int], Any]:
    """POST 并返回 (状态码, 解析后的 JSON 或错误文本)；网络异常返回 (None, 错误)"""
    try:
        r = requests.post(f"{base}{path}", json=payload, timeout=timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:300]
    except Exception as e:  # 连接失败、超时等
        return None, f"{e.__class__.__name__}: {e}"


def _get(base: str, path: str, timeout: int = 120) -> Tuple[Optional[int], Any]:
    """GET 并返回 (状态码, 解析后的 JSON 或错误文本)"""
    try:
        r = requests.get(f"{base}{path}", timeout=timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:300]
    except Exception as e:
        return None, f"{e.__class__.__name__}: {e}"


def test_health(base: str) -> None:
    """健康检查"""
    print("\n[1] 健康检查  GET /health")
    code, body = _get(base, "/health")
    _record("GET /health", code == 200, f"status={code} body={str(body)[:160]}")


def test_search(base: str) -> Optional[str]:
    """文本检索；返回 query_id 供后续步骤串联"""
    print("\n[2] 文本检索  POST /api/v1/search/query")
    code, body = _post(base, "/api/v1/search/query", {"query_text": "黑色轿车", "top_k": 5})
    if code != 200:
        _record("POST /search/query", False, f"status={code} body={str(body)[:200]}")
        return None

    candidates = body.get("candidates", []) if isinstance(body, dict) else []
    query_id = body.get("query_id") if isinstance(body, dict) else None
    _record(
        "POST /search/query",
        bool(query_id) and len(candidates) > 0,
        f"query_id={query_id} 候选数={len(candidates)} total={body.get('total_count')}",
    )
    if candidates:
        c0 = candidates[0]
        print(f"        首条候选: instance_id={c0.get('instance_id')} "
              f"track_id={c0.get('track_id')} camera={c0.get('camera_id')} "
              f"final_score={c0.get('final_score')}")
    return query_id


def test_plate_search(base: str) -> None:
    """车牌检索"""
    print("\n[3] 车牌检索  POST /api/v1/search/plate")
    code, body = _post(base, "/api/v1/search/plate", {"plate_number": "京"})
    ok = code == 200 and isinstance(body, dict) and "candidates" in body
    _record("POST /search/plate", ok, f"status={code} total={body.get('total_count') if isinstance(body, dict) else body}")


def test_confirm(base: str, query_id: Optional[str], instance_id: Optional[str]) -> None:
    """确认目标；同时验证不存在的 query_id 必须返回 404"""
    print("\n[4] 确认目标  POST /api/v1/confirm/target")
    if query_id and instance_id:
        code, body = _post(base, "/api/v1/confirm/target",
                           {"query_id": query_id, "instance_id": instance_id})
        ok = code == 200 and isinstance(body, dict) and body.get("status") == "confirmed"
        _record("POST /confirm/target (合法 query_id)", ok, f"status={code} body={str(body)[:160]}")
    else:
        _record("POST /confirm/target (合法 query_id)", False, "缺少 query_id/instance_id，已跳过")

    # 负例：不存在的 query_id 应返回 404（任务 1 的验收点）
    code, body = _post(base, "/api/v1/confirm/target",
                       {"query_id": "NONEXISTENT_QUERY_ID", "instance_id": "X"})
    _record("POST /confirm/target (非法 query_id 应 404)", code == 404,
            f"status={code} body={str(body)[:160]}")


def test_backtrack_trajectory(base: str, instance_id: Optional[str]) -> None:
    """跨镜轨迹（真实 vehicle_id 聚合路径）"""
    print("\n[5] 跨镜轨迹  POST /api/v1/backtrack/trajectory")
    iid = instance_id or "CF3_c001_V0034_000001"
    code, body = _post(base, "/api/v1/backtrack/trajectory", {"instance_id": iid})
    if code == 200 and isinstance(body, dict):
        traj = body.get("trajectory", {})
        _record("POST /backtrack/trajectory", bool(traj),
                f"vehicle_id={traj.get('vehicle_id')} 摄像头数={traj.get('total_cameras')} "
                f"检测数={traj.get('total_detections')}")
    else:
        _record("POST /backtrack/trajectory", False, f"status={code} body={str(body)[:200]}")


def test_backtrack_trace(base: str, instance_id: Optional[str]) -> None:
    """回溯 /trace：同一 instance_id 连打两次，结果必须完全一致（无随机性）"""
    print("\n[6] 回溯（确定性检查）  POST /api/v1/backtrack/trace")
    iid = instance_id or "CF3_c001_V0034_000001"
    code1, body1 = _post(base, "/api/v1/backtrack/trace", {"instance_id": iid})
    code2, body2 = _post(base, "/api/v1/backtrack/trace", {"instance_id": iid})
    if code1 == 200 and code2 == 200:
        _record("POST /backtrack/trace 可调用", True,
                f"camera_sequence={body1.get('camera_sequence')}")
        _record("POST /backtrack/trace 两次结果一致", body1 == body2,
                "一致" if body1 == body2 else "两次响应不同 → 仍存在随机性")
    else:
        _record("POST /backtrack/trace 可调用", False, f"status1={code1} status2={code2} body={str(body1)[:200]}")


def test_dashboard(base: str) -> None:
    """仪表盘接口"""
    print("\n[7] 仪表盘  GET /api/v1/dashboard/*")
    for path in ("/api/v1/dashboard/stats", "/api/v1/dashboard/cameras", "/api/v1/dashboard/health"):
        code, body = _get(base, path, timeout=120)
        # 路由不存在(404)与内部错误(500)要区分开，避免把"没这个接口"误报成通过
        if code == 404:
            print(f"        跳过 {path}（该接口不存在，status=404）")
            continue
        _record(f"GET {path}", code == 200, f"status={code} body={str(body)[:120]}")


def test_frontend_imports() -> None:
    """前端模块可导入性检查（不启动 Streamlit，只验证 import 不炸）"""
    print("\n[8] 前端模块导入检查")
    mods = ["frontend.utils"]
    for m in mods:
        try:
            __import__(m)
            _record(f"import {m}", True, "")
        except Exception as e:
            _record(f"import {m}", False, f"{e.__class__.__name__}: {e}")

    # frontend/app.py 是 Streamlit 入口，import 它需要 streamlit 运行时；
    # 这里只做语法编译检查，确保没有语法错误。
    for rel in ("frontend/app.py", "frontend/utils.py"):
        p = _PROJECT_ROOT / rel
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
            _record(f"compile {rel}", True, "")
        except Exception as e:
            _record(f"compile {rel}", False, f"{e.__class__.__name__}: {e}")


def main() -> int:
    """入口：跑全部冒烟项并打印汇总"""
    parser = argparse.ArgumentParser(description="前后端冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("=" * 66)
    print(f"冒烟测试 — base_url={base}")
    print("=" * 66)

    test_health(base)
    if not _RESULTS or not _RESULTS[0][1]:
        print("\n服务不可达，后续接口测试无法进行。")
        print("请先启动: .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000")
        return 1

    query_id = test_search(base)
    instance_id = None
    # 从检索结果里取一个真实 instance_id 用于后续串联
    code, body = _post(base, "/api/v1/search/query", {"query_text": "轿车", "top_k": 1})
    if code == 200 and isinstance(body, dict) and body.get("candidates"):
        instance_id = body["candidates"][0].get("instance_id")

    test_plate_search(base)
    test_confirm(base, query_id, instance_id)
    test_backtrack_trajectory(base, instance_id)
    test_backtrack_trace(base, instance_id)
    test_dashboard(base)
    test_frontend_imports()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [(n, d) for n, ok, d in _RESULTS if not ok]

    print("\n" + "=" * 66)
    print(f"汇总: {passed}/{len(_RESULTS)} 项通过")
    if failed:
        print("未通过项:")
        for n, d in failed:
            print(f"  - {n}: {d}")
    print("=" * 66)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
