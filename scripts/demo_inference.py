"""
scripts/demo_inference.py - 端到端推理演示（检索 → 确认 → 回溯）

用法:
    python scripts/demo_inference.py --query "白色轿车"
    python scripts/demo_inference.py --query "黑色SUV" --top_k 3
    python scripts/demo_inference.py --query "白色轿车" --no-backtrack

## 这个脚本做什么

它**直接调用生产代码**（`api/routes/search.py` 的检索函数 +
`src/trajectory/builder.py` 的轨迹构建器），不经过 HTTP，也不重新实现一遍逻辑。
因此它的输出与线上接口一致；同时它把检索/确认/回溯三步在一次运行里跑完，
便于命令行验证与排障。

`--query` 命中后，脚本会把**第一条候选当作"人工确认"的目标**（并在输出里
明确标注这是模拟确认）继续回溯 —— 因为真实系统里这一步必须由人来做。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logger import get_logger

logger = get_logger("scripts.demo_inference")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="端到端推理演示")
    parser.add_argument("--query", type=str, required=True, help="查询文本，如 '白色轿车'")
    parser.add_argument("--top_k", type=int, default=5, help="展示前 K 条候选")
    parser.add_argument("--no-backtrack", action="store_true",
                        help="只做检索，不继续回溯")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "strong", "stitch"],
                        help="回溯模式（stitch 为弱身份概率拼接）")
    return parser.parse_args()


def _fmt(value, unit: str = "", dash: str = "—") -> str:
    """格式化一个可能为 None 的值

    注意不能写 `d.get(k, '—')`：键存在但值为 None 时默认值不生效，
    会渲染成 "Nones" 这种既不是数字也不是占位符的东西。
    """
    if value is None or value == "":
        return dash
    return f"{value}{unit}"


def _print_candidates(candidates, top_k: int) -> None:
    """打印候选列表"""
    print(f"\n{'=' * 78}")
    print(f"检索候选（共 {len(candidates)} 条，展示前 {min(top_k, len(candidates))} 条）")
    print("=" * 78)
    for i, cand in enumerate(candidates[:top_k], start=1):
        score = cand.get("final_score")
        score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
        print(
            f"  {i}. 匹配度 {score_txt}  |  "
            f"摄像头 {_fmt(cand.get('camera_id'))}  |  "
            f"轨迹 {_fmt(cand.get('track_id'))}"
        )
        attrs = cand.get("attributes") or {}
        if attrs:
            shown = "、".join(f"{k}={v}" for k, v in list(attrs.items())[:4])
            print(f"     属性: {shown}")


def _print_chain(result: dict) -> None:
    """打印离散观测链（区分观测段与推断段）"""
    print(f"\n{'=' * 78}")
    print("离散观测链")
    print("=" * 78)

    seq = result.get("camera_sequence") or []
    print(f"  摄像头序列: {' -> '.join(seq) if seq else '—'}")
    conf = result.get("overall_confidence")
    print(f"  整链置信度: {conf if conf is not None else '—'}")
    mode = (result.get("identity") or {}).get("mode")
    print(f"  匹配方式  : {mode or '—'}")

    obs = result.get("observation_segments") or []
    print(f"\n  ▎观测段（摄像头实拍，共 {len(obs)} 段）")
    if not obs:
        print("     （无）")
    for seg in obs:
        print(
            f"     [{_fmt(seg.get('camera_id'))}] "
            f"{_fmt(seg.get('start_time'))} ~ {_fmt(seg.get('end_time'))}  "
            f"{_fmt(seg.get('direction'), dash='')}"
        )

    inf = result.get("inference_segments") or []
    print(f"\n  ▎推断段（摄像头之间的概率推断，共 {len(inf)} 段）")
    if not inf:
        print("     （无）")
    for seg in inf:
        print(
            f"     [{_fmt(seg.get('source_camera_id'))} -> "
            f"{_fmt(seg.get('target_camera_id'))}]  "
            f"实际 {_fmt(seg.get('actual_travel_time'), 's')} / "
            f"理论 {_fmt(seg.get('estimated_travel_time'), 's')}  "
            f"置信度 {_fmt(seg.get('confidence'))}"
        )


def main() -> None:
    """主函数"""
    args = parse_args()

    # ── 1. 检索（复用生产函数） ──
    from api.routes.search import _build_candidates_with_clip, _extract_query_features, _load_results

    data = _load_results()
    if not data:
        logger.error("数据不可用：请确认 output/datastore 或 output/cityflow_results.json 存在")
        sys.exit(2)

    logger.info(f"查询: {args.query!r}")
    features = _extract_query_features(args.query)
    logger.info(f"解析结果: color={features.get('color')}, vehicle_type={features.get('vehicle_type')}")

    candidates = _build_candidates_with_clip(data, args.query, features, top_k=args.top_k)
    if not candidates:
        print(f"\n未检索到与 {args.query!r} 匹配的候选。")
        sys.exit(0)

    _print_candidates(candidates, args.top_k)

    if args.no_backtrack:
        return

    # ── 2. 模拟人工确认（真实系统里这一步由人做） ──
    top = candidates[0]
    instance_id = top.get("instance_id")
    print(f"\n[模拟人工确认] 取第一条候选作为确认目标: {instance_id}")
    print("               注意：真实系统中确认必须由研判人员执行，本脚本自动选取仅为演示。")

    if not instance_id:
        logger.warning("候选缺少 instance_id，无法继续回溯")
        sys.exit(1)

    # ── 3. 回溯（复用 TrajectoryBuilder） ──
    from src.trajectory.builder import (
        TrajectoryDataUnavailableError,
        TrajectoryNotFoundError,
        get_trajectory_builder,
    )

    builder = get_trajectory_builder()
    try:
        result = builder.build(instance_id=instance_id, mode=args.mode)
    except TrajectoryDataUnavailableError as e:
        logger.error(f"轨迹数据不可用: {e}")
        sys.exit(2)
    except TrajectoryNotFoundError as e:
        logger.error(f"未找到该目标的轨迹: {e}")
        sys.exit(1)

    _print_chain(result)
    logger.info("演示推理完成")


if __name__ == "__main__":
    main()
