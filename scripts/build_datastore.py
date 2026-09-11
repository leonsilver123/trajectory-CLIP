"""
scripts.build_datastore - 把 cityflow_results.json 落成 datastore（离线一次性构建）

产物（默认写在 output/datastore/）:
    detections.parquet      检测表（一列一字段；嵌套字段以 JSON 文本存储）
    tracks.parquet          轨迹表
    det_image_vectors.npy   内联 768 维图像向量（从检测里抽出来单独存，占 JSON 体积近三成）
    det_text_vectors.npy    内联 768 维文本向量
    *_rows.npy              上述向量对应的检测行号
    meta.sqlite             元数据：schema 版本、源文件指纹、统计、summary、列格式自描述

设计要点
--------
1. **叠加式**：datastore 只是可选加速层。产物缺失/损坏/版本不符时，
   `src.storage.datastore.load_results()` 会自动回退 JSON 直读，行为与构建前一致。
   所以本脚本没跑过、跑失败，系统都照常工作。
2. **自描述**：哪些列是 JSON 文本、哪些列抽到了独立向量文件，都写进 meta 表，
   读取层据此还原，不写死列名——将来新增字段只要重跑本脚本即可。
3. **不写入确定性的东西**：所有产物均由源 JSON 唯一决定，无随机数、无时间戳参与数据内容
   （`built_at` 仅记录构建时刻，不参与任何还原逻辑）。

使用方式:
    .venv/Scripts/python.exe scripts/build_datastore.py                # 构建
    .venv/Scripts/python.exe scripts/build_datastore.py --verify       # 构建并与源 JSON 逐字段比对
    .venv/Scripts/python.exe scripts/build_datastore.py --force        # 已存在也重建
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.storage.datastore import (  # noqa: E402  (需先修正 sys.path)
    DATASTORE_DIRNAME,
    DETECTIONS_PARQUET,
    IMAGE_VECTORS_NPY,
    META_SQLITE,
    SCHEMA_VERSION,
    TEXT_VECTORS_NPY,
    TRACKS_PARQUET,
    VECTOR_INDEX_NPY,
    read_json_file,
)

# 这些字段一律以 JSON 文本存储：它们的取值是嵌套/不定形结构，
# 直接用 Parquet 的 struct 会把「缺失的键」补成 None，与 JSON 路径不等价。
_FORCE_JSON_COLUMNS = ("attributes", "color_analysis")

# 检测里这两个内联向量字段抽到独立 .npy，检测表里只留行号
_VECTOR_COLUMNS = ("clip_image_vector", "clip_text_vector")

_DEFAULT_RESULTS = _PROJECT_ROOT / "output" / "cityflow_results.json"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "output" / DATASTORE_DIRNAME

# 列的"存在性"列：Parquet 是列式的，无法区分「该键不存在」与「该键值为 null」，
# 而 JSON 里两者语义不同（如 67400 条检测根本没有 bbox_size 键）。
# 故对「只在部分记录里出现的列」额外存一行 JSON 名字列表，读取层据此决定删键还是留 null。
PRESENCE_COLUMN = "_present_keys"


# ============================================================
# 列编码
# ============================================================

def _encode_column(
    values: List[Any], force_json: bool
) -> Tuple[Any, bool]:
    """
    把一列 Python 值编码成 pyarrow 列

    Returns:
        (pyarrow 数组, 是否用了 JSON 文本)
    """
    import pyarrow as pa

    if force_json:
        text = [None if v is None else json.dumps(v, ensure_ascii=False) for v in values]
        return pa.array(text, pa.string()), True

    try:
        # 先按原生类型推断（标量直接成列，列表成为 list<...>）
        return pa.array(values), False
    except Exception:
        # 类型混杂等推断失败 → 退化为 JSON 文本，保证不丢数据
        text = [None if v is None else json.dumps(v, ensure_ascii=False) for v in values]
        return pa.array(text, pa.string()), True


def _all_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """记录集合里出现过的全部键（保持首次出现的顺序）"""
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    return keys


def _variable_columns(
    rows: List[Dict[str, Any]], columns: List[str]
) -> Tuple[List[str], List[str]]:
    """
    只在部分记录里出现的列（这些列需要额外的存在性标记）

    Returns:
        (absent_when_null, nullable)
        - absent_when_null: 源记录里该键要么存在且非空、要么没有 → 读出来是 None 即代表缺键
        - nullable: 源记录里存在"键在、值就是 null" → 必须逐行记录存在性才能精确还原
    """
    total = len(rows)
    absent_when_null: List[str] = []
    nullable: List[str] = []
    for col in columns:
        present = sum(1 for row in rows if col in row)
        if present in (0, total):
            continue
        if any(col in row and row[col] is None for row in rows):
            nullable.append(col)
        else:
            absent_when_null.append(col)
    return absent_when_null, nullable


def _build_table(
    rows: List[Dict[str, Any]],
    columns: List[str],
    extra: Optional[Dict[str, List[Any]]] = None,
    presence_columns: Optional[List[str]] = None,
) -> Tuple[Any, List[str]]:
    """
    按列构造 pyarrow 表

    Args:
        rows: 记录列表
        columns: 要落盘的列名（顺序即表中顺序）
        extra: 额外列（如承载 det_to_track_map 的 track_id），键为列名
        presence_columns: 需要逐行存在性标记的列（"键在但值为 null"的列）；
                          非空时才会附加 PRESENCE_COLUMN 列

    Returns:
        (pyarrow 表, 以 JSON 文本存储的列名列表)
    """
    import pyarrow as pa

    arrays = []
    json_columns: List[str] = []
    for name in columns:
        values = [row.get(name) for row in rows]
        array, used_json = _encode_column(values, force_json=name in _FORCE_JSON_COLUMNS)
        arrays.append(array)
        if used_json:
            json_columns.append(name)
    for name, values in (extra or {}).items():
        array, used_json = _encode_column(values, force_json=False)
        arrays.append(array)
        if used_json:  # extra 列属于内部列，理论上不会走 JSON 分支
            json_columns.append(name)

    names = list(columns) + list((extra or {}).keys())
    if presence_columns:
        presence = [
            json.dumps([c for c in presence_columns if c in row]) for row in rows
        ]
        names.append(PRESENCE_COLUMN)
        arrays.append(pa.array(presence, pa.string()))

    return pa.table(dict(zip(names, arrays))), json_columns


def _extract_vectors(
    detections: List[Dict[str, Any]], out_dir: Path
) -> Dict[str, List[str]]:
    """
    把内联向量抽成独立的 .npy 文件

    占 JSON 体积近三成的两个 768 维字段（949 条检测带）单独存放，
    检测表里不再有它们；读取层按 meta 表的 `vector_files` 描述回填。

    Returns:
        {列名: [行号文件名, 向量文件名]}；无该列数据时不含该键
    """
    import numpy as np

    vector_files: Dict[str, List[str]] = {}
    for column in _VECTOR_COLUMNS:
        rows: List[int] = []
        vectors: List[List[float]] = []
        for idx, det in enumerate(detections):
            value = det.get(column)
            if isinstance(value, list) and value:
                rows.append(idx)
                vectors.append(value)
        if not rows:
            continue
        # 文件名按列名派生（clip_image_vector → det_image_vectors.npy）
        suffix = "image" if "image" in column else "text"
        rows_name = VECTOR_INDEX_NPY if suffix == "image" else f"det_{suffix}_vector_rows.npy"
        matrix_name = IMAGE_VECTORS_NPY if suffix == "image" else TEXT_VECTORS_NPY
        # float64：与 JSON 里的双精度浮点逐位一致，不引入精度差异
        np.save(out_dir / rows_name, np.asarray(rows, dtype=np.int64))
        np.save(out_dir / matrix_name, np.asarray(vectors, dtype=np.float64))
        vector_files[column] = [rows_name, matrix_name]
    return vector_files


# ============================================================
# SQLite 元数据
# ============================================================

def _write_meta_sqlite(
    out_dir: Path,
    results_json: Path,
    meta: Dict[str, Any],
    counts: Dict[str, int],
    summary: Dict[str, Any],
) -> None:
    """写 meta.sqlite（meta / counts / summary 三张表），覆盖写入"""
    path = out_dir / META_SQLITE
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE counts (name TEXT PRIMARY KEY, value INTEGER)")
        conn.execute("CREATE TABLE summary (id INTEGER PRIMARY KEY, payload TEXT)")
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [(k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
             for k, v in meta.items()],
        )
        conn.executemany("INSERT INTO counts (name, value) VALUES (?, ?)", list(counts.items()))
        conn.execute(
            "INSERT INTO summary (id, payload) VALUES (1, ?)",
            (json.dumps(summary, ensure_ascii=False),),
        )
        conn.commit()
    finally:
        conn.close()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """源文件指纹（分块读，避免一次性吃下 100MB+）"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ============================================================
# 构建
# ============================================================

def build(
    results_json: Path,
    out_dir: Path,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行构建

    Args:
        results_json: 源 JSON 路径
        out_dir: datastore 输出目录
        force: 已存在时是否覆盖
        verbose: 是否打印进度

    Returns:
        构建摘要（行数、体积、耗时）

    Raises:
        FileNotFoundError: 源 JSON 不存在
        RuntimeError: 源 JSON 解析失败
    """
    if not results_json.is_file():
        raise FileNotFoundError(f"源数据不存在: {results_json}")
    if out_dir.exists() and not force and (out_dir / META_SQLITE).exists():
        raise RuntimeError(f"datastore 已存在（加 --force 覆盖）: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if verbose:
        print(f"[1/5] 读取源 JSON: {results_json} ({results_json.stat().st_size / 1e6:.1f} MB)")
    t0 = time.time()
    data = read_json_file(str(results_json))
    if not data:
        raise RuntimeError(f"源 JSON 解析失败或为空: {results_json}")
    t_json = time.time() - t0

    detections: List[Dict[str, Any]] = data.get("detections") or []
    tracks: List[Dict[str, Any]] = data.get("tracks") or []
    det_to_track: Dict[str, str] = data.get("det_to_track_map") or {}
    summary: Dict[str, Any] = data.get("summary") or {}
    if verbose:
        print(f"      检测={len(detections)} 轨迹={len(tracks)} 映射={len(det_to_track)} 耗时={t_json:.2f}s")

    # 检测表的列：源 JSON 里出现过的字段（不含被抽走的向量列），保持原顺序
    det_all_keys = _all_keys(detections)
    det_columns = [c for c in det_all_keys if c not in _VECTOR_COLUMNS]
    track_columns = _all_keys(tracks)

    if verbose:
        print(f"[4/5] 抽取内联向量到独立文件")
    t0 = time.time()
    vector_files = _extract_vectors(detections, out_dir)
    t_vec = time.time() - t0

    # 存在性标记要覆盖「被抽走的向量列」，否则读不回"该键不存在"的行
    det_absent, det_nullable = _variable_columns(detections, det_all_keys)
    track_absent, track_nullable = _variable_columns(tracks, track_columns)

    if verbose:
        print(f"[2/5] 写检测表 {DETECTIONS_PARQUET}（{len(det_columns)} 列 + 内部列 track_id）")
    t0 = time.time()
    det_table, det_json_columns = _build_table(
        detections,
        det_columns,
        extra={"track_id": [det_to_track.get(d.get("target_id", "")) for d in detections]},
        presence_columns=det_nullable,
    )
    import pyarrow.parquet as pq

    pq.write_table(det_table, out_dir / DETECTIONS_PARQUET, compression="zstd")
    t_det = time.time() - t0

    if verbose:
        print(f"[3/5] 写轨迹表 {TRACKS_PARQUET}（{len(track_columns)} 列）")
    t0 = time.time()
    track_table, track_json_columns = _build_table(
        tracks, track_columns, presence_columns=track_nullable
    )
    pq.write_table(track_table, out_dir / TRACKS_PARQUET, compression="zstd")
    t_track = time.time() - t0

    if verbose:
        print(f"[5/5] 写元数据 {META_SQLITE}")
    camera_ids = {d.get("camera_id", "") for d in detections if d.get("camera_id")}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(results_json),
        "source_size_bytes": results_json.stat().st_size,
        "source_mtime": results_json.stat().st_mtime,
        "source_sha256": _sha256(results_json),
        "detection_json_columns": det_json_columns,
        "track_json_columns": track_json_columns,
        "vector_files": vector_files,
        "map_column": "track_id",
        "presence_column": PRESENCE_COLUMN,
        "detection_absent_when_null_columns": det_absent,
        "detection_nullable_columns": det_nullable,
        "track_absent_when_null_columns": track_absent,
        "track_nullable_columns": track_nullable,
        "detection_columns": det_columns,
        "track_columns": track_columns,
    }
    counts = {
        "detections": len(detections),
        "tracks": len(tracks),
        "det_to_track_map": len(det_to_track),
        "cameras": len(camera_ids),
        "vectors": 0,
    }
    if vector_files:
        import numpy as np
        counts["vectors"] = int(
            sum(np.load(out_dir / files[0]).shape[0] for files in vector_files.values())
        )
    _write_meta_sqlite(out_dir, results_json, meta, counts, summary)

    sizes = {p.name: p.stat().st_size for p in sorted(out_dir.iterdir()) if p.is_file()}
    result = {
        "out_dir": str(out_dir),
        "detections": len(detections),
        "tracks": len(tracks),
        "det_to_track_map": len(det_to_track),
        "camera_ids": len(camera_ids),
        "json_read_seconds": round(t_json, 3),
        "detections_write_seconds": round(t_det, 3),
        "tracks_write_seconds": round(t_track, 3),
        "vectors_write_seconds": round(t_vec, 3),
        "total_seconds": round(time.time() - t_start, 3),
        "json_size_bytes": results_json.stat().st_size,
        "datastore_size_bytes": sum(sizes.values()),
        "files": sizes,
    }
    if verbose:
        print("\n产物:")
        for name, size in sizes.items():
            print(f"  {out_dir / name}  {size / 1e6:.2f} MB")
        print(f"\n源 JSON {result['json_size_bytes'] / 1e6:.1f} MB → datastore "
              f"{result['datastore_size_bytes'] / 1e6:.2f} MB "
              f"（{result['json_size_bytes'] / max(result['datastore_size_bytes'], 1):.1f}x 更小）")
        print(f"总耗时 {result['total_seconds']}s（其中读 JSON {result['json_read_seconds']}s）")
    return result


# ============================================================
# 校验：datastore 还原结果与源 JSON 逐字段比对
# ============================================================

def verify(results_json: Path, out_dir: Path) -> bool:
    """
    用读取层的还原逻辑重新加载 datastore，与源 JSON 逐字段比对

    比对的是**还原后的字典**（即在线路径真正拿到的对象），不是 Parquet 本身。
    """
    from src.storage import datastore as ds

    source = read_json_file(str(results_json))
    if source is None:
        print("源 JSON 读取失败，无法校验")
        return False
    ds.reset_cache()
    rebuilt = ds._read_datastore(out_dir)  # 直接走还原逻辑，绕开可用性缓存
    if rebuilt is None:
        print("datastore 还原失败")
        return False

    ok = True
    for key in ("detections", "tracks", "det_to_track_map", "summary"):
        src = source.get(key)
        got = rebuilt.get(key)
        if src == got:
            size = len(src) if hasattr(src, "__len__") else "-"
            print(f"  [OK]   {key}: 与源 JSON 完全一致（{size}）")
            continue
        ok = False
        print(f"  [DIFF] {key}: 与源 JSON 不一致")
        if isinstance(src, list) and isinstance(got, list):
            print(f"         长度 源={len(src)} 还原={len(got)}")
            for i, (a, b) in enumerate(zip(src, got)):
                if a != b:
                    only_src = {k: a.get(k) for k in a if a.get(k) != b.get(k)} if isinstance(a, dict) else a
                    only_got = {k: b.get(k) for k in b if b.get(k) != a.get(k)} if isinstance(b, dict) else b
                    print(f"         首个差异行 index={i}")
                    print(f"           源  多出/不同: {str(only_src)[:300]}")
                    print(f"           还原多出/不同: {str(only_got)[:300]}")
                    break
        elif isinstance(src, dict) and isinstance(got, dict):
            diff = [k for k in set(src) | set(got) if src.get(k) != got.get(k)]
            print(f"         差异键: {diff[:10]}")
    return ok


# ============================================================
# 入口
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="构建 datastore（JSON → Parquet + SQLite + 向量文件）")
    parser.add_argument("--results-json", default=str(_DEFAULT_RESULTS), help="源 JSON 路径")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR), help="datastore 输出目录")
    parser.add_argument("--force", action="store_true", help="已存在时覆盖重建")
    parser.add_argument("--verify", action="store_true", help="构建后与源 JSON 逐字段比对")
    parser.add_argument("--verify-only", action="store_true",
                        help="只校验已存在的 datastore，不重新构建")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    out_dir = Path(args.out_dir)

    if args.verify_only:
        print("=" * 66)
        print("校验 datastore（不重新构建）")
        print("=" * 66)
        ok = verify(results_json, out_dir)
        print("\n校验结果:", "全部一致" if ok else "存在差异")
        return 0 if ok else 1

    print("=" * 66)
    print("构建 datastore")
    print("=" * 66)
    try:
        build(results_json, out_dir, force=args.force)
    except Exception as e:
        print(f"\n构建失败: {e.__class__.__name__}: {e}")
        return 1

    if args.verify:
        print("\n校验（datastore 还原 vs 源 JSON）:")
        ok = verify(results_json, out_dir)
        print("\n校验结果:", "全部一致" if ok else "存在差异")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
