"""
src.storage.datastore - 离线数据契约的统一读取层（可选加速层）

背景
----
在此之前，`output/cityflow_results.json`（107MB / 68349 条检测）被 **5 处**各自读取：
api/routes/search.py、api/routes/dashboard.py、src/trajectory/builder.py、
frontend/utils.py、frontend/home.py。每次在线首查都要重新解析整个 JSON（实测 1.2–1.5 秒），
且同一进程内不同模块各存一份内存副本。

本模块把这条读取路径收敛成**唯一一处**：JSON 解析只发生在 `_read_json()`，
其余消费方一律调用 `load_results()` / `load_detections()` / `get_stats()`。

叠加式设计（**不替换**现有路径）
--------------------------------
    output/datastore/ 存在且校验通过  → 走 Parquet + SQLite（毫秒级热读）
    output/datastore/ 缺失/损坏/版本不符 → 回退到 JSON 直读，行为与改造前完全一致
    JSON 也不存在                     → 返回 None（调用方按原有语义处理）

因此**不需要**先跑 `scripts/build_datastore.py` 才能用：没构建过就自动走老路，
构建失败、pyarrow 缺失、schema 版本不匹配也都会安全降级，绝不抛异常到接口层。

返回值约定
----------
`load_results()` 返回的字典与 `json.load(cityflow_results.json)` **结构等价**
（detections / tracks / summary / det_to_track_map 四个顶层键），且被**跨调用方共享**——
调用方只读，不要原地修改其中的检测字典。

使用方式:
    from src.storage.datastore import load_results, get_stats

    data = load_results() or {}
    stats = get_stats()          # 走 SQLite，不触发大 JSON 解析
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("storage.datastore")


# ============================================================
# 布局常量（构建脚本与读取层共用，避免两边写死文件名）
# ============================================================

SCHEMA_VERSION = 1                  # datastore 布局版本，写入 SQLite 的 meta 表
DATASTORE_DIRNAME = "datastore"
DETECTIONS_PARQUET = "detections.parquet"
TRACKS_PARQUET = "tracks.parquet"
META_SQLITE = "meta.sqlite"
IMAGE_VECTORS_NPY = "det_image_vectors.npy"   # 内联 768 维图像向量（仅部分检测带）
TEXT_VECTORS_NPY = "det_text_vectors.npy"     # 内联 768 维文本向量
VECTOR_INDEX_NPY = "det_vector_rows.npy"      # 上述向量对应的检测行号

# datastore 里必须同时存在的文件（缺任一即视为不可用 → 回退 JSON）
_REQUIRED_FILES = (DETECTIONS_PARQUET, TRACKS_PARQUET, META_SQLITE)

# 项目默认位置（与 api/routes/*、frontend/* 的既有约定一致）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
_DEFAULT_RESULTS_JSON = _DEFAULT_OUTPUT_DIR / "cityflow_results.json"
_DEFAULT_DATASTORE_DIR = _DEFAULT_OUTPUT_DIR / DATASTORE_DIRNAME


# ============================================================
# 缓存（沿用项目里"模块级惰性缓存"的惯例）
# ============================================================
# 说明：RLock 是因为 get_stats() 等便捷函数内部会再调 load_results()；
# 持锁解析可避免并发首查各自解析一遍大数据。
_LOCK = threading.RLock()
_RESULTS_CACHE: Dict[str, Tuple[Any, Tuple[float, float]]] = {}   # key -> (结果, 数据指纹)
_META_CACHE: Dict[str, Dict[str, Any]] = {}         # key -> SQLite 元数据


# ============================================================
# 路径解析
# ============================================================

def results_json_path() -> Path:
    """默认的 cityflow_results.json 路径"""
    return _DEFAULT_RESULTS_JSON


def datastore_dir() -> Path:
    """默认的 datastore 目录（output/datastore）"""
    return _DEFAULT_DATASTORE_DIR


def resolve_paths(results_path: Optional[str] = None) -> Tuple[Path, Path]:
    """
    把调用方给的路径解析成 (JSON 路径, datastore 目录)

    - 不传 / 传默认 JSON 路径 → datastore 用默认目录 output/datastore
    - 传了别的路径（测试常用）→ 用它的同级 datastore 目录，找不到自然回退 JSON
    """
    if not results_path:
        return _DEFAULT_RESULTS_JSON, _DEFAULT_DATASTORE_DIR
    json_path = Path(results_path)
    if not json_path.is_absolute():
        json_path = _PROJECT_ROOT / json_path
    json_path = json_path.resolve()
    if json_path == _DEFAULT_RESULTS_JSON:
        return json_path, _DEFAULT_DATASTORE_DIR
    return json_path, json_path.parent / DATASTORE_DIRNAME


def _cache_key(json_path: Path) -> str:
    """缓存键：JSON 路径的绝对字符串"""
    return str(json_path)


# ============================================================
# datastore 可用性
# ============================================================

def datastore_available(ds_dir: Optional[Path] = None) -> bool:
    """
    判断 datastore 是否可用（当前进程调用，无异常抛出）

    校验：必需文件齐全 + schema 版本匹配 + meta.sqlite 可读。
    任一条不满足都返回 False，由调用方回退 JSON。
    """
    directory = Path(ds_dir) if ds_dir else _DEFAULT_DATASTORE_DIR
    if not directory.is_dir():
        return False
    for name in _REQUIRED_FILES:
        if not (directory / name).is_file():
            return False
    meta = _read_meta(directory)
    if not meta:
        return False
    version = meta.get("schema_version")
    if str(version) != str(SCHEMA_VERSION):
        logger.warning(
            "datastore schema 版本不匹配（文件 %s / 期望 %s），回退 JSON 直读",
            version, SCHEMA_VERSION,
        )
        return False
    return True


def data_source(results_path: Optional[str] = None) -> str:
    """当前生效的数据来源: "datastore" / "json" / "none"（用于日志与健康检查）"""
    json_path, ds_dir = resolve_paths(results_path)
    if datastore_available(ds_dir):
        return "datastore"
    if json_path.is_file():
        return "json"
    return "none"


def has_data(results_path: Optional[str] = None) -> bool:
    """数据是否可用（datastore 或 JSON 任一存在即可）"""
    return data_source(results_path) != "none"


# ============================================================
# SQLite 元数据读取
# ============================================================

def _read_meta(ds_dir: Path) -> Dict[str, Any]:
    """读 meta.sqlite 的 meta 表为扁平字典（失败返回空字典）"""
    path = ds_dir / META_SQLITE
    key = str(path)
    with _LOCK:
        cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        # 只读打开，避免在线路径意外写盘
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        finally:
            conn.close()
        meta = {str(k): v for k, v in rows}
    except Exception as e:
        logger.warning("读取 datastore 元数据失败: %s", e)
        return {}
    with _LOCK:
        _META_CACHE[key] = meta
    return meta


def _read_counts(ds_dir: Path) -> Dict[str, int]:
    """读 counts 表（检测数/轨迹数等），失败返回空字典"""
    path = ds_dir / META_SQLITE
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT name, value FROM counts").fetchall()
        finally:
            conn.close()
        return {str(k): int(v) for k, v in rows}
    except Exception as e:
        logger.warning("读取 datastore 统计失败: %s", e)
        return {}


def _read_summary(ds_dir: Path) -> Dict[str, Any]:
    """读 SQLite 中缓存的 summary 字段（JSON 文本），失败返回空字典"""
    path = ds_dir / META_SQLITE
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT payload FROM summary WHERE id = 1").fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        logger.warning("读取 datastore summary 失败: %s", e)
    return {}


# ============================================================
# datastore 读取
# ============================================================

def _meta_json(meta: Dict[str, Any], key: str, default: Any) -> Any:
    """读 meta 表里以 JSON 文本存的配置项（缺失/损坏时返回默认值）"""
    raw = meta.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _apply_json_columns(rows: List[Dict[str, Any]], columns: Any) -> None:
    """把以 JSON 文本存储的列还原成对象（保证与 JSON 路径逐字段等价）"""
    for col in columns or []:
        for row in rows:
            value = row.get(col)
            if isinstance(value, str):
                try:
                    row[col] = json.loads(value)
                except Exception:
                    row[col] = None


def _prune_absent_keys(
    rows: List[Dict[str, Any]],
    table: Any,
    absent_columns: Any,
    nullable_columns: Any,
    presence_column: str,
) -> None:
    """
    还原「键不存在」与「键值为 null」的区别

    Parquet 是列式的，缺键的行读出来一律是 None，而源 JSON 里两者的语义不同。
    构建脚本把可变列分成两类（见 scripts/build_datastore.py）：

    - absent_columns：源记录里该键要么存在且非空、要么根本没有 → 值为 None 即代表"没这个键"，
      直接用 Arrow 的空值位图取行号批量删键（绝大多数情况走这条，无逐行 JSON 解码开销）。
    - nullable_columns：源记录里存在"键在、值就是 null"的情况 → 只能读逐行存在性列表精确判断。
    """
    import numpy as np

    for col in absent_columns or []:
        try:
            nulls = table.column(col).is_null().to_numpy(zero_copy_only=False)
        except Exception as e:
            logger.warning("读取列空值位图失败（%s 列将保留 null 键）: %s", col, e)
            continue
        for idx in np.flatnonzero(nulls).tolist():
            rows[idx].pop(col, None)

    if not nullable_columns:
        return
    for row in rows:
        raw = row.pop(presence_column, None)
        try:
            present = set(json.loads(raw)) if raw else set()
        except Exception:
            present = set()
        for col in nullable_columns:
            if col not in present:
                row.pop(col, None)


def _apply_vector_columns(
    rows: List[Dict[str, Any]], ds_dir: Path, vector_files: Dict[str, Any]
) -> int:
    """
    把独立向量文件里的向量按行号回填到检测行

    向量文件由 meta 表的 `vector_files` 描述：{列名: [行号文件, 向量文件]}。
    缺文件/矩阵形状对不上时该列整体留空（None），不影响其余字段。
    """
    if not vector_files:
        return 0
    try:
        import numpy as np
    except Exception:  # numpy 不可用 → 向量留空，其余字段照常
        logger.warning("numpy 不可用，datastore 向量字段留空")
        return 0

    filled = 0
    for col, files in vector_files.items():
        for row in rows:
            row[col] = None
        try:
            row_index = np.load(ds_dir / files[0])
            matrix = np.load(ds_dir / files[1])
        except Exception as e:
            logger.warning("读取向量文件失败（%s 列留空）: %s", col, e)
            continue
        if len(row_index) != len(matrix):
            logger.warning("向量文件行数不一致（%s 列留空）: %d vs %d", col, len(row_index), len(matrix))
            continue
        for i, det_idx in enumerate(row_index.tolist()):
            if 0 <= det_idx < len(rows):
                rows[det_idx][col] = matrix[i].tolist()
                filled += 1
    return filled


def _read_datastore(ds_dir: Path) -> Optional[Dict[str, Any]]:
    """
    从 Parquet + SQLite 还原出与 JSON 等价的字典

    schema 由 meta 表自描述（哪些列是 JSON 文本、哪些列抽到了独立向量文件、
    哪一列承载 det_to_track_map），因此构建脚本新增字段时读取层无需同步改动。

    失败（pyarrow 缺失 / 文件损坏等）返回 None，由调用方回退 JSON。
    """
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        logger.warning("pyarrow 不可用（%s），回退 JSON 直读", e)
        return None

    try:
        meta = _read_meta(ds_dir)
        det_table = pq.read_table(ds_dir / DETECTIONS_PARQUET)
        track_table = pq.read_table(ds_dir / TRACKS_PARQUET)
        det_rows = det_table.to_pylist()
        track_rows = track_table.to_pylist()
        summary = _read_summary(ds_dir)

        _apply_json_columns(det_rows, _meta_json(meta, "detection_json_columns", []))
        _apply_json_columns(track_rows, _meta_json(meta, "track_json_columns", []))
        vector_count = _apply_vector_columns(
            det_rows, ds_dir, _meta_json(meta, "vector_files", {})
        )
        presence_column = meta.get("presence_column") or "_present_keys"
        _prune_absent_keys(
            det_rows, det_table,
            _meta_json(meta, "detection_absent_when_null_columns", []),
            _meta_json(meta, "detection_nullable_columns", []),
            presence_column,
        )
        _prune_absent_keys(
            track_rows, track_table,
            _meta_json(meta, "track_absent_when_null_columns", []),
            _meta_json(meta, "track_nullable_columns", []),
            presence_column,
        )

        # det_to_track_map 与检测一一对应，构建时存成检测表的一列，这里还原成映射
        map_column = meta.get("map_column") or "track_id"
        det_to_track = {
            row["target_id"]: row[map_column]
            for row in det_rows
            if row.get("target_id") and row.get(map_column)
        }
        # 该列是 datastore 的内部列，JSON 的 detection 里并没有，还原时去掉
        for row in det_rows:
            row.pop(map_column, None)

        logger.info(
            "datastore 加载完成 | 检测=%d | 轨迹=%d | 向量=%d | 目录=%s",
            len(det_rows), len(track_rows), vector_count, ds_dir,
        )
        return {
            "detections": det_rows,
            "tracks": track_rows,
            "summary": summary,
            "det_to_track_map": det_to_track,
        }
    except Exception as e:
        logger.warning("datastore 读取失败，回退 JSON 直读: %s", e)
        return None


# ============================================================
# JSON 回退读取（**全仓唯一** 解析 cityflow_results.json 的地方）
# ============================================================

def _read_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """直读 JSON（datastore 不可用时的回退路径），失败返回 None"""
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("加载 %s 失败: %s", json_path.name, e)
        return None


def read_json_file(json_path: str) -> Optional[Dict[str, Any]]:
    """
    直读源 JSON 文件（**仅供离线构建/校验工具使用**）

    在线读取请一律调用 `load_results()`，那条路径才会优先走 datastore。
    """
    return _read_json(Path(json_path))


# ============================================================
# 对外统一接口
# ============================================================

def load_results(
    results_path: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    加载完整的 CityFlow 结果（detections / tracks / summary / det_to_track_map）

    Args:
        results_path: 结果 JSON 路径；缺省用 output/cityflow_results.json。
                      非默认路径时只做 JSON 直读（其同级 datastore 若存在也会优先用）。
        use_cache: 是否复用进程内缓存（默认复用；测试需要热读对比时可关）

    Returns:
        与 json.load 结构等价的字典；数据完全不可用时返回 None。
        **返回值跨调用方共享，请只读。**
    """
    json_path, ds_dir = resolve_paths(results_path)
    key = _cache_key(json_path)

    with _LOCK:
        stamp_now = _data_stamp(json_path, ds_dir)
        if use_cache:
            cached = _RESULTS_CACHE.get(key)
            if cached is not None:
                data, stamp = cached
                # 沿用改造前的 mtime 失效语义：源文件（JSON 或 datastore 元数据）一变就重载
                if stamp == stamp_now:
                    return data

        data = None
        if datastore_available(ds_dir):
            data = _read_datastore(ds_dir)
            if data is None:
                logger.warning("datastore 不可用，回退 JSON 直读: %s", json_path)
        if data is None:
            data = _read_json(json_path)

        if use_cache:
            _RESULTS_CACHE[key] = (data, stamp_now)
        return data


def _data_stamp(json_path: Path, ds_dir: Path) -> Tuple[float, float]:
    """
    数据指纹：datastore 元数据 mtime 与 JSON mtime

    两者任一变化都会让缓存失效——包括「datastore 被删掉/改名后回退 JSON」这种切换。
    """
    return (_mtime(ds_dir / META_SQLITE), _mtime(json_path))


def _mtime(path: Path) -> float:
    """文件修改时间；不存在返回 -2.0（与任何真实 mtime 都不同）"""
    try:
        return path.stat().st_mtime
    except OSError:
        return -2.0


def load_detections(
    results_path: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """只取检测列表（等价于 load_results()["detections"]），数据不可用返回 None"""
    data = load_results(results_path, use_cache=use_cache)
    if data is None:
        return None
    return data.get("detections") or []


def get_summary(results_path: Optional[str] = None) -> Dict[str, Any]:
    """
    只取 summary 字段

    datastore 可用时直接读 SQLite，**不加载任何大文件**；
    回退 JSON 时才顺带解析（与改造前 frontend/home.py 的代价一致）。
    """
    json_path, ds_dir = resolve_paths(results_path)
    if datastore_available(ds_dir):
        summary = _read_summary(ds_dir)
        if summary:
            return summary
    data = load_results(results_path)
    if data is None:
        return {}
    return data.get("summary") or {}


def get_stats(results_path: Optional[str] = None) -> Dict[str, Any]:
    """
    轻量统计（检测数 / 轨迹数 / 摄像头数 / 数据来源）

    datastore 可用时全部来自 SQLite 元数据（毫秒级，不解析大 JSON）；
    回退 JSON 时逐条统计，数值与改造前 frontend/home.py 一致。
    """
    json_path, ds_dir = resolve_paths(results_path)
    if datastore_available(ds_dir):
        counts = _read_counts(ds_dir)
        if counts:
            return {
                "detections": counts.get("detections", 0),
                "tracks": counts.get("tracks", 0),
                "cameras": counts.get("cameras", 0),
                "source": "datastore",
            }
    data = load_results(results_path)
    if data is None:
        return {"detections": 0, "tracks": 0, "cameras": 0, "source": "none"}
    detections = data.get("detections") or []
    camera_ids = {d.get("camera_id", "") for d in detections if d.get("camera_id")}
    # 轨迹数按 det_to_track_map 的值域（= builder 生成 tracklet 的口径，实测 926）。
    # 缺陷 E3：tracks[] 混装了 CF3_TRACK_*/BL_TRACK_*/CF2S_TRACK_* 三个来源，
    # 其中 1700 条没有任何对应检测，len(tracks[]) 会虚报成 2070。
    tracklet_count = len({t for t in (data.get("det_to_track_map") or {}).values() if t})
    return {
        "detections": len(detections),
        "tracks": tracklet_count,
        "cameras": len(camera_ids),
        "source": "json",
    }


def reset_cache() -> None:
    """清空进程内缓存（测试与数据重建后使用）"""
    with _LOCK:
        _RESULTS_CACHE.clear()
        _META_CACHE.clear()
