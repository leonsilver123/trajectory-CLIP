"""
src.storage - 数据存储层

对外只暴露 src.storage.datastore 的统一读写接口，供 api/、src/trajectory
等消费方共用；离线构建脚本 scripts/build_datastore.py 也依赖这里的布局常量。
"""

from src.storage.datastore import (
    DATASTORE_DIRNAME,
    DETECTIONS_PARQUET,
    IMAGE_VECTORS_NPY,
    META_SQLITE,
    SCHEMA_VERSION,
    TEXT_VECTORS_NPY,
    TRACKS_PARQUET,
    VECTOR_INDEX_NPY,
    data_source,
    datastore_available,
    datastore_dir,
    get_stats,
    get_summary,
    has_data,
    load_detections,
    load_results,
    read_json_file,
    reset_cache,
    resolve_paths,
    results_json_path,
)

__all__ = [
    "DATASTORE_DIRNAME",
    "DETECTIONS_PARQUET",
    "IMAGE_VECTORS_NPY",
    "META_SQLITE",
    "SCHEMA_VERSION",
    "TEXT_VECTORS_NPY",
    "TRACKS_PARQUET",
    "VECTOR_INDEX_NPY",
    "data_source",
    "datastore_available",
    "datastore_dir",
    "get_stats",
    "get_summary",
    "has_data",
    "load_detections",
    "load_results",
    "read_json_file",
    "reset_cache",
    "resolve_paths",
    "results_json_path",
]
