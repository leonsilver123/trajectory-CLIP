"""
tests/test_frontend_retrieval.py - 前端检索收敛测试

测试目标: 确保检索只在后端一处实现，前端不再自行加载 CLIP/BLIP 打分
覆盖:
  - frontend/ 下不存在本地 CLIP/BLIP 模型加载点
  - frontend.utils 中已删除本地检索函数，且保留轨迹展示所需的读取函数
  - frontend.pages.search._do_search 只调用后端接口，后端不可用时如实提示
"""

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

# 本地检索管线（前端自有的 OpenCLIP/BLIP 那套）已整体删除，以下名字不应再出现
_REMOVED_FUNCS = [
    "build_search_results_from_cityflow",
    "_get_clip_extractor",
    "_get_blip_extractor",
    "_blip_score_image_text",
    "_compute_attr_consistency",
    "_batch_cosine_similarity",
    "_cosine_similarity",
    "_check_cuda",
    "_parse_cityflow_query",
    "_cityflow_det_to_candidate",
]

# 前端不应再出现的模型加载标识（注释里提到名字不算，这里匹配代码级标识）
_MODEL_MARKERS = [
    "open_clip",
    "BlipForImageTextRetrieval",
    "BlipProcessor",
    "Salesforce/blip",
    "src.perception.feature_extractor",
    "CN-CLIP-ViT-L-14",
]


def _frontend_py_files() -> list[Path]:
    """frontend/ 下全部 Python 源文件（不含 __pycache__）"""
    return [p for p in _FRONTEND_DIR.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_local_clip_blip_model_loading_in_source():
    """frontend/ 源码中不存在 CLIP/BLIP 模型加载标识"""
    offenders = []
    for path in _frontend_py_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            # 跳过注释行：本任务在注释中说明了「已删除哪些实现」，不构成调用点
            if line.lstrip().startswith("#"):
                continue
            for marker in _MODEL_MARKERS:
                if marker in line:
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{lineno}: {marker}")
    assert offenders == [], f"frontend/ 下仍有 CLIP/BLIP 模型标识: {offenders}"


def test_removed_helpers_gone_from_utils():
    """frontend.utils 中本地检索函数已删除"""
    frontend_utils = pytest.importorskip("frontend.utils")
    still_there = [n for n in _REMOVED_FUNCS if hasattr(frontend_utils, n)]
    assert still_there == [], f"这些本地检索函数应已删除: {still_there}"


def test_trajectory_helpers_still_available():
    """轨迹展示所需的读取函数仍然保留（本任务只收敛检索，不动它们）"""
    frontend_utils = pytest.importorskip("frontend.utils")
    for name in ("load_cityflow_results", "has_cityflow_results", "resolve_image_path"):
        assert hasattr(frontend_utils, name), f"{name} 不应被删除"


def test_backend_api_helpers_intact():
    """后端检索接口封装仍然存在（前端检索的唯一入口）"""
    frontend_utils = pytest.importorskip("frontend.utils")
    for name in ("api_search", "api_plate_search", "api_confirm", "api_backtrack"):
        assert callable(getattr(frontend_utils, name, None)), f"{name} 缺失"


def _load_do_search_ast() -> ast.FunctionDef:
    """解析 frontend/pages/search.py，取出 _do_search 的 AST 节点"""
    src = (_FRONTEND_DIR / "pages" / "search.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_do_search":
            return node
    raise AssertionError("frontend/pages/search.py 中找不到 _do_search")


def _called_names(node: ast.AST) -> set[str]:
    """收集节点内所有被调用者的名字（含属性调用的最后一段）"""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_do_search_only_uses_backend_api():
    """_do_search 只通过 api_search 取结果，不再调用任何本地检索实现"""
    func = _load_do_search_ast()
    called = _called_names(func)
    assert "api_search" in called, "_do_search 必须调用后端检索接口 api_search()"
    leaked = called & set(_REMOVED_FUNCS)
    assert leaked == set(), f"_do_search 仍在调用本地检索实现: {leaked}"


def test_do_search_reports_backend_failure_honestly():
    """后端不可用时 _do_search 如实提示，而不是静默换一套算法"""
    func = _load_do_search_ast()
    called = _called_names(func)
    assert "_render_warning_block" in called, "后端不可用时应渲染明确的失败提示"
    # 降级策略 = 「少一步」：失败分支直接返回 None，由调用方展示提示
    src = ast.unparse(func)
    assert "return None" in src
