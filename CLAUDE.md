# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language traffic risk-perception system ("交通风险感知子系统"): **text-driven target retrieval and cross-camera spatio-temporal backtracking** for road surveillance video. Given a natural-language description ("黑色轿车", "蓝色背包的男人") or a license plate, it returns candidate target images; once the user confirms a target, it reconstructs the sequence of cameras the target passed through, key timestamps, observed segments (real, within a camera's view) and inferred segments (probabilistic, between cameras), plus confidence scores.

The system explicitly models discontinuous camera coverage: it outputs a *discrete observation chain* (observed segments vs. inferred segments) rather than a fake continuous trajectory.

## Architecture — serving path vs. offline library

The single most important thing to understand: `src/` is a **layered algorithm library** (perception → tracking → stitching → retrieval → backtrack), but only a **subset of it is actually wired into the running service**. Do not assume an importable module is a live one.

**On the serving path** — the real dependency closure is 14 modules:

```
api/main.py            → src.common.{config,logger}
api/routes/search.py   → src.common.{logger,session_store} + src.storage.datastore
api/routes/confirm.py  → src.common.session_store
api/routes/dashboard.py→ src.common.{config,logger} + src.storage.datastore
api/routes/backtrack.py→ src.common.{logger,session_store} + src.trajectory.builder

src.trajectory.builder
  ├─→ src.common.{config,data_models,ids,logger,utils}
  ├─→ src.storage.datastore
  ├─→ src.stitching.{candidate_edge,observation_chain,scoring}
  └─→ src.data_governance.{camera_manager,road_topology}

```

**Implemented but NOT on the serving path** — `src/perception/*` (6), `src/retrieval/*` (4), `src/tracking/*` (2), `src/output/trajectory_output.py`. These have real logic (there are **no `TODO`/stub modules** in `src/`), but only tests and `scripts/pipeline_validation.py` / `scripts/*_feature_pipeline.py` import them.

Three unreachable modules were **deleted on 2026-09-21** (`src/backtrack/`, `src/data_governance/video_stream.py`, and the never-imported DI skeleton `api/dependencies.py`). If you find a reference to them, it is stale.

Two notable consequences to keep in mind when editing:

- `api/routes/search.py` implements its own query parsing (`_extract_query_features`) and attribute filtering (`_attribute_filter`) rather than using `src/retrieval/query_parser.py` / `attribute_filter.py`. The two implementations differ (the route's own comment records a "皮卡 vs 卡车" keyword bug that the `src/` version never had).
- The deployed retrieval model is Chinese-CLIP **ViT-B-16** (512-dim) via the `cn_clip` package, **not** the CN-CLIP ViT-L/14 (768-dim) that `configs/default.yaml` and most docs still name. Likewise **BLIP and OSNet appear nowhere in the code** — `docs/` describes them as part of the *designed* system (the paper's design lineage), not the deployed implementation. Trust the code, not the docs.

**Data access is unified.** All online reads go through `src/storage/datastore.py`; nothing else opens `output/cityflow_results.json` (a 75MB file with 68,349 detections). `load_results()` returns a dict structurally equivalent to that JSON (`detections` / `tracks` / `summary` / `det_to_track_map`) but reads Parquet + SQLite from `output/datastore/`, falling back to JSON parse only if the datastore is missing, corrupt, or `schema_version` != 1. This convergence is enforced by `tests/test_datastore.py::test_only_datastore_opens_cityflow_json`, which greps every `.py` under `api/` and `src/` (the Streamlit frontend it used to cover was removed).

## Package layout

- `src/` — algorithm package, imported as `src.*`. Subpackages: `common` (config, data_models, ids, logger, session_store, utils), `perception`, `tracking`, `retrieval`, `stitching` (candidate_edge, scoring, observation_chain), `backtrack`, `data_governance`, `storage` (datastore), `trajectory` (builder), `reporting` (研判报告 PDF/Excel 渲染), `output`.
- `api/` — FastAPI backend. `api/main.py` mounts five routers under `/api/v1/{search,confirm,backtrack,dashboard,report}` plus inline `/health`. If `webapp/dist/` exists it is also served at `/` with an SPA fallback route; that mount must stay **after** all API routes are registered (Starlette matches in registration order).
  - **`/static` is a whitelist, not the whole `output/`.** `api/main.py::_STATIC_IMAGE_DIRS` lists the image directories that may be served; anything else under `output/` (notably `cityflow_results.json`, `datastore/*.parquet`, `meta.sqlite`) is **404**. Adding a new image directory requires registering it there. `tests/test_static_exposure.py` guards both directions (data files stay 404, whitelisted images stay 200).
  - `api/routes/report.py` generates 研判报告 from a session via `src/reporting/report_builder.py`. It reads **only the session** (which already carries `backtrack_result`), so `src/` never depends on `api/`.
- `webapp/` — the frontend (React 18 + TypeScript + Vite + antd; pages: Search / Backtrack / Trajectory / Dashboard, axios client in `src/api/`).
  - `TrajectoryPlayer.tsx` is the 轨迹还原 playback (PLAN4). It is deliberately **non-geographic**: the dataset has only **4 distinct GPS coordinate sets for 46 cameras** (all cameras in a scene share the scene-centre coordinate), so any map animation would show the vehicle teleporting between 4 points. The player instead has three linked panels — camera lanes, real in-frame position from each detection's bbox, and a live parameter panel. **It must not interpolate across cameras**: the road between two cameras was never filmed, so a smooth glide would be fabrication. Inferred spans render a placeholder, never a vehicle position. Speed is reported in "画面比例/秒 (estimated)" — never km/h, because pixel-to-metre needs camera calibration the project does not have. Built to `webapp/dist/` and served by the backend on the same origin. Its `README.md` is still the unedited Vite template. **This is the only frontend** — the earlier Streamlit UI (`frontend/`) was removed on 2026-09-21.
- `scripts/` — CLI entry points, offline data pipelines, one-off repair utilities, and the evaluation suite (see below).
- `configs/` — `default.yaml` plus camera metadata. `configs/cityflow_camera_metadata.yaml` is the authoritative camera list (46 cameras with GPS); `configs/camera_metadata.yaml` has only a `scenes` section and no `cameras` section.
- `tests/` — pytest suite.

**Not in the repo**: `third_party/fast-reid` (395 files of vendored code — `scripts/extract_reid.py` clones it on demand and its own error message says so), `output/`, `models/`, `cityflow/`, `data/`.

## Data flow (designed, per `src/common/data_models.py`)

`TargetInstance` (one detection) → `Tracklet` (single-camera track) → `CrossCameraEdge` (candidate link between tracklets) → `TrajectoryResult` (final output: observation nodes, observed segments, inference segments, candidate paths, evidence). `ParsedQuery` / `RetrievalCandidate` / `ObservationChain` cover the retrieval side and are **not** on the serving path.

`TrajectoryBuilder.build()` is the single entry point for backtracking and picks a path by `mode`:

- `strong`: the dataset carries a real `vehicle_id` per detection, so aggregation is exact.
- `stitch`: the probabilistic path, and the only online consumer of `src/stitching`. It is what `scripts/eval_chain_idf1.py` must be run with (`mode="stitch"`), because `auto` resolves to `strong` on this dataset and makes IDF1 trivially 1.0.

Every segment carries a `basis` field (`strong_identity` vs `probabilistic_inference`) so the API can label evidence honestly.

## Commands

Run from the project root (`H:\trajectory-CLIP`). Scripts and modules insert the project root into `sys.path`, so imports like `src.common.config` and `api.main` resolve against the root directory.

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run backend API (default 127.0.0.1:8000 per configs/default.yaml; Swagger at /docs, health at /health)
python scripts/run_server.py                 # add --host, --port, --reload, --config
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Build the React frontend, then serve it from the backend at /
cd webapp && npm run build        # tsc -b && vite build → webapp/dist
npm run dev                       # dev server, proxies /api and /static to VITE_BACKEND_ORIGIN

# Run tests (pure-logic tests need no model weights)
pytest
pytest tests/test_retrieval.py
pytest tests/test_retrieval.py -k test_parse_plate

# GPU environment diagnostics
python scripts/check_gpu_env.py --verbose

# Docker (backend :8000; the UI is served by the backend at http://<host>:8000/)
docker compose up -d
```

Evaluation scripts live in `scripts/eval_*.py` and **never hit HTTP** — they import the production functions directly:

- `eval_retrieval_hit_rate.py` — retrieval hit-rate@K and MRR over the deployed search path.
- `eval_chain_idf1.py` — per-target IDF1; pass `mode="stitch"` (see above).
- `eval_global_idf1.py` — union-find global clustering with official (motmetrics) IDF1.
- `eval_cross_camera.py` — Rank-1 / Rank-5 / mAP for cross-camera ReID, over either the CLIP or the ReID FAISS index.
- `eval_contamination.py`, `eval_method_suite.py`, `eval_attribute_consistency.py`.

The offline data pipelines have hard ordering, enforced by `sys.exit` guards inside the scripts: `extract_attributes.py` → `merge_attributes.py`; `extract_reid.py` → `build_reid_tracks.py`; `unify_clip_949.py` → `merge_clip_unified.py`. The `merge_*` scripts rewrite `output/cityflow_results.json` atomically and idempotently, after which `build_datastore.py` must be re-run.

## Configuration

`configs/default.yaml` is loaded by `src/common/config.py` (`Config` class, dot-path access via `config.get("system.device")`). A global singleton is exposed through `get_config()` / `reset_config()`. Environment variables override values using the `TRAFFIC__SECTION__KEY` convention (e.g. `TRAFFIC__SYSTEM__DEVICE=cpu` → `system.device`).

Key artifacts referenced by the serving path:
- `output/datastore/` — `detections.parquet`, `tracks.parquet`, `meta.sqlite` (holds `schema_version`, `counts`, and the `summary` payload), plus vector `.npy` files. This is the primary online data source.
- `output/cityflow_results.json` — the JSON fallback and the source the datastore is built from.
- `output/clip_vectors.faiss` — 68,349 × 512 `IndexFlatIP` over L2-normalized detection CLIP vectors. `output/track_clip_vectors.faiss` (370 × 512) is loaded by helper functions that have no callers.
- Model weights: `yolov8x.pt` (repo root), `models/clip_cn_vit-b-16.pt`, `models/veri_sbs_R50-ibn.pth` + `vehicleid_bot_R50-ibn.pth` + `veriwild_bot_R50-ibn.pth` (fast-reid ensemble).
- Camera metadata: `configs/cityflow_camera_metadata.yaml`.

Stitching weights are target-type dependent and live in `configs/default.yaml` under `stitching.weights`. The `vehicle` entry currently zeroes `reid` and `attribute` — not an oversight: they measured those soft scores as noise on this data and moved appearance/attribute to hard gates in `src/stitching/candidate_edge.py` (rules 7 and 8, gated by `stitching.min_appearance_score`). `reweight_missing_dimensions` redistributes weight away from dimensions with no evidence, keeping the total at 1.0 instead of adding a constant offset to every candidate.

## Conventions

- Docstrings and comments are in **Chinese**; identifiers, module names, and log messages are in English.
- Config-driven thresholds/weights live in `configs/default.yaml`; don't hardcode them. Fallback constants in `src/stitching/scoring.py` must stay numerically identical to the config.
- The serving path uses module-level lazy-loaded caches (CLIP model, FAISS indices) to avoid startup cost — `api/main.py`'s `lifespan` deliberately preloads nothing. Follow that pattern when adding model-backed code.
- Never fabricate data to fill a field. When a fact isn't in the dataset (e.g. camera online status), return `None` and let the UI show `--`. This rule is documented in-line in several places and has been the subject of repeated bug fixes.

## Gotchas

- **Anchor gitignore rules for root-level directories.** `output/` (unanchored) once also matched `src/output/`, hiding 481 lines of tracked-worthy code from git *and* from ripgrep (which skips ignored paths). Rules for root dirs must be written `/output/`. Use `git check-ignore -v <path>` before concluding a file doesn't exist.
- **`pytest.ini` sets `testpaths = tests`** — that keeps collection out of `.venv/` and `output/` (without it, collection alone takes 7+ minutes). The hand-written `test_*.py` scripts that used to sit in the repo root were removed on 2026-09-21.
- `pytest.ini`'s `addopts` disables the user-level `langsmith` plugin by name (`-p no:langsmith_plugin`). Do not "fix" the resulting problems with `PYTHONNOUSERSITE` — that changes `sys.path` and silently zeroes CLIP scores.
- `tests/test_api.py` and `tests/test_trajectory_builder.py` build a `TestClient` and do exercise real CLIP + FAISS; data-dependent tests elsewhere `skipif` when `output/` artifacts are absent.
- Several guards scan source text rather than executing it (`test_static_exposure.py`, `test_frontend_no_fabrication.py`, `test_datastore.py`). When you move or rename a file they scan, they fail loudly — that is the intent; update the path, don't silence the test.
