# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language traffic risk-perception system ("交通风险感知子系统"): **text-driven target retrieval and cross-camera spatio-temporal backtracking** for road surveillance video. Given a natural-language description ("黑色轿车", "蓝色背包的男人") or a license plate, it returns candidate target images; once the user confirms a target, it reconstructs the sequence of cameras the target passed through, key timestamps, observed segments (real, within a camera's view) and inferred segments (probabilistic, between cameras), plus confidence scores.

The system explicitly models discontinuous camera coverage: it outputs a *discrete observation chain* (observed segments vs. inferred segments) rather than a fake continuous trajectory.

## Architecture — two parallel implementations

This is the single most important thing to understand. There are **two code paths** that are only loosely connected:

1. **Designed modular pipeline (`src/`)** — a clean, layered architecture (perception → tracking → stitching → retrieval → backtrack → output) with rich dataclasses in `src/common/data_models.py` (the shared contract between all modules). Many `src/` modules and `scripts/` entry points (`build_index.py`, etc.) are **stubs with `TODO` comments** — the full offline-structured pipeline is only partially implemented. The `tests/` suite exercises these pure-logic modules (query parsing, attribute filtering, reranking, vector recall) with **no model weights required**.

2. **Actually-deployed serving path (`api/`)** — the running FastAPI server reads directly from precomputed JSON + FAISS indices in `output/`, and does not go through the `src/` pipeline. `api/routes/search.py` is the canonical example: it loads `output/cityflow_results.json`, runs attribute pre-filtering, then a Chinese-CLIP + FAISS vector re-rank, all inline.

The docs (`PROJECT_FINAL.md`, `RESUME_PROJECT.md`, `docs/USER_GUIDE.md`, `docs/ALGORITHM_REPORT.md`) describe the *designed* system and its target metrics; the working code in `api/`/`frontend/` is a slimmer subset. When the two disagree, trust the code, not the docs. Documentation references models (CN-CLIP ViT-L/14 768-dim, BLIP, OSNet) that the actual `api/routes/search.py` does not use — it uses Chinese-CLIP **ViT-B-16** (512-dim) with a FAISS `IndexFlatIP`.

## Package layout

- `src/` — core algorithm package, imported as `src.*`. Subpackages: `common` (config, data_models, logger, utils), `perception` (detector, tracker, attribute, plate_ocr, feature_extractor, quality), `tracking`, `retrieval` (query_parser, attribute_filter, vector_recall, reranker), `stitching` (candidate_edge, scoring, observation_chain), `backtrack` (anchor_backtrack, chain_expander), `data_governance` (camera_manager, road_topology, video_stream), `output`.
- `api/` — FastAPI backend. `api/main.py` creates the app and mounts routes under `/api/v1/{search,confirm,backtrack,dashboard}`; `output/` is served as static files at `/static`.
- `frontend/` — Streamlit UI. `frontend/app.py` is the entry point with a hand-rolled sidebar navigation; pages live in `frontend/pages/`. Falls back to demo/mock data (`frontend/mock_data.py`, hardcoded values in `frontend/home.py`) when the backend is unreachable.
- `scripts/` — CLI entry points and one-off preprocessing/verification utilities. Several are exploratory (`*_fix_*`, `*_test_*`).
- `configs/` — YAML config (`default.yaml`) plus camera metadata.
- `tests/` — pytest suite for the `src/` modules.

## Data flow (designed, per `src/common/data_models.py`)

`TargetInstance` (one detection) → `Tracklet` (single-camera track) → `CrossCameraEdge` (candidate link between tracklets) → `ObservationChain` / `TrajectoryResult` (final output with observation nodes, segments, inference segments, candidate paths, evidence). `ParsedQuery` and `RetrievalCandidate` cover the retrieval side.

## Commands

Run from the project root (`H:\trajectory-CLIP`). Scripts and modules insert the project root into `sys.path`, so imports like `src.common.config`, `api.main`, `frontend.*` resolve against the root directory.

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run backend API (default :8000, Swagger at /docs, health at /health)
python scripts/run_server.py                 # add --port, --reload, --config
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run frontend (default :8501)
streamlit run frontend/app.py --server.port 8501

# Run tests (no model weights required; pure-logic tests)
pytest
pytest tests/test_retrieval.py
pytest tests/test_retrieval.py -k test_parse_plate

# GPU environment diagnostics
python scripts/check_gpu_env.py --verbose

# Quick demo: extract frames from AICity22, run YOLOv8, write output/results.json
python scripts/quick_demo.py

# Docker (backend :8000, frontend :8501, qdrant :6333)
docker compose up -d
```

## Configuration

`configs/default.yaml` is loaded by `src/common/config.py` (`Config` class, dot-path access via `config.get("system.device")`). A global singleton is exposed through `get_config()` / `reset_config()`. Environment variables override values using the `TRAFFIC__SECTION__KEY` convention (e.g. `TRAFFIC__SYSTEM__DEVICE=cpu` → `system.device`).

Key artifacts referenced by the serving path:
- `output/cityflow_results.json` — detections + tracks + det-to-track map (input to `api/routes/search.py`).
- `output/clip_vectors.faiss`, `output/track_clip_vectors.faiss` — precomputed detection/track-level CLIP indices.
- Model weights: `yolov8x.pt` (repo root), `models/clip_cn_vit-b-16.pt`.
- Camera metadata: `configs/cityflow_camera_metadata.yaml` (loaded by `api/routes/search.py` for ID→name mapping).

## Conventions

- Docstrings and comments are in **Chinese**; identifiers, module names, and log messages are in English.
- Config-driven thresholds/weights live in `configs/default.yaml` (stitching weights differ by target type: vehicle vs. pedestrian); don't hardcode them.
- The server path uses module-level lazy-loaded caches (CLIP model, FAISS indices) to avoid startup cost — follow that pattern when adding model-backed code.
- `api/routes/` also contains a `search_backup.py` (an older route implementation); the live route is `search.py`.
