#!/bin/bash
cd "$(dirname "$0")/.."
streamlit run frontend/app.py --server.port 8501
