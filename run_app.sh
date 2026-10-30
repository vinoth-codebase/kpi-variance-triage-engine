#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating local Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[INFO] Verifying Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[INFO] Initializing SQLite Star-Schema warehouse if needed..."
python3 -m src.data_generator

echo "[INFO] Launching Automated KPI Variance & Root-Cause Triage Engine..."
exec streamlit run app.py --server.headless true

