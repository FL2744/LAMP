#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import torch, numpy' >/dev/null 2>&1; then
    .venv/bin/python -m pip install -r requirements.txt
fi
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python toy_demo.py "$@"
