#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python_path=".venv-gpu/bin/python"
if [[ ! -x "$python_path" ]]; then
    echo "GPU environment not found. Run ./setup_gpu.sh first." >&2
    exit 1
fi

"$python_path" scripts/check_gpu.py
exec "$python_path" app.py
