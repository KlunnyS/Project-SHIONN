#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode=auto
if [[ $# -gt 0 ]]; then mode="$1"; fi
if [[ $# -gt 1 ]]; then
    echo "Usage: ./setup_gpu.sh [auto|legacy|modern]" >&2
    exit 2
fi

if [[ "$mode" == auto ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; specify legacy or modern." >&2
        exit 1
    fi
    capability="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
    if [[ -z "$capability" ]]; then
        echo "Could not read GPU capability; specify legacy or modern." >&2
        exit 1
    fi
    mode="$(awk -F. '{if ($1 < 7 || ($1 == 7 && $2 < 5)) print "legacy"; else print "modern"}' <<< "$capability")"
fi

if [[ "$mode" != legacy && "$mode" != modern ]]; then
    echo "Usage: ./setup_gpu.sh [auto|legacy|modern]" >&2
    exit 2
fi
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is required." >&2
    exit 1
fi

python3.11 -m venv .venv-gpu
python_path=".venv-gpu/bin/python"
"$python_path" -m pip install --upgrade pip

if [[ "$mode" == legacy ]]; then
    "$python_path" -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
else
    "$python_path" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
fi

"$python_path" -m pip install -r ../requirements.txt
"$python_path" -m pip install -r requirements-gpu.txt
echo "GPU environment ready. Start it with ./run_gpu.sh"
