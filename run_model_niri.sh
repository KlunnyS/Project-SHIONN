#!/usr/bin/env bash
# Run a SHIONN checkpoint in an existing Niri desktop session.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ -z "${NIRI_SOCKET:-}" ]] || ! command -v niri >/dev/null 2>&1; then
  printf 'Start this from a Niri desktop session (NIRI_SOCKET and niri are required).\n' >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  printf 'Python environment missing; set up the project environment first.\n' >&2
  exit 2
fi

exec .venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --record-video \
  --keep-focused \
  --verbose \
  "$@"
