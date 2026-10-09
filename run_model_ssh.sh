#!/usr/bin/env bash
# SSH variant: use the desktop's running game and maintain its focus.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Share checkpoint validation, desktop environment, and capture defaults.
exec "$script_dir/run_model.sh" \
  --no-launch \
  --keep-focused \
  "$@"
