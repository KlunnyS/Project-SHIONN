#!/usr/bin/env bash
# Local Portal 2 model launcher. Usage: ./run_model.sh --checkpoint PATH [runner options]
set -euo pipefail

# Resolve paths from the repository, even when launched from another directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Validate the selected model before the runner can launch or control the game.
args=("$@")
checkpoint=""
show_help=false
for ((index = 0; index < ${#args[@]}; index++)); do
  case "${args[index]}" in
    --checkpoint)
      if ((index + 1 >= ${#args[@]})) || [[ "${args[index + 1]}" == --* ]]; then
        printf 'Missing path after --checkpoint\n' >&2
        exit 2
      fi
      checkpoint="${args[index + 1]}"
      ((index += 1))
      ;;
    --checkpoint=*) checkpoint="${args[index]#--checkpoint=}" ;;
    --help|-h) show_help=true ;;
  esac
done

if [[ "$show_help" != true ]]; then
  if [[ -z "$checkpoint" ]]; then
    printf 'Choose a model with --checkpoint PATH (see docs/TRAINING_RUNBOOK.md).\n' >&2
    exit 2
  fi
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
    exit 2
  fi
fi

runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
# SSH sessions may lack these desktop variables; keep existing values when set.
export XDG_RUNTIME_DIR="$runtime_dir"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-1}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${runtime_dir}/bus}"

# Extra arguments are passed last so they can override launcher defaults.
exec .venv/bin/python run_imitation.py \
  --map dataset_test1 \
  --record-video \
  --verbose \
  "$@"
