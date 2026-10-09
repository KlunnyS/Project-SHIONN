#!/usr/bin/env bash
# Open Hammer++ from the workstation's Portal 2 bin directory.
set -euo pipefail

# Override this path for a different Steam installation.
hammer_dir="${SHIONN_HAMMER_DIR:-/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/bin}"
if [[ ! -f "$hammer_dir/hammerplusplus.exe" ]]; then
    printf 'Hammer++ not found in %s\nSet SHIONN_HAMMER_DIR to its bin directory.\n' "$hammer_dir" >&2
    exit 1
fi
if ! command -v wine >/dev/null 2>&1; then
    printf 'wine is required to launch Hammer++.\n' >&2
    exit 1
fi

cd "$hammer_dir"
exec wine ./hammerplusplus.exe
