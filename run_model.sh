#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
wayland_display="${WAYLAND_DISPLAY:-wayland-1}"
dbus_address="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${runtime_dir}/bus}"

exec env \
  XDG_RUNTIME_DIR="${runtime_dir}" \
  WAYLAND_DISPLAY="${wayland_display}" \
  DBUS_SESSION_BUS_ADDRESS="${dbus_address}" \
  .venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints_v3/best.pt \
  --device auto \
  --output DP-1 \
  --map dataset_test1 \
  --max-seconds 60 \
  --record-video \
  --verbose \
  "$@"
