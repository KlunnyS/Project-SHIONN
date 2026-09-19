#!/usr/bin/env -S fish --no-config

set -l script_dir (path dirname (status filename))
cd "$script_dir"; or exit 1

set -l runtime_dir
if set -q XDG_RUNTIME_DIR
    set runtime_dir "$XDG_RUNTIME_DIR"
else
    set runtime_dir "/run/user/"(id -u)
end

set -l wayland_display
if set -q WAYLAND_DISPLAY
    set wayland_display "$WAYLAND_DISPLAY"
else
    set wayland_display wayland-1
end

set -l dbus_address
if set -q DBUS_SESSION_BUS_ADDRESS
    set dbus_address "$DBUS_SESSION_BUS_ADDRESS"
else
    set dbus_address "unix:path=$runtime_dir/bus"
end

exec env \
    XDG_RUNTIME_DIR="$runtime_dir" \
    WAYLAND_DISPLAY="$wayland_display" \
    DBUS_SESSION_BUS_ADDRESS="$dbus_address" \
    .venv/bin/python run_imitation.py \
    --checkpoint /mnt/extra/Project-SHIONN/checkpoints_450_v3/best.pt \
    --device auto \
    --output DP-1 \
    --map evaluation1 \
    --max-seconds 60 \
    --record-video \
    --verbose \
    $argv
