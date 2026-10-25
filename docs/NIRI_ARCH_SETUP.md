# Running SHIONN from Arch Linux with Niri

The runner now has basic Niri window discovery and focus support. This gives a
model-only run a Niri-aware launch path. It is not yet a validated replacement
for the workstation's Hyprland recovery workflow: current Niri IPC does not
report global cursor coordinates, and the Gamescope pointer-grab refresh used
for recovery has not been tested under Niri.

## Arch setup

Install or update Niri and the required Wayland capture tools with Arch's
package manager. Niri is available from the official `Extra` repository.

```bash
sudo pacman -Syu niri gamescope wf-recorder pipewire wireplumber \
  xdg-desktop-portal-gnome python python-pip git xautomation
```

The Niri screencasting docs require a running D-Bus and PipeWire session with
`xdg-desktop-portal-gnome`, or support for older `wlr-screencopy`; verify that
`wf-recorder -l` lists the laptop display. `xautomation` provides `xte` for
direct XWayland mouse activation. The Niri session must export `NIRI_SOCKET`;
do not synthesize it in a shell launched outside the desktop session.

Clone the repo, set up its virtual environment and Python dependencies, then
check the local requirements:

```bash
git clone <repository-url> Project-SHIONN
cd Project-SHIONN
./install_dependencies.sh
./check_dependencies.sh
```

For PyTorch on the laptop, install the build appropriate to its GPU from the
official PyTorch instructions if the default dependency install does not match
the device. CPU inference is available with `--device cpu`.

The model files are ignored by Git. Copy both `best.pt` **and its matching
`best.json`** from the workstation to the laptop; the loader verifies the JSON
configuration against the checkpoint. For example, put them in the same
relative run directory under `models/imitation/checkpoints/runs/`. Live
inference does not need the training caches or dataset recordings.

## Model-only run

Niri's fullscreen-game guidance recommends Gamescope with cursor grabbing on
multi-monitor setups. Keep the existing Pressure Vessel mount and netconsole
arguments, and wrap Steam's `%command%` with Gamescope rather than replacing
those project-specific arguments. Adapt the resolution and use the existing
mount paths if that laptop needs them:

```text
PRESSURE_VESSEL_FILESYSTEMS_RW="<existing-mounts>" gamescope -f \
  -w 1920 -h 1080 -W 1920 -H 1080 --force-grab-cursor --backend sdl \
  -- %command% -netconport 8020
```

Start Portal 2 through Steam/Gamescope in the Niri session, then launch from a
terminal in that same session:

```bash
./run_model_niri.sh \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 --device auto --max-seconds 60
```

The wrapper defaults to the current V5 candidate; passing another
`--checkpoint` overrides it. It preserves the Niri session environment rather
than guessing `WAYLAND_DISPLAY`. `--keep-focused` asks Niri to focus the Portal
2 or Gamescope window and restore focus after loss. Niri recommends
`--force-grab-cursor --backend sdl` for Gamescope on multi-monitor setups; that
does not replace SHIONN's future capture verification. Use `--record-video` and
`--log-actions` when reviewing a short run.

The Niri adapter reports focused window and active workspace. Cursor position
and cursor containment are stored as `null` because the supported Niri window
and workspace IPC responses do not expose the global cursor location. Do not
interpret null as proof that the cursor stayed inside the game.

## Recovery recording status

Do not use `run_recovery_session.py` with automated handoff under Niri yet. The
runner will stop early if recovery-handoff is requested. The Hyprland path
currently relies on compositor cursor coordinates and a live-tested workspace
transition to reacquire Gamescope's relative pointer. Niri has dynamic
workspaces with different behavior, so copying that transition would be unsafe.

Before enabling recovery on the laptop, add and validate an Niri-specific way
to monitor focus/cursor capture, reacquire Gamescope relative mouse, and prove
camera movement in both directions. First do this with a no-save test and no
long recovery session. Keep a physical emergency stop available.

## Troubleshooting

- `NIRI_SOCKET` missing: open a terminal from the logged-in Niri session.
- `niri msg --json focused-window` fails: check `niri --version` and IPC socket
  availability. The adapter tolerates extra JSON fields but requires current
  window IDs, app IDs, and titles.
- `wf-recorder` reports no display: confirm D-Bus, PipeWire, portal service,
  and Wayland capture support; inspect `wf-recorder -l`.
- Model stays still or camera does not turn: stop after a short test. A focused
  window is not proof of mouse capture. Check Gamescope/XWayland, the Steam
  launch options, and the SHIONN action log before recording demonstrations.

## References

- [Arch Linux Niri package](https://archlinux.org/packages/extra/x86_64/niri/)
- [Niri IPC and event stream](https://niri-wm.github.io/niri/IPC.html)
- [Niri screencasting requirements](https://niri-wm.github.io/niri/Screencasting.html)
- [Niri fullscreen-game guidance](https://niri-wm.github.io/niri/Application-Issues.html#fullscreen-games)
- [Niri window fields](https://docs.rs/niri-ipc/latest/niri_ipc/struct.Window.html)
- [Niri workspace fields](https://docs.rs/niri-ipc/latest/niri_ipc/struct.Workspace.html)
