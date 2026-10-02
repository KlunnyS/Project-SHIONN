# Chamber authoring and verification

This is the maintained procedure for creating an instrumented Portal 2 chamber
for SHIONN. Run shell commands from the repository root. Replace
`dataset_test13` with the chosen map name everywhere; keep it lowercase and use
the same name for the VMF, BSP, recorder `--map` value, and episode metadata.

The navigation curriculum geometry is documented separately in
[the chamber layout sheet](NAVIGATION_CHAMBER_LAYOUTS.md).

## 1. Check paths and choose a name

The current workstation installation is under:

```text
/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/
├── sdk_content/maps/          editable Hammer VMFs
├── portal2/maps/              compiled BSPs loaded by `map`
└── portal2/scripts/vscripts/  game-side event scripts
```

Confirm that the proposed source and compiled map do not already exist:

```bash
game_dir='/mnt/game-main/SteamLibrary/steamapps/common/Portal 2'
find "$game_dir/sdk_content/maps" -maxdepth 1 -iname 'dataset_test*.vmf'
find "$game_dir/portal2/maps" -maxdepth 1 -iname 'dataset_test*.bsp'
```

Portal 2 must run through Proton on this machine. The native Linux build has
previously produced an invalid `GAME/home/...` export path. If Hammer reports a
missing stock instance such as `instances/p2editor/door_entrance_4.vmf`, verify
Portal 2's game files before changing the chamber.

## 2. Build the geometry

Choose one route:

1. Open a known-good instrumented VMF in Hammer++ and immediately **Save As**
   the new map name. Change the geometry and reposition every inherited start,
   goal, timeout, and failure trigger.
2. Build and test the geometry in Puzzle Maker. Test/Simulate exports
   `sdk_content/maps/preview.vmf`. Open it in Hammer++, immediately **Save As**
   the permanent map name, and then add SHIONN's entities.

Puzzle Maker overwrites `preview.vmf`, and Hammer edits do not round-trip back
into the voxel editor. Treat the export as a one-way handoff. Start the current
Hammer++ installation with:

```bash
./hammerpp-home.sh
```

The VMF is editable source; Portal 2 runs the compiled BSP.

## 3. Install and wire the event script

Copy the repository version whenever the VScript changes:

```bash
game_dir='/mnt/game-main/SteamLibrary/steamapps/common/Portal 2'
mkdir -p "$game_dir/portal2/scripts/vscripts"
cp portal_assets/scripts/vscripts/shionn_events.nut \
  "$game_dir/portal2/scripts/vscripts/shionn_events.nut"
```

Create exactly one `logic_script` in Hammer++:

```text
Name / targetname: shionn_event_script
Entity Scripts:    shionn_events.nut
```

Use player-only `trigger_once` brushes and wire these outputs:

| Purpose | Source/output | Parameter | Delay |
|---|---|---|---:|
| Recording start | Start trigger `OnStartTouch` | `SignalChamberReady()` | 0 |
| Successful finish | Goal trigger `OnStartTouch` | `SignalGoalReached()` | 0 |
| Optional timeout | Start trigger `OnStartTouch` | `SignalTimeout()` | chosen limit |
| Optional failure volume | Failure trigger `OnStartTouch` | `SignalOutOfBounds()` | 0 |

For each row, target `shionn_event_script` and use input `RunScriptCode`. A
delayed `logic_auto` `OnMapSpawn` can replace the start trigger, but it must
allow enough time for the player to spawn. The shared script suppresses
duplicate ready and terminal events, but the goal should still be a
`trigger_once`, never `trigger_multiple`.

The expected event order for each attempt is:

```text
EVT|chamber_ready|...
EVT|goal_reached|1
```

or one `EVT|episode_failed|reason` instead of the goal event.

## 4. Compile the map

Compile the permanent VMF with VBSP, VVIS, and VRAD. Confirm that Hammer's
output is available at:

```text
/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/portal2/maps/dataset_test13.bsp
```

If Hammer writes elsewhere, copy that newly compiled BSP into `portal2/maps/`.
Do not copy Puzzle Maker's `puzzlemaker/preview.bsp` after adding Hammer
entities; it does not contain those edits. Recompile after every VMF change.

## 5. Verify events by hand

Launch Portal 2 with netconsole enabled:

```bash
steam -applaunch 620 -netconport 8020
```

Use this temporary listener to load and hand-walk the chamber:

```bash
.venv/bin/python - <<'PY'
import time
from wrapper import Portal2Controller

controller = Portal2Controller(port=8020, log_file=None)
if not controller.connect():
    raise SystemExit("Portal 2 is not listening on port 8020")
controller.send_command("map dataset_test13")
try:
    while True:
        controller.read_console(print_to_terminal=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    controller.disconnect()
PY
```

Confirm exactly one ready event after spawn and one goal event at the finish.
Reload at least once to ensure the start trigger works on every attempt. If the
map has timeout or failure triggers, test each one as well. Fix and recompile
before collecting any demonstrations.

## 6. Record and inspect a pilot

Run the recorder as the desktop user, never with `sudo`:

```bash
.venv/bin/python record_dataset.py --map dataset_test13 --episodes 3
.venv/bin/python dataset_stats.py
```

The three-episode target counts successful runs; failed attempts are retained
and retried. Inspect at least one saved `video.mp4`, confirm that `actions.csv`
is aligned, and verify that `metadata.json` contains the expected map name
before recording a larger batch.

If capture selects the wrong monitor, use `wf-recorder -L` and pass
`--output NAME`. If input devices are unavailable, run
`./install_dependencies.sh --permissions`, then log out and back in. The
[CLI reference](CLI_REFERENCE.md) documents recorder timing and capture flags.

## Acceptance checklist

- The permanent VMF and BSP names match the recorder map name.
- The chamber is solvable by hand and its intended path matches the layout.
- Every reload emits one ready event.
- Every outcome emits one terminal event.
- Timeout values agree with the recorder's intended duration.
- Pilot videos contain only the intended game view.
- Pilot action rows and metadata match the recording.
