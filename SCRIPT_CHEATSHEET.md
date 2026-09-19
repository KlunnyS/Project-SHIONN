# Script cheat sheet

For component behavior and data formats, see [the program guide](docs/PROGRAM_GUIDE.md). For every command-line flag and default, see [the CLI reference](docs/CLI_REFERENCE.md).

Run these commands from the repository root. Use `.venv/bin/python` on this Linux machine. The recorder and live runner require a working Wayland desktop, Portal 2, and the appropriate input permissions. Training and dataset reports can run without the game.

## Common workflow

| What you want | Command | Result |
|---|---|---|
| Check the setup | `./check_dependencies.sh` | Reports Python, capture tools, Steam, and input access. |
| Install dependencies | `./install_dependencies.sh --all` | Creates/updates `.venv`, installs system tools, and configures input permissions. May require `sudo`; log out and back in after group changes. |
| Open the current Hammer++ installation | `./hammerpp-home.sh` | Opens the editor from `/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/bin`. |
| Record 20 successful runs on a chamber | `.venv/bin/python record_dataset.py --map dataset_test2 --episodes 20` | Saves every attempt, retries failures, and returns to the game menu after 20 successes. Omit `--episodes` to run until Ctrl+C. |
| Check recorded outcomes and chambers | `.venv/bin/python dataset_stats.py` | Counts completed episodes and action rows by outcome, date, and chamber. |
| Prepare successful episodes for training | `.venv/bin/python -m models.imitation.preprocess --recordings-dir episodes/goal_reached` | Creates missing cached RGB frames and action arrays under `data/datasets/cached_frames`. |
| Train a new candidate | `.venv/bin/python -m models.imitation.train_bc --checkpoint-dir /mnt/extra/Project-SHIONN/checkpoints_new_v3 --checkpoint-every 0` | Trains from the default cache; stores checkpoints on the extra drive and skips periodic step files. |
| Run the current model | `./run_model.sh` or `./run_model.fish` | Runs a 60-second attempt on `dataset_test1`, saving a review video and diagnostics. |
| Run the model from an SSH session | `./run_model_ssh.sh` | Uses the already-running game and Hyprland focus recovery. |
| Run automated tests | `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` | Runs the isolated Python test suite. |

The raw `episodes/` and `data/datasets/cached_frames/` paths are symlinks to `/mnt/extra/Project-SHIONN/` on this machine. Mount `/mnt/extra` before recording, preprocessing, or training.

## Shell and Fish launchers

| Script | Usage | Notes |
|---|---|---|
| `check_dependencies.sh` | `./check_dependencies.sh` | Read-only setup check. |
| `install_dependencies.sh` | `./install_dependencies.sh [--system\|--permissions\|--all]` | Installs Python requirements by default; optional flags install `wf-recorder` and configure `/dev/input`/`uinput` access. `--help` lists options. |
| `hammerpp-home.sh` | `./hammerpp-home.sh` | Starts the active Hammer++ editor through Wine. |
| `run_model.sh` | `./run_model.sh [run_imitation.py options]` | Bash desktop launcher. Defaults: `checkpoints_v3/best.pt`, output `DP-1`, map `dataset_test1`, 60 seconds, video and verbose diagnostics. Later arguments override defaults. |
| `run_model.fish` | `./run_model.fish [run_imitation.py options]` | Fish equivalent of `run_model.sh`; requires Fish. |
| `run_model_ssh.sh` | `./run_model_ssh.sh [run_imitation.py options]` | Bash launcher for an already-running game; adds `--no-launch` and `--keep-focused`. |

Example override: `./run_model.sh --map dataset_test2 --checkpoint models/imitation/checkpoints_new_run/best.pt`. List live-runner flags with `.venv/bin/python run_imitation.py --help`.

## Python entry points

| Script/module | Usage | Purpose and useful options |
|---|---|---|
| `record_dataset.py` | `.venv/bin/python record_dataset.py --map dataset_test2 --episodes 20` | Human demonstration recorder. `--episodes` counts only `goal_reached`; failed attempts are saved and retried. `--duration` sets the local attempt limit, `--output` selects a monitor from `wf-recorder -L`, and `--mouse-device` selects the physical pointer. `--help` lists all flags. |
| `dataset_stats.py` | `.venv/bin/python dataset_stats.py [episodes_directory]` | Reports outcome/date/chamber counts. The default directory is `episodes`; pass `episodes/goal_reached` to count only successes. |
| `models/imitation/preprocess.py` | `.venv/bin/python -m models.imitation.preprocess --recordings-dir episodes/goal_reached` | Converts each completed MP4 and CSV into 320×180 RGB frame arrays plus cached action arrays. Skips existing cache files; add `--overwrite` to rebuild them. `--cache-dir` changes the destination. |
| `models/imitation/train_bc.py` | `.venv/bin/python -m models.imitation.train_bc --checkpoint-dir models/imitation/checkpoints_new_run` | Behavior-cloning trainer. Common flags: `--cache-dir`, `--epochs`, `--batch-size`, `--device auto\|cuda\|cpu`, `--workers`, and `--resume PATH`. Use a new checkpoint directory for a new dataset run. |
| `run_imitation.py` | `.venv/bin/python run_imitation.py --checkpoint models/imitation/checkpoints_v3/best.pt --map dataset_test2` | Direct live-policy runner. Use `--dry-run --max-seconds 10` to inspect predictions without sending actions. `--record-video` saves model attempts; `--keep-focused` supports Hyprland. `--help` lists all flags. |
| `recorder.py` | `.venv/bin/python recorder.py` | Lower-level event listener. It waits for `EVT` signals from a game already connected on netconsole port 8020 and does not load/reset a map; use `record_dataset.py` for normal data collection. |
| `wrapper.py` | `.venv/bin/python wrapper.py` | Manual netconsole and input smoke test. Loads `puzzlemaker/preview` and tries a jump. It can wait if the map/player is unavailable; it is also imported by active scripts. |

## Python modules imported by the entry points

These files are library code, so there is no separate command to run them:

| File | Role |
|---|---|
| `models/__init__.py` | Marks the model package. |
| `models/imitation/__init__.py` | Exposes the imitation dataset/model package. |
| `models/imitation/checkpoint.py` | Saves and loads resumable `.pt` checkpoints and matching JSON configs. |
| `models/imitation/dataset.py` | Discovers cached episodes, splits whole episodes, and builds memory-mapped training samples. |
| `models/imitation/inference.py` | Applies the trained policy to live frames. |
| `models/imitation/network.py` | Defines the current and legacy neural networks. |

`portal_assets/scripts/vscripts/shionn_events.nut` is game-side VScript, not a shell/Python command. Copy it to Portal 2's `portal2/scripts/vscripts/` when setting up a chamber; the chamber's `logic_script` invokes its ready, goal, and failure functions.

## Automated tests

Run them together with the test command in **Common workflow**. Each is an automated test module, not a live-game launcher:

| File | Checks |
|---|---|
| `tests/test_record_dataset.py` | Recorder events, input-device selection, episode metadata/outcome handling, and success quota. |
| `tests/test_dataset_stats.py` | Outcome/date/chamber counts and incomplete episode handling. |
| `tests/test_imitation_pipeline.py` | Frame preparation, model/data contracts, inference helpers, and timeout behavior. |

## Manual hardware diagnostics

These are kept under `scripts/diagnostics/` so they are not mistaken for automated tests. Run them only while the intended Wayland desktop/game is active; they operate on the real screen or input devices.

| File | Usage and effect |
|---|---|
| `scripts/diagnostics/input_movement.py` | `.venv/bin/python scripts/diagnostics/input_movement.py` — waits three seconds, then presses and releases `W` through a virtual keyboard. |
| `scripts/diagnostics/input_mouselook.py` | `.venv/bin/python scripts/diagnostics/input_mouselook.py` — waits three seconds, then sends a short relative mouse movement. |
| `scripts/diagnostics/capture_bytes.py` | `.venv/bin/python scripts/diagnostics/capture_bytes.py` — reads five seconds of raw `wf-recorder` bytes from output `DP-1` at 1280×720/20 FPS. |
| `scripts/diagnostics/capture_motion.py` | `.venv/bin/python scripts/diagnostics/capture_motion.py` — captures two 1280×720 Wayland frames and compares visual change. |

## Older one-off helpers

These were moved from the repository root to `scripts/legacy/`. Run the Python files as modules from the repo root so imports of `wrapper.py` still work.

| File | Usage and effect |
|---|---|
| `scripts/legacy/run_sequence.py` | `.venv/bin/python -m scripts.legacy.run_sequence` — loads `puzzlemaker/preview` and runs a hard-coded movement/use sequence; not part of the dataset workflow. |
| `scripts/legacy/example_usage.py` | `.venv/bin/python -m scripts.legacy.example_usage` — replays `sequences/TEST_0_mimic_sequence/actions.csv` at 60 FPS. |
| `scripts/legacy/hammerpp_notas.sh` | `./scripts/legacy/hammerpp_notas.sh` — alternate Hammer++ launcher for `/home/user/.local/share/Steam/...`; update that path before use on a different install. |
