# Program guide

This guide describes the implemented SHIONN pipeline. For copyable commands, see [the script cheat sheet](../SCRIPT_CHEATSHEET.md). For every command-line option and default, see [the CLI reference](CLI_REFERENCE.md). Run commands from the repository root unless stated otherwise.

## What runs today

SHIONN records human Portal 2 demonstrations, caches aligned video and actions, trains a behavior-cloning policy, and runs that policy against the live game. The policy receives screen pixels only. Netconsole events are used by the surrounding program to start and stop attempts; they are not model inputs. A Gym-style API, reward calculation, and reinforcement-learning trainer remain future work.

```text
Portal 2 + chamber VScript
    ├─ netconsole EVT messages ────────┐
    ├─ Wayland screen frames ───────────┤ record_dataset.py + recorder.py
    └─ physical keyboard/mouse events ─┘            │
                                               episodes/<outcome>/
                                                       │
                                    models.imitation.preprocess
                                                       │
                                      cached_frames/*.npy + *.json
                                                       │
                                        models.imitation.train_bc
                                                       │
                                           best.pt / last.pt
                                                       │
                   Portal 2 ← wrapper.py ← run_imitation.py ← inference.py
```

## 1. Game, chamber, and control channel

[`wrapper.py`](../wrapper.py) owns the `Portal2Controller`. It connects to the game's local netconsole, normally started with `-netconport 8020`, sends commands such as `map dataset_test3`, reads console output, and translates predicted actions into game input. If Portal 2 is not running, the recorder and live runner can launch it through Steam; an already-running game must have the matching netconsole port enabled.

Each instrumented chamber contains a `logic_script` that loads [`shionn_events.nut`](../portal_assets/scripts/vscripts/shionn_events.nut). A start trigger calls `SignalChamberReady()`, a goal trigger calls `SignalGoalReached()`, and optional failure triggers call functions such as `SignalTimeout()` or `SignalOutOfBounds()`. The script emits lines of the form `EVT|chamber_ready|...`, `EVT|goal_reached|1`, and `EVT|episode_failed|reason`. Its guards prevent duplicate start and terminal signals within an attempt. The chamber must send ready on every reload or the recorder will wait until `--ready-timeout` expires.

`Portal2Controller.apply_action()` holds/release movement, jump, use, and portal-fire commands only when their predicted binary state changes. Mouse look is sent as relative `uinput` movement. `release_policy_actions()` releases held controls at boundaries and shutdown. The controller also has `play_csv()` for replaying a saved action sequence.

## 2. Screen and physical input capture

[`recorder.py`](../recorder.py) provides the capture primitives used by both recording and live inference:

| Component | Responsibility |
|---|---|
| `find_input_devices()` | Finds readable physical keyboards and pointing devices; `--mouse-device` can select a pointer. |
| `InputTracker` | Reads physical `evdev` events on background threads, keeps key-hold state, catches short presses, and accumulates mouse deltas between ticks. |
| `WaylandCamera` | Runs `wf-recorder` as a long-lived raw BGR video source and retains the latest frame. |
| `FFmpegVideoWriter` | Streams those BGR frames to an audio-free H.264 MP4 using `libx264`. |
| `EpisodeRecorder` | Writes one frame and action row per tick, then publishes a finished episode under its outcome. |

The normal recording contract is 1920×1080 at 24 FPS with CRF 20. These values are flags on the recorder. On this Linux/Wayland setup, the desktop user needs access to `/dev/input` and `/dev/uinput`; run the recorder as that user, not with `sudo`. `./install_dependencies.sh --permissions` sets the persistent input permissions, and `wf-recorder -L` lists display outputs.

## 3. Human demonstration recorder

[`record_dataset.py`](../record_dataset.py) is the normal data-collection entry point. It connects to netconsole, starts input and camera capture, waits for the focus delay, loads the selected map, then waits for `EVT|chamber_ready`. After ready, each 24 Hz tick writes a frame and its aligned input snapshot. The first goal or failure event, or the local `--duration` limit, ends the attempt. The chamber reloads if the success target has not been reached.

`--episodes N` counts **successful** `goal_reached` episodes. Failed attempts are still saved and retried; they do not advance the success quota. The terminal prints successful count and total attempts after each completed recording. At the target, the recorder sends `disconnect` to return Portal 2 to its menu. With `--episodes 0` (the default), recording continues until Ctrl+C. An interrupted active episode is finalized under `interrupted`.

Episodes start in `episodes/.in_progress/` and move to `episodes/<outcome>/episode_<timestamp>/` only after the video writer finalizes. A finished episode contains:

| File | Contents |
|---|---|
| `video.mp4` | Audio-free H.264 screen frames, one per recorded tick. |
| `actions.csv` | `timestamp`, `frame_idx`, four movement columns, `jump`, `crouch`, `use`, two portal-fire columns, `mouse_dx`, and `mouse_dy`. |
| `metadata.json` | `{"map": "dataset_test3"}` or another map name supplied by `--map`. |

The folder name records the outcome, while `metadata.json` records the chamber. The video and CSV remain the source data for training. Recording artifacts are ignored by Git.

## 4. Dataset report and offline cache

[`dataset_stats.py`](../dataset_stats.py) scans completed episodes and reports counts and action rows by outcome, recording date, and chamber. It excludes `.in_progress` entries and episodes without `video.mp4`. A legacy episode without a map label appears as `unknown`.

[`models/imitation/preprocess.py`](../models/imitation/preprocess.py) discovers episodes with both `video.mp4` and `actions.csv`. Point `--recordings-dir` at `episodes/goal_reached` when preparing expert demonstrations; the default `episodes` includes all outcomes. Before publishing a cache, it checks that `frame_idx` starts at zero and is contiguous and that the declared video frame count equals the CSV row count. It decodes BGR video, converts frames to RGB, and resizes them to 320×180 with area interpolation.

For each episode, the cache contains:

| File | Contents |
|---|---|
| `<episode>.npy` | Memory-mappable `uint8` RGB frames shaped `(N, 180, 320, 3)`. |
| `<episode>.actions.npy` | `float32` action labels shaped `(N, 10)`. |
| `<episode>.json` | Source paths, frame count, source size/FPS, cached action filename, and preprocessing contract. |

The ten training labels are `move_w`, `move_a`, `move_s`, `move_d`, `jump`, `use`, `fire_left`, `fire_right`, `mouse_dx`, and `mouse_dy`. The recorded `crouch` column is retained in the CSV but excluded from this policy. Existing complete caches are skipped; `--overwrite` rebuilds them. [`models/imitation/dataset.py`](../models/imitation/dataset.py) memory-maps cached arrays, stacks four frames into `(12, 180, 320)`, and excludes leading idle frames before the first recorded action. Splits are made by whole episode, not by individual frame.

On this machine, `episodes/` and `data/datasets/cached_frames/` are symlinks into `/mnt/extra/Project-SHIONN/`. The drive must be mounted before recording, preprocessing, or training. The paths used in commands stay the same.

## 5. Network, training, and checkpoints

[`models/imitation/network.py`](../models/imitation/network.py) defines the current `shionn_imitation_v3` policy. It takes four 320×180 RGB frames as 12 channels. A convolutional trunk with GroupNorm and SiLU produces features for eight independent binary action heads and one Gaussian mouse head (mean and log standard deviation for `dx` and `dy`). `LegacyImitationPolicy` remains available to load older v1/v2 checkpoints. The model does not read the chamber label or netconsole messages.

[`models/imitation/train_bc.py`](../models/imitation/train_bc.py) trains the v3 model on cached expert episodes. It chooses CPU or CUDA, splits complete episodes into train and validation sets, computes class weights for supported binary actions, and standardizes mouse targets using statistics from the training split. Its loss adds eight binary cross-entropies and a Gaussian mouse negative log likelihood. The optimizer is AdamW; CUDA training uses mixed precision. Validation loss selects `best.pt`.

[`models/imitation/checkpoint.py`](../models/imitation/checkpoint.py) saves a `.pt` file with model weights, optimizer state, epoch/step, best validation loss, and configuration, plus a matching `.json` contract. `last.pt` is written after each epoch; `best.pt` is replaced when validation improves. By default, resumable `step_*.pt` files are also written every 1,000 optimizer steps. `--checkpoint-every 0` disables only those step files. `--resume PATH` restores optimizer and model state but requires the architecture, preprocessing, action columns, and target-processing values to match; an expanded dataset can change class weights and mouse scale. Use a new checkpoint directory for a new dataset run.

The trainer's episode-level random split measures performance on held-out recordings from the available data. It does not automatically hold out an entire chamber. Test the trained policy in the game on each chamber before judging its generalization.

## 6. Live policy runner and diagnostics

[`run_imitation.py`](../run_imitation.py) loads a checkpoint through [`models/imitation/inference.py`](../models/imitation/inference.py), connects to Portal 2, optionally loads a map, starts the same Wayland capture path, and predicts actions at `--fps`. Inference applies the same frame conversion and normalization contract used for training. It keeps a four-frame history, repeats the first frame until the history fills, and uses the controller to apply key transitions and relative mouse motion.

`--dry-run` still captures and predicts but does not apply predicted controls. `--record-video` writes a model-attempt MP4 under `model_attempts/` by default and a matching JSONL log. `--log-actions` writes JSONL without video. Logs contain metadata, per-tick actions and model diagnostics, focus state, console events, errors/stops, and a final summary. These are **model-generated attempts**, not expert training labels.

The runner normally stops at its `--max-seconds` limit or a goal/failure event. When `--max-seconds` is positive, a chamber `episode_failed|timeout` message is logged but does not end the run; the runner's own time limit controls it. With `--max-seconds 0`, chamber timeout is terminal. Escape from a readable physical keyboard and Ctrl+C are stop paths. Held actions are released in cleanup. `--keep-focused` asks Hyprland to focus Portal 2 and restore input if focus is lost.

`run_model.sh` and `run_model.fish` provide local defaults; `run_model_ssh.sh` also sets `--no-launch` and `--keep-focused` for an existing desktop/game reached over SSH. Their extra arguments are passed to `run_imitation.py`.

## 7. Setup, maintenance, and tests

[`install_dependencies.sh`](../install_dependencies.sh) creates/updates `.venv`, installs Python packages from `requirements.txt`, optionally installs `wf-recorder`, and optionally configures input-device permissions. [`check_dependencies.sh`](../check_dependencies.sh) reports the current setup. `hammerpp-home.sh` opens the local Hammer++ installation. See [the script cheat sheet](../SCRIPT_CHEATSHEET.md) for the older one-off helpers under `scripts/legacy/` and manual hardware probes under `scripts/diagnostics/`.

Run the automated suite with `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`. These tests cover recorder events and quota behavior, dataset reporting and preprocessing, model contracts, inference helpers, and stop/timeout handling. Hardware probes in `scripts/diagnostics/` run against real devices and are not part of that suite.
