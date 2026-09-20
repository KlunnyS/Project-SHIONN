# Command-line reference

Run commands from the repository root. Python commands use `.venv/bin/python` on this Linux installation. `-h` or `--help` prints the current parser for each Python entry point; the defaults below reflect the current code. See [the program guide](PROGRAM_GUIDE.md) for what each component does and [the script cheat sheet](../SCRIPT_CHEATSHEET.md) for short workflows.

## `record_dataset.py`

Example: `.venv/bin/python record_dataset.py --map dataset_test3 --episodes 20`

For round-robin collection, repeat `--map`; `--episodes` is the successful-run target **for each map**:

```bash
.venv/bin/python record_dataset.py --map dataset_test5 --map dataset_test6 --map dataset_test7 --episodes 50
```

This records one success on each map before starting the next round. A failed attempt is saved and retried on the same map, so failures do not advance the round.

| Flag | Default | Meaning |
|---|---:|---|
| `--map NAME` | `dataset_test1` | Chamber sent as `map NAME`; repeat this flag to cycle distinct maps in order. Saved in each episode's `metadata.json`. |
| `--port N` | `8020` | Portal 2 netconsole TCP port. An existing game must have been launched with this port. |
| `--duration SECONDS` | `30` | Local maximum length of one recording; reaching it saves the attempt as `timeout`. |
| `--focus-delay SECONDS` | `5` | Time to focus Portal 2 before the first map load. |
| `--restart-delay SECONDS` | `1` | Pause after a completed attempt before loading the chamber again. |
| `--ready-timeout SECONDS` | `60` | Maximum wait for `EVT|chamber_ready` after a map load. |
| `--episodes N`, `--episode N` | `0` | Stop after N **successful** `goal_reached` episodes per map. Failures are saved and retried on that map. `0` cycles until Ctrl+C. After every map meets a finite target, return Portal 2 to the menu. |
| `--fps N` | `24` | Recording tick and video frame rate. |
| `--width N` | `1920` | Captured/video frame width. |
| `--height N` | `1080` | Captured/video frame height. |
| `--crf N` | `20` | H.264 encoder CRF passed to FFmpeg. |
| `--episodes-root PATH` | `episodes` | Recording root. Use `episodes_eval` for held-out evaluation references. |
| `--output NAME` | first detected output | Wayland monitor name from `wf-recorder -L`. |
| `--mouse-device PATH_OR_NAME` | auto-detected pointer | Exact `/dev/input/event*` path or case-insensitive fragment of a device name. |
| `--no-launch` | off | Require Portal 2 to be running instead of launching it through Steam. |

The recorder validates positive duration, ready timeout, FPS, width, and height; delays and episode target must be nonnegative, and map names must be unique. Run it as the desktop user, not with `sudo`.

## `dataset_stats.py`

Example: `.venv/bin/python dataset_stats.py` or `.venv/bin/python dataset_stats.py episodes/goal_reached`

| Argument | Default | Meaning |
|---|---:|---|
| `dataset_dir` | `episodes` | Directory to scan for completed episodes. Passing the success folder limits the report to successful recordings. |

## `models.imitation.preprocess`

Recommended command: `.venv/bin/python -m models.imitation.preprocess --recordings-dir episodes/goal_reached`

| Flag | Default | Meaning |
|---|---:|---|
| `--recordings-dir PATH` | `episodes` | Source tree. The default includes failed outcomes; choose `episodes/goal_reached` for expert-only training data. |
| `--cache-dir PATH` | `data/datasets/cached_frames` | Output directory for frame, action, and metadata cache files. |
| `--overwrite` | off | Rebuild episodes already present in the cache. Without it, complete cached episodes are skipped. |

The command checks frame/action alignment and stops with an error when a source episode violates the cache contract.
New caches store the map name. Training can also read it from the source recording for older caches.

## `models.imitation.train_bc`

Example: `.venv/bin/python -m models.imitation.train_bc --checkpoint-dir models/imitation/checkpoints/runs/new_v3 --checkpoint-every 0`

| Flag | Default | Meaning |
|---|---:|---|
| `--cache-dir PATH` | `data/datasets/cached_frames` | Cached frame/action dataset. |
| `--train-extra-cache-dir PATH` | none | Repeat to add cached correction episodes to training only, after the base episode split. |
| `--correction-sampling-fraction FLOAT` | `0` | Expected fraction of optimizer draws from the extra caches, such as `0.10`; requires extra cached episodes. |
| `--start-window-frames N` | `8` | Usable frames after each episode's first action eligible for opening sampling boost. |
| `--start-sampling-boost FLOAT` | `1` | Multiply opening-frame sampling weight; `5` gives openings more training exposure. |
| `--map NAME` | all cached maps | Repeat to train on only the named chambers. An unknown or duplicate name is an error. |
| `--epochs N` | `20` | Total number of epochs in the run; on resume this remains the total target. |
| `--batch-size N` | `8` | Training and validation batch size. |
| `--learning-rate FLOAT` | `3e-4` | AdamW learning rate. |
| `--validation-fraction FLOAT` | `0.2` | Fraction of complete episodes used for validation. |
| `--seed N` | `0` | Shuffle seed for episode split and epoch order. |
| `--sampling chamber-balanced\|uniform` | `chamber-balanced` | Draw equal expected numbers of usable frames from each training map per epoch, or use legacy uniform frame sampling. The epoch still contains one draw per usable training frame. |
| `--binary-class-weighting balanced\|none` | `balanced` | Reweight binary action losses, or leave every class at weight 1 so recorded action frequencies determine the loss. |
| `--jump-positive-weight FLOAT` | class weighting default | Set the positive jump weight as a multiple of the no-jump weight; use separate runs to test values such as 3 and 5. |
| `--mouse-head gaussian\|binned` | `gaussian` | Existing Gaussian head or independent dx/dy classification heads. |
| `--mouse-bins N` | `15` | Requested odd class count per axis for the binned head. Quantile edges are fitted from usable training actions only, with an exact zero class. |
| `--workers N` | `0` | PyTorch data-loader worker processes. |
| `--device auto\|cuda\|cpu` | `auto` | Training device; `auto` uses CUDA when available. |
| `--checkpoint-dir PATH` | `models/imitation/checkpoints_v3` | Directory for `best.pt`, `last.pt`, step checkpoints, and matching JSON files. Use `models/imitation/checkpoints/runs/<name>` for a new run on the extra drive. |
| `--checkpoint-every N` | `1000` | Save a `step_*.pt` checkpoint and refresh `last.pt` every N optimizer steps. `0` disables step checkpoints; epoch-end `last.pt` and improved `best.pt` remain. |
| `--log-every N` | `100` | Print average running loss every N training batches. `0` suppresses batch logs. |
| `--early-stop-patience N` | `3` | Stop after N complete epochs without a lower validation loss. `0` disables early stopping; `--epochs` remains the maximum. The counter starts fresh when resuming. |
| `--resume PATH` | none | Resume model and optimizer from a compatible checkpoint. The trainer checks the architecture and data/target-processing contract. |

The default cache is not filtered by chamber or outcome; prepare it intentionally. With the default opening and correction weights, `--sampling uniform` shuffles every usable training frame once per epoch, retaining the recorded frame proportions between maps. Boosted openings or a correction fraction use weighted draws with replacement. `--binary-class-weighting none` disables action-class reweighting. Mouse deltas are still scaled numerically for Gaussian training; this does not change sample frequencies. On a larger dataset, frequent step checkpoints can consume substantial disk space. Each completed epoch also appends train/validation totals and per-action losses to `metrics.jsonl` in the checkpoint directory. The number of epochs and batches determines training time; this command does not run the live game.

For the mouse-bin experiment and its jump-weight follow-ups, see [the mouse policy workflow](MOUSE_POLICY_EXPERIMENT.md). Binned checkpoints keep the fitted edges and representative deltas inside the checkpoint. They use argmax independently on each axis during inference.

For the 450-episode cutdown experiment (`dataset_test2` and `dataset_test5` through `dataset_test12`), use a separate checkpoint directory:

```bash
.venv/bin/python -m models.imitation.train_bc \
  --map dataset_test2 \
  --map dataset_test5 --map dataset_test6 --map dataset_test7 --map dataset_test8 \
  --map dataset_test9 --map dataset_test10 --map dataset_test11 --map dataset_test12 \
  --sampling uniform --binary-class-weighting none \
  --device cuda \
  --checkpoint-dir models/imitation/checkpoints/runs/cutdown_uniform_v1 \
  --checkpoint-every 0
```

## `run_imitation.py`

Example: `.venv/bin/python run_imitation.py --checkpoint models/imitation/checkpoints_v3/best.pt --map dataset_test3 --dry-run --max-seconds 10`

| Flag | Default | Meaning |
|---|---:|---|
| `--checkpoint PATH` | `models/imitation/checkpoints/best.pt` | Checkpoint to load. Bash/SSH launchers choose `checkpoints_v3/best.pt`; Fish chooses `checkpoints_450_v3/best.pt`. |
| `--device auto\|cuda\|cpu` | `auto` | Device for policy inference. |
| `--jump-threshold FLOAT` | binary argmax | Use a validation-selected threshold between 0 and 1 for the jump action. |
| `--move-w-threshold FLOAT` | binary argmax | Override the forward action threshold between 0 and 1 for calibration tests. |
| `--port N` | `8020` | Portal 2 netconsole port. |
| `--fps FLOAT` | `24` | Prediction tick rate; screen capture/video FPS use a rounded positive integer. |
| `--width N` | `1920` | Capture/video frame width. |
| `--height N` | `1080` | Capture/video frame height. |
| `--output NAME` | first detected output | Wayland monitor from `wf-recorder -L`. |
| `--map NAME` | none | Optionally load a chamber after connecting. |
| `--countdown SECONDS` | `3` | Delay before the policy loop starts; Escape can cancel during this period. |
| `--max-seconds SECONDS` | `60` | Local run limit. `0` removes the local limit; goal/failure events can still stop the run. |
| `--dry-run` | off | Compute predictions without applying them to the game. Capture, map loading, and optional logging still run. |
| `--no-launch` | off | Require an already-running Portal 2 process. |
| `--continue-after-event` | off | Continue the run after a goal/failure event, resetting policy history and releasing held actions. |
| `--record-video` | off | Save a timestamped model-attempt MP4 and matching JSONL diagnostics. |
| `--recording-dir PATH` | `model_attempts` | Directory for attempt videos and default diagnostic logs. |
| `--log-actions` | off | Write per-tick JSONL diagnostics without requiring `--record-video`. |
| `--log-file PATH` | none | Write diagnostics to this JSONL file; implies logging. |
| `--verbose` | off | Print a compact status line periodically. |
| `--status-every SECONDS` | `1` | Interval for status output and focus checks. |
| `--keep-focused` | off | On Hyprland, focus Portal 2 and try to restore focus/input after a loss. |
| `--hyprland-instance VALUE` | `auto` | Hyprland instance signature; `auto` discovers candidates, including for SSH sessions. |

Positive `--max-seconds` makes the runner ignore chamber-only `episode_failed|timeout` as a stop signal; its own time limit remains authoritative. `--max-seconds 0` lets that event end the run. Escape and Ctrl+C stop the policy; held controls are released during cleanup.

## `run_model_sequence.py`

Example: `.venv/bin/python run_model_sequence.py --checkpoint old=models/imitation/checkpoints_v3/best.pt --checkpoint new=models/imitation/checkpoints_450_v3/best.pt --map dataset_test1 --map evaluation1 --repeats 3 --output DP-1`

The runner makes a full checkpoint × chamber matrix. By default, each repeat visits one chamber with every model before moving to the next chamber. Add `--plan-only` to print the jobs without opening Portal 2. All checkpoints and matching `.json` files must exist before the sequence starts.

| Flag | Default | Meaning |
|---|---:|---|
| `--checkpoint PATH` or `--checkpoint LABEL=PATH` | required, repeatable | Model checkpoint. A label names its result folders; without one, the checkpoint directory and filename become the label. |
| `--map NAME` | required, repeatable | Chamber to load for every model. |
| `--repeats N` | `1` | Number of attempts per model/chamber pair. |
| `--order map-first\|model-first` | `map-first` | Visit models within each chamber or chambers within each model, repeated N times. |
| `--recording-root PATH` | `model_attempts/sequences` | Parent directory for timestamped sequence folders. |
| `--max-seconds SECONDS` | `60` | Maximum live-policy time per attempt; `0` waits for an event or manual stop. |
| `--countdown SECONDS` | `3` | Delay before each policy attempt. |
| `--pause-seconds SECONDS` | `1` | Pause between attempts. |
| `--output NAME` | first detected output | Wayland monitor passed to each run. |
| `--device auto\|cuda\|cpu` | `auto` | Policy inference device. |
| `--jump-threshold FLOAT` | binary argmax | Pass a validation-selected jump threshold to every live attempt. |
| `--move-w-threshold FLOAT` | binary argmax | Pass a forward action threshold to every live attempt. |
| `--port N` | `8020` | Portal 2 netconsole port. |
| `--fps FLOAT` | `24` | Prediction/capture rate. |
| `--width N`, `--height N` | `1920`, `1080` | Screen/video size. |
| `--status-every SECONDS` | `1` | Status and focus-check interval passed to each run. |
| `--hyprland-instance VALUE` | `auto` | Hyprland instance signature for focus handling. |
| `--no-video` | off | Keep JSONL diagnostics but omit MP4 videos. |
| `--no-launch` | off | Require an already-running game. |
| `--keep-focused` | off | Restore Portal focus/input on Hyprland. |
| `--dry-run` | off | Predict without applying controls; still loads maps and records attempts. |
| `--verbose` | off | Print periodic live-runner status. |
| `--continue-on-error` | off | Try later jobs after a runner process fails; the sequence still exits nonzero. |
| `--plan-only` | off | Show the job order without creating files or launching the game. |

Every job gets a separate `attempt_*.jsonl` and, unless `--no-video` is used, MP4. `sequence.jsonl` in the timestamped folder records each job's checkpoint, map, return code, stop reason, and paths. A completed sequence also writes `benchmark_summary.json` with per-model, per-map success rates. `goal_reached` and `episode_failed` require chamber event signals; `time_limit` means no terminal event was received before the local deadline. Escape or Ctrl+C cancels the remaining jobs. The sequence stops on a process error unless `--continue-on-error` is set.

## Benchmark reports

See [the benchmark workflow](BENCHMARKING.md) for complete commands and how to keep evaluation recordings outside the training cache.

`.venv/bin/python benchmark_expert.py --checkpoint PATH` scores held-out expert frames named by the checkpoint's training split. `--cache-dir PATH` selects a cache, `--map NAME` limits maps, `--subset all` scores every episode in a separate evaluation cache, `--final-frames N` changes the final-frame segment, and `--output-dir PATH` chooses where `summary.json` and `action_comparison.csv` are written. The default output is a timestamped folder under `model_attempts/benchmarks/`.

`.venv/bin/python benchmark_sequence.py PATH` summarizes a `run_model_sequence.py` directory or its `sequence.jsonl`. It writes `benchmark_summary.json` beside the manifest by default; `--output PATH` changes the report path.

`.venv/bin/python benchmark_jump.py PATH/TO/action_comparison.csv` sweeps jump thresholds on an expert-frame comparison CSV and writes per-map and overall precision/recall to `jump_thresholds.json`. Repeat `--threshold FLOAT` to use custom thresholds.

`.venv/bin/python analyze_mouse_rollout.py PATH/TO/attempt.jsonl --wall-window START:END` reports bin entropy, peak gap, opposing-direction peaks, and jump rates for manually annotated wall-facing seconds of a binned live attempt. Repeat `--wall-window` for separate sections.

`.venv/bin/python analyze_mouse_validation.py PATH/TO/action_comparison.csv --wall-ranges PATH/TO/wall_ranges.csv` measures those mouse-bin statistics and jump false positives on annotated expert validation frames. The annotation CSV needs `episode,start_frame,end_frame` columns with inclusive frame ranges. `--jump-threshold` selects a calibrated threshold for the error comparison.

`.venv/bin/python record_recovery.py --map NAME --source-attempt PATH/TO/attempt.jsonl` records a human recovery from the current Portal 2 position without reloading the chamber. It saves under `episodes/recovery/goal_reached` on success. `--max-seconds`, `--focus-delay`, `--episodes-root`, `--port`, `--fps`, `--output`, and `--mouse-device` adjust capture settings.

## Shell, Fish, and direct diagnostics

| Command | Options/defaults |
|---|---|
| `./install_dependencies.sh` | Installs/updates `.venv` and Python requirements. `--system` installs `wf-recorder` if missing, `--permissions` configures input/uinput access, `--all` does both, and `-h`/`--help` prints usage. Without the flags, it may prompt interactively for missing system setup. |
| `./check_dependencies.sh` | No flags. Reports environment and hardware setup. |
| `./hammerpp-home.sh` | No flags. Opens Hammer++ from the configured Steam path. |
| `./run_model.sh [options]` | Bash wrapper for `run_imitation.py`: `checkpoints_v3/best.pt`, `--device auto`, `--output DP-1`, `--map dataset_test1`, `--max-seconds 60`, `--record-video`, and `--verbose`. Appended runner flags override values. |
| `./run_model.fish [options]` | Fish wrapper with the same capture/map defaults; its checkpoint is `models/imitation/checkpoints_450_v3/best.pt`. |
| `./run_model_ssh.sh [options]` | Bash wrapper with the same defaults plus `--no-launch` and `--keep-focused`. |
| `.venv/bin/python recorder.py` | No flags. Low-level recorder that listens for game events without loading/resetting the chamber. |
| `.venv/bin/python wrapper.py` | No flags. Manual preview-map and jump smoke test. |

The scripts under `scripts/diagnostics/` and `scripts/legacy/` have no argument parsers. Their exact invocations and effects are listed in [the script cheat sheet](../SCRIPT_CHEATSHEET.md). Automated tests run with `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.
