# Benchmarking a navigation checkpoint

Run these commands from the repository root after training a candidate. Replace the checkpoint path with the actual `best.pt`. The updated trainer saves its validation episode names inside the checkpoint, so the offline scorer uses the same held-out recordings that were excluded from training.

The [frozen split](DATA_SPLIT.md) defines test1–10 as training, test11/12 as validation, and evaluation1/2 as evaluation-only. The latter two have had prior live runs, which should be disclosed with results; neither is part of the training cache. Check the checkpoint's split ID and manifest hash when comparing candidates trained after the split was introduced.

For the latest V5 comparison, limitations of the 2026-10-20 live sequence, and
the next model/data phases, see the [model roadmap](MODEL_ROADMAP.md).

## 1. Compare actions on held-out expert frames

```bash
.venv/bin/python benchmark_expert.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --cache-dir data/datasets/cached_frames
```

This uses the checkpoint's validation episodes by default. It creates `summary.json` with per-chamber binary-action precision, recall, F1, exact binary-action match, and mouse mean absolute error. `action_comparison.csv` contains each human action and the model's prediction and probability on the **same recorded frame**. The report also breaks out the final 24 frames of each episode; those approximate the goal approach only when the source episodes were successful. `--map NAME` can limit the report to one or more maps.

For checkpoints made before validation episode names were saved, use a new checkpoint from the updated trainer. `--subset all` is for an independent evaluation cache; using it on training data produces an optimistic score.

## 2. Measure autonomous success

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint candidate=models/imitation/checkpoints/runs/candidate/best.pt \
  --map dataset_test1 --map dataset_test5 --map dataset_test10 \
  --map evaluation1 --map evaluation2 \
  --repeats 10 --keep-focused
```

Add `--output NAME` if the default Wayland output is wrong. Each attempt writes a video and an action/diagnostic JSONL. The sequence writes `sequence.jsonl` and, when it finishes, `benchmark_summary.json` under a timestamped `model_attempts/sequences/sequence_*` directory. For an interrupted or older sequence, generate the summary afterward:

```bash
.venv/bin/python benchmark_sequence.py model_attempts/sequences/sequence_TIMESTAMP
```

Sequences now enable focus handling and a bidirectional mouse preflight by
default. Before policy inference, the runner sends three small horizontal pulses
in each direction and checks game yaw (0.25–20 degrees per direction), opposing
responses, pitch stability, and game focus. It aborts the sequence if capture is
weak or unknown, even with `--continue-on-error`. After passing, it reloads the
chamber and restores focus before recording model frames. Preflight pulses are
logged separately and never become expert demonstrations. `--dry-run` skips
preflight and sends no policy controls. `--no-mouse-preflight` is available for
diagnosis, but those results require capture review.

For a single short capture check followed by a model smoke run:

```bash
.venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 --keep-focused --mouse-preflight \
  --max-seconds 10 --log-actions
```

Logged runs sample desktop state in a separate thread at a target 15 Hz
(`--telemetry-hz 1..20`), independent of inference and status printing. Every
record includes a monotonic timestamp. `desktop_state` records actual collection
start/end times, focused window identity, workspace, cursor position and
containment. Tick records include sample age and staleness, raw and limited
actions, submission times, and socket/uinput submission results. With
`--keep-focused`, unknown or stale focus suppresses policy input; observed focus
loss releases held actions and triggers reacquisition. Geometry from a previously
focused game is marked cached; containment is unknown until the game is observed
focused again. Unsupported Niri cursor values remain `null`.

`camera_response` records use marked, asynchronous `getpos` queries through the
runner's existing console reader, preserving terminal-event handling. Request
and receipt timestamps and tick IDs bracket each observation; angle deltas span
successive observations, not a guaranteed one-action response. Missing, late,
or malformed replies remain unknown. Camera sampling follows runner ticks and
can fall below its target during slow inference; desktop sampling continues.
Socket/uinput submission success does not establish game consumption. Inspect
mouse actions between camera observations and review video before diagnosing
capture loss; `policy_freeze` remains a policy-output signature only.

The summary reports actual sampling gaps, unknown observations, focus losses,
cursor exits, injection failures and preflight outcome. Benchmark reports expose
`capture_review_reasons` per attempt and counts of attempts passing capture
checks versus requiring review. Missing legacy telemetry requires review.
Sampling gaps or collection latency above 250 ms, missing camera data, unknown
cursor state, or observed focus/cursor faults also require review. Existing
success rates still describe completed attempts; they are not automatically
clean checkpoint rankings. Passing these checks does not prove uninterrupted
capture between samples.

The report gives successes, valid attempts, success rate, median successful run duration, outcome counts, invalid attempts, and focus losses by model and map. A runner error or manual cancellation is marked invalid instead of counting as a navigation failure. The summary also retains each attempt's video and log path. Compare multiple checkpoints by repeating `--checkpoint LABEL=PATH` in the sequence command.

Review the beginning and at least one later frame before accepting a sequence. With `--keep-focused`, Hyprland should report a nonzero fullscreen state and the runner requests the same visible portal-gun viewmodel used in the demonstrations. Reject runs containing a zoomed opening, desktop bars, window borders, an open Steam overlay, or other large visual mismatches. Input activation requires `xte`: the runner briefly unbinds `MOUSE1`, delivers an XWayland left click to capture input, and restores the prior binding so no portal is fired. An enabled but closed Steam overlay is harmless; hide any visible performance HUD before the authoritative repeated benchmark.

### Guard against hover-driven window switching on Hyprland

The 2026-10-25 sequence recorded focus switching to Firefox after cursor
escapes, followed by the runner restoring game focus. The workstation config
has `input:follow_mouse = 1`, which makes hover focus a plausible contributor.
This is a working hypothesis, not proof of the initial grab failure.

Add `--hyprland-focus-guard` to a direct run or sequence to temporarily set
`input:follow_mouse` to zero. It reads the live value, checks that the write
took effect, and restores the original value on normal shutdown, Escape,
Ctrl+C, SIGTERM, or an error. The flag requires `--keep-focused` on Hyprland
and is unavailable during recovery handoff. It implies diagnostics logging.
It does not edit desktop configuration files or enable pointer confinement.
The previously disabled `steam_app_620` confinement rule does not match the
outer Gamescope window in the recorded attempts; leave that separate setting
out of this first controlled experiment.

During guarded inference, an observed focus loss (even if focus returns before
the next policy tick), stale/unknown focus, a different game window address,
or a workspace/monitor change stops the attempt. The runner releases held
actions and does not try to refocus. A sequence also stops on guard violations
or restoration failure, including with `--continue-on-error`.

Test the repeated-start path that previously showed window switching:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint balanced=models/imitation/checkpoints/runs/v5_balanced_recovery_v1/best.pt \
  --map dataset_test1 --map dataset_test11 --repeats 3 \
  --max-seconds 60 --keep-focused --hyprland-focus-guard \
  --no-launch --verbose --recording-root /tmp/shionn-focus-guard
```

This is a capture diagnostic, not a checkpoint-promotion benchmark. Compare
focus/cursor observations and camera responses with the original sequence;
an idle model cannot establish continuous mouse-capture reliability. Hover
suppression does not establish cursor confinement or relative-input capture,
so keep the mouse preflight and telemetry enabled.

The guard serializes ownership per Hyprland session. It logs the original
value and exact restore command, with a recovery note under
`/tmp/shionn-focus-guard-<uid>-<instance-hash>.json`. SIGKILL, a power loss,
or loss of compositor IPC can prevent restoration; an active stale note with
a mismatched setting blocks another guard from silently accepting the leftover
value. Use the saved command to restore the original setting, then retry.
An explicit preference change during the run is preserved and reported for
capture review. Guard setup/cleanup are recorded in attempt logs and benchmark
reports. The six-attempt guarded sequence at
`/tmp/shionn-focus-guard/sequence_20261025_162021_137292/` completed with
no observed focus losses, and restored the hover setting after every attempt.
Cursor containment still failed in all three test11 runs, starting at about
23.2, 46.5, and 34.5 seconds. No exits were recorded during test1 inference;
one test1 run commanded no mouse movement at all. This separates the observed
focus-switching improvement from the unresolved pointer-grab problem.

Reference: [Hyprland 0.56.2 focus and window actions](https://github.com/hyprwm/Hyprland/blob/v0.56.2/src/config/shared/actions/ConfigActions.cpp).

### Test game-window pointer confinement

Use `--hyprland-confine-pointer` for a separate confinement experiment. It
implies the hover-focus guard, and requires mouse preflight, `--keep-focused`,
and a map reload before inference. Dry-run and recovery handoff are excluded.
The implementation uses Hyprland's named `windowrule` configuration with
`confine_pointer = on`, targeting only the selected window via a unique static
tag. `match:tag` takes the literal tag name, without regex anchors. This covers
the outer Gamescope window rather than the nested Portal window class. It
verifies the tag before enabling the rule. It also toggles the window's readable
`no_follow_mouse` effect through the same named rule and checks that effect
with `getprop`, proving the rule matched. Global hover focus is already disabled
during this check. Disabling the rule restores the underlying window property.
On cleanup it
disables the named rule, removes and verifies removal of its tag, and restores
hover focus. No desktop config file is edited. Disabled named entries remain
in compositor runtime memory until its next config reload.

Start with two runs of the chamber where all guarded attempts escaped:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint balanced=models/imitation/checkpoints/runs/v5_balanced_recovery_v1/best.pt \
  --map dataset_test11 --repeats 2 --max-seconds 60 \
  --keep-focused --hyprland-confine-pointer --no-launch --verbose \
  --recording-root /tmp/shionn-confined
```

Unknown or outside pointer containment during the initial check or inference
stops the attempt. A sampled cursor exit also stops it even if the pointer
returns before the next policy tick. Setup failures stop the entire sequence
even if logging has not started or `--continue-on-error` is set. The rule's
acceptance does not establish working relative mouse input: require the
bidirectional yaw preflight and review camera response during commanded turns.
Hyprland 0.56.2's installed config verifier accepts the rule syntax; live
confinement remains unverified until this pilot is inspected.

The recovery note/log includes `confinement_cleanup_commands` as well as the
hover restore command. If killed without cleanup, run those commands, verify
the unique tag is gone from `hyprctl clients -j`, and remove the stale recovery
note before retrying. Startup refuses a stale confinement note instead of
silently leaving a prior constraint active.

Reference: [Hyprland 0.56.2 named-rule keyword handling](https://github.com/hyprwm/Hyprland/blob/v0.56.2/src/debug/HyprCtl.cpp).

The user-run confinement pilot on 2026-10-25 completed one 60-second attempt
without an exit, then stopped the second at about 18 seconds when the pointer
crossed onto DP-2 while Gamescope retained focus. Cleanup succeeded. The user's
Steam options already included `--backend sdl --force-grab-cursor`, so adding
those flags again would not test a new configuration.

The next driver comparison prefixes `SDL_VIDEODRIVER=x11 SDL_VIDEO_DRIVER=x11`
to the same Steam options, with the user fully exiting and relaunching Portal
2/Gamescope before the next test. Keep resolution, model, map, and runner flags
fixed, saving results under `/tmp/shionn-confined-x11`. The SDL2/SDL3 variables
select the X11 video driver; this is a diagnostic experiment, not a demonstrated
capture remedy. New attempt window snapshots log `xwayland`; confirm the outer
game window reports true. Keyboard-assisted unsticking is allowed for capture
stress tests, but those runs must not be scored as autonomous model benchmarks.

The X11 driver trial confirmed `xwayland: true`, but still recorded exits at
about 7.3 seconds in an incomplete earlier attempt and 21.9 seconds in the next
sequence's second attempt; that sequence's first attempt completed 60 seconds
without an exit. Subsequent source review found a bug in the SHIONN rule setup:
`match:tag` was given `^tag$`, while Hyprland matches literal tag names. Thus
previous "confined" runs did not establish an effective host confinement rule.
The matcher is corrected, and startup now verifies the same rule's readable
effect rather than only checking tag presence. Reports flag older confinement
logs without this evidence as `confinement_rule_match_unverified`. Keep the
current Steam settings for the next user-run pilot to isolate this code fix;
save it under `/tmp/shionn-confined-tagfix`. Effective matching still does not
replace the yaw preflight or prove uninterrupted capture.

References: [SDL2 video-driver selection](https://wiki.libsdl.org/SDL2/SDL_HINT_VIDEODRIVER),
[SDL3 video-driver selection](https://wiki.libsdl.org/SDL3/SDL_HINT_VIDEO_DRIVER).
Literal tag behavior: [Hyprland 0.56.2 TagKeeper](https://github.com/hyprwm/Hyprland/blob/v0.56.2/src/helpers/TagKeeper.cpp).

### Isolate virtual-mouse lifecycle between sequence attempts

After correcting tag matching, the first test11 run completed 60 seconds
without observed cursor exits and with effective-rule verification. The second
failed its yaw preflight before policy inference: left yaw was about +2.70
degrees and right yaw was zero despite submitted pulses and observed focus.
This differs from a mid-run cursor escape. The default sequence starts a fresh
Python process and creates/destroys a virtual mouse for every attempt.

`run_model_sequence.py --persistent-mouse` tests that lifecycle hypothesis by
creating one virtual mouse for the whole sequence before loading/focusing the
first map. Attempts call `run_imitation.main` in the same Python process and
borrow that device. Each attempt still loads its selected checkpoint afresh,
reloads the chamber, starts/stops capture and keyboard monitoring, runs mouse
preflight, and restores the temporary focus/confinement guard. The sequence
destroys the virtual mouse once on completion, failure, or interruption.
Python/CUDA process reuse is also a difference from the default mode, so a
successful test would not alone prove which part of process/device teardown
caused the original problem. No threshold is relaxed and no failed check is
automatically retried.

Keep the current Steam options and other settings fixed:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint balanced=models/imitation/checkpoints/runs/v5_balanced_recovery_v1/best.pt \
  --map dataset_test11 --repeats 2 --max-seconds 60 \
  --keep-focused --hyprland-confine-pointer --persistent-mouse \
  --no-launch --verbose --recording-root /tmp/shionn-persistent-mouse
```

Leave all physical controls alone until `Policy control started` is printed.
Keyboard-assisted unsticking after that point is permitted for this capture
diagnostic, but it makes the run unsuitable as an autonomous model score.
The manifest records `execution_mode`; each attempt records `runner_pid` and
`shared_virtual_mouse` so the selected execution path is auditable. The mode
has isolated lifecycle/cleanup coverage. The user-run pilot at
`/tmp/shionn-persistent-mouse/sequence_20261025_172543_365196/` completed
both 60-second test11 attempts with no observed cursor exits, focus losses,
or submission failures. Both yaw checks and effective rule probes passed;
the second attempt exercised 680 mouse-active policy ticks and 418 moving
camera observations. Guard cleanup succeeded on both attempts. The second
had two camera-query timeouts out of 700 observations, requiring telemetry
review; the first had no automatic capture flags. This supports retaining the
shared process/device mode for further capture tests, while two attempts do
not establish long-run reliability or identify which lifecycle change helped.

The follow-up five-repeat test at
`/tmp/shionn-persistent-mouse-confirmation/sequence_20261025_175754_528314/`
passed all automatic capture checks on every attempt: zero recorded cursor
exits, focus losses, unknown camera replies, or submission failures, with
successful preflights, effective rule probes, and cleanup. Steering was
exercised over 2,720 mouse-active policy ticks. The sequence retained one PID
and shared mouse mode. `capture_confirmation.json` records this inspection.
Keyboard assistance was allowed, so the two goal events are not an autonomous
model score. Keep the current X11 Steam configuration, verified confinement
and persistent mouse for local comparisons. No further capture-only reruns
are needed until the configuration changes or another fault occurs; use
unassisted runs for subsequent model-performance comparisons.

## 3. Record separate references for evaluation chambers

Freeze the candidate checkpoint first. Human reference runs on `evaluation1` and `evaluation2` belong outside the training recording and cache directories:

```bash
.venv/bin/python record_dataset.py \
  --map evaluation1 --map evaluation2 --episodes 10 \
  --episodes-root episodes_eval

.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes_eval/goal_reached \
  --cache-dir data/datasets/evaluation_cached_frames

.venv/bin/python benchmark_expert.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --cache-dir data/datasets/evaluation_cached_frames \
  --subset all
```

Do not pass `episodes_eval` to the training preprocessor or merge its cache into `data/datasets/cached_frames` while it is being used as a held-out evaluation set. The offline comparison asks which actions the model would take on expert images. Once the model controls the game, its images can diverge from the human recording; use autonomous success and attempt videos to judge those runs.

## 4. Diagnose wall-facing and mouse failures

For a binned-mouse checkpoint, annotate visually confirmed wall-facing ranges in
the expert comparison. `action_comparison.csv` provides episode names and frame
indices; the annotation uses inclusive ranges:

```csv
episode,start_frame,end_frame
episode_YYYYMMDD_HHMMSS_000000,42,58
```

Then run:

```bash
.venv/bin/python analyze_mouse_validation.py \
  model_attempts/benchmarks/candidate/action_comparison.csv \
  --wall-ranges model_attempts/benchmarks/candidate/wall_ranges.csv
```

The report compares mouse-bin entropy, peak separation, opposing-direction
peaks, and jump false positives inside and outside the annotated ranges. The
ranges must come from video inspection; the dataset has no automatic wall
label.

For a live failure, save video and diagnostics, note every wall-facing time
window, and analyze the corresponding ticks:

```bash
.venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --map dataset_test9 --max-seconds 30 --record-video --verbose

.venv/bin/python analyze_mouse_rollout.py \
  model_attempts/attempt_TIMESTAMP.jsonl \
  --wall-window 12:18 --wall-window 23:27
```

Opposing peaks may indicate conflicting demonstrations, while a flat
distribution may indicate low confidence. Neither result proves a cause by
itself; interpret it alongside the video, expert-frame metrics, visual motion,
and focus diagnostics.

The runner also writes explicit `diagnostic` records when one of these applied-
action patterns lasts at least one second:

- `policy_freeze`: no movement, interaction, firing, jumping, or mouse action;
- `wall_stuck`: movement is commanded while sparse-frame visual delta remains
  at or below `0.05`; or
- `turn_loop`: movement is combined with horizontal mouse output of at least
  `45` in the same direction.

Each record has a `start` or `end` phase, and the final summary counts detected
starts. These are triage signals, not proof of a cause; confirm them against
the matching video.

To isolate whether extreme mouse bins cause circular failures, compare the
unchanged baseline against the same sequence with a cap:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint v5_best=models/imitation/checkpoints/runs/v5_holdout_11_12/best.pt \
  --map dataset_test1 --map dataset_test10 --map dataset_test11 \
  --map dataset_test12 --map evaluation2 \
  --repeats 3 --keep-focused --mouse-max-abs 28
```

The cap is opt-in and does not modify the checkpoint. Every tick logs the raw
model output as `raw_action`, the capped action sent to the controller as
`action`, and whether they differ as `mouse_limited`. Keep all other settings
and chamber order fixed when comparing it with the uncapped baseline.

On 2026-10-06, an experimental `74 × tanh(raw/74)` response was compared with
the uncapped balanced candidate over three repeats of `dataset_test1`,
`dataset_test11`, and `dataset_test12`. It reduced steering too aggressively in
closed-loop use: zero-mouse raw predictions rose from 84.0% to 98.3%, freeze
events rose from 7 to 25, `dataset_test11` fell from 1/3 to 0/3 successes, and
`dataset_test12` fell from 3/3 to 2/3. The experiment was rejected and its CLI
option was not retained. Use uncapped behavior for future recovery-data work.

## 5. Calibrate jump and forward decisions

Sweep jump thresholds only on held-out expert predictions:

```bash
.venv/bin/python benchmark_jump.py \
  model_attempts/benchmarks/candidate/action_comparison.csv
```

Choose a precision/recall trade-off before live evaluation, then pass that
threshold consistently to `run_imitation.py` or `run_model_sequence.py` with
`--jump-threshold`. Thresholds are checkpoint-specific; do not promote one from
a different model merely because it performed well there.

If a policy remains idle despite expert-frame evidence that forward movement is
correct, use `--move-w-threshold FLOAT` for a short calibration run. Change one
decision rule at a time and leave mouse decoding and other thresholds fixed.
Threshold overrides test calibration; they do not retrain the model or prove
that it can complete a chamber. Confirm any candidate with repeated autonomous
runs.

## 6. Record targeted recovery demonstrations

Use recovery recording only after calibration shows a genuinely unfamiliar
policy-induced state. Stop a model attempt while Portal 2 is still at the
failure state; the runner releases held controls during cleanup. Without
reloading the chamber, record a human recovery:

```bash
.venv/bin/python record_recovery.py \
  --map dataset_test9 \
  --source-attempt model_attempts/attempt_TIMESTAMP.jsonl \
  --max-seconds 30
```

For a supervised collection session, the handoff and chamber cycling can be
automated instead:

```bash
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_balanced_recovery_v1/best.pt \
  --map dataset_test1 --map dataset_test5 --map dataset_test9 \
  --target-corrections 30
```

The session loads a training chamber and runs the model. Press **P** at a useful
failure: the model releases its controls and the recovery recorder starts at
the *same position*. On the current Gamescope setup, the handoff briefly
switches to an empty Hyprland workspace and back to refresh the mouse grab,
then gives a five-second focus wait. In automatic-focus mode, sweep the physical
mouse toward the other monitor and back during that wait, then return the view
within about 8 degrees of its starting angle. The runner checks cursor position,
Portal focus, and camera response; a detected escape or large view shift retries
the same chamber without writing a correction. Manual-focus mode lets you align
the view at your own pace and checks that physical mouse movement turns the
camera when you press P to start the clip. If confinement looks wrong once
recording starts, press **R** to reject that correction; its partial clip is
kept under `episodes/recovery/discarded/` for review, not training. Reach the
goal to save a successful correction and automatically start the model on the
next chamber.
The five-second handoff also logs physical mouse counts, game angles, and
Hyprland cursor/focus to `handoff_diagnostics.jsonl` in the attempt folder.
If substantial mouse movement produces almost no camera turn, the runner
reacquires input once and then retries the chamber without a clip.
If automated focus verification blocks collection, run the session with
`--manual-focus`: after the first P handoff, check confinement and re-align the
view at your own pace, then press P again to begin the recovery MP4 and actions
CSV. No pre-start mouse movement is included in that recovery clip. This mode
does not automatically reject an escaped pointer, though it rejects clearly
ineffective physical mouse movement; use R if the take is invalid.
For a feature check without filling the dataset with unusable clips, add
`--test-mode` (optionally alongside `--manual-focus`). The policy still sees
live frames, but no model or recovery MP4/CSV is written. Test diagnostics
go under `model_attempts/recovery_tests/` instead of the normal session folder.
The Gamescope workspace-refresh handoff passed a live no-save test on
2026-10-13, followed by one successful saved pilot on `dataset_test1`. A later
session on 2026-10-18 saved 30 additional corrections across
`dataset_test1`–`dataset_test10`; all 30 have been preprocessed into the
recovery cache. Check source/cache coverage with:

```bash
.venv/bin/python dataset_stats.py episodes/recovery/goal_reached \
  --cache-dir data/datasets/recovery_cached_frames
```

The expanded cache does not change the frozen v1 split. Generate and review a
new split manifest before using these corrections in training. See
[`handoff.md`](../handoff.md) for recording paths and current status.
For a real correction run without the preceding model-attempt MP4, use
`--no-model-video`; this helps isolate whether model-video encoding affects
the handoff while keeping recovery MP4/CSV collection enabled.
**R** restarts the current chamber from a new model attempt, whether pressed
during model control or human correction. **K** stops the whole session;
during correction it saves the partial clip under `interrupted`. Esc remains
the model runner's emergency stop, not the handoff key. During recovery recording,
Esc pauses/resumes; capture waits three seconds after the resume press so the
Portal 2 pause menu is not written to the clip. P is read from the
physical keyboard and may also reach Portal 2, so ensure it has no in-game
binding. The runner temporarily unbinds Portal 2's R key and restores its
original binding at the end of each phase.
The `wall_stuck` diagnostic remains a hint to the human, not an automatic
handoff trigger. If the model reaches the goal before takeover, the same
chamber is reloaded for another attempt; no recovery is recorded or counted.
Model time limits also retry the same chamber. Session logs and model videos
are saved under `model_attempts/recovery_sessions/`; corrections remain under
`episodes/recovery/`. The command accepts only `dataset_test1`–`dataset_test10`
because validation/evaluation chambers must stay out of recovery training.
The model keeps its configured `--model-seconds` limit (60 seconds by default),
regardless of the chamber's built-in timer. The session requires the updated
`shionn_events.nut` installed in Portal 2: it disables the chamber timeout
with an acknowledged command before running the model and rechecks the guard
before human recording. The correction has no local code time limit by default
(`--recovery-seconds 0`); K and genuine goal/failure events still end it.
An unexpected map reload invalidates the clip. If timeout suppression is not acknowledged, the session
stops rather than collecting a potentially invalid correction.

Keep successful corrections separate from the base dataset and its validation
split:

```bash
.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes/recovery/goal_reached \
  --cache-dir data/datasets/recovery_cached_frames

.venv/bin/python -m models.imitation.train_bc \
  --cache-dir data/datasets/cached_frames \
  --train-extra-cache-dir data/datasets/recovery_cached_frames \
  --correction-sampling-fraction 0.10 \
  --start-window-frames 8 --start-sampling-boost 5 \
  --checkpoint-dir models/imitation/checkpoints/runs/candidate_recovery
```

Extra caches are added to training only after the original episode split, so
they do not contaminate the preserved base validation set. Use a new checkpoint
directory, review every correction video, and include several distinct
recoveries before assigning a large sampling fraction. After preprocessing new
corrections, generate and review a *new* split manifest as described in
[DATA_SPLIT.md](DATA_SPLIT.md), then pass it with `--split-manifest` when
training. The original frozen v1 manifest must not be silently overwritten.

## 7. Record proactive-turn corrections

Use this mode for a useful turn that begins **before** the agent would touch a
wall. It is distinct from #4-style recovery after the policy is already
stuck. The recorder loads a training chamber, then waits without writing data
while you manually move to the chosen approach point. Press physical **P** to
start aligned video/action capture from that position. Reach the goal to save a
successful clip; failures are kept under their outcome and retried. P must be
unbound in Portal 2. Only `dataset_test1`–`dataset_test10` are accepted, to
protect the frozen validation chambers.

```bash
.venv/bin/python record_dataset.py --proactive-turn --turn-direction left --map dataset_test1 --map dataset_test2 --map dataset_test3 --map dataset_test4 --map dataset_test5 --episodes 2 --duration 60
```

Repeat for `--turn-direction right`. This example targets 20 successful clips
total (10 per direction, two on each of five maps). Across attempts, vary how
far from the turn you begin and the approach angle; steer gradually and add
lateral movement when it supports the path. Inspect every MP4/CSV pair and
metadata before caching. Keep these files under
`episodes/corrections/proactive_turns/` and use a distinct proactive-turn cache;
do not combine them with the policy-induced recovery cache. Before training,
extend the split generator to register this additional cache group in a new
manifest revision, preserving v1 and the existing validation set.

## Acceptance criteria

A candidate is ready for promotion only when:

- offline reports use data excluded from training;
- live trials use clean, training-compatible frames;
- runner errors and manual cancellations are marked invalid rather than failed;
- each evaluation chamber has repeated autonomous attempts;
- videos and diagnostics support the reported outcomes; and
- evaluation recordings remain isolated from training data.

The [CLI reference](CLI_REFERENCE.md) documents every benchmark, calibration,
and recovery option.
