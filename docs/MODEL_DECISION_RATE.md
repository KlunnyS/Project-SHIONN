# Model decision rate experiment

`run_imitation.py` and `run_model_sequence.py` accept optional `--model-fps`.
Omitting it preserves the existing path. With `--fps 24 --model-fps 12`, the
policy observes every control tick and predicts every second tick; `8` predicts
every third tick. The model rate must be a finite positive integer divisor of
the control rate. Rates are tick based, so actual wall-clock throughput can be
lower if capture, inference or input submission stalls. No catch-up predictions
are made after a stall.

Keyboard/button states and mouse deltas repeat on held ticks. A mouse delta
continues to mean displacement per control tick: a constant prediction produces
the same total camera displacement at 24,12,8 decisions per second. Short button
predictions may consequently last longer. Focus checks, stop controls, telemetry,
capture, video writing and input submission retain their existing control rate.
Missing observations do not advance the decision schedule. Every existing policy
reset also discards held actions and restarts with a fresh decision.

Frame history is updated at control rate, including observations between model
decisions. GRU hidden states update only on decision ticks; that changes recurrent
temporal semantics. Start with the deployed feedforward checkpoint. No dataset,
checkpoint format, loss or training-label change is part of this inference-only
experiment. It measures the effect of holding an existing policy's commands;
it does not establish that training at a lower decision rate improves learning.

Attempt metadata records effective model FPS, interval and mouse-repeat units.
On reduced-rate runs each tick's policy diagnostics identify whether a fresh
prediction was made and its source observation. Cached probabilities refer to
that source, not the newest frame. Held ticks have null neural-inference timing
fields; their action-submission timings are still recorded.

## Controlled gameplay comparison

Keep capture/input at24Hz and the established scored protocol unchanged: deployed
checkpoint, maps test1/test11/test12, persistent mouse, preflight,3s post-preflight
settle,60s attempts. Use fresh recording roots for each rate. Do not pool these
runs with training data or previous timing protocols. Test11 remains validation;
test12 is the regression check. Run the same sequence with `RATE` set to24,12,8
and distinct `RATE` recording roots (these are placeholders to replace):

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint deployed=models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 --map dataset_test11 --map dataset_test12 \
  --repeats 3 --max-seconds 60 --countdown 3 \
  --fps 24 --model-fps RATE --post-preflight-settle 3 \
  --output DP-1 --persistent-mouse --keep-focused \
  --hyprland-confine-pointer --no-launch --verbose \
  --recording-root model_attempts/sequences/model_fps_RATE
```

Game launch and these live input runs are user-operated. Review capture/input
validity first, then completion, timeouts, completion time, startup inactivity,
wall stalls, wandering, camera overshoot and actual decision timing. A successful
screen needs balanced run order and fresh repetitions before drawing conclusions.
Do not promote a rate on smoother video or offline action agreement alone.
Generalization requires separate unseen scenarios and cannot be inferred from
this initial three-map screen. Roll back by omitting `--model-fps`.

## Engineering validation (2026-11-11)

- Focused adapter, sequence and mocked runner tests:45 passed.
- Complete regression suite:363 passed (21.362s).
- Read-only independent verifier passed the rate, default, reset, CLI and
  main-loop logging contracts. No live desktop input was used.
- Actual deployed-checkpoint CPU smoke with24 seeded synthetic observations:
  24/12/8 decisions took1.347/0.673/0.458 seconds respectively (two CPU threads).
  These are single synthetic batch-of-observations checks, not live FPS or
  stable latency benchmarks. The24Hz adapter produced exactly the original
  action sequence after reset. Lower rates produced the expected decision counts.
- Gameplay completion, live scheduling/latency and generalization are untested.


## Read-only timing audit

After a live attempt, run `.venv/bin/python audit_model_fps.py PATH_TO_ATTEMPT.jsonl`.
This reports configured and observed control/decision rates separately, fresh and
held counts, median neural inference time, and held ticks incorrectly claiming
fresh inference. Rates use intervals between logged ticks, so short runs and
focus resets can affect the estimate. It does not establish capture validity or
behavioral improvement. Empty attempts report unknown rates rather than zero.
The audit and rate adapter focused suite passed7 tests; the previous full363-test
suite predates the two audit tests.

## First live 12 Hz screen (2026-11-11)

Source: `model_attempts/sequences/model_fps_12_screen/sequence_20261111_113855_611196`.
All three attempts passed existing capture/input checks: preflight passed, no
focus loss/cursor exits/injection failure, guard cleanup restored. Actual logged
control/decision rates were24.008/12.004Hz (test1),24.005/12.001Hz (test11), and
23.963/11.982Hz (test12). Fresh/held counts were707/706,720/720,75/74. No held tick
claimed neural inference and every logged action was applied.

Test1 and test11 timed out at60s. Test12 reached the goal in6.192s of policy time.
This verifies live rate integration; one attempt per map does not establish
improved navigation or generalization. Automated diagnostics flagged one inactivity
event on each timeout and three turn-loop events on test1; these are heuristics,
not a visual diagnosis of wall contact. No video behavior review is claimed here.
Next collect the same three-map screen at24Hz, then8Hz, with fresh roots; use
balanced repeated runs only if the initial screens justify further comparison.

## Matched 24 Hz screen (2026-11-12)

Source: `model_attempts/sequences/model_fps_24_screen/sequence_20261112_132348_971872`.
All three attempts passed capture/input checks, with no focus losses, cursor
exits or injection failures. Observed control and decision rates were24.011,
24.004,23.990Hz for test1/test11/test12. Outcomes matched the first12Hz screen:
test1 and test11 timed out; test12 completed in6.151s versus6.192s at12Hz.
These single attempts, collected on different days, do not demonstrate a
performance difference. Earlier24Hz sequence20261111_115053_628078 failed
preflight with zero policy ticks and is excluded. Next screen:8Hz decisions
with24Hz capture/input, same checkpoint, maps and safeguards.

## Completed 8 Hz integration screen (2026-11-12)

Source: `model_attempts/sequences/model_fps_8_screen/sequence_20261112_132817_952610`.
All three attempts passed the existing capture/input checks, with no focus losses,
cursor exits or injection failures. Control/decision rates were24.013/8.004Hz,
24.005/8.001Hz,24.034/8.011Hz for test1/test11/test12. Fresh/held counts were
478/956,480/960,50/98; no held tick claimed fresh neural inference and all logged
actions were applied. Test1 and test11 timed out; test12 completed in6.148s.

The integration has now been exercised live at24,12,8Hz while retaining24Hz
control. All three screens have the same chamber outcomes: timeout/timeout/goal.
Single attempts per map/rate, including rates collected on different days, cannot
establish an improvement or statistical equivalence. Retain24Hz as the default;
12/8Hz remain optional experimental settings. No performance promotion is
supported and no additional repetitions are required merely to validate the
integration. Visual smoothness, route recovery and generalization were not
established by the log audit. Training a policy with longer action targets would
be a separate experiment requiring an explicit data/action-duration design.
