# Startup preservation during route correction — 2026-11-09

## Evidence

The completed route-expansion pair improved its new TRAIN examples but failed the unchanged opening-idle guard: source175 idle predictions among1600 first-eight frames, original-correction control193, expanded192 (limit191). Both candidates acquired the same12 newly idle first-active episode predictions, all with expert/source W=1. No gameplay improvement is established; deployed remains unchanged.

Reproducing the actual8000 shared base draws by hash found42 first-active targets and334 first-eight targets from284 of1100 original TRAIN episodes.816 episodes supplied no first-eight target. This is limited exposure, not proof of causation.

A predeclared TRAIN calibration picks ten episodes/map by SHA256 ranking of names, with first-eight active frames (800 total). It excludes all validation episodes. Read-only CUDA inference found source47 false-idle frames versus51 for each candidate, and20 versus24 missed expert W labels. The same four new W misses occur in both candidates, with source W probabilities0.526–0.552 falling below0.5. No labels, thresholds or weights were changed. These checks concern active-window openings, which are not necessarily spawn views or proof of sustained gameplay freezes.

Reproduce exposure: `.venv/bin/python analyze_route_openings.py`; add `--inference` to reproduce the fixed TRAIN calibration on CUDA. This performs no optimization/game control. Results are in the `route_expansion_train_opening_*_20261109.json` annotations.

## Single-factor experiment

`run_opening_rehearsal.py` is separate from existing pinned runners. It reuses the complete expanded-correction arm from `route_expansion_v1/report.json` as control only after matching the full parent plan except mode/time budget, successful1000 updates, complete evaluation and endpoint hash. Source/data/code/draw hashes remain enforced.

The new arm starts fresh from deployed, NOT the control endpoint: same architecture, source bins/losses, seed0, float32, fresh AdamW LR1e-5, batch10,1000 updates and augmentation stream. Replace only slot7 in every batch with an original TRAIN first-eight-frame example. Keep all other slots identical:7000 original base targets,1000 opening targets and2000 unchanged expanded-human targets. No validation/held-out frames enter gradients. Normal four-frame history is preserved. No GRU, gating, backward rehearsal, threshold changes or automatic retries.

The8800-frame opening pool is frame-uniform with deterministic seed4 draws, not equal-map weighting. The validated replacement schedule covers678 episodes; map exposure is recorded in the plan (test1 has177 draws, other maps77–107). Existing original base draws still contain some openings. The opening intervention changes the base distribution, so success would support this sampling intervention rather than establish a particular causal network mechanism.

## Gates and interpretation

Evaluate unchanged full39769-frame navigation, original/expanded TRAIN correction diagnostics and272 development-held-out frames. Preserve source-relative guards: binary exact drop<=0.005, mouse-axis MAE ratio<=1.05, first-eight idle increase<=0.01 (maximum191). Do not relax the failed guard.

Before considering gameplay, require the navigation guard, no additional development false backward predictions versus expanded control (2), at least its9/40 correct backward targets, and no drop in its held-out episode-mean binary exact (0.6838632922). Report all metric changes rather than treating one passing number as recovery evidence. Review TRAIN route and exit steering too. These are development diagnostics, not untouched final testing; satisfying them alone cannot establish chamber completion or justify promotion.

If eligible, use the same user-run timing-v2 scored comparison described in `ROUTE_EXPANSION_EXPERIMENT_2026_11_09.md`, with unchanged source versus candidate and explicit repeats/order. If the startup intervention still fails to transfer recovery, stop extending this sampling family and review task/trajectory coverage before another training job. No general recurrent experiment begins automatically.

## Execution

Plan-only default (no training):

```sh
.venv/bin/python run_opening_rehearsal.py
```

User-run execution, estimated6–9 minutes,10-minute internal budget and12-minute external cap:

```sh
systemd-run --user --unit=shionn-opening-rehearsal-v1 --collect --same-dir \
  --property=RuntimeMaxSec=12min --property=LimitNOFILE=8192 \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python \
  run_opening_rehearsal.py --execute
```

Follow with `journalctl --user -u shionn-opening-rehearsal-v1 -f`; stop with `systemctl --user stop shionn-opening-rehearsal-v1`. Portal and a graphical session are unnecessary. The estimate is based on earlier single-arm probes, not a guarantee.

Output: `models/imitation/checkpoints/runs/opening_rehearsal_v1`. No overwrite/resume. Use an endpoint only with report `comparison_complete=true`; exception paths roll endpoint names back to pending, while abrupt process termination still requires report verification. Generic optimizer resume is disabled. Existing source, control and raw data remain unchanged.

## Verification

Actual plan-only preparation passed, current input hashes rechecked, no new training output present. Saved plan: `data/annotations/opening_rehearsal_validated_plan_v1.json`. Full suite330 tests passed in19.387s; eight focused tests cover exact changed slots, reference provenance, TRAIN/boundary constraints, checkpoint metadata, default no-training behavior and publication-failure rollback. Independent read-only falsification passed. No real training or gameplay was launched by this preparation.

## Completed result and predeclared gameplay screen

User-run rehearsal completed1000 updates in285.236s. Endpoint SHA256 `3d735bdb8eba1ffffdbb374d87d29f4159ef9e166e39045cab1d30ebdccbeef4`, complete report/source/control/input hashes verified. All navigation guards pass: first-eight idle185/1600 versus source175 and expanded control192. Held-out binary exact remains0.6838632922, backward9/40 and false backward2, exactly matching control; therefore all additional development checks also pass. New route TRAIN binary exact rises from79.90% to80.41%; horizontal mouse MAE10.63 to10.45. New exit binary exact stays47.37%. These results establish eligibility for a gameplay screen, not improved recovery or completion.

Stage1 is a fixed18-job map-first sequence: deployed then opening candidate on test1/test11/test12, repeated three times,60s maximum each,3s countdown and post-preflight settle, DP-1, persistent mouse and all existing safeguards. `run_model_sequence.py --plan-only` validated this exact matrix without gameplay. Save under `model_attempts/sequences/opening_rehearsal_screen_20261109`.

After the complete stage, inspect all valid attempts and invalid capture records. Candidate must improve test1 completion over the contemporaneous source and preserve test12's3/3 completion, without increased long startup freezes or wall stalls. Test11 remains validation-only, reported in full. If the source's expected test12 behavior fails, investigate setup/variance before inferring candidate regression or promotion. If candidate fails this screen, stop and retain deployed; no selective rerun. If promising, run a separately recorded18-job confirmation matrix with candidate FIRST and deployed SECOND, same maps/repeats/settings, before any promotion decision. Combined stages have balanced model order. Review route return and sustained exit progress in videos, not just aggregate success or offline action agreement.

Game launch/control remains user-only. The agent did not detect a Portal process during the preparation check; run Portal first and verify desktop capture/console readiness before executing the game sequence. Remote shell commands should explicitly inherit the home display environment (XDG_RUNTIME_DIR=/run/user/1000, WAYLAND_DISPLAY=wayland-1, DISPLAY=:1 on the currently inspected host); never disable mouse preflight to get past a setup failure.

## Stage1 gameplay reviewed

Sequence `opening_rehearsal_screen_20261109/sequence_20261109_194633_332161` completed18/18 attempts. All passed mouse preflight, timing-v2 three-second settling, focus/cursor/injection checks and video/tick-count alignment. No capture failure was excluded. Review manifest: `data/annotations/opening_rehearsal_gameplay_screen_20261109.json` with raw log/video hashes, idle intervals and visual notes.

| Map | Deployed | Opening rehearsal |
| --- | ---: | ---: |
| test1 |0/3|1/3,20.87s goal|
| test11 |0/3|0/3|
| test12 |3/3,median6.19s|3/3,median6.16s|

The candidate's successful test1 run (job14) was additionally sampled every1s: wall approach around6–8s, reorientation by9s, another wall visit around13–15s, return toward stairs/exit17–20s and entry to the exit corridor by20.87s. Console goal confirms completion. Other test1 candidate runs still wander or become idle; test11 still fails. The failed test1 job8 has the exit visible around15s but does not maintain progress to it.

On test1/test11, long all-zero-action intervals (>=1s) occur in5/6 deployed attempts versus2/6 candidate attempts, totaling154.98s versus40.20s. These later policy-inactivity intervals are not proof of improved startup behavior or a physical wall-stall count. Automated wall-stuck counts are zero in both arms despite visible wall approaches; do not use that heuristic alone to assert no wall problems.

Stage1 meets the predeclared completion/regression screen sufficiently for confirmation: candidate test1 advantage and preserved test12, with fewer long inactivity intervals. A single extra success does not establish a reliable improvement or isolate the effects of correction data versus startup rehearsal. No promotion. Next is the already specified18-job candidate-FIRST confirmation under identical settings, saved separately at `model_attempts/sequences/opening_rehearsal_confirmation_20261109`. Plan-only matrix validation passed. The user executes gameplay; the agent did not launch/control it. All scored autonomous trajectories remain excluded from expert correction targets.

## Confirmation and combined decision

Candidate-first confirmation `opening_rehearsal_confirmation_20261109/sequence_20261109_201622_516140` completed all18 attempts with preflight, timing-v2 settling, focus/cursor/injection and video alignment checks passing. Candidate test1 was0/3 (the initial goal did not repeat); candidate test11 was1/3 with19.29s completion; both models retained test12 3/3. Deployed test1/test11 were0/3. The successful test11 trajectory was sampled every1s and shows departure from a dark-wall view, reorientation toward the exit and corridor entry; console goal agrees. This scored validation trajectory is NOT expert correction data.

Across both balanced-order stages (36 valid attempts, no invalid captures):

| Map | Deployed | Candidate |
| --- | ---: | ---: |
| test1 |0/6|1/6|
| test11 |0/6|1/6|
| test12 |6/6|6/6|

Long all-zero-action intervals occur in11/12 deployed runs on test1/test11 versus5/12 candidate runs, totaling375.89s versus143.98s. This is descriptive evidence of fewer later inactive states in this cohort, not a causal or statistical proof of reliable navigation and not a startup-freeze result. Continuing actions can still wander or hit walls. Sampled confirmation test1 failures revisit walls/stairs/unmarked-door views without sustained exit progress.

Decision: retain deployed unchanged, keep the candidate isolated as experimental, and stop extending this sampling family. No promotion or further confirmation runs: the predeclared reliable test1 improvement is not established. Correction data and opening rehearsal have not produced reliable chamber completion. Keep these two stages separate from older timing protocols and do not retrospectively pool them into a favorable historical comparison.

Machine-readable confirmation and combined reviews: `data/annotations/opening_rehearsal_gameplay_confirmation_20261109.json` and `opening_rehearsal_combined_gameplay_20261109.json`. Next work is an offline trajectory-coverage review and a separate bounded GRU comparison design, keeping data and architecture effects distinguishable. No real training or gameplay is automatically started.
