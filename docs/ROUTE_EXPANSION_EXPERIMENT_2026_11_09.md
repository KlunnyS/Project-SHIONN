# Route correction expansion — 2026-11-09

## Evidence and scope

The new human collection provides one substantive test9 wall-to-route-to-exit trajectory (194 usable frames) and two short test1 exit-finalization trajectories (31 and26 frames). Structural timing/action checks and sampled visual review passed; the roughly10–11s handoff transition remains unrecorded. See `data/annotations/route_collection_review_20261109.json`. Raw files and that review remain unchanged.

The experiment split explicitly adds these251 frames to the original four TRAIN corrections (550 frames), making801 frames. The original two development holdouts remain272 frames and never enter gradients. These holdouts have already informed diagnosis. Adding dark-wall TRAIN views changes the question from disjoint-view transfer to within-view episode transfer; this is not a new untouched test. No test11 or test12 demonstration enters training.

## Bounded comparison

`run_route_expansion.py` is separate from the previous pinned runner. Both arms start from the unchanged deployed feedforward checkpoint, with original mouse bins and losses, gating/memory disabled, fresh AdamW LR1e-5, seed0, float32, batch10 and at most1000 successful updates per arm.

- Control: eight original base targets plus two original human-correction targets.
- Expanded: the same eight base targets plus two targets from the expanded human pool.

Human targets are balanced over usable frames, not equally over episodes. Consequently the two short exit clips have less weight than the longer route correction. Original history frames, including leading waiting context, are preserved; only reviewed active windows supply targets. Episodes and the handoff gap are never joined. The plan records input/code/cache/draw hashes, per-episode human exposure and backward-target counts. Both arms train from the source, not from earlier fine-tuned candidates.

This tests replacing part of the old correction exposure with the expanded pool at a fixed20% share. It does not isolate each new episode's effect or guarantee navigation improvement. Neither arm has backward rehearsal enabled.

## Validation and acceptance

Before and after each arm, score all39769 navigation validation frames, original corrections, all expanded corrections and the unchanged development holdouts. New route/exit categories are reported separately as TRAIN diagnostics. Existing navigation guard: binary exact accuracy drop at most0.005, mouse MAE ratio at most1.05, and initial-eight-frame idle fraction increase at most0.01 versus source.

A complete matched pair and passing navigation guard are required before considering gameplay. Review route/exit diagnostics and development holdouts for steering, backward false positives and idle regressions; these cannot establish chamber completion. Do not tune thresholds on the held-out failures to claim success.

If an offline candidate merits gameplay, the user runs a separately declared timing-v2 comparison against deployed on test1, test11 (validation only), and test12 (regression), three repeats each,60s, mouse preflight,3s post-preflight settle, persistent mouse and identical capture settings. Compare candidate/source order in separate counterbalanced sequences. Report all valid attempts, capture failures, completion, timeouts, freezes, wall stalls and reviewed return-to-route intervals. Require better test1 completion across repeated attempts and preserve test12 completion; report test11 results without using them as demonstration targets. No automatic promotion or claim from offline accuracy alone.

## Commands and cost

Plan-only, no optimizer or gameplay:

```sh
.venv/bin/python run_route_expansion.py
```

User-run training after plan review; estimated12–18 minutes total with an18-minute internal soft budget and20-minute external hard cap. The estimate comes from the previous paired experiment, not a fresh speed benchmark. Preparation and evaluation count toward the soft budget. An incomplete run is retained as incomplete; never automatically retry or resume. Choose a new output path for a deliberately authorized repeat.

```sh
systemd-run --user --unit=shionn-route-expansion-v1 --collect --same-dir \
  --property=RuntimeMaxSec=20min --property=LimitNOFILE=8192 \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python run_route_expansion.py --execute
```

Follow output:

```sh
journalctl --user -u shionn-route-expansion-v1 -f
```

Stop:

```sh
systemctl --user stop shionn-route-expansion-v1
```

The default output is `models/imitation/checkpoints/runs/route_expansion_v1`. Endpoints are published only after both arms finish and final hashes pass; a report is authoritative only when `paired_complete` is true. Existing output is never overwritten. Generic optimizer resume is disabled in endpoint metadata. Training does not require Portal or a Wayland display and does not control gameplay.

## Subsequent memory comparison

Only after reviewing this data-expansion result: freeze the same episode memberships and compare feedforward versus GRU, gating off, matching supervised target-frame budgets and reporting compute separately. Preserve complete human trajectories and recurrent reset/burn-in boundaries. Do not conflate added-data effects with recurrent architecture effects. No general recurrent training is started by this increment.

## Engineering validation

Plan-only preparation against actual files passed and its current input hashes were rechecked; saved as `data/annotations/route_expansion_validated_plan_v1.json`. No training output directory existed at verification. All322 unittest tests passed in23.032s; eleven focused tests include distinct human-pool selection, validation-map rejection, metadata, source preservation and publication-failure rollback. Independent read-only review passed after the publication fix. A hard process kill can still interrupt filesystem publication: always require a complete paired report before consuming either endpoint. No real optimizer run or gameplay evaluation was performed for this experiment.

The first user launch failed during dataset initialization because the user service soft open-file limit was1024. No training output directory was created. The launch command now explicitly sets LimitNOFILE=8192 for the memory-mapped episode arrays; the manager hard limit is524288. This changes only the transient service limit, not system defaults or model/training code.

## Completed user run and decision

The corrected service launch (LimitNOFILE8192) completed both1000-update arms in726.567s; report `models/imitation/checkpoints/runs/route_expansion_v1/report.json`, paired_complete=true, endpoint hashes verified. The initial launch failed before creating output. No model was promoted and no gameplay evaluation was performed.

| Metric | Deployed source | Original-correction control | Expanded corrections |
| --- | ---: | ---: | ---: |
| Navigation binary exact |84.60%|85.00%|84.91%|
| Opening idle predictions /1600 |175|193|192|
| Development holdout episode-mean binary exact |69.47%|69.11%|68.39%|
| Development backward correct /40 |8|10|9|
| New route TRAIN binary exact |78.87%|78.87%|79.90%|
| New route TRAIN horizontal mouse MAE |11.72|11.44|10.63|
| New exit TRAIN binary exact |36.84%|43.86%|47.37%|

Both arms fail the opening-idle guard (maximum191), despite passing other navigation checks. Improvements on newly trained clips are fitting evidence, not generalization or chamber completion. The expanded arm did not improve held-out recovery relative to control.

The existing per-episode first-active-frame records identify exactly the SAME12 newly idle episode starts in both arms; none of the source's43 idle starts were corrected, making55 idle starts for each. All12 new misses have expert W=1 and source W=1 with no other action. This is a real first-frame imitation regression, not merely extra correct idle labels. It suggests a shared fine-tuning effect rather than a regression uniquely introduced by the new clips; it does not identify a causal layer/loss or prove sustained in-game freezes. Full first-eight-frame identities and action probabilities were not recorded by this evaluator.

Next: inspect TRAIN opening coverage and model confidence on a predeclared TRAIN-only calibration cohort before proposing a bounded startup-preserving correction comparison. Do not lower the validation guard, alter mouse/control thresholds or train on the twelve validation failures. Keep deployed unchanged. Machine-readable result and first-frame diagnostics are in `data/annotations/route_expansion_result_review_20261109.json` and `route_expansion_opening_diagnostic_20261109.json`.
