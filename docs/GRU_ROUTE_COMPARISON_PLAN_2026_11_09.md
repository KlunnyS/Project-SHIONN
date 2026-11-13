# Bounded GRU route comparison — prepared, not launched

## Question and current evidence

Does persistent recurrent state improve off-route recovery and sustained route progress when the visual encoder is allowed to adapt, using the SAME expert trajectories and target exposure as feedforward and reset-memory controls?

This follows the evaluated feedforward correction experiment; it is not the previously proposed general recurrent training run. The deployed model remains unchanged. The earlier frozen-encoder/readout pilot failed steering eligibility, so do not simply rerun it under a new name or claim that its result rules out end-to-end memory. The new experiment changes encoder trainability and ordered state reconstruction; compare matched new arms, not historical runs, to assess it.

## Reuse and integration constraints

Reuse the residual visual encoder, existing GRU512 policy, binary and binned mouse losses, source bins/class weights/smoothing, four-frame stacking, frozen split, reviewed human loader, raw hashes, checkpoint interfaces and inference-owned state/reset logic. Keep observation gating off. No PPO, value head, attention, hierarchy, world model, new action space or online weight updates.

The existing SequenceDataset safely constructs episode-local chunks. Existing run_recurrent_epoch resets hidden state for each independently sampled chunk and only reconstructs configured burn-in. Its default T8/burn4 covers a short interval, not an entire route. The proposed experiment needs an additional experiment-local ORDERED iterator and stateful TBPTT loop; do not silently change that trainer's defaults or reinterpret its checkpoints. The ordinary trainer also counts attempted AMP batches; this comparison must count successful updates explicitly or use float32.

## Fixed data

Original1100 TRAIN base episodes and seven approved human TRAIN corrections (801 active-window frames) are eligible. Preserve complete trajectories and raw preceding four-frame context. Keep original two development-held-out correction episodes272 frames and navigation validation200 episodes39769 frames excluded from gradients. No scored autonomous recording is an expert target. In particular, test11 gameplay success cannot become demonstration data.

Use the exact role/membership in `route_expansion_split_20261109.json`, with its raw hashes and source handoff linkage. Never bridge the unrecorded10–11s transition. The two one-second test1 exit clips retain their exit-finalization category; do not count them as long recoveries. The corpus only contains33.375 nominal seconds of human TRAIN targets. A negative result may be data- or budget-limited; it cannot establish that memory cannot help.

## Matched arms and initialization

1. Feedforward residual CNN, trained on the same ordered chunks.
2. GRU512 with state carried within each episode, detached between chunks.
3. Identical GRU512 reset at EACH timestep, a diagnostic capacity/nonlinearity control.

All arms copy the SAME deployed visual-encoder tensors. Reuse the existing temporal-pilot construction of identical newly initialized512-dimensional action heads, including the feedforward control. Do not compare a fully pretrained feedforward head against an untrained recurrent readout and call the difference a memory effect. Persistent/reset GRU weights start identical. The source with its original trained heads is an untrained reference only.

Unlike the earlier pilot, unfreeze the encoder in ALL three new arms. Proposed optimizer groups: fresh AdamW, pretrained encoder LR1e-5, new heads/memory LR1e-3, source weight decay and clip1; these rates must be pinned before real training. No architecture-specific learning-rate tuning. All use float32, source target bins/class weights/smoothing and no augmentation for the first comparison. Encoder dropout train/eval behavior remains the same across arms. Reset execution RNG after construction and use a private schedule generator so initialization/evaluation cannot perturb target order. Initial outputs need not match because the representations differ; record initial calibration/held-out metrics, not just improvement from each arm's own starting loss.

This comparison isolates memory/capacity effects conditional on one fixed data set and initialization protocol. It does not measure the independent benefit of adding corrections. A later2x2 data/architecture experiment is needed for that claim.

## Ordered schedule and state contract

Proposed supervised segment length32, batch1 episode stream, no leading burn-in: earlier chunks provide the context. Precompute a whole-episode schedule approaching16000 supervised frames per arm, with approximately80/20 original/human target exposure. Balance base chamber exposure and report actual map/episode/category counts. Whole-episode lengths can prevent exactly80/20 exposure or a round nominal target/update budget. All arms must nevertheless have IDENTICAL target IDs, masks and successful update counts for the resulting schedule. Do not cut a human correction before its exit just to hit a round number. Pin the schedule, actual target count, resulting optimizer updates and estimated runtime before execution. If this exceeds the approved budget, revise the plan before training, not during it.

Within each scheduled episode, walk consecutive supervised chunks in order. Initialize hidden=None at episode start. Preserve returned state between chunks; detach it before the next optimizer segment. No graph/gradient crosses the32-target boundary, but detached state contains earlier context. Reset state on episode changes, independent stream starts and evaluation resets. Right padding has no loss and cannot advance state. Feedforward and reset controls consume the same valid targets and masks. Each complete human trajectory remains continuous to the exit. An event more than32 steps back does not receive direct long-range gradient; this is stateful TBPTT, not full-episode backpropagation.

No active hidden state is saved as a deployed checkpoint. Record ordered-state training semantics, sequence length and budget in experiment provenance; recurrent architecture ID stays `shionn_imitation_v5_gru_binned`. Generic optimizer resume is disabled for these experimental endpoints. The reset control is saved only as a diagnostic artifact and must be rejected by PolicyInference.

## Evaluation, gates and stopping

At initialization evaluate the fixed TRAIN calibration and complete human heldouts for all arms. At the fixed endpoint evaluate full navigation and all complete correction trajectories. Recurrent inference carries state across each usable expert episode and resets before the next; compare with reset-at-each-step diagnostics. No independent shuffled-frame evaluation of a persistent GRU.

Report completion is an OFFLINE prerequisite only. Required navigation guards stay source-relative: binary exact drop<=0.005, each mouse-axis MAE ratio<=1.05, first-eight idle fraction increase<=0.01. Also report target loss, idle runs, backward true/false positives, steering direction/magnitude, route/exit TRAIN metrics, complete held-out episode metrics and memory-reset sensitivity. Do not promote from action accuracy or select a convenient intermediate checkpoint. A source-relative guard failure stops gameplay eligibility; an underfit new-head run is inconclusive, not proof against GRU.

Persistent memory must offer a meaningful held-out advantage over BOTH matched feedforward and reset controls without steering/startup regression before considering a user-run gameplay comparison. Predeclared development margin: persistent GRU episode-mean held-out binary exact must exceed BOTH matched controls by at least0.02, meet or exceed source0.6947328575, retain at least8/40 true backward targets and at most2 false backward outputs. Report steering metrics too; this tiny-set margin is an eligibility screen, not statistical significance. Do not change it after seeing the new result. Report this tiny, previously reviewed development set honestly. Keep test1 scored, test11 validation-only and test12 regression in the unchanged timing-v2 gameplay protocol. No more automatic confirmation of the existing feedforward candidate.

If the comparison fails, retain deployed and review data/task coverage before increasing budget or architecture complexity. No automatic retries, extended runs, checkpoint promotion or GRU general-training job.

## Hardware evidence and budget readiness

`benchmark_recurrent.py --device cuda --sequence-length32 --hidden-size512 --iterations10` ran one synthetic squared-logit update per architecture, not a real demonstration optimizer run. Baseline peak allocated2507MiB, GRU2568MiB; step inference2.56/2.41ms, cold update0.439/0.404s. Full report and script hash: `data/annotations/gru_t32_synthetic_feasibility_20261109.json`. It supports model-size feasibility on8GB, not comparative speed, convergence or a full pipeline runtime guarantee. It excludes state-stream scheduling, real losses, data loading, repeated warm updates and full validation overhead.

Before issuing a real training command, run synthetic tests of state carry/detach/masks and an end-to-end plan-only pass, measure a few synthetic real-loss updates and episode-evaluation throughput, and derive a user-visible time estimate and hard ceiling. Expected order of magnitude is tens of minutes for three end-to-end arms plus full validation, not the seconds needed by frozen feature readouts. The bounded command and measured preparation results are now recorded below. Do not silently increase the approved budget or launch automatically.

## Implementation/verification checklist

- Add a separate plan-only runner and ordered episode iterator; preserve existing trainer, dataset, inference and CLI defaults.
- Validate source, original/new review roles, raw data/code/cache and schedule hashes; reject all overlaps with heldouts/maps11/12 and existing outputs.
- Test exact common target exposure, all episode boundaries, carried-state versus one-pass equivalence in eval mode, no cross-chunk gradients, padding invariance, reset control, and real-loss forward/backward passes.
- Test identical initialized heads/encoder across arms, source immutability, accurate current-gradient versus parent-ancestry metadata, diagnostic rejection and checkpoint step inference consistency.
- Preserve failed-run reports and stage outputs. Publish endpoints only after the entire comparison and final hashes complete; roll back partial endpoint names on exceptions and require the completed report even after hard termination.
- Run focused and full regression tests plus independent read-only falsification before asking the user to execute a bounded real job.

Implemented: the separate ordered stateful comparison runner, tests, evidence/trajectory audit and synthetic profiling. NOT performed: real end-to-end comparison training or demonstrated GRU gameplay improvement.


## Implemented preparation and exact exposure

`run_ordered_gru_comparison.py` now implements this plan as a separate experiment; the normal trainer, sequence dataset, policy/inference interfaces and earlier runners are unchanged. Actual plan-only preparation against the reviewed files passed. Saved plan: `data/annotations/ordered_gru_route_validated_plan_v1.json`; source/data/code/schedule/target-trace hashes were rechecked afterward. No real output directory existed at verification.

Each arm receives16090 supervised frames over545 successful float32 updates and90 whole-episode occurrences. These comprise62 distinct original base episodes12886 targets, plus seven human corrections sampled four complete times each3204 targets (19.91%). Category exposure:2200 original recovery,776 new test9 route recovery and228 exit-finalization targets. All arms share the exact schedule, per-chunk target trace, masks and successful update count. Tail masks cause the average target count/update to be below32. The checkpoint records69 episodes in current gradient membership, separately from the full source encoder ancestry; it does not claim that all1100 base episodes were sampled in this small budget.

The encoder is trainable with LR1e-5; new heads/memory use1e-3. No scheduler or augmentation is active. Inherited source extra-cache paths are cleared in endpoint metadata; raw expert inputs remain referenced by pinned reviews/report. Initial heads and source encoder equality are checked before training an arm, with persistent/reset GRU initialization equality checked too. Generic resume remains disabled. Reset control remains a non-policy diagnostic. Exceptions during final rename roll endpoints back; hard termination still requires checking the completed comparison report.

Human corrections begin with zero hidden state at their own recorded episode boundaries. Their preceding autonomous gameplay and unrecorded handoff gap are NOT reconstructed or spliced into memory. This limits claims about recovery from a long autonomous history. Full original base trajectories train ordered state carry; complete human trajectories teach recovery from their independently recorded starting states. This prototype does not repair the capture gap.

## Validation and synthetic profile

All345 regression tests passed in24.175s, including15 focused tests. Coverage includes common target exposure, consecutive chunk/tail bounds, state carry/reset for repeated episode draws, gradient truncation, real-loss synthetic encoder updates, initialization identity, native recurrent equivalence, actual exported-config/public PolicyInference action/state consistency, provenance rejection, source preservation, accurate metadata, diagnostic rejection, complete-pair publication and timeout/rename rollback. Independent read-only falsification passed, including a separate backend-flag restoration failure check.

Canonical synthetic command: `.venv/bin/python benchmark_ordered_gru.py`. It uses scratch weights/random pixels and synthetic W/zero-mouse targets only, four updates/arm with three warm timings, and no data/checkpoint/game writes. Saved report: `data/annotations/ordered_gru_route_synthetic_profile_20261109.json`, including current runner/benchmark hashes. Warm32-frame updates measured approximately0.24–0.26s; normal-path peak allocated GPU memory2.72–2.75GiB and reserved3.32–3.41GiB. Encoder changes/finite gradients were checked. These small samples exclude real data loading/full metrics and cannot guarantee runtime or convergence.

GPU parity checks replay one32-frame synthetic image sequence16 times (512 observations, not512 distinct images). All three arms had zero decoded-action differences versus native single-step/FF inference in both tested precision modes. Runtime-default TF32 produced maximum logit differences up to0.000329; the separate TF32-disabled numerical check passed rtol1e-4/atol1e-5 with max logit errors about1e-6 and persistent hidden error4.4e-7. The verification restores both precision flags in finally, including on failure; production inference/math defaults are unchanged. These tests establish the tested stream, not universal parity for arbitrary trained checkpoints.

The stricter IEEE-only diagnostic emitted recoverable cuDNN workspace-allocation retry warnings before completing. The reported training-memory peaks were measured BEFORE that diagnostic and do not describe its maximum workspace. Normal-path synthetic training completed; a real-data allocation or budget failure must still be retained as incomplete, never automatically retried.

## User-run bounded execution

Estimated12–20minutes for all three arms and complete evaluation, based on the above model timings plus data/evaluation overhead;24-minute internal budget and30-minute outer ceiling. This is a controlled end-to-end memory experiment with matching newly initialized heads, not a continuation of the deployed policy's optimizer or an extended general-training run. Real convergence/runtime remains unverified. Portal and Wayland are unnecessary. From the project directory:

```sh
systemd-run --user --unit=shionn-ordered-gru-route-v1 --collect --same-dir \
  --property=RuntimeMaxSec=30min --property=LimitNOFILE=8192 \
  --property=MemoryHigh=16G --property=MemoryMax=20G \
  --property=MemorySwapMax=2G --setenv=PYTHONUNBUFFERED=1 \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python \
  run_ordered_gru_comparison.py --execute
```

Follow: `journalctl --user -u shionn-ordered-gru-route-v1 -f`.
Stop: `systemctl --user stop shionn-ordered-gru-route-v1`.

Output: `models/imitation/checkpoints/runs/ordered_gru_route_v1/report.json`. Require `comparison_complete=true` before consuming any endpoint. Review the fixed development/navigation gates before gameplay; no automatic promotion. Existing output is never overwritten, and no failed run is automatically extended/resumed. No real training or gameplay was launched during preparation.

## Completed authorized run — 2026-11-09

After the user explicitly asked the agent to run the prepared training command, the agent launched and monitored `shionn-ordered-gru-route-v1`. All three arms completed545 updates on identical16090 targets; report `comparison_complete=true`, elapsed942.378s (15min42s). Service wall time15min45s, peak host memory6.1GiB. Endpoint/diagnostic hashes, source/reference/code/data hashes and realized target traces were rechecked after completion. Encoders changed in all arms, confirming that this was not another frozen-encoder pilot. No gameplay was launched.

| Model | Navigation binary exact | Horizontal mouse MAE | Opening idle /1600 | Held-out episode-mean binary exact | Held-out backward correct /40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deployed reference, original trained heads |84.60%|6.292|175|69.47%|8|
| Matched feedforward, new heads |84.07%|6.726|150|63.27%|4|
| Persistent GRU |83.86%|6.929|90|59.96%|0|
| Reset-GRU control |83.35%|6.871|99|60.71%|2|

All three trained arms fail binary-exact and horizontal steering guards; vertical steering and opening-idle checks pass. Persistent GRU fails the fixed development screen, including improvement over both matched controls, source held-out agreement and backward true positives. Fewer opening idle predictions do not override these regressions. Training peak allocated VRAM2.72/2.75/2.75GiB and actual optimization time156.7/150.3/140.2s respectively; these are this run's timings, not proof of architecture speed advantage.

Decision: retain deployed unchanged. No candidate is eligible for gameplay or promotion, and the job is not extended, resumed or repeated automatically. The matched new-head feedforward control also failed to recover the deployed policy's quality, so this one-seed, small-budget result cannot establish that memory is intrinsically unhelpful. It establishes that this precise ordered, end-to-end comparison did not produce a usable improvement. The33s unique human correction corpus remains a substantial limitation; repeating it four times does not add failure-state coverage.

Machine-readable review: `data/annotations/ordered_gru_route_result_review_20261109.json`. Keep all endpoints isolated as research artifacts. Further work should address actual route-recovery coverage and initialization/retention explicitly before another training proposal; do not simply increase the same job's budget.
