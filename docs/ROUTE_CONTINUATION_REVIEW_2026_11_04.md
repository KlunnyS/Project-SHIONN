# Route continuation review — 2026-11-04

## Decision

Keep the deployed model. Do not request more broad gameplay comparisons or start another training run yet. Existing demonstrations include complete route endings; test the model's fit to those endings before choosing a correction.

## Live evidence

Original screen: `/tmp/shionn-turn-onset-screen/sequence_20261104_111302_251893`.
Repeat sequences: `/tmp/shionn-turn-onset-repeat/sequence_20261104_115344_364310` and `sequence_20261104_150119_517565`.

Across these sequences, deployed completes test1 0/5, test11 1/5, test12 5/5. Onset completes test1 0/5, test11 2/5, test12 5/5. All scored attempts passed recorded capture checks. Small cohorts and fixed model order limit inference; onset is not established as reliably superior.

Latest onset test11 attempts 004 and 010 reach views of the exit corridor around video seconds 10–14, then face nearby walls and wander. Longest all-zero stretches are approximately 0.5s and 3.58s, estimated from tick counts at 24Hz. An earlier failed onset attempt has approximately 12.96s of all-zero output. Thus both active wandering and intermittent idle output occur. Sparse visual sampling does not prove goal recognition, precise collisions, or a causal weight/data defect.

## Existing training coverage inspected

Frozen v3 base TRAIN membership: test1 has 200 episodes; test7, test9 and test10 have 100 each. These are membership counts, not counts of individually reviewed successful examples.

One midpoint episode per map was visually sampled at approximately 45%, 65%, 85% and 99% of its cached frames:

- test1: `episode_20260915_193018_130856` (324 frames).
- test7: `episode_20261004_150352_857611` (308 frames).
- test9: `episode_20261004_150427_167217` (312 frames).
- test10: `episode_20261004_150445_661153` (294 frames).

All four sampled routes show an exit approach and a final view beyond the door. This establishes existence of relevant training behavior, not adequate diversity or model mastery. Montage: `/tmp/shionn-training-route-audit.jpg` (temporary local artifact).

Validation example `episode_20260920_140231_897826` on test11 also includes the complete approach and crossing. It remains validation-only, never a correction training source. Its use in diagnosis means it is not an untouched final assessment.

## Next bounded diagnostic

1. Select and manually review final-approach windows from existing TRAIN routes on maps 1, 7, 9 and 10. Do not assume the last fixed number of frames is the approach: post-door footage must be distinguished.
2. Replay deployed and onset on identical normal-history windows without optimization. Measure forward/strafe agreement, dx/dy sign and magnitude, unwanted turn predictions when experts continue forward, and idle output. Report per episode and map.
3. Compare relevant held-out test11 expert windows separately. They may inform diagnosis but must stay outside gradients; failed live trajectories have no expert action labels and cannot be treated as corrections automatically.
4. If models fail on familiar expert approaches, propose a short route-continuation replay experiment with existing navigation rehearsal and fixed update/runtime limits. Include post-turn continuation rather than only turn onset. State cost before execution.
5. If expert approaches are predicted well but live off-route views fail, document the distribution gap and request targeted human corrections on TRAIN chambers when the user can operate the game. More of the same clean routes would not directly address that gap.

No claim of a planning/memory limitation is established. No neural architecture change, LLM weight editing, training execution or model promotion follows from this review.

## Completed sparse replay diagnostic

Read-only CUDA inference completed on deployed and onset, with no optimizer or game actions. For each of the four TRAIN examples above and the test11 validation example, sampled 16 unique frames uniformly from 60–80% of the recording. Constructed each input from its true four-frame cached RGB history, divided by 255, and used existing checkpoint action decoding. Both models were loaded through PolicyInference in evaluation mode. This is an exploratory middle/late-route screen, not a manually bounded exit-only assessment.

| Map | Deployed binary exact | Onset binary exact | Deployed dx MAE | Onset dx MAE |
|---|---:|---:|---:|---:|
| test1 | 81.25% | 68.75% | 15.125 | 11.000 |
| test7 | 68.75% | 75.00% | 9.750 | 10.063 |
| test9 | 81.25% | 75.00% | 1.188 | 2.125 |
| test10 | 93.75% | 81.25% | 1.375 | 1.063 |
| test11 validation diagnostic | 100% | 100% | 0.063 | 0.625 |

Among the four TRAIN examples, 37 sampled frames have expert forward movement and absolute dx <=3. Neither model predicts an absolute dx >=40 on any of these. On the test11 example, all16 frames meet this straight criterion; both match every binary control, neither idles and neither predicts a large horizontal turn. These thresholds describe requested input deltas, not measured yaw.

Interpretation: these sampled clean approaches do not reproduce the strong wrong-turn behavior seen in live attempts. This supports investigating off-route observation coverage and error accumulation rather than assuming that clean exit-continuation fitting is the main bottleneck. It does not establish a global causal diagnosis: five selected episodes and80 inputs/model cannot establish full coverage, and the histories follow the expert rather than each model's own actions.

Raw results with per-frame targets/predictions and source hashes: `/tmp/shionn-route-probe-results.json`. Reproduction script: `/tmp/shionn_route_probe.py`. No weights or existing helpers were modified.

### Recommended next development step

Prepare a targeted human-correction session on TRAIN chambers (start with test1) when desktop interaction is available. Stage or pause at a recognizable failed route position, record the human steering and movement back onto the route and continue through the exit; keep assisted collection separate from scored autonomous runs. Reuse the existing recorder and provenance controls. Human actions must not be inferred from unlabeled failed videos. Review current correction coverage against those off-route viewpoints first; reserve whole correction episodes before any fitting. Test11 remains validation-only. Do not start a further generic replay or onset oversampling run from these results.

## Collection preparation — 2026-11-05

See [TARGETED_NAVIGATION_RECOVERY_PROTOCOL.md](TARGETED_NAVIGATION_RECOVERY_PROTOCOL.md) for the inspected checkpoint identities, existing takeover mechanism, user-only environment rehearsal, three-correction pilot command, review/quarantine gate, blocked correction-training draft and later GRU comparison. Twelve existing session workflow tests passed offline. Live capture/control readiness remains unverified. No new targeted correction recordings were found in the existing recovery tree, so no training was started. The current workflow records linked autonomous/human segments with an explicit handoff gap, not gapless whole-episode media; only human CSV targets may enter corrections.
