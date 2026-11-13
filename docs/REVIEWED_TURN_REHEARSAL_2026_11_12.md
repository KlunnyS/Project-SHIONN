# Focused turn rehearsal — 2026-11-12

The completed expanded-correction candidate did not improve chamber completion.
Read-only label replay found missing camera turns even on its own TRAIN clips.
A separate100-update diagnostic on40reviewed turn frames reduced zero horizontal
predictions14->0 and improved horizontal sign agreement26/40->38/40. This is TRAIN
fit with augmentation disabled, not gameplay or generalization evidence. Magnitude
error remained high (98.05->92.95dx MAE). No weights were exported. Diagnostic
report: `docs/reviews/recovery_turn_fit_probe_20261112/report.json`.

This new fixed-budget experiment tests concentrated exposure under normal
training settings. It does not extend or resume the completed training run.
`run_reviewed_turn_rehearsal.py` defaults to plan-only. Its reference is the
hash-verified completed `reviewed_recovery_pair_20261112/correction` arm. Exact
source/config/data/draw-plan equality is required before reusing that control.

The candidate starts fresh from deployed source with fresh AdamW, LR1e-5,
1000successful float32 updates, original augmentation/loss/preprocessing/bins.
Only slot9 changes: a uniform human sample becomes one of the first8active frames
of each of5reviewed new corrections (40targets total). Every other sample and
setting is identical to the completed control. Composition per batch of10:
7original TRAIN,1original opening,1uniform human,1reviewed turn. Total human
fraction remains20%. All complete episode memberships/heldout roles are preserved.

Source and reference checkpoints remain unchanged. Output:
`models/imitation/checkpoints/runs/reviewed_turn_rehearsal_20261112`.
External15min runtime limit and soft840s shared deadline; no overwrite, automatic
retry, extension, promotion or generic resume. Checkpoint stays pending until the
full navigation/development evaluation and final provenance checks complete.

Eligibility uses unchanged navigation/startup guards; review complete human
trajectories and development metrics against source and reference, not only
40rehearsed TRAIN frames. Eligibility is not proof of learned recovery. A live
user-operated screen is considered only if offline checks pass; otherwise retain
the deployed model and diagnose the regression rather than asking for repeated
same-family tests.

## Completed offline result

Code validation:14focused tests and379full regression tests passed. Independent
pre-execution review passed. The new job finished successfully in311.403s
(service wall5min14s, host memory peak2.5GiB),1000updates/10000targets, within
the15min cap. Source/reference/input hashes and candidate load were verified.
Machine-readable review: `docs/reviews/reviewed_turn_rehearsal_result_20261112.json`.

|Measure|Source|Uniform expanded control|Turn rehearsal|
|---|---:|---:|---:|
|Navigation binary exact|84.604%|84.943%|84.978%|
|Development episode-mean binary exact|69.473%|68.013%|69.836%|
|Development horizontal mouse MAE|20.368|20.640|21.053|
|New full-clip zero dx predictions /503nonzero targets|144|147|131|
|First8TRAIN zero dx predictions /40targets|14|12|0|
|First8TRAIN correct horizontal sign|65%|70%|95%|
|Navigation/startup guard|reference|pass|pass|

The40TRAIN rehearsal targets are fitted more successfully under normal training
settings; this does not prove generalization. Remaining first8horizontal mouse
MAE91.325 is large. Development binary agreement improves modestly, while held
horizontal magnitude error worsens. These mixed results support only a limited
user-operated gameplay screen, not promotion or a claim of solved navigation.
Two already inspected development episodes are weak evidence for transfer.

Next screen: deployed versus turn-rehearsal candidate on test1/test11/test12,
one repeat each (6attempts),60s,24Hz, existing capture/input safeguards and3s
post-preflight settle. Fresh root:
`model_attempts/sequences/reviewed_turn_rehearsal_20261112_screen`.
No additional training, automatic retry, promotion or budget extension follows
from offline metrics. Gameplay completion and route continuation remain unknown.

## Autonomous screen completed

Source: `model_attempts/sequences/reviewed_turn_rehearsal_20261112_screen/sequence_20261112_145845_838813`.
All6attempts valid with no capture-review flags. Both models timed out on test1
and test11; test12 goals deployed6.1452s, turn6.1475s. No completion improvement.
Detailed review: `docs/reviews/reviewed_turn_gameplay_20261112.json`.

All-zero action ticks dropped in this screen from1064/1434 to534/1440 on test1
and1147/1440 to84/1440 on test11. This single-run inactivity difference is
encouraging but does not establish reliable recovery. Sampled candidate test1
frames show stairs/exit visibility, wall returns and a late wall-facing freeze;
test11 shows multiple wall/stair/door views and exit visibility in the last
sampled frame. Final logs include large turns, so a late door view is not proof
of sustained exit pursuit or a near-certain finish with more time.

Candidate horizontal largest-bin commands (absolute134) occur454/1440ticks on
test1 and661/1440 on test11; respective deployed counts193/1434 and25/1440.
Longest uninterrupted largest-bin stretches are0.92s and3.88s for the candidate.
These are different closed-loop trajectories, so counts alone do not isolate
which training change caused the difference. They motivate testing camera-rate
control rather than automatically collecting more data or extending training.

Next exploratory intervention: existing `--mouse-max-abs 40`, applied to BOTH
deployed and turn checkpoints with the same3maps/1repeat/60s protocol. This
reduces the camera command while preserving its sign and the model rate. Use
fresh root `model_attempts/sequences/reviewed_turn_cap40_20261112_screen`.
No code/checkpoint/training change. Compare capture validity, completion, camera
turning and freezes. This is a screen with one attempt per condition; do not pool
with prior cohorts or promote based on visual smoothness. No guarantee that
clamping improves route completion, and it can slow necessary recovery turns.

## Camera-cap screen completed: reject cap40

Source: `model_attempts/sequences/reviewed_turn_cap40_20261112_screen/sequence_20261112_152240_816868`.
All6attempts valid, configured cap40 verified and maximum applied horizontal
delta40. Both models again timed out on test1/test11 and completed test12
(deployed6.193s, turn6.163s). No completion gain. Turn-candidate all-zero action
fractions were1117/1440 (77.6%) on test1 and913/1440 (63.4%) on test11, versus
37.1%/5.8% in its uncapped screen. Distinct single trajectories are not a
controlled statistical estimate, but there is no evidence supporting the cap.

Do not retain cap40 or promote the turn candidate. No further camera-cap/rate
sweep, same-data training extension or repeated confirmation is justified by
these screens. Keep deployed defaults. Detailed artifact:
`docs/reviews/reviewed_turn_cap40_gameplay_20261112.json`.

Taken together: fitting a few demonstrated turns can reduce TRAIN action errors
and sometimes prolong live movement, but has not produced sustained route-to-exit
behavior. The next offline investigation should inspect trajectory supervision
and corrective-action representation before another candidate: turn timing,
camera target magnitude/quantization, and whether frame-level target agreement
captures sustained progress. Do not infer an architecture defect or prescribe
another data-collection batch solely from these failed screens.
