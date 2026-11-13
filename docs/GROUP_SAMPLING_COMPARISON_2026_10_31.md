# V5 correction-group sampling comparison — 2026-10-31

## Recovery history probe result — 2026-11-02

The user completed `recovery_history_probe_v1` in 227.339 seconds. The report
has `comparison_complete: true`, 1,000 optimizer updates, 8,000 draws with the
same 7,173/424/403 group counts as the reused control, and 221 repeated-history
recovery draws. All 39,769 validation and 879 assessment frames were evaluated.
Parent/reference/helper/script hashes match; the control was not retrained.

| Offline metric | Reused control | History repetition |
|---|---:|---:|
| Idle at first assessment action | 6/6 | 5/6 |
| Idle in first8 frames/clip | 13/48 | 11/48 |
| First8 binary exact match | 83.33% | 79.17% |
| First8 horizontal MAE | 45.729 | 45.729 |
| First24 backward recall | 8/11 | 8/11 |
| All assessment horizontal MAE | 20.854 | 20.940 |
| Validation binary exact match | 84.946% | 84.978% |
| Validation horizontal MAE | 6.225 | 6.268 |

The changed first action is episode `episode_20261101_141321_386858`, frame18:
expert turns (`dx=-13, dy=7`), while the history arm presses W with zero camera
movement. Thus the reduction in idle is not a demonstrated correct recovery.
These expert-frame predictions do not establish live navigation improvement.
No promotion, automatic extension or game test is justified by this probe alone.
Next review should audit history construction and image/action timing before
another training decision. Historical library versions and complete pixel
hashes remain unavailable, so reused-control equivalence is exploratory.

## Bounded recovery probe result review — 2026-11-02

The user completed `recovery_onset_probe_v1`. Its report confirms
`paired_complete: true`, both fixed endpoints at 1,000 optimizer updates, and
full validation/assessment evaluation. Total recorded runtime was 687 seconds
(11m27s), within budget. The parent checkpoint hash is unchanged. Both arms
consumed exactly 7,173 base, 424 recovery and 403 proactive frames; control drew 75
recovery-opening frames versus 276 for boosted, confirming the intended change.

| Offline metric | Source candidate | Control | Opening boost |
|---|---:|---:|---:|
| All-zero output at first assessment action frame | 6/6 clips | 6/6 | 6/6 |
| Idle outputs in first8 assessment frames/clip | 15/48 | 13/48 | 13/48 |
| Backward recall in first24 assessment frames/clip | 5/11 | 8/11 | 8/11 |
| Assessment horizontal MAE, all879 frames | 21.264 | 20.854 | 20.829 |
| Assessment horizontal MAE, first8 frames/clip | 42.90 | 45.73 | 45.73 |
| Validation exact binary match | 84.60% | 84.95% | 84.93% |
| Validation horizontal MAE | 6.292 | 6.225 | 6.195 |

The 11 backward frames belong to one pilot, not independent successes. Normal
validation opening-frame idle predictions also increased (first 8: 175/1,600
source, 198 control, 199 boosted). Both fine-tunes have modest broad improvements
and improved later backward imitation, but the recovery-opening boost adds no
clear initiation benefit over control under this budget. All six initial
assessment decisions remain idle. These are offline predictions, not six new
autonomous game trials. Do not promote or extend this recipe automatically.

Loss definition: the probe applies source training label smoothing 0.01 in its
evaluation; the full trainer's ordinary validation defaults to 0. Thus source
probe validation loss 2.924 is not directly comparable with the historical
2.618 checkpoint metric. All three within-probe losses share the same protocol,
and action metrics above are directly paired. Raw reports are preserved.

### Read-only follow-up: input-history dependence

An exploratory boosted-model feature comparison finds some close-wall queries
near training examples with different expert actions (backwards, turning and
pauses), while stair-adjacent examples have weaker matches. These nearest
neighbors use the model's own 512D features and do not prove state equivalence,
contradictory labels, or a memory limitation.

A more direct input ablation held the newest image and labels fixed on the
first 24 usable frames of each of the six assessment clips. Replacing the older
three images with copies of that newest image increased all-zero predictions
from 19/144 to 84/144 and horizontal MAE from 56.74 to 102.95. No weights were
changed. This shows strong sensitivity to motion/history on these examples.
Removing history also removes legitimate cues and creates a distribution shift,
so it is not proof of a pathological shortcut or that frame-repeat augmentation
will improve live behavior.

Next hypothesis: test recovery-specific repeated-frame/history augmentation,
without more recording or a larger architecture by default. First check whether
the completed short control can be reused under exactly matched source, data,
sampling, optimizer, RNG and evaluation contracts; do not require another
unexplained baseline run. Any new arm still needs an explicit short budget and
keeps all six clips out of gradients. No new training or live benchmark was
launched during this review.

Raw probe: `models/imitation/checkpoints/runs/recovery_onset_probe_v1/report.json`.
Endpoints are `control/endpoint.pt` and `boosted/endpoint.pt`; neither is selected
as the deployed candidate. Interpretation, scripts, neighbor images and history
ablation are archived at
`model_attempts/diagnostics/recovery_probe_results_20261102/`.

## Six fresh assessment clips completed — 2026-11-01

All six manually staged test1 recoveries reached the goal:1,049 video frames
and matching contiguous action rows at24FPS, without timestamp reversals.
They include wall, stair/corner and tilted-view starts. There are879 usable
frames after leading-idle exclusion. All six episodes are absent from frozen-v3
and every compared model's training list. They remain assessment-only; none
entered a training cache. Sampled images confirm the intended start views and
successful human recoveries, without proving exact acquisition/input timing.

Scored the same saved human frames with the matched control, grouped90/5/5
and previous candidate:

| Offline diagnostic | Control | Grouped | Previous candidate |
|---|---:|---:|---:|
| All-zero output at first expert-action frame | 5/6 clips | 4/6 clips | 6/6 clips |
| All-zero outputs in first8 frames/clip (48 frames) | 14 | 10 | 15 |
| Correct backward outputs when expert backs (first24/clip) | 3/11 | 8/11 | 5/11 |
| Horizontal MAE on all879 frames | 20.38 | 22.09 | 21.26 |
| Exact binary-action match | 73.7% | 71.1% | 75.3% |

The11 expert backward frames all come from the pilot, so that improvement is
one case, not11 independent recovery successes. These are teacher-forced
predictions, not six autonomous model attempts. First nonzero human actions
can also include small adjustments; a one-frame mismatch alone is not proof
of an inevitable freeze. The pilot static-view check and actual live stalls
provide additional, separate evidence for the recovery-initiation concern.

Collection is complete. The grouped model improves some initial/backward
predictions but does not give a reliable overall recovery gain, and predicts
larger horizontal errors on this small set. Do not promote it or collect
another batch by default. Future tuning on these clips makes them development
assessment data; they cannot then be described as an untouched final test.

## Offline zero-bin decoding check

As a diagnostic only, re-decoded grouped saved mouse probabilities: choose
zero when P(zero) >= threshold, otherwise choose the highest-probability
nonzero bin. Swept fixed thresholds0.5,0.65,0.8,0.9 and0.95 on the existing
39,769 validation frames and879 assessment frames. No runner code/default was
changed, and no live test used this rule.

The0.9 threshold eliminates all-zero first-frame outputs in these six clips,
but validation horizontal MAE worsens from6.744 to7.687, and nonzero horizontal
output on expert-zero frames rises from3.83% to38.99%. All swept thresholds
worsen validation horizontal MAE. Therefore making the policy move more is
not a sufficient calibration fix; do not deploy this global override.

Next prepare a bounded recovery-initiation training experiment using existing
training clips, retaining these six for assessment. State the warm-start
checkpoint, matched comparison, step/time cap and expected runtime before
launch; do not give another full four-hour command as an implicit prerequisite.
Existing `--resume` requires the same recipe and is not a valid way to change
sampling during a new fine-tune. No further training is authorized by merely
completing these recordings. Full independent assessment and navigation tests
remain necessary before any model promotion.

All results are under `model_attempts/recovery_assessment_20261101/`, including
`six_clip_integrity.json`, `six_clip_review.jpg`, `scores/six_clip_comparison.json`
and `scores/zero_gate_diagnostic.json`; scripts are preserved alongside scores.

## Fresh recovery assessment pilot — 2026-11-01

The user manually staged a close-wall start on test1, backed away, turned and
reached the exit. `episode_20261101_132850_051907` contains213 video frames
and213 contiguous CSV rows at24FPS (8.875seconds), without timestamp reversals.
Sampled images show the intended wall-facing start and successful recovery.
There are53 leading idle frames, excluded from scoring targets using the
existing dataset behavior, leaving160 usable frames. Timing alignment is
supported by the sampled action/image sequence, not proved per acquisition.

The episode is absent from frozen-v3 and all three models' training lists.
It is kept at `model_attempts/recovery_assessment_20261101/`, with an assessment-
only cache and scores. This is a single fresh recording on a familiar map,
not an unseen-map benchmark or an autonomous success. Its standalone recorder
metadata uses the generic `model_recovery` source with null source-attempt;
the collection plan explicitly records its actual manually staged provenance.

At the first active frame53, the expert presses S; all three policies predict
all-zero actions. On the first eight active-target frames, the expert backs
away8/8, compared with control3/8, grouped6/8 and previous candidate4/8.
The grouped policy improves after the human trajectory has changed the view;
this teacher-forced score does not establish autonomous escape from the start.

As an additional deterministic check, each policy was given four identical
copies of the saved wall-facing frame53. Every policy again produced an
all-zero action. This supports a static-view idle failure on this exact image:
without an initial action, the model may not receive the changed views on
which its later predictions improve. It is not a full closed-loop proof;
real rendering/physics and frame history can vary.

Do not lower thresholds, add forced backwards motion or start new training
from this one case. The pilot is suitable for continuing the planned small
assessment: one further close-wall example, two angled corner/stair starts
and two tilted-view starts, all kept out of training. Compare the same models
on all six before choosing an intervention. The user alone operates the game.

Reports: `scores/pilot_comparison.json`, `scores/pilot_static_start.json` and
`pilot_review.jpg` under the assessment folder. No model training occurred.

## Recovery coverage and training-fit check — 2026-11-01

The user clarified that some current freezes match existing recovery examples,
some are similar, and some are different. Reviewed the first usable cached
image from each of31 recovery clips: close frontal walls and some angled
wall/corner views are visibly represented. Strong floor/ceiling views in the
latest failures are less clearly represented in this sampled starting-view
sheet. Visual resemblance is not equivalence of full state/history or evidence
that a route requires the same action; this is not an exhaustive coverage audit.

Scored both models on those31 clips (4,477 usable frames), verifying that all
clips were training episodes for both models, with no overlap in their saved
validation lists. This measures reproduction of trained examples, not held-out
recovery or autonomous success. On the first eight usable frames per clip:

| Metric (248 training frames) | Expert | Control | Grouped90/5/5 |
|---|---:|---:|---:|
| All-zero action frames | 2.8% | 27.0% | 13.3% |
| Horizontal mouse active | 77.0% | 58.9% | 66.1% |
| Correct horizontal sign when expert turns | — | 72.8% | 82.7% |
| Backward recall when expert presses S | — | 67.2% | 86.9% |
| Horizontal mouse MAE | — | 41.98 | 34.91 |

Across all recovery training frames, horizontal MAE also improves from15.76
to13.34. Group sampling therefore improved imitation of these trained recovery
clips, even though broader validation regressed and live wall freezing remains.
It is inaccurate to call the recipe uniformly ineffective; its observed benefit
has not become reliable closed-loop recovery. Even on trained opening images,
grouped still predicts all-zero actions substantially more often than the expert.

Next distinguish already represented versus novel failed states, and establish
independent recovery examples for assessment before adding another broad batch
or tuning the recipe further. Do not reuse these31 training clips as a held-out
test. Any new correction session should target measured gaps and reserve
independent assessment clips before training. Source expert actions are useful;
the model's all-zero failure labels are not replacement demonstrations.

Reports and raw predictions are at
`model_attempts/benchmarks/recovery_fit_20261101/`; the first-view sheet, frame
metadata and comparison script are archived at
`model_attempts/diagnostics/recovery_coverage_20261101/`. These checks used saved
data only; no model training or game interaction was launched.

## Live screen completed 2026-11-01

Primary sequence:
`model_attempts/sequences/group_sampling_screening/sequence_20261101_125422_631650/`.
The user reported many wall freezes after following the requested hands-off
screen. Twelve attempts completed: control/grouped, three maps, two repeats,
interleaved,60-second limits, persistent mouse and verified confinement.

| Model | test1 | test11 | test12 | Recorded total |
|---|---:|---:|---:|---:|
| Control | 0/2 | 0/2* | 2/2 | 2/6* |
| Grouped90/5/5 | 0/2 | 1/2 | 2/2 | 3/6 |

*Control job3/test11 has one camera-query timeout and a0.554-second observation
gap near the end; it requires telemetry review and cannot count as a clean
capture-confirmed failure for promotion. Excluding it yields control2/5 clean
attempts and0/1 clean test11, versus grouped3/6 and1/2. Eleven of twelve
attempts passed all automatic capture checks. Across the completed sequence,
there were zero recorded focus losses, cursor exits, unknown desktop samples,
input submission failures, or suppressed policy ticks. Every guard cleanup
restored settings. Camera telemetry is observational; passing samples cannot
prove uninterrupted capture or game consumption.

All attempts first requested a nonidle action within0.74seconds. Startup
hesitation is therefore not the main failure in this screen. Six of the seven
timeouts ended in long all-zero raw policy output, not input suppression:

| Job / model / map | Terminal idle interval | Approximate duration |
|---|---|---:|
| 1 control/test1 | 16.4–60.0s | 43.6s |
| 2 grouped/test1 | 26.3–60.0s | 33.7s |
| 4 grouped/test11 | 11.7–60.0s | 48.3s |
| 7 control/test1 | 22.5–60.0s | 37.5s |
| 8 grouped/test1 | 16.0–60.0s | 44.0s |
| 9 control/test11 | 13.1–60.0s | 46.9s |

Sampled saved frames from all seven timeouts show close wall/floor views in
these terminal idle states. Control job3 remained much more active and ended
in a strongly tilted view; its longest idle interval was only1.5seconds.
Thus not every timeout is the same failure. Built-in `wall_stuck` counts were
zero despite visual/user evidence of wall stalls; do not use that detector
as a true collision/recovery count. Both models solved test12 using essentially
forward-only output, so its success does not establish steering ability.

Two earlier sequence starts are retained separately, not silently pooled:
`sequence_20261101_122328_755134` stopped its first attempt with Escape;
`sequence_20261101_122522_198588` recorded ten completed attempts then stopped
job11 on stale desktop telemetry (sample age0.275seconds), with no recorded
focus loss/cursor exit and successful cleanup. That earlier guard stop is
not evidence of an observed window switch. Restarted runs are supplemental
evidence and not balanced complete comparison blocks.

Decision: no navigation-readiness or promotion gate is met. One grouped test11
success is insufficient to establish reliable improvement; both still fail
test1 and idle in failed visual states. Do not start another broad training
or full benchmark by default. Next inspect whether existing expert recovery
clips cover the observed close-wall, floor and stair-adjacent states, while
continuing the reviewed onset audit. Model-generated all-zero labels are
failure evidence, not expert targets; any new labels must come from reviewed
human corrections. Puzzle training remains deferred.

Analysis script, per-attempt statistics and failure-frame sheet are archived
under `model_attempts/diagnostics/group_sampling_screening_20261101/`.

## What was trained

The user completed two fresh V5 training runs. The control repeated the
existing sampling recipe; the second run applied the proposed increase in
correction exposure. These are separate models, not two stages of one run.

| Run | Expected base / proactive / recovery draws | Best epoch | Best validation loss | Last epoch loss |
|---|---|---:|---:|---:|
| `v5_binned_sampling_control_v1` | about 97.01% / 1.11% / 1.88% | 3 | 2.58747 | 2.76430 |
| `v5_binned_turn_recovery_v1` | 90% / 5% / 5% | 3 | 2.74413 | 2.85681 |

Both completed four epochs and 121,276 optimizer steps. Saved model/target
configuration and all training settings match except the group fractions and
recorded group membership. Both use frozen-v3 (1,151 training / 200 validation
episodes), seed 0, batch 8, fixed learning rate 0.0002, no LR scheduler,
opening boost 1 and identical augmentation. The validation split digest is
`2680429b46e154c23ae0a8c651575c6704cbf8a860c049059eb017e8abaaee4e`.

Group fractions are expected replacement-sampling shares; realised epoch
counts fluctuate. Reweighting groups also changes global map exposure because
correction clips cover a subset of chambers. Within-group relative weights are
preserved; this is not simultaneous exact group and global map balancing.

The grouped run's best validation loss is higher by 0.15666 (about 6.1%).
That is evidence against improving this aggregate held-out objective. It does
not measure closed-loop turning/recovery and does not establish live failure.
Both runs worsened validation in epoch 4 despite decreasing training loss;
use epoch-3 `best.pt`, not `last.pt`, for subsequent comparisons.

## Paired expert-action comparison

Scored each new epoch-3 checkpoint on the same 200 validation episodes and
39,769 usable frames. Also compared the previous candidate's saved per-frame
predictions. Frame IDs and expected action labels were verified equal across
all three reports. This is open-loop expert-frame evaluation, not gameplay.

| Model | Exact binary-action match | Horizontal mouse MAE | Vertical mouse MAE |
|---|---:|---:|---:|
| Previous `v5_recovery_proactive_v3` | 84.60% | 6.292 | 2.361 |
| Matched control | 84.43% | 6.503 | 2.399 |
| 90/5/5 grouped | 83.52% | 6.744 | 2.365 |

Per map, control versus grouped exact match was 82.19% versus 82.02% on test11
and 87.82% versus 85.80% on test12. Horizontal mouse MAE worsened from 7.862
to 8.104 and from 4.445 to 4.684 respectively. Forward recall increased
slightly and overall vertical error decreased slightly; these limited gains
do not establish improved navigation or offset the horizontal/action results.

The first 24 usable frames of each episode produced 4,800 opening examples.
All-idle predictions were about 7.5% for both new policies, while opening
horizontal mouse MAE worsened from 1.085 to 1.196. No start-freeze fix is
demonstrated by these expert-frame results; live spawn-state coverage can differ.

A provisional label-only first-turn detector found 1,074 frames in six-frame
windows across 179 episodes. It requires at least three of the next five
frames to have abs(dx) >= 3, at most one of the previous six, then takes the
first qualifying six-frame window per episode. Expert horizontal mouse was
nonzero on 72.0% of those frames. Control predicted nonzero horizontal mouse
on 37.6%, grouped on 28.4%, and the previous candidate on 36.2%. Horizontal
MAE was 16.204, 16.866 and 15.264 respectively. This detector is exploratory,
not a visually validated annotation set or proof of the cause of failure.

Evidence is archived under
`model_attempts/benchmarks/group_sampling_comparison_20261031/`, including
`control/summary.json`, `grouped/summary.json`, both per-frame CSVs and
`comparison.json`. The earlier candidate's source report remains at
`model_attempts/benchmarks/validation_cuda_v5_recovery_proactive_v3/`.

## Evaluation limits and next decision

A later eight-example, visually reviewed main-turn diagnostic found fewer zero
horizontal predictions for grouped than control, with slightly worse mouse
magnitude error and worse approach error. See [the review](TURN_ONSET_REVIEW_2026_10_31.md).
That limited signal motivates a short12-attempt paired live screen, rather
than another training run. It does not reverse the broad offline regression
or establish improved live navigation.

No autonomous gameplay result exists yet for either of these two checkpoints.
The earlier 7/15 candidate and 5/15 balanced results used different checkpoints;
they must not be attributed to the new runs.

The 90/5/5 recipe is not supported as an improvement by this one-seed,
fixed-budget comparison. Do not promote it or launch a larger correction-share
run by default. Retain the previous candidate while reviewing image/action
alignment and visually annotating representative first-turn windows for #25.
The next intervention should test that narrower hypothesis rather than
increase correction shares again without evidence. A live screen is optional
if there is a specific recovery question; no live superiority is claimed.
Offline expert scoring cannot evaluate recovery from policy-induced wall
states. Any label-only onset detector remains provisional until visually
reviewed and cannot satisfy #25's annotation criterion by itself.
