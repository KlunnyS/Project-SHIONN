# Turn-onset and alignment review — 2026-10-31

## Reviewed validation score — 2026-11-01

The remaining proactive marker was resolved using the user's interpretation,
the sampled images and a CSV recheck: `150633_718452` has a small right burst
at35–39, zero dx at40–44, then sustained increasing right input at45. Its main
onset is45. All20 pilot onsets now have review provenance; endpoints remain
provisional. This supersedes the ambiguity noted in earlier sections.

Marked ten systematically selected test11 validation examples (every tenth
sorted frozen episode ID). Sampled route-context review excluded two uncertain
boundaries, leaving eight left-turn windows. Three initial small adjustments
were replaced with the later main-turn markers before scoring predictions.
The [validation annotation file](../data/annotations/validation_main_turns_v1.json)
records selections, exclusions and provisional endpoints. All eight episodes
were verified in the validation list and absent from training for every model
scored. This is development validation, not an untouched final evaluation.

Using existing saved predictions on the same 48 onset frames (six per turn):

| Model | Zero horizontal output | Left output | Horizontal MAE |
|---|---:|---:|---:|
| Previous candidate | 37.5% | 62.5% | 34.56 |
| Matched control | 35.4% | 64.6% | 37.98 |
| 90/5/5 grouped | 27.1% | 72.9% | 38.06 |

The expert requested left movement on93.75% of these frames. For the 192
early-turn frames (24 per example), grouped horizontal MAE also worsened
slightly (29.85 versus28.96 for control). Before onset, grouped approach error
was14.00 versus10.53 and it predicted more left movement than the expert.
Thus grouping has a limited direction-initiation signal in this small reviewed
sample but does not establish better calibration or navigation. Forty-eight
correlated frames are not48 independent trials, and all eight turns are left
turns from one map. The broader held-out regression remains a negative result.

This differs from the earlier provisional first-burst score because it assesses
visually selected main route turns, not incidental camera movements. It is not
a reason to promote or train a higher correction fraction. Next, perform a
short, user-operated paired live screen: control/grouped, test1/test11/test12,
two repeats each,60-second limit, interleaved, unchanged capture settings.
Twelve attempts can screen for obvious practical gains/regressions; they do
not replace the later five-repeat promotion protocol. No additional training
is requested. The plan-only command verified all12 jobs without game input.

Scores, diagnostic scripts and reviewed image sheets are archived under
`model_attempts/diagnostics/reviewed_validation_turns_20261101/`.

## Frame-indexed annotations added 2026-11-01

The assistant marked all 20 proactive training clips using the existing CSV
`frame_idx` values and sampled corresponding cached RGB sequences. See
[the JSON annotations](../data/annotations/proactive_turn_onsets_v1.json) and
[the spreadsheet-friendly CSV](../data/annotations/proactive_turn_onsets_v1.csv).
No user video marking or new recording session is required for this pilot.

Nineteen main-onset candidates have sampled-image/action review. The test4
right clip `episode_20261018_150633_718452` retains an explicit 35–45-frame
uncertainty and is excluded from onset training until further local review.
The earlier incidental movements in the two identified counterexamples are
excluded: the main left turn in `150437_239955` starts with small input at20,
and the main right turn in `150816_693808` starts at88.

Context begins 12 frames before each onset; it is a fixed review window, not
the earliest visible cue. Turn-end columns are provisional mouse-burst ends,
not exact semantic turn-completion labels. Do not use those endpoints as
verified ground truth. Each record has a source CSV hash; all episode IDs,
maps, training roles and frame bounds were checked against frozen v3. Input
labels and frozen split membership remain unchanged. No model was trained.

This is the first training-side annotation batch. Representative base and
independent validation windows remain necessary before a held-out onset claim.

## Scope

Read-only analysis of the 20 proactive-turn clips and three selected base
test1 recordings. No live game actions, label edits, preprocessing changes,
training or checkpoint updates were performed. These are training examples,
not independent evaluation data.

Artifacts: `model_attempts/diagnostics/turn_onset_audit_20261031/` contains
the audit script, optical-flow statistics, sampled-frame sheets and a review
manifest. The manifest is explicitly not approved as training annotations yet.

## Timing check

Feature tracking compared signed horizontal scene displacement with recorded
mouse dx across all 20 proactive clips, excluding the lower third of the
image to reduce portal-gun interference. Input frame i was compared with
scene motion into frame i+L, for L from -4 through +4. Pooled correlations
were 0.846 at lag 0, 0.884 at +1 and 0.886 at +2. Nineteen clips peaked
at +1 or +2 and one at -1. At 24 FPS, one or two frames are about 42–83 ms.

This supports an association between recorded mouse input and subsequent
image movement. It does not prove exact alignment or establish that any
particular fixed shift is correct: smooth inputs, translation, depth,
tracking errors and asynchronous capture all affect this statistic.

The recorder snapshots accumulated physical actions and reads the latest
asynchronously captured image. It timestamps the CSV row after writing the
image; the camera does not retain acquisition timestamps or frame sequence
numbers. Thus existing files cannot prove per-frame acquisition/input timing.
Do not globally shift labels from these correlations alone.

## Sampled visual review

Inspected six frames around each detector-selected candidate in all 20
proactive clips and three base test1 examples. Negative/positive dx generally
corresponds to visible left/right turning in the subsequent samples. This is
sampled inspection, not a full collision/trajectory or synchronization audit.
The user previously approved full proactive clip quality and modest approach
variation.

The first qualifying mouse burst is not always the main proactive turn:

| Episode | Detector selection | Reviewed later candidate | Finding |
|---|---:|---:|---|
| `episode_20261018_150437_239955` | 6 | about 21 | Brief early left adjustment and pause precede the sustained left turn. |
| `episode_20261018_150816_693808` | 38 | about 88 | A small left adjustment is selected in a right-labelled clip; the later main turn is right. |
| `episode_20261018_150633_718452` | 35 | unresolved | Weak right movement near/under stairs needs a full route review before calling it the main turn. |

The original detector used three of the next five frames with abs(dx) >= 3
and at most one of the previous six. These counterexamples show why its
first-onset metric was provisional. They do not establish corruption or a
wrong direction label for the full episode, and must not be extrapolated
as an estimated error rate for the 200-episode validation set.

## Next work before another training run

1. Create explicit, visually reviewed approach/main-turn windows, including
   the visible route cue and exclusion of incidental camera adjustments.
   Review the ambiguous staircase clip through the main turn.
2. Add representative base training windows and independent validation windows;
   keep split membership unchanged. Train-only examples cannot establish an
   improvement on held-out turn states.
3. Re-score the existing checkpoints on those windows before changing training.
   Compare zero-bin probability, correct turn direction, mouse error and
   forward prediction. No training is needed for this comparison.
4. Only then prepare one bounded onset-weighting experiment, with its settings,
   runtime budget and comparison model stated before launch. Keep the prior
   group-sampling result as a negative result; do not combine another group
   fraction change or a global label shift into the onset experiment.

The 90/5/5 experiment increased exposure to entire correction clips. It did
not specifically establish exposure to reviewed decision-onset states, and
its negative held-out result remains valid for the measurements reported in
[the group-sampling comparison](GROUP_SAMPLING_COMPARISON_2026_10_31.md).
