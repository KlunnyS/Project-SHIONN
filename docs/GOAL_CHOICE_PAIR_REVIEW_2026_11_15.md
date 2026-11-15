# goal_choice_v1 pair review — 2026-11-15

## Evidence and boundaries

Reviewed every decoded frame on dense 320×180 contact sheets with synchronized
W/A/D/mouse-dx overlays, all action CSV rows and the metadata waypoint events.
This is frame-complete downscaled review, not display-exposure or collision
certification. All source metadata/actions/video SHA-256 values still match the
previous feasibility manifest. Originals and development_pending membership are
unchanged. The versioned annotation ledger is
`data/annotations/goal_choice_visual_annotations_v1_20261115.json`; timing results
are `data/annotations/goal_choice_pair_timing_review_v1_20261115.json`.

Ranges below are half-open [start,end), preserving raw waiting and continuity:

| Phase | A, right floor button | B, left cube |
|---|---|---|
| Initial waiting | [0,111) | [0,29) |
| Forward doorway approach | [111,130) | [29,44) |
| Goal steering and approach | [130,177) | [44,104) |
| Terminal event | A received after last sample by 11.9 ms | B received after last sample by 0.9 ms |

A's first mouse action at 130 is a one-count adjustment; substantive right turn
begins at 131. A additionally strafes right at 145–151 and makes later left camera
corrections. B first turns left at 44, later corrects right, releases forward at
93 and briefly strafes left at 97–99. These terminal adjustments are legitimate
observations but imperfect imitation targets, not proven optimal actions. No
use/fire/jump rows occur. Neither clip demonstrates a settled final stop: capture
ends at trigger entry, with nonzero actions remaining. Do not synthesize stopping
labels or train indefinite motion from an assumed complete stop.

No gross discontinuity, menu or object interaction was observed. Lighting visibly
changes on entering the room in both clips. Initial idle frames have gun/scene
animation; zero keyboard/mouse actions do not imply byte-identical frames.
The different waiting lengths must not become goal cues. Any later training
must mask waiting losses while retaining declared history/burn-in, and compare
identical history lengths across goal arms. No cache/sampler has been changed.

## Visual labels and shared start

281 rows cover all frames, with stable goal IDs and approximate visible landmark
centers. A is the red floor-button landmark; B is the cube. Labels identify the
landmark associated with the approach area, NOT the invisible trigger center.
Manual anchor estimates are linearly filled and checked against the complete
contact sheets; uncertainty is approximately ±8 pixels at review resolution.
They are suitable for feasibility, not pixel-accurate tracking truth. Fine
per-frame center refinement is required before precision-coordinate training.
The schema validator passes, but it cannot certify this visual truth or accuracy.

The two first frames have mean absolute pixel difference 1.349/255 after resizing
to 320×180. Their doorway/landmark arrangement matches visually; this supports an
approximately shared view, not exact pose equality. Previous user getpos checks
reported identical XYZ and <0.5° camera differences, but those readings are not
attached to these specific recordings. Map hash, per-clip engine pose, FOV and
capture-output identity are absent from metadata. Both decoded streams and action
counts match at 177/177 and 104/104. No clean held-out comparison is claimed.

## Timing and earlier failure

A/B contain zero skipped camera IDs and zero reused camera IDs. Maximum frame
ages are 36.40/36.50 ms; maximum sampling intervals are 49.15/54.69 ms at nominal
24 FPS (41.67 ms). Event receipt occurs 40.69/27.06 ms after the final image was
received. These are receive-time relationships, not exact trigger exposure or
individual keypress alignment. Input snapshots and frame receipt cannot prove
sub-frame causality; preserve this uncertainty for turn-onset analysis.

The two previous invalid A captures stopped on 216→218 and 231→233. Their saved
rows had ages up to at least 46–47 ms immediately before termination. Camera code
publishes only the latest sample; IDs count complete frames read, not source
engine frames. Two independently scheduled 24-Hz loops can sample twice or skip
an intermediate latest frame even if the producer is healthy. Thus an ID skip
alone is not proof of broken camera capture or a manual-map-load fault. The
rejected sample/timing was not saved, so the exact cause cannot be reconstructed.
The strict safeguard remains unchanged and the invalid captures remain excluded.
A future focused fix should distinguish producer EOF/staleness, consumer deadline
gaps and benign resampling, record dropped counts and timestamps, and test jitter,
pauses and sustained stalls. Do not simply disable gap checks. These successful
clips show no observed gaps; that does not remove receive-time alignment limits.

## Persistent visual target proposal

Keep the current 11-value adapter/checkpoint ID unchanged. x/y and visibility can
be visual inputs only when supplied by a human or validated visual tracker; world
projection would be privileged. Kind embedding is semantic, and the external
string goal ID is not a network identity input. Both invisible waypoint vectors
collapse to the same input, so the current interface cannot alone encode which
occluded landmark is selected. Present metadata waypoint events are evaluator-only.

Minimal new opt-in variant: select a target crop from the initial screenshot,
encode it once into a persistent 128-dimensional embedding, and condition the
existing GRU readout adapter on that vector. Use the existing visual encoder with
a documented crop preprocessing path (repeat RGB crop across frame-stack channels
if reusing the stacked encoder); freeze it initially and compare the same crop
encoder in the control. Keep policy action semantics, GRU and optional feature
gating intact. Zero-initialize new residual action heads and use a distinct
architecture/config ID. Persistent crop survives camera changes; reset only at
new episode or explicit goal replacement. No engine pose or trigger features.

Proposed descriptor: source frame hash, normalized crop bounds, target-kind label,
semantic goal ID, encoding/preprocessing version and crop hash. Initial approximate
320×180 crop bounds: A [178,94,203,108], B [124,93,139,108]. Both are small in the
start view; inspect full-resolution crops and context padding before freezing
preprocessing. Do not implement a visual tracker yet: a fixed embedding plus
current scene/GRU tests whether visual identity can control action choice. Later
cross-image localization is a separate measured capability. Human crop choice
is goal following, not autonomous goal selection. This is a design, not an
implemented visual-crop model or runnable trainer.

## Controlled pilot and smallest data increment

Current pair remains exposed development/annotation-only. It is eligible for
protocol debugging and qualitative counterfactual checks, not gradients or clean
evaluation. No eligible goal-training cohort currently exists.

A minimal split-feasibility expansion is SIX new trajectories: two new whole
start groups × A/B (four proposed TRAIN), plus a third distinct start group × A/B
(two proposed held-out). Reserve the latter before inspection/model selection;
use distinct collection sessions per split. Current default-spawn pair remains
exposed development. These are proposed memberships requiring explicit review,
not automatic promotion. This cohort could only support a within-room exploratory
pilot; one held-out start cannot establish generalization. Vary safe lateral start
position and modest camera angle while keeping both landmarks visible and verifying
pose. Do not create these numeric starts from unverified image estimates.

Matched arms: same eligible recurrent source/hash, frozen visual/GRU base, same
new readout capacity, data, batch order, losses and seed; one receives the selected
crop embedding, control receives a constant embedding. Keep a shuffled-goal
negative control offline. Proposed 200-update cap, batch 2, sequence 16/burn-in16,
Adam 1e-3, seed 20261115; no automatic runtime extension or promotion. Runtime and
VRAM are unmeasured; measure bounded CPU shape checks then user-authorized GPU
profiling before issuing a training command. Compare longer pilot only if this
feasibility budget actually fits without forgetting. Source compatibility and
source validation exposure must be audited before choosing a clean holdout claim.

Counterfactual test: same frame/history and identical initial hidden state, swap
A/B crops, roll each branch independently, report signed steering/probabilities
and requested-goal correspondence. Do not require opposite steering at the doorway:
both human trajectories initially move straight. Test goal divergence after the
common approach using history-prefix windows; a swapped goal at an already
committed trajectory is not a human counterfactual label. Training fit is diagnostic
only. For later held-out live evaluation: two requested goals × three valid repeats
from the reserved start per arm; exploratory gate ≥5/6 correct targets, no wrong
waypoint, and at least two more successes than constant-goal control. Report
invalids separately; stop on capture faults. These small-sample thresholds are
preregistered exploratory gates, not statistical generalization evidence.

Need fresh same-start paired observations at decision points for reliable offline
steering labels; current branches diverge visually when steering begins. Current
pair supports crop-swap sensitivity tests, not expert-correct counterfactual
accuracy. Test target persistence with changed camera/brief occlusion separately;
this open pair contains no dedicated recovery or occlusion case.

## Next action and schedule

No additional recording is required immediately. First Codex should prepare
full-resolution persistent crop descriptors, refine coordinate labels if retained,
and implement bounded crop-conditioned readout/sequence validation with explicit
split/provenance checks. The user should confirm whether stopping at first trigger
entry is the intended task or a settled stop is required; the latter needs a
separate dwell/stop collection protocol, not relabeling these clips.

Prepare the six-trajectory expansion only after crop/annotation contracts and
per-recording start/config evidence are ready. Proposed dates: Nov 15–17 offline
review/crop protocol; Nov 18–22 supervised start-registration and six short clips,
about 45–90 minutes setup/collection and 60–120 minutes annotation/review; Nov
23–26 ingestion/profile and explicitly authorized pilot. Estimates are provisional.
Do not request a large batch. Keep test11/test12 and recovery exclusions unchanged;
retain February freeze/March reporting buffer. Navigation recovery and interactions
remain later separately measured milestones.
