# Registered interior pair review — 2026-11-15

## Completed review

Every decoded frame of A (84) and B (76) was viewed on dense 320×180 contact sheets
with synchronized W/A/D/mouse-dx overlays. All CSV rows and metadata events were
checked. The full raw videos remain untouched; no training membership changed.
Source hashes are pinned in `data/annotations/goal_choice_interior_01_annotations_v1_20261115.json`.
Both clips stay development_pending and annotations are annotation_only.

A waits [0,34), begins forward at 34 and turns right at 36. B waits [0,25), begins
forward at 25 and turns left at 28. A later strafes right and corrects camera left;
B corrects camera right near arrival. These are human demonstrations, not optimal
action labels. Both show purposeful approaches to the requested region, no
interaction/jump/fire, no menu or gross discontinuity. Success is region entry,
as explicitly agreed; no settled stop is required or invented. Both record a
single correct waypoint event, received about 6.1/6.6 ms after the last sample.

Versioned annotations cover 160 frames. Initial-wait masks leave 50 A and 51 B
movement frames without deleting history. Approximate landmark centers are
manually anchored, linearly filled and compared with the frame-complete sheets;
uncertainty is about ±8 review pixels. They are not precise invisible-trigger
projections. Use persistent initial visual crops for the planned pilot; do not
claim these approximate coordinates are certified tracking labels.

## Pose and timing evidence

Both recordings reference goal_choice_interior_01 and config hash
f3b95bda035efc498a9d4e59920d61d321951be2c0aef564c864347a1c25415f.
Final measured XYZ is identical: [320,26.44379,64.03125], yaw=90 and roll=0.
Pitch A=.001061, B=.001023 degrees. Both passed restore and final verification.
These observations support a common start; they do not certify matching physics,
map version, FOV or lighting configuration. Lighting visibly shifts on approach
in both clips. Map hash/FOV still need explicit future recording provenance.

No skipped or reused camera IDs in either clip. Max frame age is 37.42 ms A,
23.27 ms B; max sample interval 52.33/53.78 ms at nominal 24 Hz. Receive timestamps
and input snapshot brackets retain the alignment limitations documented in the
first pair review. No failure evidence justifies reclassifying the previous
invalid captures. Strict frame-gap resampling behavior remains a separate issue.

Timing summary: `data/annotations/goal_choice_interior_01_review_v1_20261115.json`.

## Persistent visual evidence and readiness

New native-resolution button/cube crops and source-frame/video/crop hashes are in
`data/annotations/visual_targets/goal_choice_interior_01_v1/targets.json`.
Both crops were visually inspected. They are persistent target selections, not
trained embeddings. Human selection does not demonstrate autonomous goal choice.
Existing offline loader verified source hashes/counts and generated wait masks
for both clips. Schema validation passes for two episodes; source metadata still
explicitly forbids training.

The default-spawn pair and this interior pair are exposed development examples,
not a train/test split. Protocol/reset feasibility is now stronger, but training
is still blocked by approved independent data, crop-conditioned architecture and
sequence-batching integration, pinned source compatibility and held-out scoring.
No new recordings are needed immediately. Do not repurpose these clips as a
clean holdout or silently approve training to remove the block.

Next Codex step is an opt-in persistent crop-conditioned readout interface and
sequence dataset with matched-history sampling, split/hash enforcement and CPU
forward/backward tests. Keep the existing coordinate adapter/default checkpoint
intact. No GPU run is authorized. After this infrastructure is concrete, preregister
new start/session groups and obtain explicit approval for the small collection
expansion; six fresh trajectories remain a provisional minimum, not a claim of
sufficient generalization data.

## Validation

Existing goal validator reports valid=true, episode_count=2, errors=[]. Real
loader/crop extraction verified all six original file hashes and decoded exactly
84/76 frames; resulting loss-mask counts are 50/51. Crop extraction does not
modify originals and refuses overwrite. No runtime/model code changed during
this review; broader tests were not repeated without a new code change. No
Portal 2 commands, recording, training, promotion or deployed checkpoint edits
were performed by Codex.

### Subsequent implementation

The next Codex software step listed above is now implemented and CPU-tested;
see `VISUAL_GOAL_IMPLEMENTATION_2026_11_15.md`. No new data was collected or
approved and no trained candidate exists. Training/evaluation conclusions remain
blocked by eligible independent data and explicitly authorized experiments.
