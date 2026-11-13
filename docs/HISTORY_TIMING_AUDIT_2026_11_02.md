# History and timing audit — 2026-11-02

The recovery-opening and repeated-history probes did not establish useful
initiation improvement. This audit ran offline, without training, loading a
learned model, running capture, accessing input devices or controlling the game.

## Saved-data checks

The audit selected all six assessment recordings, all 31 recovery recordings,
and the first eight alphabetically sorted recordings from each of the base and
proactive caches: 53 episodes total. This is a diagnostic sample, not a random
generalization evaluation. It decoded through the first 24 usable action frames
per clip (including earlier idle frames): 2,691 source frames checked.

| Check | Result |
|---|---|
| Cached action arrays against source CSV, full selected episodes | 0 mismatched episodes |
| Raw video preprocessing against cached pixels | 0/2,691 mismatches |
| Live inference input against training input at first 24 usable frames, with matching preceding history | 0 mismatched tensors |
| Exact repeated cached image pairs within those 24-frame windows | 0 |

A tensor-capture stub replaced the learned network. Actual
`PolicyInference.predict_with_diagnostics()` constructed the normalized inputs;
the audit compared them against `BehaviorCloningDataset` outputs. This checks
colour conversion, resizing, channel layout, normalization and frame order. It
does not measure policy quality or capture freshness.

Cold inference repeats its initial image until four images arrive, while the
trimmed training dataset retains the preceding idle images. This affects the
first three inputs; on the six assessment clips their mean absolute pixel
differences were only 0.035, 0.018 and 0.007 on the 0–255 scale. This establishes
a small input difference on these saved clips, not its effect on predictions.

Saved logs from the 12-attempt November 1 group-sampling screen have median
tick intervals of 41.29–42.38 ms, close to 24 Hz, with four history images after
the third tick. One attempt has one interval over 100 ms; the remaining eleven
have none. Tick timing does not establish image freshness or uniform image
sampling, since the camera independently publishes its latest image.

## Timing uncertainty

The recorder samples physical input accumulated since the previous reset, then
selects the latest asynchronously received image. Cached training pairs use
that row's action with its image. Inference selects an image, predicts, then
submits an action. Without frame arrival IDs/timestamps, the old records cannot
establish exposure age, repeated capture samples or a precise causal label lag.

An exploratory Farneback-flow check used central-image median flow on the six
assessment clips through 60 frames after action onset, excluding button/key
activity at each compared index. It compared incoming scene displacement with
mouse labels shifted from −3 to +3 frames. The strongest per-clip offsets varied
from −3 to +3; aggregate horizontal correlations were only 0.297–0.343 and the
number of usable pairs varied by shift. Perspective, ongoing motion, clipping
and capture buffering confound this proxy. It does not justify choosing a
global shift or rebuilding training caches.

Scripts and JSON results are archived, outside gradient data, in
`model_attempts/diagnostics/history_alignment_20261102/`.

## Added diagnostics and next manual check

`WaylandCamera` now publishes a coherent image, increasing frame ID and local
monotonic receive time after a complete raw-frame read. The old image-only API
is retained. Receive time is when Python reads the frame, not compositor
exposure time; upstream buffering remains unmeasured.

Future recording CSVs append input-snapshot call brackets, selected frame ID,
receive/sample times, receive age, reuse flag and input-discard count. A changed
discard count identifies input resets through pause/resume; do not infer one
continuous input interval across it. The first sample's preceding window is
unknown. Existing action columns and caching semantics remain unchanged.

Future policy tick JSONL adds selected ID, receive/sample times, reuse/skipped
IDs, inference brackets and receive age at inference/action boundaries. First
reuse/skipped comparisons are unknown. IDs count delivered pipe frames, not
game render frames. Timing is diagnostic-only: no freshness gate, skipped-frame
repair, action change, label shift or scheduling change was introduced.

The next user-run check is one 20-second manual mouse recording, not a model
benchmark or another training-data collection. Load `dataset_test1` yourself,
close its console, and run:

```bash
.venv/bin/python record_recovery.py --map dataset_test1 \
  --manual-focus --session-controls --max-seconds 20 \
  --episodes-root model_attempts/timing_check_20261102 \
  --result-file model_attempts/timing_check_20261102/result.json
```

Focus the game and press P after the five-second minimum wait. Remain in place
without movement keys: hold still for two seconds, make a brief left mouse
movement, stop for two seconds, then make a brief right movement and stop.
Repeat until the recording ends automatically. Do not solve the chamber. These
isolated motion bursts are easier to compare with video movement onset than a
continuous sweep. The clip stays outside frozen training/assessment caches.
This diagnostic does not require chamber timeout suppression; the recorder's
local 20-second limit is sufficient. If a chamber terminal event arrives first,
the clip can end early and should be reviewed as such. The initial command with
timeout suppression failed before recording because no acknowledgement arrived;
this does not establish a missing script or the cause of the empty response.
Review timing columns and video onset before deciding whether a future-target
label experiment is warranted. No automatic training follows this check.

## Completed manual calibration and learning check

The user completed `timing_check_20261102/timeout/episode_20261102_211109_841589`.
The `timeout` outcome is expected for this bounded diagnostic: input snapshots
span 20.0001 seconds, with 481 matching CSV/video frames at 24 FPS. There are no
movement keys/buttons or input discards. Metadata says generic `model_recovery`,
but actual provenance is the manually staged mouse-only calibration described
above; it has no source model attempt and stays outside frozen caches.

Selected-frame age since Python receipt was 11.76 ms median, 14.74 ms at the
95th percentile and 51.68 ms maximum. Five of 480 successive selections reused
a receive ID and five delivered IDs were skipped. These are occasional sampling
irregularities, not evidence of a prolonged capture freeze. Upstream frame age
and the behavior under live policy inference remain unmeasured by this clip.

There are five substantial isolated horizontal mouse bursts. With incoming
scene movement measured by central-image median Farneback flow (threshold
0.1 pixels/frame) and mouse onset at abs(dx) >= 5 counts, the first four visible
onsets follow their input rows by two frames and the fifth by one. Whole-clip
horizontal correlation peaks when incoming image movement is compared with
mouse labels two rows earlier: 0.923, versus 0.894 with the same row. Per-burst
correlation peaks at two earlier rows for four bursts and one for the remaining
burst. This is consistent with a short input/render/capture response delay;
it does not identify its components or establish an optimal training-target
offset. Since the image can already lag physical input, a retrospective target
shift could make the task easier without making control more causal. Do not
shift labels or rebuild frozen caches on this evidence alone.

A subsequent read-only GPU inference check compared the parent, reused control
and history fine-tune on the first 24 usable frames of all 31 recovery training
clips and six independent assessment clips. It used normal recorded histories
and, separately, newest RGB repeated four times with the same targets.

| Metric | Control | History fine-tune |
|---|---:|---:|
| Training, normal history: idle first action | 21/31 | 18/31 |
| Training, repeated history: idle first action | 22/31 | 19/31 |
| Training, repeated history: idle first24 frames/clip | 403/744 | 274/744 |
| Assessment, repeated history: idle first24 frames/clip | 81/144 | 53/144 |
| Assessment, repeated history: binary exact match | 47.92% | 42.36% |
| Assessment, repeated history: horizontal MAE | 102.91 | 96.57 |
| Assessment, repeated history: backward recall | 0/11 | 0/11 |

The augmentation changes predictions under stationary-history intervention,
but reduced idle alone does not demonstrate correct recovery. Even on the
training recordings, 19/31 first repeated-history decisions remain idle despite
non-idle expert labels. Normal-history assessment results reproduce the probe
report. Repeated-history intervention removes legitimate motion cues and is not
a live success benchmark. No model is promoted, no optimizer was used, and no
training or game action was started by the agent.

The next proposed experiment is a bounded recovery-fit sanity check on existing
training clips: establish that correct actions at stationary recovery starts can
be learned before another full mixed-data run. The user subsequently completed
it in 34.651 seconds: all 31 full action vectors correct after 400 updates, versus
3/31 before, with idle 19→0 and backward recall 2/8→8/8. This establishes fitting
capability for known stationary starts; it does not establish transfer or isolate
which training-setting change was responsible. The diagnostic saved only a report;
no model was promoted. See the [runbook](TRAINING_RUNBOOK.md#recovery-fit-sanity-check--2026-11-02).
Keep all six assessment clips and this calibration outside gradients; do
not automatically extend the failed 90/5/5 recipe.

Analysis scripts, raw JSON, arrays and the timing plot are preserved in
`model_attempts/diagnostics/timing_calibration_20261102/`.

## Replay transfer review — 2026-11-03

Offline comparison of the six assessment first-action images with all31 replay
starts (80×45 RGB pixel MAE, three nearest examples per query) shows similar
close-wall views paired with different actions, and weaker visual coverage of
stairs and tilted views. Pixel distance does not establish state equivalence,
contradictory supervision, or the only valid recovery action. The first assessment
expects backward motion but the candidate turns slightly left; the corner
expects a larger turn but the candidate presses W while turning slightly left;
the last tilted view expects a left/up turn but the candidate strafes left.
Three remaining initial outputs are idle. Exact first-frame mismatch alone
cannot establish that all alternative actions fail at recovery.

Some initial expert targets are tiny movements (one assessment starts at dx=-1),
so exact initial-action imitation is an incomplete recovery criterion. Before
another training recipe, review sustained recovery windows and establish an
episode-level recovery holdout from existing training recordings. This can
separate memorization from transfer without putting the six assessment clips
into gradients or asking for additional recordings. No new training/game work
was performed. Exploratory montage/JSON: /tmp/shionn-replay-start-comparison.*.
