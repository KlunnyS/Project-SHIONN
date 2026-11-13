# Recovery calibration review — 2026-11-12

Reviewed session: `model_attempts/targeted_navigation_20261109/context_calibration_sessions/session_20261112_134846_503160`.
Human episode: `context_calibration_episodes_pending/goal_reached/episode_20261112_134935_742233`
under the same investigation root. Source deployed policy handed off normally;
human recorder reported goal_reached. Earlier session134654_823846 has no
completed recovery row/result and is excluded from correction selection.

Successful context audit: all14 structural checks passed,1369 timeline frames
and1369 decoded video frames. Context spans the model and human records,
including265 observations during the11.060s handoff interval. Maximum sampled
gap0.177s; maximum context frame age0.084s. The auxiliary context remains
unlabelled and training_approved=false. Its startup includes desktop/loading
and focus-check imagery; do not treat this entire stream as expert observations.
Sampled scene review establishes useful gameplay coverage, not exact cross-camera
frame synchronization. No automatic context ingestion or action joining.

Human video decoded156frames, matching156action rows and sequential frame
indices. Binary actions valid; no discarded input events; maximum frame age
0.0665s and sample gap0.0857s. First active action index28, leaving128 frames
from that point through the exit. Preserve all raw frames; record index28 as a
proposed training-start exclusion, to be applied explicitly during dataset
preparation. Camera/input snapshot timestamps are existing recorder estimates,
not hardware exposure/event alignment guarantees.

A nine-frame visual contact sheet shows initial wall-facing view, a turn toward
stairs, ascent, exit approach and doorway crossing. This is a short meaningful
wall-to-route-to-exit correction rather than only an exit crossing. The continuous
context includes preceding autonomous wandering. It does not demonstrate learned
model recovery or justify a training run by itself.

Next: collect five additional complete human corrections across TRAIN maps1 and9,
at varied representative off-route positions. Preserve complete episodes and
source links. Review all captures before freezing whole-episode TRAIN/development
assignments; do not choose assignments by candidate performance. Keep test11 out
of gradient collection and test12 for regression. Bounded feedforward correction
training follows only after review; no GRU/gating/context targets are enabled.
