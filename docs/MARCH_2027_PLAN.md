# Development plan through March 2027

Planning baseline: 2026-10-31. Target a working demonstration by **2027-03-01**;
reserve March for evaluation, fixes, and the final report. These are delivery
targets, not claims that the current model already solves puzzles. Until the
user supplies an availability estimate, budget 6–10 hours of human work per
week, plus unattended training time. Replan if that budget is unavailable.

## Deliverable and scope

Deliver a vision-only learned policy that completes short, instrumented custom
Portal 2 puzzles from screen images without keyboard/mouse assistance. The
core target is two puzzle families: activate a button to open a door, and pick
up a cube, place it on a pressure button, then exit. The preferred third family
requires aiming and placing a portal to reach the exit. This is a constrained
custom-chamber result, not a claim to solve arbitrary Portal 2 levels.

Use one selected multi-task checkpoint for the final core demonstration. Keep
task-specific checkpoints as diagnostic baselines. If only specialist policies
work by the deadline, present that honestly as a reduced-scope result; do not
claim that externally selecting a policy demonstrates learned task selection.

Each puzzle must require its mechanic: a finish trigger reachable by walking
around the button/cube/portal is a map defect. Keep task state, positions,
rewards, and netconsole events out of policy observations. Use events only for
reset, assessment, and (if later implemented) training rewards. Do not add
scripted button/portal sequences to inflate learned-policy success.

Minimum delivery gate: **at least 8/10 autonomous completions on each of the
two core families' reserved evaluation variants**, within 120 seconds per
attempt. Report the raw denominator and all failures; ten trials are a small
sample, not a precise reliability estimate. Apply the same gate to the portal
family if it is included in the final claimed scope. Report performance on
training layouts separately. March's larger evaluation should use 20 attempts
per claimed family if time permits.

## Why the current plan needs to change

The recorder, trainer, runner, binary use/portal action heads, and event script
already exist. The tested local capture profile passed five consecutive
confirmation attempts; those assisted runs did not establish autonomous
navigation quality. Current V5 policies still show wall idling and pitch drift.
The base navigation data does not teach positive use or portal-fire behavior.
More navigation-only training cannot establish puzzle-solving ability.

Start interaction curriculum work before navigation is perfect. Use short
approaches in early puzzle maps so navigation errors do not consume the entire
schedule. Behaviour cloning with targeted human corrections is the primary
route. A compact temporal variant is a bounded contingency. PPO, Windows
support, broad Niri support, momentum, and general multi-room reasoning are
outside the critical path. Niri becomes necessary only if the final demo must
run on that laptop; otherwise use the verified workstation configuration.

## Dated milestones

| Dates | Work and concrete output | Decision / completion gate | Existing issues |
|---|---|---|---|
| Nov 1–14 | Preserve baseline reports; run the clean 2-model × 3-map × 5-repeat navigation comparison. Audit idle/turn/action alignment and correction sampling. Build and hand-verify one button-door map in parallel. | Publish 30 valid unassisted attempts, with capture faults reported separately, and a working button pilot. Stop navigation experimentation at the end of this window and carry the best usable baseline forward. | #31, #27, #25, #13 |
| Nov 15–30 | Implement explicit base/proactive/recovery sampling groups if needed; compare baseline, 5%+5%, and 10%+10% correction draws at fixed training budget. Record button demonstrations, train the first button policy, and review failures. | By Nov 30, button mechanic learned on training layouts: target ≥8/10 autonomous completions. If it fails, reduce approach length, inspect use-label timing and visual coverage; do not collect hundreds of navigation clips by default. | #24–26, #13, #5 |
| Dec 1–20 | Build cube pickup/place/exit maps and independent variants. Verify use press/release behaviour during carrying, record positive interactions and failure-state corrections, and train a joint navigation/button/cube candidate. | By Dec 20, target ≥6/10 autonomous cube completions on development layouts and no material button regression. Pick/place/exit must all be observed in video. | #13, #5, #26 |
| Dec 21–Jan 3 | Light holiday buffer: data quality review, artifact backup, reproducibility, and queued training. | No essential new feature is scheduled in this buffer. | #22, #27 |
| Jan 4–24 | Improve joint-policy cube results; build a simple portal chamber with one preplaced portal and one required shot. If time permits, add a second chamber requiring both portal colours. Record aiming/fire examples and corrections. | By Jan 24, core button/cube families should reach ≥8/10 on development variants; portal candidate should show repeatable real traversals. If core families fail, pause extra portal complexity and repair the core. | #13, #14, #5 |
| Jan 25–Feb 14 | Integrate the best compact policy, correct measured failures, and run repeated development evaluations. Compare one temporal variant only if the decision rule below was triggered. | By Feb 14, select final candidate(s), declared scope, task limits, maps and evaluation protocol. If portals remain unreliable, publish the two-family core result and describe portals as incomplete. | #26, #27, #14 |
| Feb 15–28 | Freeze checkpoint, preprocessing, input settings, and evaluation map versions. Run reserved evaluation variants; record videos, per-task metrics and limitations. Prepare a reproducible demo and backup machine/files. | By Mar 1, a runnable demo and preliminary report meet the declared delivery gate. Freeze features. | #27, documentation |
| Mar 1–14 | Larger repeated evaluation, failure analysis, report/diagrams, setup verification and demo rehearsal. | Evidence package links every result to its checkpoint, split, map version and attempt log. | final evaluation |
| Mar 15–31 | Submission buffer and critical fixes only. | Submit code, report, videos, reproducibility instructions and honest scope/limitations. | final delivery |

The 30-run navigation comparison belongs in the first window; it must not
become a requirement to solve every navigation map before opening puzzle work.
If invalid capture persists, investigate that specific fault and record it as
a blocker rather than converting assisted runs into benchmark successes.

## Data and training protocol

For each family, prepare two training layout variants, one development variant,
and one reserved evaluation variant. Mirror or reposition objects and start
poses enough to require visual decisions. A camera angle alone is not an
independent layout. Build only the next family's maps rather than all twelve
at once. The user authors, compiles and hand-tests maps using
[the chamber guide](CHAMBER_AUTHORING.md); code support can prepare specifications,
instrumentation, and validation tools. Archive VMF/BSP versions and map hashes.

Begin with three inspected pilot episodes per training variant. Then target
40–60 successful demonstrations per family across its training variants,
covering different approaches and action timing. This is a first batch budget,
not an assurance of sufficient data. Reserve 10–20 independent development
episodes before training. Freeze a new manifest rather than changing frozen
v3; no existing training clip becomes a validation clip. Do not record final
evaluation demonstrations into training caches.

After each candidate, review roughly ten autonomous development attempts and
collect 10–20 targeted correction clips from common failure states. Label
human takeover intervals, family, map version, objective stage and outcome.
Autonomous rollout data and human corrections have different provenance;
only reviewed human-labelled segments become expert targets. Keep interaction
and portal caches separate from base navigation and recovery caches.

Count positive use/left-fire/right-fire frames and inspect alignment before
training. Evaluate action precision/recall on interaction windows, aiming
error, and actual completion of pickup, placement, door activation and portal
traversal. A large average action-match score dominated by walking/idle frames
does not meet a puzzle milestone. Check that a learned policy releases use
between separate interactions instead of holding it throughout a task.

For the initial navigation sampler experiment, compare natural group draws,
90/5/5 base/proactive/recovery and 80/10/10 at fixed architecture, split, seed,
and update budget. Report measured draws as well as configured weights. The
existing correction fraction combines correction caches; independent group
control needs verification/implementation. For puzzle training, introduce an
explicit puzzle group and tune a small, recorded set of recipes rather than
allowing 1,100 navigation episodes to drown rare use/fire actions. Start by
comparing 50% navigation / 50% puzzle draws against 25% / 75%; balance puzzle
families within their group. Recheck navigation and previously learned tasks
after every joint fine-tune. Keep the prior working checkpoint.

Budget two or three complete training jobs per week, not a broad hyperparameter
search. The recorded older V5 run took about three hours; measure new dataset
and architecture costs before booking further experiments. Run the best recipe
with a second seed before claiming a stable training improvement when feasible.
Training, dataset manifests, checkpoint sidecars and logs must be resumable.

## Architecture and reinforcement-learning decisions

Keep V5 as the first puzzle baseline. If two correction rounds still produce
repeatable failures that depend on task history (for example dropping a cube
after pickup or forgetting an activated door), document paired examples and
compare one small CNN+GRU variant. Give that experiment at most two weeks;
implement recurrent dataset boundaries, inference reset, checkpoint and
latency checks together. Use previous visual/action history only, without
privileged task-stage labels at inference. A GRU is a hypothesis to evaluate,
not a guaranteed solution. Do not wait until February to begin this decision:
review at the Dec 20 cube milestone, or earlier if the evidence is clear.

Do not make PPO a dependency for puzzles. Revisit it only if button and cube
imitation already work, reset/outcome events are dependable, and a measured
rollout budget fits the remaining calendar. Issues #6, #10 and #15 remain a
later track otherwise. If PPO is required academically, schedule a small
environment/reward prototype after the imitation demo is secured and report
it separately; do not promise a trained RL improvement without rollout data.

## Evaluation, ownership and weekly cadence

The user runs all game-involved work: map hand-tests, recordings, live policy
attempts and final demos. The assistant handles code, isolated tests, saved
artifact analysis, experiment commands, documentation and issue updates when
requested. Do not launch/control the game or alter the live compositor on the
user's behalf. Keyboard-assisted runs remain capture/data diagnostics.

Until availability is confirmed, reserve about 2–3 hours/week for maps and
recordings, 1–2 for unassisted tests and clip review, 2–3 for implementation
and experiment review, and up to 2 for reporting and buffer. Keep long training
outside these interactive sessions. If only 3–5 hours/week are available,
prioritize button and cube families, one training layout each initially,
and defer portal/multi-platform work. This fallback reduces demonstrated
generality; retain a separate evaluation variant per claimed family.

Each week publish one short evidence entry: changed code/maps/data, checkpoint
and manifest, family completion counts, capture-invalid and human-assisted
counts, dominant failure, next experiment, and hours spent. Triage priorities
weekly. Every two weeks require a saved video of the current best autonomous
behaviour, even if it fails. This prevents training loss becoming the only
progress signal.

Freeze a deterministic list of evaluation starts before candidate selection.
Keep untouched evaluation layouts hidden from model selection until the final
protocol; select checkpoints on development variants. If final-test failures
are used to change training, label that set as development and create a fresh
reserved set before making unseen-layout claims. Report per-family successes,
timeouts, wall stalls, interaction/aiming failures, elapsed time, and input
faults. Log every scheduled attempt: replace capture-invalid trials to meet
the valid-trial target, and report the invalid count and reason separately.
Archive evidence outside `/tmp`, which is not durable storage.

## Next seven days

1. Preserve the current working capture profile and mount/verify local model
   storage. Copy useful existing evidence out of `/tmp` if it still exists.
2. Run the 30-attempt unassisted comparison using the documented persistent
   mouse/confinement profile; review input flags before interpreting results.
3. Audit sampler group draws, positive interaction labels and turn-onset
   alignment. Define the three bounded navigation recipes above.
4. Specify and build the first button-door map: short approach, visible button,
   exit physically blocked until activation, exactly one ready/terminal event.
5. Inspect three human pilot recordings, then begin the first button batch.

See [the model roadmap](MODEL_ROADMAP.md) for historical evidence and
[the board](https://github.com/orgs/SPSE-Zochova-2627-4c/projects/5) for task status.
Existing issues retain their acceptance criteria; this schedule does not close
unfinished tasks or authorize additional platform work automatically.
