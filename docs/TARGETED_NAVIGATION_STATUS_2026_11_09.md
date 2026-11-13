# Targeted navigation investigation — status at 2026-11-09

Deployed remains `v5_recovery_proactive_v3/best.pt` (SHA2562ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483). Optional GRU and observation gating have engineering tests but no demonstrated gameplay benefit. Do not begin a general recurrent training run.

The correction-only investigation is now evaluated, including matched original versus expanded correction training and a separate startup-rehearsal intervention. Expanded corrections fit TRAIN examples somewhat better but did not improve development-held-out recovery. Opening rehearsal passed offline eligibility; its balanced-order gameplay comparison produced test1 1/6 versus deployed0/6, test11 1/6 versus0/6, and test12 6/6 for both. Candidate later inactivity occurred less often, but reliable chamber completion and repeatable test1 improvement are not established. No promotion, no additional confirmation or sampling-family training. See `OPENING_REHEARSAL_EXPERIMENT_2026_11_09.md` and its linked immutable result reviews.

## Collection/data findings

Existing assisted recording already supports autonomous navigation, physical P handoff, human recovery through exit, source-log linkage and human action labels. Separate model/human files preserve ownership but leave approximately10–11s unrecorded at takeover. Never join that interval or treat model actions as expert labels. Reviewed raw recordings remain intact and separately quarantined from generic caches.

The split now has seven TRAIN corrections (801 frames,33.375 nominal seconds at24fps) and two complete development holdouts (272 frames,11.33 nominal seconds). Original four TRAIN clips contribute550 frames. New test9 contributes194 frames of substantive wall-to-route-to-exit recovery, while two new test1 clips contribute only57 frames of exit finalization. Most TRAIN corrections are short and do not show the entire autonomous20–60s wandering history. No observation/action repair is inferred from outcome directories alone.

First-eight-frame rehearsal can reduce one offline eligibility regression without teaching a route. The live candidate still revisits walls/stairs/unmarked doors and sometimes turns away after the exit becomes visible. Frame-level improvement cannot establish objective tracking. Conversely, these failures do not prove that the visual encoder lacks particular objects or that GRU memory is the required remedy.

All test11 demonstrations remain excluded from gradients, including the candidate's successful scored trajectory. Test12 remains a regression check. Existing two heldouts are already reviewed development diagnostics, not untouched final tests. If more data is later requested, target early off-route-to-route-to-exit trajectories on TRAIN maps, not additional one-second exit crossings. No collection is requested now.

## Immediate next work

The separate bounded sequence comparison is prepared in `GRU_ROUTE_COMPARISON_PLAN_2026_11_09.md`. It uses a separate plan-only runner and addresses state across complete expert trajectories and keeps architecture/data effects distinguishable. This is a plan and an integration contract, not authorization for automatic training. No PPO, world model, hierarchy, online updates or new action semantics.

## Ordered memory comparison completed

The user authorized direct agent execution of the prepared bounded training job. It completed all three arms in15min42s, with matched16090 target frames and545 updates each; artifacts/provenance verified. All arms failed source-relative navigation/steering guards. Held-out action agreement was63.27% matched FF,59.96% persistent GRU and60.71% reset control, versus69.47% pretrained-head deployed reference. Persistent memory did not improve this comparison. No gameplay, promotion, retry or budget extension follows. See the completed-result section in `GRU_ROUTE_COMPARISON_PLAN_2026_11_09.md` and its machine-readable review. Existing deployed checkpoint remains unchanged.

The agent may run the approved bounded offline training/check commands directly; the user's earlier reservation concerns game interaction and human demonstrations. Do not ask the user merely to launch an already approved offline command. Preserve explicit runtime limits and the rule against automatic extensions/retries.

## Optional handoff observation-context calibration (2026-11-10)

An opt-in observer now records an unlabelled Wayland camera stream around the model-to-human subprocess handoff. It writes timestamped frames and orchestration links without action labels; no dataset loader consumes it. The `--record-context --no-model-video` path is documented in `OBSERVATION_CONTEXT_CALIBRATION_2026_11_10.md`. Regression suite:356 tests passed; focused session/context suite:23 passed. The exact one-run command parses successfully without starting a capture. Live display permission, encode throughput, camera alignment, and visual continuity are not established. A single user-operated calibration is required before deciding whether to collect more corrections. Do not start recurrent or correction training from these context files.
