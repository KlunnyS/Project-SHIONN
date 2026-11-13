# Targeted navigation recovery collection — 2026-11-05

## Status and sources

No targeted correction recordings dated November4 or later were found in the existing `episodes/recovery` collection. Existing goal_reached/discarded/interrupted/unused records include earlier recovery work, not reviewed evidence for this new off-route-to-exit experiment. No new experiment data has been validated and no training is authorized to start from this preparation alone.

Deployed source: `models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt`, SHA256 `2ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483`, architecture `shionn_imitation_v5_residual_binned`, no GRU or optional post-encoder feature gate. Its original training metadata records LR0.0002; a correction fine-tune requires a separately declared smaller LR, not blindly inheriting that value.

Prior candidates remain isolated:

- Whole: `models/imitation/checkpoints/runs/turn_onset_comparison_v1/whole_episode/endpoint.pt`, SHA256 `a273bbb48ec6369595f5dd1b7b83493d7ce34a724970ce067f87bac0e95ffb6d`.
- Onset: `models/imitation/checkpoints/runs/turn_onset_comparison_v1/onset_window/endpoint.pt`, SHA256 `4c8d28a9427de4d46e231933957ac457a5137ba2fa11cbbc38191fdca3701ee2`.

Baseline/onset results and interpretation remain in ROUTE_CONTINUATION_REVIEW_2026_11_04.md: test1 0/5 each; test11 deployed1/5/onset2/5; test12 5/5 each. Preserve frozen v3 roles: test1 TRAIN, test11 validation-only, test12 regression/validation. Later old recovery splits were already exposed to this deployed checkpoint; they cannot be relabeled clean holdouts.

## Existing workflow and its limits

`run_recovery_session.py` runs the policy, accepts physical P-key handoff, releases policy controls and invokes `record_recovery.py` without loading a new map. It writes a session.jsonl linking model JSONL/MP4, human recording and outcome. Recorder metadata labels human clips `source=model_recovery` and records `source_attempt`. Model actions stay in the source attempt; only the human CSV may become correction targets. Physical key/mouse tracking and frame-receive/input-snapshot timing are reused unchanged.

This is a **segmented assisted episode**, not a gapless recording. Model video ends before the recorder process starts; focus/mouse checks and waiting are outside both segments. Keep that gap explicit. Do not concatenate videos/actions into a continuous episode or invent labels during it. Keep still during transition and do not navigate before the human recording starts. If the view is substantially changed during preparation, note it and reject/retry rather than claiming recovery starts at the original failure. The full HUMAN recovery-to-exit trajectory is continuous when no pause/discard occurs; the complete logical episode is represented by linked artifacts. Gapless whole-attempt video would require another recorder integration and is not implemented here.

## Environment confirmation: user only

The user operates all game/control work. Live environment is NOT verified by this document or offline tests. Use the home desktop with physical keyboard/mouse; SSH alone cannot demonstrate human corrections.

Before a session, confirm Portal2 is open under the previously working Gamescope configuration, console port8020 works, DP-1 contains the game, capture permission works and physical mouse movement turns the camera. P must be unbound in Portal2; the session temporarily unbinds/restores R. Keep Esc/K emergency controls available. Existing Gamescope grab problems may require the user to refresh workspace focus before starting. Do not disable checks to force a capture.

The session requires `shionn_events.nut` chamber-timeout suppression acknowledgement. The earlier empty-ack failure remains a hard stop: inspect installed event script/console before retrying, do not silently continue with a chamber timer. The handoff mode does not support the scored sequence's automatic mouse-preflight/focus-guard combination; its physical handoff verification/test mode is a separate collection check, never evidence of a scored autonomous attempt.

First run one disposable rehearsal, only after the above confirmation:

```fish
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --target-corrections 1 \
  --max-model-attempts 1 \
  --model-seconds 60 \
  --recovery-seconds 0 \
  --output DP-1 \
  --test-mode \
  --session-root model_attempts/targeted_navigation_20261105/rehearsal
```

Test mode writes diagnostics, not training videos/CSVs. Allow the model to reach a representative failure, press P once and follow physical mouse/focus prompts. Confirm control can be recovered and the chamber can be completed. Stop on camera jumps, lost capture, timeout-ack failure or unexpected game actions. Do not proceed to collection until this works.

## Collect a small pilot

After successful rehearsal, run:

```fish
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --target-corrections 3 \
  --max-model-attempts 6 \
  --model-seconds 60 \
  --recovery-seconds 0 \
  --output DP-1 \
  --episodes-root model_attempts/targeted_navigation_20261105/episodes_pending \
  --session-root model_attempts/targeted_navigation_20261105/sessions
```

Do not use `--no-model-video`: retain both segments. Three successful corrections is a review pilot, not sufficient evidence for a training run. A zero correction limit means no recorder timer; the user may stop at any point. Maximum autonomous time is6×60s plus loading; human duration is unrestricted.

1. Leave controls untouched while the model navigates. Choose a position where it is wandering/off-route, approaching the wrong wall or abandoning a useful corridor. Do not intervene merely because a turn is imperfect.
2. Press P before the model's60s limit; a timeout restarts rather than offering a correction. Hold keyboard/mouse still while policy inputs are released and focus checks run.
3. Follow the physical mouse verification prompts. Start purposeful navigation only after `STARTING RECORDING`. Record a note of the failure category and any preparation adjustment. If physical capture is unreliable, use R to discard/retry, not an unrecorded workaround.
4. Demonstrate the whole route back to useful progress and through the exit. Use natural steering and necessary movement, not a prescribed90-degree turn. Do not stop immediately after escaping a wall.
5. Let `goal_reached` finish the recording. R discards/restarts; K stops the session. Esc pauses/resumes during human recording and is emergency stop during model control; any paused clip requires segmentation/review and is not a continuous recurrent trajectory.

## Review/quarantine gate

`episodes_pending` is quarantine by location: no automatic cache discovery or frozen-manifest addition. Existing recorder outcome directories separate goal_reached, discarded, interrupted and timeout, but outcome alone is not approval. There is no general automatic semantic reviewer; explicit per-episode review is required.

For each pilot retain source model log/video, human metadata/video/CSV, recovery_result.json, handoff diagnostics and session.jsonl. Verify links and checkpoint SHA, map, source=model_recovery, source handoff reason and human goal outcome. Check video decoding/frame count against CSV, sequential frame_idx, finite timestamps/action fields, monotonic sampling brackets, nonnegative frame age, binary keyboard/button values and plausible mouse changes. Review timing gaps/reused frames and focus/input diagnostics. Current receive timestamps do not measure compositor exposure time, so perfect physical event alignment is not claimed.

Watch the full human clip: require meaningful off-route recovery followed by exit crossing, no model input overlap, focus errors, discontinuous pauses, substantial unrecorded route correction or corrupted labels. A clean replay from spawn alone does not meet the target. Preserve original files; mark rejected reasons rather than rewriting action labels.

Create a versioned review/split manifest after inspection: episode ID, raw hashes, source log/checkpoint, takeover timestamp from model log, human start timestamp from CSV, explicit unrecorded transition, human-only target ownership, map, review status/reason, recovery-to-route and route-to-exit frame windows. Reserve WHOLE new correction episodes for held-out diagnostics before gradients. Separate nearby/repeated same-state trajectories together where possible; small same-chamber holdouts are not unseen-map generalization. Do not split one recording between train and holdout. Keep full approved human trajectories in cache for later recurrent experiments.

## Draft correction-only experiment (blocked on reviewed data)

Source is the deployed checkpoint above; strict existing architecture, preprocessing, mouse bins and action semantics. No GRU or optional feature gate. Fresh optimizer, isolated candidate directory and parent/data/code hashes. Do not invoke old pinned experiment helpers with a substituted source or rewrite their guards.

Proposed pilot after enough reviewed TRAIN/held-out correction episodes exist: matched source control and candidate, batch8, fresh AdamW LR1e-5, at most1000 successful updates, mix correction frames with original base TRAIN rehearsal (initial proposal20% corrections/80% base). Preserve full trajectories even though this feedforward arm samples frames. Fit no new mouse bins on validation data. Select whole-episode membership and correction weighting only after measuring the approved cohort. Review whether original correction rehearsal is also needed to avoid losing existing recovery behavior.

Before execution provide a plan-only validation, fixed seed/draw schedule, no-grad held-out assessment and frozen navigation checks, an actual runtime estimate and hard ceiling, explicit AMP skipped-update accounting and no automatic retry/promotion. The proposed1000updates and mixture are a draft, not a prepared executable experiment. No training command is issued while pilot recordings are absent/unreviewed.

## Autonomous evaluation and acceptance

Compare unchanged source and candidate through the established map-first persistent-mouse sequence on test1/test11/test12, with identical60s limits, outputDP-1, capture/focus guards and mouse preflight. The user runs gameplay, hands off. Assisted runs never enter scored completion totals. Declare model order/repeats before execution and balance order across repeated sequences where possible; do not selectively rerun failed policies.

Report valid/invalid attempts, goals/timeouts, completion times, startup idle/freezes, wall stalls and sustained return-to-route progress from video/action review. Mark capture failures invalid and retain them. Log observed route recovery with reviewed time windows rather than inferring it from movement alone. Require repeatable completion/recovery gains on test1, preserved test12 performance and reported test11 behavior; a tiny offline accuracy gain or single successful run cannot authorize promotion. Set exact acceptable regression margins and number of repeats before the candidate is tested, based on the available run budget.

## Subsequent GRU comparison

Only after correction-only results: compare feedforward versus GRU on identical approved TRAIN episodes and complete held-out trajectories, with gating disabled for both initially. Keep preprocessing/action targets fixed; match supervised target-frame exposure and report updates/compute separately, since a sequence update contains more targets than a frame update. Use original-data-only versus original+correction arms if budget permits to separate architecture and data effects. Freeze episode membership, burn-in/chunk settings, seeds and budgets before training. A GRU requires a separate architecture/optimizer and is not a direct resume from the feedforward source. Use comparable initialization protocols and report transfer differences rather than attributing them to memory. Test memory-dependent behavior, wandering and sustained exit progress through the same user-run scored gate.

## Rehearsal observations — 2026-11-05

`session_20261105_161129_737452` completed a P-key handoff and test-mode goal_reached with no episode saved. The earlier `session_20261105_161021_892404` logged handoff and model summary, but no parent model_attempt completion record. This places its failure after model summary and before recovery invocation; it does not prove the specific blocked operation.

Inspection found unbounded wf-recorder shutdown in that cleanup interval. WaylandCamera.stop now waits3s after terminate, then kills the capture subprocess and waits at most another3s. No game process is killed. New tests cover normal and stalled capture termination. This bounds a plausible hang source, but the incident's exact cause and the live fix remain unverified. Startup progress messages now identify inference-library/checkpoint initialization before the console connection; loading order is unchanged and no speedup is claimed.

## Three-clip pilot received

Session `session_20261105_161705_075500` produced three human goal_reached clips (attempts3/4/5), one discarded pre-recording handoff (attempt2), and one model goal_reached (attempt1). That model success occurred in assisted collection and is not added to scored evaluation totals.

Structural audit:145/159/209 CSV rows match decoded video frames at24FPS; durations5.95/6.53/8.62s. Action values are finite/binary where expected, frame indices contiguous, row timestamps increasing, input snapshot brackets ordered and discard counters all0. Reused-frame counts1/37/27 have longest consecutive reuse1/2/1; max frame ages33/85/69ms. This flags timing variability without evidence of a prolonged capture freeze. Handoff gaps10.69/10.81/11.74s are unrecorded and must stay explicit.

Sampled human views show a wall departure, stair approach and exit progress; all three terminate goal_reached. First two starts are visually very similar, third pitched down. Hashes and per-clip metrics are in `data/annotations/targeted_navigation_pilot_20261105.json`. Status remains quarantined, structurally checked with sampled visual review; final full trajectory approval/split is not yet assigned. Never describe these as three independent recovery situations or train from a goal directory automatically.

Next collection can add three genuinely different failure viewpoints on test1 (different side of the stairs, heading or route position), using the same collection command. Do not manufacture failures by controlling the autonomous part. Reserve entire episodes and group near-duplicate starting views together before training; no split is assigned from the tiny pilot alone.

## Final pilot review and split

All six complete trajectory time spans were visually inspected at six-frame intervals (0.25s), including final frames; this is not frame-perfect semantic certification. Raw hashes were rechecked. The six model sources terminate in recovery_handoff and report zero injection failures. The human trajectories show wall-facing starts, purposeful reorientation, stair traversal and exit crossing, with no gross discontinuity visible in those samples. Approve the six clips for a bounded pilot with these stated limits.

Lock TRAIN:161823_785804,161859_257631,161954_188319,162638_835978 (all episode_20261105_ prefixes). Lock development holdout:162522_556646 and162601_434118. Keep both dark-wall starts out of gradients and all three similar white-wall starts together in TRAIN. The split has550 usable TRAIN frames and272 held-out frames after excluding leading waiting. First-action indices are25,39,34,33,35,16 in chronological order. Preserve full raw recordings and episode-local history; never concatenate across handoffs.

Timing variability (up to97ms frame age, reused frames and a104ms row gap in the last clip) remains documented. No prolonged capture freeze was found; acquisition/exposure alignment is not proven by receive timestamps. These are narrow, local wall-to-stair recoveries on one familiar map, not demonstrations of broad exploration or goal planning. The held-out clips were visually reviewed and are development diagnostics, not untouched final assessment. Autonomous scored comparison remains necessary.

Review/split is now in `data/annotations/targeted_navigation_pilot_20261105.json`; the draft experiment pins its hash. Training runner, timing budget and plan-only validation remain to be prepared. No training started and no more collection is requested at this point.

## Prepared bounded paired experiment

`run_targeted_navigation_correction.py` now defaults to read-only planning. It decodes the six reviewed human clips into RAM using the existing preprocessing, preserving full normal four-frame history; it writes no cache or model in plan mode. It validates review/raw/source hashes, source payload/sidecar contract, frozen base membership and new correction non-exposure. It rejects existing outputs.

The matched control draws10 original base TRAIN frames per update; correction draws8 identical shared base frames plus2 new TRAIN human frames. Both start from the same deployed weights, fresh AdamW LR1e-5 and fixed1000 successful float32 updates. Source bins/loss weights/smoothing/dropout/augmentation remain fixed. Neither GRU nor optional feature gate is enabled. Private data RNG and reset execution RNG protect the matched comparison. The correction sampler cycles all550TRAIN frames in random orders; episode exposures are439/432/634/495 in chronological TRAIN order for seed0. This is frame-balanced correction sampling, not equal-episode sampling. Original base sampling is chamber-balanced.

Both arms receive10000targets total; the correction arm receives8000base+2000correction targets. No prior recovery/proactive targets are rehearsed in this first pair, so old correction skills could regress; navigation and subsequent game evaluation must not be skipped. Source ancestry is recorded separately from current optimizer membership. Some expert mouse motions exceed existing decoded bin magnitudes; bins remain fixed and quantization limitations must be considered when interpreting MAE.

The unchanged source and both endpoints are evaluated on all39769navigation frames and all550TRAIN/272development-holdout correction frames, with per-episode/window metrics. Holdouts never enter training or checkpoint selection. Endpoints are staged until both arms, evaluations, final input/code/cache guards and hashing finish. `paired_complete=true` in report.json is authoritative; filenames alone do not prove completion. A timeout can leave staging or partial publication artifacts, which must not be promoted. Source metadata and original frozen manifest remain unchanged. Generic optimizer resume is disabled for these endpoints.

Planning:

```fish
.venv/bin/python run_targeted_navigation_correction.py
```

Estimated12–18minutes total on the4060, including full evaluation, with a shared18minute soft deadline and20minute outer ceiling. Estimate is based on earlier runs, not a completed run of this experiment. Do not automatically extend, retry or promote on timeout.

User execution, only after reviewing the plan:

```fish
systemd-run --user \
  --unit=shionn-targeted-navigation-correction-v1 \
  --collect \
  -p WorkingDirectory="$PWD" \
  -p LimitNOFILE=8192 \
  -p MemoryHigh=16G \
  -p MemoryMax=20G \
  -p MemorySwapMax=2G \
  -p RuntimeMaxSec=20min \
  -p TimeoutStopSec=1s \
  "$PWD/.venv/bin/python" -u run_targeted_navigation_correction.py --execute
```

Monitor with `journalctl --user -u shionn-targeted-navigation-correction-v1 -f`. No Portal2 process or desktop access is needed for this offline run. Output: `models/imitation/checkpoints/runs/targeted_navigation_correction_v1/report.json`, plus per-arm endpoint files only after full pair completion. Leave fingerprinted code, source checkpoints and data unchanged while it runs. Stop on incomplete reports; review before another action.

Engineering checks exercise the8+2matched schedule, role/group/hash guards, tiny CPU paired training/publication, source immutability and incomplete/final-hash deadline rejection. These checks are not real correction-model training. The actual-data plan passed before execution; no real optimizer run was launched by the agent.

## Completed paired experiment — 2026-11-05

Report: `models/imitation/checkpoints/runs/targeted_navigation_20261105_195411/report.json`. Paired completion true,1000successful float32 updates/arm, elapsed702.623s. Earlier stopped `targeted_navigation_correction_v1` remains incomplete and is excluded.

Equal-episode correction binary agreement: source TRAIN80.52% / held69.47%; control TRAIN80.90% / held69.84%; correction TRAIN87.08% / held69.11%. Held dx MAE20.368/20.671/20.651 source/control/correction. Held idle5.14%/4.78%/4.78%. Thus correction training fits its TRAIN clips but does not demonstrate benefit over matched rehearsal on the two development holdouts.

Navigation binary agreement84.604%/84.946%/84.998%; dx MAE6.292/6.164/6.320. Both arms pass these ordinary regression checks but fail the declared opening-idle check: source175/1600frames, control195/1600, correction193/1600; allowed maximum191 under the+1percentage-point tolerance. No threshold relaxation, automatic promotion or immediate gameplay comparison follows.

Post-hoc location diagnostic (not a predeclared selection metric): first24held active frames have binary agreement25.0%/27.1%/25.0%, next24have41.7%/39.6%/41.7%, and the remainder89.2%/89.8%/88.7%. The main imitation gap is at the beginning of off-route recovery, while later continuation is much better. This supports focused investigation of initial recovery transfer and starting-view coverage, not a claim that exit following is universally learned or that memory would solve the problem.

Keep deployed source. No new training or live runs are scheduled from this result. Held-out cohort is only two visually reviewed same-map episodes; negative transfer evidence is limited to this budget/source/data setting and does not establish that human correction learning cannot work.

## Follow-up diagnosis: backward-action exposure

Read-only normal/repeated-newest history inference on source/control/correction is summarized in `data/annotations/targeted_start_transfer_diagnostic_v1.json`. Reproduce the specific GPU diagnostic with `analyze_targeted_start_transfer.py`; full raw predictions are written to `/tmp/shionn-targeted-start-audit.json`. No gradients or game actions were used.

New TRAIN corrections contain only6backward targets/550frames; the completed2000correction draws contain23backward targets. Reconstructing the exact base schedules gives63backward targets in the shared8000base draws and18in the extra2000control draws. Thus the completed control saw81backward targets/10000 total, and correction saw86/10000. The experiment barely changed exposure to backing away. The two development holdouts contain40backward targets in their first96active frames. View-group separation and action requirements are confounded.

Held first48 backward recall is8/40source and10/40for both trained arms. All three give0/40when every input history is replaced by repeated newest RGB. Older data already has11episodes with76backward targets in first48 and94backward targets over all4477usable recovery frames. Source normal-history recall is45/76, correction50/76; repeated-newest recall6/76and10/76 respectively. There is no evidence here of catastrophic backward-skill forgetting. Repeated history removes legitimate motion and changes distribution; it does not prove a defective shortcut or justify history augmentation by itself.

At the first backward target, source predicts backward in only2/11older examples. Eight of11start backward at their first active frame; two others begin within3frames. Most of those starts have almost stationary image history. These examples exist and can support a controlled exposure test without asking for more recordings. Exact expert-action agreement is supporting evidence only: alternative turns could also recover.

### Prepared5% rehearsal test, with reused completed control

`run_backward_rehearsal.py` defaults to planning. It verifies the entire completed correction arm's source/settings/draw/data/code provenance (ignoring only mode and time budget), checkpoint hash and completion, and reuses that arm as the control. It does not initialize from that endpoint: the new arm starts fresh from the same original deployed weights with the same fresh optimizer and RNG.

Only input slot7on odd-numbered zero-based batches changes:500previous base inputs are replaced by actual backward-labeled normal-history frames from old source-exposed TRAIN recovery recordings. Other7500base and2000new human draws are unchanged from the reference. All94old backward frames are shuffled in balanced cycles; their labels are not fabricated or thresholds changed. No new recordings or holdout gradients are involved. This is a sampling/rehearsal experiment, not new architecture or memory training. The old backward examples are source-exposed and are not clean transfer tests.

Fixed1000float32updates, B10, LR1e-5, same source bins/weights/augmentation/smoothing. Evaluate full navigation and all TRAIN/held correction trajectories once at the endpoint. Shared10minute soft deadline and12minute external ceiling; estimated6–9minutes, based on the completed pair's runtime. No control retraining, automatic retry/extension or promotion. The reused control's startup-idle guard already failed; the new arm must independently pass the original source-relative guard. Initial48held backward recall should improve in BOTH episodes without increased false backward output, and full trajectory/navigation regressions must be reported. These tiny diagnostic improvements would justify at most a user-run live comparison, not promotion.

Plan: `.venv/bin/python run_backward_rehearsal.py`.

User execution after reviewing this plan:

```fish
systemd-run --user \
  --unit=shionn-backward-rehearsal-v1 \
  --collect \
  -p WorkingDirectory="$PWD" \
  -p LimitNOFILE=8192 \
  -p MemoryHigh=16G \
  -p MemoryMax=20G \
  -p MemorySwapMax=2G \
  -p RuntimeMaxSec=12min \
  -p TimeoutStopSec=1s \
  "$PWD/.venv/bin/python" -u run_backward_rehearsal.py --execute
```

Monitor `journalctl --user -u shionn-backward-rehearsal-v1 -f`. Output: `models/imitation/checkpoints/runs/backward_rehearsal_v1/report.json`. Only `comparison_complete=true` establishes completion; partial endpoints are unusable. Existing source/control/old experiments remain unchanged. Earlier partial attempts must not be overwritten. The agent prepared and validated this runner but did not launch real training.

## Completed5% rehearsal and threshold diagnosis — 2026-11-05

Report: `models/imitation/checkpoints/runs/backward_rehearsal_v1/report.json`. Comparison complete;1000updates;499.147s. Endpoint SHA matches report. Navigation guard passes: keyboard agreement85.026%, dx MAE6.252, opening idle190/1600 (source175, reused control193). Backward false positives on navigation rise9source/12control/47rehearsal, while true positives rise48/55/71 of155expert backward frames. No gameplay benefit has been tested.

Held-out whole-trajectory keyboard agreement:69.47%source/69.11%control/70.56%rehearsal. First48held backward matches:8/40source,10/40control,13/40rehearsal. Per episode, control4/29and6/11 becomes6/29and7/11. False backward predictions rise from1to4total; these occur near human backward-release boundaries (relative frame31in first clip and11–13in second). First predicted backward remains at offsets2and3after the expert begins at0, identical to control. Thus the predeclared no-increased-false-backward requirement is not met. Keep source deployed; do not relax that criterion after seeing results.

Read-only confidence diagnostic: use10,000fixed-seed chamber-balanced original TRAIN samples,550new TRAIN frames,94old backward TRAIN frames and11old first-backward frames. Threshold selection uses only original TRAIN calibration negatives, constrained to the unchanged source's6false positives/9940negative samples; held-out scores are evaluated only afterward. Rehearsal default threshold0.5has22false positives and48/60backward recall on that calibration cohort versus source6false positives/32of60. Matching the source's6false positives requires threshold0.8967for rehearsal; held-out recall falls from13/40to8/40. Source's similarly selected threshold0.4391also gives8/40held recall. This small training calibration is not a full navigation safety guarantee. No thresholds or checkpoint weights were changed.

At default threshold, rehearsal recognizes10/11old first-backward examples (source2/11), yet neither new held-out first-backward command. This is consistent with fitting existing examples without reliable transfer to the new views, not proof of a unique root cause. The new TRAIN set contains no dark-wall starting-view group; both dark starts are development holdout. Reusing the held-out clips as TRAIN would invalidate the experiment. A global threshold does not solve the observed trade-off under the chosen calibration constraint.

Reproduction: `analyze_backward_threshold.py` (bounded240s read-only GPU diagnostic). Compact results: `data/annotations/backward_threshold_diagnostic_v1.json`. Full score arrays: `/tmp/shionn-backward-threshold-audit.json`.

### Next required human contribution

Collect THREE additional TRAIN demonstrations from naturally occurring wall-facing mistakes where backing away is useful before reorientation. Prefer dark-wall/stair-side views with different headings/distances; the key is a purposeful back-away/reorient/route/exit trajectory, not mechanically pressing S in every scene. Leave the autonomous part untouched and do not manufacture the failure with manual input. Keep all six reviewed clips and their roles unchanged; new clips go to a separate quarantine root. These new similar-view TRAIN examples change the next question to within-view-group episode transfer, not the prior unseen-view-group test. The two existing holdouts are already reviewed development diagnostics, never untouched final evaluation.

With Portal2 running onDP-1 and physical camera control confirmed, the USER can run:

```fish
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --target-corrections 3 \
  --max-model-attempts 8 \
  --model-seconds 60 \
  --recovery-seconds 0 \
  --output DP-1 \
  --episodes-root model_attempts/targeted_navigation_20261105/backward_episodes_pending \
  --session-root model_attempts/targeted_navigation_20261105/backward_collection_sessions
```

PressPbefore60s at a suitable failure, wait for physical mouse checks and STARTING RECORDING, then demonstrate recovery through the exit. Rdiscards/retries; Kstops. Do not move before recording begins. Natural corrective action takes precedence over forcing backward labels. No new training is prepared or launched until these clips are checked for actual starting-state/action coverage and timing.

## Remote temporal-readout pilot completed

See TEMPORAL_READOUT_COMPARISON.md for the2026-11-06frozen-encoder comparison. Persistent GRU improves held backward recall over newFF/reset controls but fails source-relative mouse regression and still misses initiation. No model promoted. Existing backward-recovery collection instructions and separate quarantine root remain the next human-data step; no further generic recurrent training is scheduled.

## Startup protocol audit

The2026-11-06review found that scored preflight reloaded the map after the countdown and waited only0.25s. See STARTUP_SETTLE_REVIEW_2026_11_06.md for the verified timing fix and three-attempt USER-run deployed baseline. No-preflight collection startup is unchanged. Old and new timing cohorts must remain separate; no timing-related performance gain is established yet. This is a useful SSH-only task while human corrections remain unavailable.
