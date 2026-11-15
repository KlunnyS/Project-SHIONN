# Goal-directed navigation: current status

Updated 2026-11-15. Start here for the active navigation work. Dated audits and
reviews preserve what was known at that time; they are not current command lists.

## What is implemented

- Existing imitation policies and action semantics remain loadable; the deployed
  `v5_recovery_proactive_v3/best.pt` checkpoint has not been replaced.
- Optional observation gating, GRU memory, coordinate goals and persistent visual
  crop goals have separate architecture/config contracts. None of these additions
  has demonstrated an autonomous navigation improvement in this work.
- The visual-crop path has checkpoint/inference support, episode-local sequences,
  wait-loss masks, source/crop/split checks, bounded BC helpers and CPU
  counterfactual tools. Current goal datasets are not approved for training.
- Recorder failures retain invalid partials; registered starts use fresh bounded
  telemetry verification. Setup waypoint/terminal events prevent recording.
- The user built goal_choice_v1 and verified ready, A/B, exit and timeout events.
  **A is the RIGHT floor button; B is the LEFT cube.** Success means region entry,
  without interaction or a required settled stop.

## Current data and next action

Two A/B pairs have been captured and reviewed: the original doorway pair and the
registered interior_01 pair. All four trajectories remain exposed development
material, outside the proposed clean train/evaluation cohort. Per-episode crops,
annotation ledgers, source hashes and limitations are in the dated reviews below.

The next dataset is SIX individual trajectories: two complete A/B training-candidate
pairs and one complete A/B reserved evaluation pair. First register a fresh
left-side start with both targets visible and validate just its pair. Later
commands stay gated until the first pair's source-linked crop/annotation review
passes. Numeric new poses require user getpos/visibility/collision evidence.

Use the [paired collection procedure](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md)
for exact registration, recording and validation commands. No larger batch,
training, live model run, membership approval or promotion is automatic.

## Maintained references

| Topic | Maintained document |
|---|---|
| First new pair and full collection matrix | [Paired visual-goal collection](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md) |
| Reset configuration, tolerances and default behavior | [Registered starts](REGISTERED_STARTS.md) |
| Model/data/checkpoint interfaces and prepared experiment tools | [Visual-goal implementation](VISUAL_GOAL_IMPLEMENTATION_2026_11_15.md) |
| Goal-choice map and working Hammer calls | [Chamber specification](GOAL_CHOICE_CHAMBER_SPEC.md) |
| Proactive quantities, effort and dependencies | [Data requirements](DATA_REQUIREMENTS_AND_COLLECTION_PLAN.md) |
| General build/install workflow | [Chamber authoring](CHAMBER_AUTHORING.md) |
| Broader delivery scope | [March plan](MARCH_2027_PLAN.md), [research roadmap](SHIONN_Goal_Directed_Navigation_Roadmap_v2.md) |

## Evidence and limitations

- [Initial navigation audit](GOAL_DIRECTED_NAVIGATION_AUDIT_2026_11_13.md).
- [Doorway pair review](GOAL_CHOICE_PAIR_REVIEW_2026_11_15.md).
- [Interior pair review](GOAL_CHOICE_INTERIOR_01_REVIEW_2026_11_15.md).
- [Archived early designs/protocols](archive/navigation/README.md).

There is no trained visual-goal candidate or demonstrated goal-directed gameplay
improvement yet. One held-out start/two trajectories is only a within-room pilot;
frames are not independent trials. FOV is unmeasured, live reset/capture checks
remain necessary for each new start, and strict frame-gap timing behavior remains
an open issue. Privileged pose/events are setup/evaluation data, never policy
inputs. PPO, a world model and hierarchical planning remain future milestones.
