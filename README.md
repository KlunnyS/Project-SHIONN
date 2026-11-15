# SHIONN — Self-learning Hybrid Independent Optical Neural Network

SHIONN is an experimental vision-based AI agent for *Portal 2*. It records
human demonstrations, learns a behavior-cloning policy from screen pixels, and
drives the live game through keyboard and mouse actions on Linux/Wayland.

The policy is vision-only. A separate training harness uses Source netconsole
events to detect chamber readiness and outcomes, but those events are never
provided to the model.

## Current status

Status updated **2026-11-15**:

- The existing imitation/capture pipeline remains available. The deployed
  `v5_recovery_proactive_v3` checkpoint has not been replaced.
- Optional GRU memory, observation gating and persistent visual-goal conditioning
  are implemented with separate checkpoint contracts and CPU tests. No gameplay
  improvement is established for these additions.
- The user built and instrumented `goal_choice_v1`. Two A/B pairs are reviewed
  development material; no visual-goal dataset is approved for training yet.
- Current work is a fresh three-start collection pilot: six individual
  trajectories, grouped into two training-candidate A/B pairs and one reserved
  evaluation pair. Validate the first pair before collecting the rest.
- PPO, a world model and hierarchical puzzle planning remain future work.

Start with the [current navigation guide](docs/NAVIGATION.md),
[first-pair procedure](docs/PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md), and
[data plan](docs/DATA_REQUIREMENTS_AND_COLLECTION_PLAN.md).
The [documentation index](docs/README.md) separates maintained guides from
historical evidence. Existing training records remain in the
[runbook](docs/TRAINING_RUNBOOK.md); do not execute historical commands as new
training prescriptions. See the [March plan](docs/MARCH_2027_PLAN.md) for delivery
scope and the [frozen split](docs/DATA_SPLIT.md) for existing membership rules.

## Pipeline

```text
Portal 2 + chamber VScript
    ├── screen frames ────────────────┐
    ├── physical keyboard/mouse ──────┼── record_dataset.py
    └── netconsole outcome events ────┘          │
                                          episodes/<outcome>/
                                                  │
                                models.imitation.preprocess
                                                  │
                                   cached RGB/action arrays
                                                  │
                                  models.imitation.train_bc
                                                  │
                                         best.pt / last.pt
                                                  │
                     Portal 2 ← run_imitation.py ← inference.py
```

The current policy consumes four 320×180 RGB frames stacked as 12 channels. A
residual CNN produces eight independent binary controls (`W`, `A`, `S`, `D`,
jump, use, and both portal buttons) plus horizontal and vertical mouse actions.
The recorded crouch column is not part of the current ten-output policy.

## Technology

- Python, PyTorch, NumPy, and OpenCV
- `wf-recorder`/FFmpeg for Wayland screen capture and video
- `evdev` and Linux `uinput` for physical-input recording and virtual input
- Portal 2, Source netconsole, VScript, Puzzle Maker, and Hammer++
- Proton and Hyprland on the current workstation

## Quick start

Run commands from the repository root. The live recorder and runner require
Portal 2, a Wayland session, and access to `/dev/input` and `/dev/uinput`.

```bash
./install_dependencies.sh --all
./check_dependencies.sh
```

Record demonstrations and inspect the dataset:

```bash
.venv/bin/python record_dataset.py --map dataset_test1 --episodes 20
.venv/bin/python dataset_stats.py
```

Prepare successful expert demonstrations:

```bash
.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes/goal_reached
```

Train into a new run directory:

```bash
.venv/bin/python -m models.imitation.train_bc \
  --checkpoint-dir models/imitation/checkpoints/runs/candidate
```

Run or compare checkpoints:

```bash
./run_model.sh --checkpoint models/imitation/checkpoints/runs/candidate/best.pt

.venv/bin/python run_model_sequence.py \
  --checkpoint candidate=models/imitation/checkpoints/runs/candidate/best.pt \
  --map dataset_test1 --map evaluation1 --repeats 3 --keep-focused
```

Use the [CLI reference](docs/CLI_REFERENCE.md) for all flags and defaults. Use
the [training runbook](docs/TRAINING_RUNBOOK.md), rather than the generic command
above, for the current workstation dataset and resource limits.

A local Plotly dashboard is available in [`dashboard`](dashboard/README.md).
It provides guarded controls for recording, benchmarking, and frozen-v3
training while continuing to call the same project CLIs.

## Data and generated artifacts

A finished recording contains:

```text
episodes/<outcome>/episode_<timestamp>/
├── video.mp4
├── actions.csv
└── metadata.json
```

Recordings, cached datasets, checkpoints, model-attempt videos, logs, and other
runtime artifacts are intentionally ignored by Git. On the current machine,
the large recording, cache, and checkpoint paths are symlinks into
`/mnt/extra/Project-SHIONN/`; mount that drive before any data or model work.

## Documentation

| Document | Purpose |
|---|---|
| [Documentation index](docs/README.md) | Current guides, evidence and archives. |
| [Navigation status](docs/NAVIGATION.md) | Current goal-directed milestone and next action. |
| [Program guide](docs/PROGRAM_GUIDE.md) | Implemented components, data contracts, and end-to-end data flow. |
| [CLI reference](docs/CLI_REFERENCE.md) | Common workflows plus every supported command and option. |
| [Training runbook](docs/TRAINING_RUNBOOK.md) | Current run status, resource limits, monitoring, resume, and checkpoint policy. |
| [Benchmarking](docs/BENCHMARKING.md) | Offline scoring, live trials, calibration, recovery data, and evaluation isolation. |
| [Chamber authoring](docs/CHAMBER_AUTHORING.md) | Build, instrument, compile, verify, and pilot a chamber. |
| [March 2027 plan](docs/MARCH_2027_PLAN.md) | Dated puzzle milestones, success criteria, scope decisions, and weekly work. |
| [Navigation layouts](docs/NAVIGATION_CHAMBER_LAYOUTS.md) | Geometry and intent of the navigation curriculum chambers. |
| [Changelog](CHANGELOG.md) | Notable repository behavior and compatibility changes. |
| [Dashboard](dashboard/README.md) | Local Dash control room for recording, training, and benchmarks. |

Each subject has one maintained home: current navigation status is in the
navigation guide, training records are in the runbook, and implementation details
belong in the relevant guide. Other documents should link rather than duplicate
those records.

## Repository map

```text
Project-SHIONN/
├── docs/                         maintained guides and runbooks
├── models/imitation/             dataset, network, training, and inference
├── portal_assets/scripts/        game-side VScript events
├── scripts/diagnostics/          manual hardware probes
├── tests/                        isolated Python tests
├── record_dataset.py             demonstration recorder
├── run_imitation.py              live policy runner
├── run_model_sequence.py         checkpoint/chamber comparison runner
├── benchmark_*.py                evaluation reports
├── wrapper.py                    netconsole and input control
└── recorder.py                   capture and episode primitives
```

## Development direction

1. Make behavior cloning reliable on held-out navigation chambers.
2. Add targeted recovery demonstrations for policy-induced failure states.
3. Teach button/door and cube puzzles through demonstrations and corrections.
4. Add simple portal puzzles; keep momentum and PPO outside the March critical
   path unless the core demonstration is already secured.
5. Add recurrence only when experiments show that the four-frame window is the
   limiting factor.

The main constraint is that Portal 2 is a real-time game rather than a
high-throughput simulator. Autonomous results must therefore be judged with
repeated live trials and reviewed videos, not offline action accuracy alone.

## License and third-party notice

Original SHIONN source code and documentation are licensed under the
[GNU General Public License version 3 or later](LICENSE). You may redistribute
and modify them under GPL version 3 or, at your option, any later version.

Third-party dependencies retain their own licenses. *Portal 2*, Source, and
related trademarks and game assets belong to Valve Corporation and their
respective rights holders. This independent project is not affiliated with or
endorsed by Valve, and the SHIONN license does not grant rights to third-party
game assets or gameplay media.
