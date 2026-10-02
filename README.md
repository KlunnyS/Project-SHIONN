# SHIONN — Self-learning Hybrid Independent Optical Neural Network

SHIONN is an experimental vision-based AI agent for *Portal 2*. It records
human demonstrations, learns a behavior-cloning policy from screen pixels, and
drives the live game through keyboard and mouse actions on Linux/Wayland.

The policy is vision-only. A separate training harness uses Source netconsole
events to detect chamber readiness and outcomes, but those events are never
provided to the model.

## Current status

Status last recorded on **2026-09-28**:

- The recording, preprocessing, training, checkpoint, and live-inference
  pipeline works end to end.
- The base dataset contains 850 successful demonstrations across 12 navigation
  chambers: 221,883 aligned rows, or about 2.57 hours at 24 Hz. Generated data
  is stored outside Git.
- The current candidate family is the 13.2M-parameter residual
  `shionn_imitation_v5` policy with binned mouse actions and complete-chamber
  validation.
- The completed v5 baseline stopped after epoch 6 and retained epoch 2 as its
  best checkpoint (`2.677965` validation loss).
- A corrected live check completed held-out `dataset_test12` once; repeated
  clean trials are still required before the model can be treated as reliable.
- Reinforcement learning, reward shaping, recurrence, and portal-mechanics
  curricula remain future work.

The dated operational record, including exact checkpoint paths and safe
commands, lives in the [training runbook](docs/TRAINING_RUNBOOK.md).

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
| [Program guide](docs/PROGRAM_GUIDE.md) | Implemented components, data contracts, and end-to-end data flow. |
| [CLI reference](docs/CLI_REFERENCE.md) | Common workflows plus every supported command and option. |
| [Training runbook](docs/TRAINING_RUNBOOK.md) | Current run status, resource limits, monitoring, resume, and checkpoint policy. |
| [Benchmarking](docs/BENCHMARKING.md) | Offline scoring, live trials, calibration, recovery data, and evaluation isolation. |
| [Chamber authoring](docs/CHAMBER_AUTHORING.md) | Build, instrument, compile, verify, and pilot a chamber. |
| [Navigation layouts](docs/NAVIGATION_CHAMBER_LAYOUTS.md) | Geometry and intent of the navigation curriculum chambers. |
| [Changelog](CHANGELOG.md) | Notable repository behavior and compatibility changes. |

Each subject has one maintained home: avoid copying flags into architecture
documents or copying current run state outside the training runbook.

## Repository map

```text
Project-SHIONN/
├── docs/                         maintained guides and runbooks
├── models/imitation/             dataset, network, training, and inference
├── portal_assets/scripts/        game-side VScript events
├── scripts/diagnostics/          manual hardware probes
├── scripts/legacy/               retained one-off helpers
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
3. Introduce a Gym-style environment and actor-critic/value head for PPO.
4. Expand the curriculum to interaction, cubes, portals, momentum, and
   multi-step chambers.
5. Add recurrence only when experiments show that the four-frame window is the
   limiting factor.

The main constraint is that Portal 2 is a real-time game rather than a
high-throughput simulator. Autonomous results must therefore be judged with
repeated live trials and reviewed videos, not offline action accuracy alone.
