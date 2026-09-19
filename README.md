# SHIONN — Self-learning Hybrid Independent Optical Neural Network

SHIONN is an experimental AI agent designed for the *Portal 2* environment, built to operate as a fully autonomous test subject capable of learning directly from visual input and interacting with puzzle-based physics systems.

Start with the [program guide](docs/PROGRAM_GUIDE.md) for the implemented pipeline, the [CLI reference](docs/CLI_REFERENCE.md) for every flag, and the [script cheat sheet](SCRIPT_CHEATSHEET.md) for quick commands.

The goal of this project is to explore reinforcement learning, imitation learning, and curriculum learning in a structured physics-puzzle environment using only pixel-based perception and keyboard/mouse action control.

---

## 🧠 Concept

SHIONN (**Self-learning Hybrid Independent Optical Neural Network**) is a single-player autonomous robot designed for iterative problem solving inside Portal-style test chambers.

Unlike scripted bots or rule-based agents, SHIONN:

* Observes the environment through raw screen pixels
* Builds an internal model of spatial relationships
* Learns from trial-and-error interactions
* Continuously improves through reinforcement feedback
* Combines multiple learning paradigms into a unified system (imitation + reinforcement + curriculum learning)

**Design principle — asymmetric privileged information.** The *policy* is strictly vision-only: no game-state API feeds into what the model perceives. The *training harness* underneath it, however, is allowed a privileged, structured side-channel (the Source engine remote console) for three purposes only: verifying a chamber has loaded before an episode starts, computing reward/termination during RL, and logging ground-truth outcomes for debugging. The policy never sees this channel — only the wrapper does.

---

## 🎯 Project Goals

* Train an AI agent to complete Portal-style puzzle chambers
* Use **visual-only input (no game state API)**
* Output real-time keyboard and mouse actions
* Support generalization across custom test chambers
* Learn progressively from simple movement to advanced portal mechanics
* Combine:
  * Imitation Learning
  * Reinforcement Learning
  * Curriculum Learning

### Current progress (2026-09-19)

* **Environment and recording pipeline:** operational end to end on Linux/Wayland, including screen capture, physical-input recording, virtual-input playback, netconsole events, automatic map loading, and outcome-grouped episodes.
* **Dataset:** 450 completed `goal_reached` demonstrations across four chambers, containing 128,957 aligned frame/action rows. All 450 are cached for training. Generated recordings and caches are intentionally excluded from Git.
* **Behavior cloning:** architecture `shionn_imitation_v3` has been trained and deployed in Portal 2 on an earlier dataset. The expanded-data candidate reached a best validation loss of `2.8933` at epoch 6; its epoch-20 `last.pt` is retained separately from `best.pt`. The existing checkpoints do not incorporate the newest 450-episode dataset.
* **Live evaluation:** the policy can complete the initial navigation chamber, but still sometimes enters wall-facing states and fails to recover. Its performance on the newer chambers needs live evaluation.
* **Next milestone:** stabilize Stage 2 navigation and recovery behavior before adding the Stage 3 actor-critic/value head. PPO, reward shaping, recurrence, and portal-mechanics curricula remain planned work.
* **Automated checks:** 33 unit tests currently cover recording, preprocessing, checkpointing, inference utilities, dataset statistics, timeout handling, the Escape-key stop path, and model/chamber sequence planning.

---

## 🧩 Environment

### Target Platform

* *Portal 2*, running through Proton on Linux/Wayland; the live runner includes Hyprland-specific focus recovery
* Custom chambers built with Puzzle Maker, hand-edited in Hammer++
* Community workshop maps for later evaluation

### Environment Wrapper

The current wrapper provides the low-level pieces needed by a vision-based environment: capture, input injection, map control, and terminal events. A formal Gym-style `reset()`/`step()` API is the Stage 3 target:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

Features:

* Fixed-rate screen capture
* Kernel-level keyboard and mouse control
* Bidirectional low-latency game console channel
* Episode logging
* Automatic chamber reset (cold-start and warm-reset paths)
* Terminal-event monitoring via a privileged side-channel, invisible to the policy

Dense reward calculation and the actor-critic training loop are not implemented yet.

### Key Limitation

Portal 2 is not a high-speed simulator.

Training speed is limited by:

* Game execution speed
* Physics simulation
* Rendering overhead
* Environment reset time

---

## 🔌 Technical I/O Architecture

This section documents the concrete, tested plumbing connecting Python to the running game — validated end-to-end before any model code was written.

### Bidirectional console channel — `-netconport`

Portal 2 supports `-netconport <port>`, opening a raw TCP socket onto the developer console.

* **Game → program:** VScript `printl()` calls (e.g. `EVT|goal_reached|1`) stream live over the socket as printed — no polling delay.
* **Program → game:** console commands are sent directly over the same socket, executing the same tick — no `exec`-file round-trip.

This replaced two slower alternatives that were prototyped and discarded: `-condebug` log-file tailing (one-way only) and `exec`-file polling (introduces main-thread stutter if polled faster than ~100 ms). `-netconport` is event-driven and bidirectional, with no such tradeoff.

### Input injection — `evdev` / `uinput`

Wayland deliberately blocks compositor-level synthetic input for security reasons, ruling out X11-style tools. The working solution operates one layer below the compositor: `python-evdev`'s `UInput` class creates a genuine virtual input device through the kernel's `uinput` module — indistinguishable from real hardware to both the compositor and the game.

* One-time setup: a udev rule (`KERNEL=="uinput", GROUP="input", MODE="0660"`) plus adding the user to the `input` group. Persistent across reboots — not a per-session step.
* Mouse-look requires **relative** motion events (`EV_REL`), not absolute positioning or discrete "turn" keys — Source engine mouselook reads continuous analog deltas each tick, and portal placement needs that same precision. Human demonstration data is recorded at this same granularity so imitation learning has a consistent target.
* **Key-hold precision fix:** an early bug under-reported hold duration because key state was sampled reactively (on event receipt) instead of continuously. Fixed by maintaining a persistent `held_keys` dict, updated by a dedicated blocking event-reader thread, and merely *read* (not driven) by the fixed-rate tick loop.
* **Mouse-delta precision fix:** fast mouse movement arrives as a burst of many small `REL_X`/`REL_Y` events between ticks; overwriting instead of accumulating them silently dropped most of a fast flick. Fixed by summing all events since the last tick, resetting only after the tick snapshots the total.

### Screen capture — `wlr-screencopy` via `wf-recorder`

The target Wayland compositors support the `wlr-screencopy` protocol. `grim` (one-shot screenshots) was tried first but its per-frame process-spawn/handshake overhead capped throughput; `wf-recorder` is used in production instead, streaming raw BGR frames through a long-lived pipe at a fixed rate with no per-frame handshake cost.

### Chosen resolutions and rates

| Parameter | Value / rationale |
|---|---|
| Recording resolution | 1920×1080 by default; `wf-recorder` scales the selected output when its native resolution differs |
| Model input resolution | 320×180 RGB — an offline cache preserves visual detail without decoding video during training |
| Frame stacking | Last 4 frames stacked as channels, giving implicit motion/velocity information |
| Capture rate | 24 Hz fixed tick loop |

These are tunable hyperparameters, not fixed decisions.

### Resolved deployment issues

| Issue | Cause | Fix |
|---|---|---|
| Puzzle Maker VMF export failure on an NTFS-backed Steam library (`Can't write file .../maps/preview.vmf`) | Steam library on a `fuseblk` (NTFS-via-FUSE) mount; writes through the Proton/Wine file layer onto FUSE-mounted NTFS are unreliable even though host-side writes succeed | Format a fixed-size disk image as ext4 and loop-mount it directly over `sdk_content/maps`; made persistent via an `/etc/fstab` entry with `loop,nofail` |
| Identical export failure even on genuine ext4 (`strace` showed the literal path `"GAME/home/.../sdk_content/maps"`) | Hardcoded Source engine `GAME` path-alias concatenated directly onto an absolute path with no separator by the native Linux build — independent of filesystem or permissions | Force Portal 2 to run under **Proton** instead of its native Linux binary; Proton resolves the alias through Wine's translation layer, which does not exhibit the bug |
| `Could not open instance file instances/p2editor/door_entrance_4.vmf` during VBSP compile, after the `GAME`-alias bug was fixed | Puzzle Maker's stock instance library was deleted as a side effect of replacing `sdk_content/maps` while chasing the NTFS fix above | Steam → Verify Integrity of Game Files, which redownloads the missing stock instance library onto the (now ext4-backed) mount |
| Steam Linux Runtime sandbox (`pressure-vessel`/`bwrap`) path visibility — considered, not the actual cause here | Portal 2 launches inside a sandboxed container; only bind-mounted paths are visible to it | `PRESSURE_VESSEL_FILESYSTEMS_RW` as a launch-option env var was tried and did not resolve this specific issue (the real cause was the `GAME`-alias bug above); documented since it remains a plausible cause for similar symptoms elsewhere |

---

## 🗺️ Chamber Authoring Pipeline

Puzzle Maker alone cannot express custom VScript triggers and always compiles to a fixed `preview.bsp` with no rename option. The working pipeline hands off from Puzzle Maker to Hammer++:

1. **Puzzle Maker** — build chamber geometry/puzzle logic with the voxel editor (fast, testable).
2. **Export** — run test/simulate, producing `sdk_content/maps/preview.vmf`, a standard Hammer-editable VMF (voxel chambers are "baked" into ordinary Source brushes on export).
3. **Hammer++** — open `preview.vmf` directly; add a `logic_script` entity and trigger brushes (below).
4. **Compile** — VBSP → VVIS → VRAD from Hammer with an explicit output name, landing directly in `portal2/maps/<name>.bsp`.

> Once a chamber is hand-edited in Hammer, re-importing it into Puzzle Maker's voxel editor is not practical. Treat the Puzzle Maker export as one-way, and finalize puzzle geometry before moving to Hammer.

### VScript event hooks

A single shared `.nut` script, referenced by every chamber's `logic_script` entity, exposes guarded signal functions. The repository copy is `portal_assets/scripts/vscripts/shionn_events.nut`; copy it to Portal 2's `portal2/scripts/vscripts/shionn_events.nut` before compiling or running the map.

```squirrel
function SignalChamberReady() { SHIONNEmitEvent("chamber_ready", Time()); }
function SignalGoalReached() { SHIONNEmitEvent("goal_reached", 1); }
function SignalEpisodeFailed(reason) { SHIONNEmitEvent("episode_failed", reason); }
```

Create one `logic_script` named `shionn_event_script` with **Entity Scripts** set to `shionn_events.nut`. Wire chamber entities to it with these outputs:

| Source entity | Output | Target | Input | Parameter |
|---|---|---|---|---|
| Delayed `logic_auto` or start `trigger_once` | `OnMapSpawn` or `OnStartTouch` | `shionn_event_script` | `RunScriptCode` | `SignalChamberReady()` |
| Player-only goal `trigger_once` | `OnStartTouch` | `shionn_event_script` | `RunScriptCode` | `SignalGoalReached()` |
| Player-only fail volume | `OnStartTouch` | `shionn_event_script` | `RunScriptCode` | `SignalOutOfBounds()` |
| Timeout `logic_timer` | `OnTimer` | `shionn_event_script` | `RunScriptCode` | `SignalTimeout()` |

Use a short delay on `OnMapSpawn` so ready is emitted after the player has spawned, or place a player-only start trigger just beyond the spawn point. Keep the goal as a player-only `trigger_once`, never a `trigger_multiple`. The shared script suppresses duplicate start and terminal events. Every new chamber is manually verified — watch raw netconsole output, walk it by hand, confirm each signal fires exactly once — **before** it enters the recorder's rotation.

### Chamber registration for named loading

Puzzle Maker's own compiler always emits `preview.bsp`. A lightweight registration step copies each finalized compile to a permanent, distinctly-named file so it loads via a plain `map <name>` command:

```python
import shutil
from pathlib import Path

MAPS_DIR = Path(".../portal2/maps")
PREVIEW_BSP = MAPS_DIR / "puzzlemaker" / "preview.bsp"

def register_chamber(name: str):
    if not PREVIEW_BSP.exists():
        raise FileNotFoundError("preview.bsp not found - compile first")
    dest = MAPS_DIR / f"{name}.bsp"
    shutil.copy2(PREVIEW_BSP, dest)
    return dest
```

Chambers compiled directly in Hammer++ skip this step — Hammer allows a distinct output filename from the start.

---

## 🎮 Action Space Design

A single mutually-exclusive softmax over all actions was **rejected**: it forces "do nothing" to compete against every other action for probability mass, and it structurally prevents simultaneous actions (strafing while aiming while firing a portal), which Portal 2 requires continuously.

### Factored (multi-discrete) action space

* Four independent binary heads: `move_w`, `move_a`, `move_s`, `move_d`
* Independent binary heads: `jump`, `use`, `fire_left` (portal 1), `fire_right` (portal 2)
* A continuous head: `mouse_dx`, `mouse_dy`, modeled as a **Gaussian** (mean + log-std) rather than deterministic regression

**"Do nothing" is not a modeled category.** It emerges for free as the zero-state across all heads — no dedicated no-op logit competes with real actions.

**No-op collapse risk.** Removing the *structural* pressure toward a no-op doesn't remove a *reward-shaping* risk: under sparse terminal-only reward, an undertrained policy can collapse to always-no-op, since standing still avoids risking a negative outcome. Mitigations: a small negative per-timestep reward so standing still is never free; distance-to-goal delta shaping for signal before the sparse terminal reward is reached; and behavior-cloning pretraining before RL, so the initial policy already has a human-like prior against standing still.

**Why the Gaussian mouse head matters beyond Stage 2.** During RL, sampling from the distribution gives exploration noise on aim for free — useful for exploring portal placement options — while during behavior cloning it trains as ordinary regression to the demonstrator's recorded mouse delta.

**Action-to-input translation.** Because the network emits a fresh prediction every tick but real key semantics are edge-triggered, the translation layer diffs against previous state and only emits an event on a transition — not every tick — to avoid event spam and to guarantee held keys are correctly released.

---

## 📹 Data Collection — The Recorder

### Recommended Dataset Size

| Purpose | Duration |
|---|---|
| Pilot batch (validate training pipeline) | 20–30 minutes |
| Minimum viable | 2–8 hours |
| Ideal | 10–20 hours |

The pilot batch is recorded and used to build/debug the Stage 2 training script *before* committing to full recording volume, so a data-format bug is caught cheaply rather than after hours of recording.

### Initial Skills (Stage 1 target)

* Walking
* Looking around
* Jumping
* Interacting with objects

### Recorder design

1. **Input reading** — `evdev.list_devices()` auto-scans for the physical keyboard/mouse (by `REL_X`/`REL_Y` and key capabilities) and reads them live on a dedicated blocking thread, separate from the `UInput` *output* device used for playback/inference.
2. **Fixed-rate capture** — `wf-recorder` piping raw BGR frames at 24 Hz and 1920×1080.
3. **Sync loop** — a precise 24 Hz tick snapshots the most recent frame together with the current persistent key-hold state and the accumulated-then-reset mouse delta, writing one aligned row.
4. **Storage** — completed recordings are grouped by outcome, for example `episodes/goal_reached/episode_<timestamp>/` and `episodes/timeout/episode_<timestamp>/`. Each contains a compressed, audio-free `video.mp4`, an `actions.csv` with `timestamp`, `frame_idx`, movement, jump, crouch, use, portal-fire, and mouse-delta columns, and a `metadata.json` containing the recorded map name. The current ten-output policy does not train on the recorded `crouch` column. Active captures remain under `episodes/.in_progress/` until finalized.
5. **Episode boundaries** — driven by the same `EVT|chamber_ready` / `EVT|goal_reached` / `EVT|episode_failed` netconsole hooks used by the environment wrapper. The terminal event selects the completed episode's result directory.

A playback function re-executes a recorded episode's actions through the same input-injection path used for live inference, validating the full record → store → replay loop independent of any model.

For continuous human-demonstration capture on `dataset_test1`, run:

```bash
.venv/bin/python record_dataset.py
```

The command launches Portal 2 with netconsole enabled when needed, waits five seconds for the user to focus the game, and repeatedly loads `dataset_test1`. Each `EVT|chamber_ready` starts a synchronized 1920×1080 video/action recording at 24 FPS; audio is disabled. `EVT|goal_reached`, `EVT|episode_failed`, or a local 30-second safety limit ends the episode. The chamber reloads while the success target remains unmet. Recording continues until `Ctrl+C`; use `--map NAME` to select and label a different chamber, `--episodes N` to stop after N successful episodes, `--focus-delay SECONDS` to change the initial delay, `--restart-delay SECONDS` to change the pause between attempts, `--output NAME` to select a monitor reported by `wf-recorder -L`, and `--mouse-device PATH_OR_NAME` to override pointer detection. Failed attempts are saved under their outcome but do not count toward `--episodes`; the recorder reports progress after each attempt and returns Portal 2 to the main menu when the target is reached.

To count completed episodes and aligned action rows by outcome, recording date, and chamber, run:

```bash
.venv/bin/python dataset_stats.py
```

Pass another dataset directory as the optional first argument. In-progress recordings and entries missing `video.mp4` are not included in the totals.

---

# 🛣️ Development Roadmap & Network Architecture

Each stage below lists its goal, its network structure, and its success criteria. **The core rule: never jump to a more complex architecture than the current stage demonstrably needs.** A feedforward CNN is the default; recurrence is an upgrade earned by evidence, not assumed upfront.

---

## Stage 0 — Environment Setup ✅

**Goal:** Create a stable AI training environment around Portal 2.

Requirements: screen capture pipeline, kernel-level input control, automatic chamber reset (cold-start + warm-reset), episode logging, dataset recording tools.

**Network:** none — this stage builds the environment wrapper only.

**Delivered:** tested netconsole control, `wf-recorder` capture, `evdev` recording, `uinput` action injection, automatic chamber loading, event-driven episode boundaries, playback, and live-policy execution. The formal Gym-style API and reward calculation remain Stage 3 work.

---

## Stage 0.5 — Curriculum Chamber Authoring 🚧

**Goal:** Build a small set of chambers with verified VScript event hooks before any real recording begins.

* Empty room, reach a visible goal (pure movement)
* Simple corridor / turn (navigation, looking around)
* Single button opens a door (first interaction entity)
* Carry a cube onto a button (cube mechanics)

**Network:** none.

**Current status:** `dataset_test1` through `dataset_test4` have successful recorded demonstrations. The interaction, cube, and portal chamber curricula still need evaluation as separate milestones.

---

## Stage 1 — Data Collection (Imitation Learning) ✅

**Goal:** Teach basic human-like movement and camera control by recording demonstrations.

**Implemented network contract:**

```
Input: 4 × 320×180 RGB frames stacked as 12 channels
  ↓
Seven-layer CNN trunk (64 → 128 → 256 → 384 channels,
GroupNorm + SiLU, adaptive 4×4 pooling, FC 1024 → 512)
  ↓
Factored action heads (see Action Space Design):
  move_w, move_a, move_s, move_d   → binary
  jump, use, fire_left, fire_right → binary
  mouse_dx, mouse_dy               → Gaussian (mean + log-std)
```

No recurrence and no value head are used at this stage. The recording, cache, training, checkpoint, and live-inference formats are now validated end to end.

**Success criteria:** demonstration data is being recorded reliably (validated via the playback function), in the exact factored-action format the model will be trained on.

**Current status:** achieved for the recorded navigation chambers. The local dataset contains 450 successful demonstrations and 128,957 aligned action rows; the existing model checkpoints predate the newest recordings.

---

## Stage 2 — Behavior Cloning 🚧

**Goal:** Learn to imitate human gameplay via supervised learning.

**Network — identical architecture to Stage 1, now actually trained:**

```
Input: 4 × 320×180 RGB frames stacked as 12 channels
  ↓
CNN trunk (shared)
  ↓
┌─────────────┬─────────────┬─────────────┬──────────────┐
move heads    button heads   mouse head    (no value head)
(4 × binary)  (4 × binary)   (Gaussian)
```

**Loss:** weighted cross-entropy on each binary head plus Gaussian negative log-likelihood on standardized mouse deltas, summed.

**Why no recurrence here:** short-horizon skills (walking, looking, jumping, interacting) are well covered by the 4-frame stack's implicit velocity information. Behavior cloning is also where a no-op-collapse-resistant prior gets baked in for free, since human demonstrations rarely sit idle.

**Expected result:** stable movement, basic navigation, natural camera control, reliable interaction — a baseline that substantially reduces the instability and exploration difficulty of RL trained from scratch.

**Current status:** the v3 policy is trained and can complete the navigation chamber in live attempts. Recovery from unfamiliar wall-facing states is inconsistent, so Stage 2 is still active; model-failure videos are used to identify states for new expert recovery demonstrations, not as positive behavior-cloning labels.

---

## Stage 3 — Reinforcement Learning

**Goal:** Transition from imitation to autonomous learning via reward.

**Algorithm:** PPO (Proximal Policy Optimization). Later candidates: SAC, Dreamer, IMPALA.

**Network — same CNN trunk and action heads as Stage 2, extended with a value head for actor-critic PPO:**

```
Input: 4 × 320×180 RGB frames stacked as 12 channels
  ↓
CNN trunk (shared, initialized from Stage 2's behavior-cloned weights)
  ↓
┌───────────────────────────┬──────────────┐
Factored action heads        Value head
(actor)                      (critic, scalar)
```

Still **no recurrence** — this stage's chambers (empty room, maze, button+door, cube-to-button) don't require memory beyond the frame-stack window.

**Reward source:** the privileged netconsole channel (see Environment section) — the policy never sees this data, only the wrapper computes reward/termination from it.

**Reward shaping:** goal completion, progress-toward-objective, successful interactions; small per-timestep penalty (addresses no-op collapse, see Action Space Design), penalties for excessive time and failure states.

**Chamber curriculum:**
1. Empty room, reach visible goal
2. Simple maze navigation
3. Button opens door
4. Carry cube to button

**Success criteria:** agent reliably solves training chambers without demonstrations.

---

## Stage 4 — Portal Mechanics Learning

**Goal:** Introduce Portal-specific reasoning — single portal, dual portal, momentum, multi-step chambers.

**Network — same feedforward actor-critic as Stage 3 by default:**

```
Input: 4 × 320×180 RGB frames stacked as 12 channels
  ↓
CNN trunk
  ↓
Factored action heads (actor) + Value head (critic)
```

**Upgrade condition, not a default:** only introduce recurrence (jump ahead to the Stage 5 architecture below) if a specific chamber demonstrably fails under the feedforward policy *because* the portal or relevant geometry has scrolled outside the 4-frame stack's window — e.g., placing portal A, turning a corner, and needing to still reason about its position. Don't add memory speculatively; let a concrete failure mode justify it.

**Curriculum:**
5. Single portal usage
6. Dual portal usage
7. Momentum puzzles
8. Multi-step chamber solving

**Major challenges:** sparse rewards, long planning horizons, portal-placement reasoning, momentum prediction.

---

## Stage 5 — Memory and Planning

**Goal:** Enable long-term reasoning once Stage 4 has produced concrete evidence that feedforward memory (the 4-frame stack) is insufficient.

**Network — CNN trunk feeding a recurrent core, not a plain feedforward trunk:**

```
Input: 4 × 320×180 RGB frames stacked as 12 channels
  ↓
CNN trunk
  ↓
GRU (hidden state persists across steps within an episode,
     reset on env.reset())
  ↓
Factored action heads (actor) + Value head (critic)
```

**GRU over LSTM, deliberately:** fewer parameters, one fewer gate, faster to train, and performs comparably to an LSTM at this scale of policy network — LSTM's extra cell-state complexity rarely earns its keep here.

**Recurrence is not "undo."** A GRU carries a compressed hidden state *forward* through time so the agent can remember something no longer on screen (e.g. a placed portal's location). It gives no ability to revisit or reconsider a past action — that would be a planning/search capability (e.g. lookahead or model-based rollouts), which is explicitly out of scope for this stage and noted only as a long-term direction.

**Optional alternative/upgrade — light self-attention over recent frame embeddings:**
```
CNN trunk (per-frame embeddings) → self-attention over last N embeddings → action/value heads
```
Worth considering *instead of* the GRU once there's a concrete need for "which specific past moment is this decision drawing on" — attention weights are directly inspectable, unlike a GRU's opaque hidden state. More data-hungry and harder to debug than a GRU, so it's a later upgrade, not a Stage 5 starting point.

**Required memory:** portal locations, room layout, previous interactions, puzzle state, long-term objectives.

**Expected result:** improved multi-step puzzle solving and strategic behavior.

---

## Stage 6 — Curriculum Learning (Philosophy)

**Goal:** never train "Portal AI" directly — train individual skills first, and never skip a stage.

**Network:** no new architecture — this stage is a training-order discipline that governs how Stages 1–5 are sequenced, not a model to build.

**Skill progression:**
1. Movement
2. Navigation
3. Interaction
4. Cube mechanics
5. Single portal usage
6. Dual portal usage
7. Momentum puzzles
8. Full chamber solving

**Rule:** never skip curriculum stages. Each stage becomes the foundation for the next.

---

# 🏗️ System Architecture Summary

## 1. Perception Layer

* Screen capture (`wf-recorder` / `wlr-screencopy`) at a fixed 24 Hz
* Offline resize/cache to 320×180 RGB
* 4-frame stacking for implicit motion information
* Observation encoding via CNN trunk

Possible future upgrades: Vision Transformers, self-supervised visual representations — not adopted by default; same "earn it with evidence" rule as recurrence.

## 2. Optical Neural Network (Core Reasoning)

Spatial understanding, object/portal recognition, navigation awareness. Default: CNN. Upgrade path: CNN + GRU (Stage 5), then optionally light self-attention.

## 3. Decision Policy

Factored action space, not a single categorical action:

* **Movement:** `move_w`, `move_a`, `move_s`, `move_d` (independent binary)
* **Mouse:** `mouse_dx`, `mouse_dy` (continuous, Gaussian head)
* **Interaction:** `jump`, `use`, `fire_left`, `fire_right` (independent binary)

"Do nothing" is the natural zero-state across these heads, not a modeled category.

## 4. Learning System

* **Imitation learning** — human demonstrations, Stage 1–2
* **Reinforcement learning** — reward/self-play via PPO, Stage 3+, reward computed through a privileged netconsole channel the policy never sees
* **Curriculum learning** — progressive unlocking, Stage 6 philosophy applied throughout

## 5. Environment Wrapper

Implemented low-level control over the real Portal 2 process: launch/connect and map loading over `-netconport`, netconsole-driven ready/termination events, `evdev`/`uinput` action handling, and `wf-recorder` observation capture. A Gym-like API and dense reward computation are the next-layer Stage 3 deliverables.

---

# 🧪 Training Pipeline

## Phase 1 — Human Demonstrations (Stage 1–2)

Train: movement, camera control, navigation.
Network: feedforward CNN + factored heads (see Stage 2).
Result: stable baseline behavior.

## Phase 2 — Autonomous Learning (Stage 3)

Train: exploration, interaction, puzzle solving.
Network: same trunk + factored actor heads + value head (actor-critic PPO).
Result: independent chamber completion.

## Phase 3 — Curriculum Scaling (Stage 4–5)

Introduce: buttons, cubes, doors, portals, momentum mechanics, multi-step puzzles.
Network: feedforward actor-critic by default; GRU-augmented only once a concrete memory-limited failure is observed.
Result: general-purpose Portal reasoning.

---

# 📦 Project Structure

```text
Project-SHIONN/
├── docs/
│   ├── PROGRAM_GUIDE.md       # implemented components and data flow
│   └── CLI_REFERENCE.md       # flags and defaults
├── SCRIPT_CHEATSHEET.md       # commands for all project scripts
├── models/imitation/
│   ├── network.py             # v3 CNN and legacy checkpoint model
│   ├── dataset.py             # episode-aware memory-mapped loader
│   ├── preprocess.py          # MP4/CSV → RGB/action .npy cache
│   ├── train_bc.py            # behavior-cloning trainer
│   ├── inference.py           # four-frame live inference
│   └── checkpoint.py          # portable atomic checkpoints
├── portal_assets/scripts/vscripts/
│   └── shionn_events.nut      # ready/goal/failure events
├── tests/                          # automated unit tests
├── scripts/diagnostics/            # manual hardware probes
├── scripts/legacy/                 # older one-off helpers
├── wrapper.py                      # netconsole and action control
├── recorder.py                     # capture/input/episode primitives
├── record_dataset.py               # continuous demonstration recorder
├── dataset_stats.py                # outcome/date/chamber dataset report
├── run_imitation.py                # live policy runner
├── run_model_sequence.py           # model/chamber comparison runner
├── run_model.sh                    # local Bash launcher
├── run_model.fish                  # local Fish launcher
└── run_model_ssh.sh                # SSH/Hyprland launcher
```

Generated `episodes*/`, `data/datasets/cached_frames/`, `model_attempts/`, checkpoint directories, and model backups are excluded by `.gitignore`. On this machine, recordings, cached frames, and model checkpoints live on `/mnt/extra/Project-SHIONN/` behind the project paths.

---

# Imitation Model Usage

Install dependencies in the project virtual environment, then convert completed demonstrations into the offline cache. Point `--recordings-dir` at `episodes/goal_reached` when training only from successful expert demonstrations; pointing it at `episodes/` includes every completed outcome category and should be done only deliberately.

```bash
.venv/bin/python -m models.imitation.preprocess --recordings-dir episodes/goal_reached
.venv/bin/python -m models.imitation.train_bc \
  --cache-dir data/datasets/cached_frames \
  --batch-size 8 \
  --checkpoint-dir models/imitation/checkpoints/runs/candidate
```

Preprocessing skips already cached episodes unless `--overwrite` is supplied, so the same command safely adds new recordings. The trainer splits complete episodes rather than adjacent frames, excludes ambiguous waiting frames before the demonstrator's first action, balances supported binary heads, and standardizes mouse deltas from the training split. The normalized GroupNorm/SiLU trunk avoids the constant-feature collapse observed in the earlier ReLU model.

### `.npy` cache contract

Each episode produces two NumPy binary arrays:

* `<episode>.npy`: pre-convolution RGB pixels with shape `(N, 180, 320, 3)` and type `uint8`.
* `<episode>.actions.npy`: aligned policy targets with shape `(N, 10)` and type `float32`.

The cache stores resized pixels, not CNN features. `BehaviorCloningDataset` memory-maps the arrays, stacks four frames into `(12, 180, 320)`, and the current model performs convolution again on every training batch. This is necessary because convolution weights change after every optimizer update. Live inference applies the same BGR→RGB resize and normalization path to `wf-recorder` frames.

### Training and checkpoints

The trainer reports running batch loss, per-epoch train/validation loss, and a final summary containing elapsed time, average epoch time, dataset sizes, parameter count, optimizer steps, best/final losses, checkpoint paths, per-head validation components, and peak CUDA memory when available. It stops after three epochs without a new best validation loss by default (`--early-stop-patience 0` disables this), while `--epochs` remains the maximum. Per-epoch train and validation metrics are appended to `metrics.jsonl` in the checkpoint directory; TensorBoard event files are not currently written.

Every checkpoint contains the model weights, AdamW optimizer state, epoch, global step, best validation loss, and the architecture/preprocessing/action configuration. The same configuration is also written as `*.json` beside the `*.pt` file. `last.pt` is saved after each epoch, `best.pt` tracks the lowest validation loss, and `step_*.pt` is saved every 1,000 optimizer steps by default. Resume a run on another machine with:

```bash
python -m models.imitation.train_bc --cache-dir cached_frames --device cuda --resume models/imitation/checkpoints_v3/last.pt
```

Resume only against the same cached dataset contract: changing the dataset can change class weights and mouse scales, and compatibility checks will reject a mismatched resume. Train an expanded dataset into a new checkpoint directory instead. Directory names such as `checkpoints_expanded_v1` are run labels; the architecture version is stored inside the checkpoint configuration.

Checkpoints and backups are excluded from Git. Back up both `best.pt` and its matching `best.json` before promoting a new candidate.

### Live evaluation

For live policy use, construct `PolicyInference` with a checkpoint, supply raw BGR frames from `WaylandCamera`, and call `policy.apply(controller, frame)`. It keeps the four-frame history and calls `Portal2Controller.apply_action()` with the predicted factored action. The runner resets policy history and releases held controls at chamber boundaries; player position remains intentionally deferred to Stage 3 reward shaping.

The Bash and Fish launchers provide the current machine defaults. Arguments appended at launch override those defaults, so candidate checkpoints can be evaluated without editing the scripts:

```bash
./run_model.sh --checkpoint models/imitation/checkpoints/runs/candidate/best.pt
./run_model.fish --checkpoint models/imitation/checkpoints/runs/candidate/best.pt
```

`run_model_ssh.sh` adds `--no-launch` and Hyprland focus recovery for an already-running game. The launchers select output `DP-1`, map `dataset_test1`, and a 60-second run. The Bash/SSH launchers use `models/imitation/checkpoints_v3/best.pt`; the Fish launcher uses `models/imitation/checkpoints_450_v3/best.pt`. Override these defaults when the machine layout or candidate changes. Launchers do not automatically discover the newest checkpoint.

To compare several models on several chambers, run the sequence tool. It tries every model/chamber pair and saves a separate video, diagnostic log, and summary entry for each attempt:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint old=models/imitation/checkpoints_v3/best.pt \
  --checkpoint new=models/imitation/checkpoints_450_v3/best.pt \
  --map dataset_test1 --map evaluation1 --repeats 3 --output DP-1
```

Append `--plan-only` to inspect the 12-job order without launching the game. Results go under `model_attempts/sequences/sequence_<timestamp>/`; Escape or Ctrl+C stops the remaining jobs. See the [CLI reference](docs/CLI_REFERENCE.md) for recording and focus options.

The live runner launches Portal 2 with `-netconport 8020` if needed, captures the selected Wayland output at 24 Hz, and releases every held action on exit. While a positive `--max-seconds` deadline is configured, `EVT|episode_failed|timeout` from the chamber is logged but ignored so the runner's own deadline remains authoritative. Use `--max-seconds 0` for an unlimited run, where chamber timeout events remain terminal. A global physical-keyboard monitor makes `Esc` an emergency stop even while Portal has focus; `Ctrl+C` remains available from the terminal. Use `wf-recorder -L` followed by `--output OUTPUT_NAME` if the wrong monitor is captured. Before allowing input, a useful capture-only check is:

```bash
.venv/bin/python run_imitation.py --checkpoint models/imitation/checkpoints_v3/best.pt --no-launch --dry-run --max-seconds 10
```

Add `--record-video` to save the exact frames used for an inference attempt as a timestamped H.264 MP4 under `model_attempts/`, separate from demonstration episodes and cached training data. Each recorded attempt also gets a matching `.jsonl` diagnostic log containing per-tick actions, policy probabilities, mouse distribution, visual motion, focus state, and console events. Use `--recording-dir PATH` to select a different review-video directory, or `--log-actions` to save diagnostics without video. `--verbose` prints a compact status line once per second. On Hyprland, `--keep-focused` focuses Portal, activates its XWayland input grab, unpauses it before inference, and restores input if another window takes focus. The current XWayland/uinput path can still lose the raw mouse grab on some setups; check `focus_losses` in the JSONL summary when diagnosing an attempt.

Model-attempt videos and model-generated actions are diagnostic data, not expert labels. When a rollout becomes stuck, reproduce or preserve that visual state and record a successful human recovery; do not add the model's failed actions to the behavior-cloning cache as if they were correct.

Recording remains a Linux/Wayland job (`wf-recorder`, `evdev`, and `ffmpeg` with `libx264` are required). Training is independent of those tools and works on a headless Debian/Ubuntu system, either Arch desktop, or Windows. Use `python -m ...` instead of the Linux-specific `.venv/bin/python` prefix on Windows. The portable defaults are `--device auto --workers 0`; CUDA is selected when available. On the headless server, request it explicitly after confirming the NVIDIA driver and PyTorch CUDA build are installed:

```bash
python -m models.imitation.train_bc --cache-dir cached_frames --device cuda --batch-size 8 --workers 4
```

Keep `--workers 0` on Windows unless a later benchmark shows a higher value is stable. The recorder writes 24 Hz, 1920×1080 H.264 video at CRF 20; capture and training can therefore live on different systems.

---

# ⚠️ Known Limitations

* Portal 2 is not a high-speed simulator: training throughput is bounded by real-time game execution, physics simulation, rendering overhead, and chamber-reset time — unlike vectorized simulators used in typical RL benchmarks.
* The deployed behavior-cloning policy has only been validated on the current navigation chamber. Completion there does not yet demonstrate generalization to unseen geometry or puzzle mechanics.
* Behavior cloning is vulnerable to distribution shift: once the policy reaches a wall-facing or otherwise unfamiliar state, its own next actions can move it farther outside the expert dataset. Targeted expert-recovery demonstrations are the current mitigation.
* The checked-in launchers contain machine-specific `DP-1` output defaults. Always verify the captured attempt video; feeding the policy another desktop produces meaningless actions even when Portal itself is focused.
* Stages 4–6 represent a multi-month-to-multi-year research effort; even large-scale prior work on Portal-style tasks has struggled with momentum and multi-portal reasoning. Treated as a long-term direction, not a near-term deliverable.
* The `GAME`-path-alias bug (see Resolved Deployment Issues) is an apparent engine bug in Portal 2's native Linux build; it may resurface on new installs and is mitigated, not fixed, by forcing Proton.
* Recurrence (Stage 5) is explicitly *not* a mechanism for backtracking or undoing actions — that would require a planning/search capability, which is out of scope for the current roadmap and noted only as a possible long-term direction.

---

# 🚀 Long-Term Vision

SHIONN aims to become a fully autonomous Portal test subject capable of:

* Navigating unfamiliar chambers
* Understanding puzzle mechanics
* Using portals strategically
* Solving multi-step challenges
* Generalizing beyond its training environment

The ultimate objective is not merely to create a Portal-playing bot, but to investigate how visual learning, memory, planning, and reinforcement learning can combine to produce a capable autonomous problem-solving agent.
