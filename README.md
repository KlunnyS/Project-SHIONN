# Project SHIONN

# SHIONN — Self-learning Hybrid Independent Optical Neural Network

SHIONN is an experimental AI agent designed for the *Portal 2* environment, built to operate as a fully autonomous test subject capable of learning directly from visual input and interacting with puzzle-based physics systems.

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

---

## 🧩 Environment

### Target Platform

* *Portal 2*, running natively on Linux (Wayland / Niri compositor)
* Custom chambers built with Puzzle Maker, hand-edited in Hammer++
* Community workshop maps for later evaluation

### Environment Wrapper

The game is treated as a vision-based reinforcement learning environment, exposed as a Gym-like interface:

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
* Reward monitoring via a privileged side-channel, invisible to the policy

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

Niri natively supports the `wlr-screencopy` Wayland protocol. `grim` (one-shot screenshots) was tried first but its per-frame process-spawn/handshake overhead capped throughput; `wf-recorder` is used in production instead, streaming raw BGR frames through a long-lived pipe at a fixed rate with no per-frame handshake cost.

### Chosen resolutions and rates

| Parameter | Value / rationale |
|---|---|
| Game render resolution | 800×600, windowed, low settings, `fps_max` capped to the capture rate |
| Model input resolution | 128×128 RGB — color kept deliberately (portal blue/orange is a strong, cheap signal grayscale would discard) |
| Frame stacking | Last 4 frames stacked as channels, giving implicit motion/velocity information |
| Capture rate | 20 Hz fixed tick loop |

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

A single shared `.nut` script, referenced by every chamber's `logic_script` entity, exposes generic signal functions:

```squirrel
function SignalChamberReady() { printl("EVT|chamber_ready|" + Time()) }
function SignalGoalReached()  { printl("EVT|goal_reached|1") }
function SignalEpisodeFailed(reason) { printl("EVT|episode_failed|" + reason) }
```

Each chamber wires these to specific brushes: a start-of-chamber trigger (fired shortly after spawn, not immediately on map load) calls `SignalChamberReady`; a `trigger_once` (never `trigger_multiple`) at the goal calls `SignalGoalReached`; fail-condition volumes or a `logic_timer` timeout call `SignalEpisodeFailed`. Every new chamber is manually verified — watch raw netconsole output, walk it by hand, confirm each signal fires exactly once — **before** it enters the recorder's rotation.

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
2. **Fixed-rate capture** — `wf-recorder` piping raw BGR frames at 20 Hz.
3. **Sync loop** — a precise 20 Hz tick snapshots the most recent frame together with the current persistent key-hold state and the accumulated-then-reset mouse delta, writing one aligned row.
4. **Storage** — one folder per episode under `episodes/`, containing `actions.csv` (columns: `frame_idx, move_w, move_a, move_s, move_d, jump, use, fire_left, fire_right, mouse_dx, mouse_dy`) alongside a compressed `video.mp4`.
5. **Episode boundaries** — driven by the same `EVT|chamber_ready` / `EVT|goal_reached` / `EVT|episode_failed` netconsole hooks used by the environment wrapper; on episode end, the folder is renamed to embed the outcome (e.g. `episode_20260906_134500_goal_reached`).

A playback function re-executes a recorded episode's actions through the same input-injection path used for live inference, validating the full record → store → replay loop independent of any model.

---

# 🛣️ Development Roadmap & Network Architecture

Each stage below lists its goal, its network structure, and its success criteria. **The core rule: never jump to a more complex architecture than the current stage demonstrably needs.** A feedforward CNN is the default; recurrence is an upgrade earned by evidence, not assumed upfront.

---

## Stage 0 — Environment Setup ✅

**Goal:** Create a stable AI training environment around Portal 2.

Requirements: screen capture pipeline, kernel-level input control, automatic chamber reset (cold-start + warm-reset), episode logging, dataset recording tools.

**Network:** none — this stage builds the environment wrapper only.

**Deliverable:**
```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

---

## Stage 0.5 — Curriculum Chamber Authoring

**Goal:** Build a small set of chambers with verified VScript event hooks before any real recording begins.

* Empty room, reach a visible goal (pure movement)
* Simple corridor / turn (navigation, looking around)
* Single button opens a door (first interaction entity)
* Carry a cube onto a button (cube mechanics)

**Network:** none.

---

## Stage 1 — Data Collection (Imitation Learning)

**Goal:** Teach basic human-like movement and camera control by recording demonstrations.

**Network — perception + control layer used for *inference sanity-checking* during recording, not yet trained on real data:**

```
Input: 4 × 128×128 RGB stacked frames (12 channels)
  ↓
CNN encoder (Conv 8×8/4 → Conv 4×4/2 → Conv 3×3/1 → FC 512)
  ↓
Factored action heads (see Action Space Design):
  move_w, move_a, move_s, move_d   → binary
  jump, use, fire_left, fire_right → binary
  mouse_dx, mouse_dy               → Gaussian (mean + log-std)
```

No recurrence. No value head yet (no RL at this stage). This is the same architecture Stage 2 trains for real — Stage 1's job is only to get demonstration data flowing through it once, end-to-end, to confirm shapes and formats line up.

**Success criteria:** demonstration data is being recorded reliably (validated via the playback function), in the exact factored-action format the model will be trained on.

---

## Stage 2 — Behavior Cloning

**Goal:** Learn to imitate human gameplay via supervised learning.

**Network — identical architecture to Stage 1, now actually trained:**

```
Input: 4 × 128×128 RGB stacked frames
  ↓
CNN trunk (shared)
  ↓
┌─────────────┬─────────────┬─────────────┬──────────────┐
move heads    button heads   mouse head    (no value head)
(4 × binary)  (4 × binary)   (Gaussian)
```

**Loss:** cross-entropy on each binary head + negative log-likelihood (or MSE, as a simpler starting point) on the Gaussian mouse head, summed.

**Why no recurrence here:** short-horizon skills (walking, looking, jumping, interacting) are well covered by the 4-frame stack's implicit velocity information. Behavior cloning is also where a no-op-collapse-resistant prior gets baked in for free, since human demonstrations rarely sit idle.

**Expected result:** stable movement, basic navigation, natural camera control, reliable interaction — a baseline that substantially reduces the instability and exploration difficulty of RL trained from scratch.

---

## Stage 3 — Reinforcement Learning

**Goal:** Transition from imitation to autonomous learning via reward.

**Algorithm:** PPO (Proximal Policy Optimization). Later candidates: SAC, Dreamer, IMPALA.

**Network — same CNN trunk and action heads as Stage 2, extended with a value head for actor-critic PPO:**

```
Input: 4 × 128×128 RGB stacked frames
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
Input: 4 × 128×128 RGB stacked frames
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
Input: 4 × 128×128 RGB stacked frames
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

* Screen capture (`wf-recorder` / `wlr-screencopy`) at a fixed 20 Hz
* Resize/normalize to 128×128 RGB
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

Gym-like interface over the real Portal 2 process: cold-start/warm-reset bootstrap over `-netconport`, netconsole-driven reward/termination, `evdev`-based action injection, `wf-recorder`-based observation capture.

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
SHIONN/
│
├── data/
│   ├── recordings/
│   └── datasets/
│
├── environment/
│   ├── screen_capture/       # wf-recorder pipe
│   ├── input_control/        # evdev / uinput
│   ├── netconsole/           # -netconport client, reward/reset hooks
│   └── portal_wrapper/       # Gym-like env: reset()/step()
│
├── models/
│   ├── imitation/            # Stage 1-2: CNN + factored heads
│   ├── rl/                   # Stage 3-4: + value head (PPO)
│   └── memory/               # Stage 5: + GRU / attention
│
├── training/
│   ├── behavior_cloning/
│   ├── ppo/
│   └── curriculum/
│
├── chambers/
│   ├── navigation/
│   ├── interaction/
│   ├── portals/
│   └── advanced/
│
└── docs/
```

---

# ⚠️ Known Limitations

* Portal 2 is not a high-speed simulator: training throughput is bounded by real-time game execution, physics simulation, rendering overhead, and chamber-reset time — unlike vectorized simulators used in typical RL benchmarks.
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
