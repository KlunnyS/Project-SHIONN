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

* *Portal 2*
* Custom chambers created using the Portal 2 Puzzle Maker
* Community workshop maps for evaluation

### Environment Wrapper

The game is treated as a vision-based reinforcement learning environment.

Features:

* Frame capture
* Keyboard and mouse control
* Episode logging
* Chamber reset automation
* Reward monitoring

### Key Limitation

Portal 2 is not a high-speed simulator.

Training speed is limited by:

* Game execution speed
* Physics simulation
* Rendering overhead
* Environment reset time

---

# 🛣️ Development Roadmap

---


## Stage 0 — Environment Setup

Goal: Create a stable AI training environment around Portal 2.

Requirements:
* Screen capture pipeline
* Input control system
* Automatic chamber reset
* Episode logging
* Dataset recording tools

Deliverable:
A Gym-like environment:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

---

## Stage 1 — Data Collection (Imitation Learning)

### Goal

Teach basic human-like movement and camera control.

### Data Collection

Record:

* Game frames (64×64 or 84×84)
* Keyboard input
* Mouse movement
* Interaction events
* Portal placement actions (later stages)

### Recommended Dataset Size

Minimum:

* 2–8 hours

Ideal:

* 10–20 hours

### Initial Skills

* Walking
* Looking around
* Jumping
* Interacting with objects

### Initial Model

#### Perception Layer

* CNN encoder

#### Control Layer

* MLP or small LSTM

Outputs:

* WASD movement
* Mouse movement
* Jump
* Interact

### Success Criteria

The agent can move through simple chambers without random behavior.

---

## Stage 2 — Behavior Cloning

### Goal

Learn to imitate human gameplay.

### Training Method

Supervised learning from recorded gameplay.

Input:

* Visual observations

Output:

* Human actions

### Expected Result

* Stable movement
* Basic navigation
* Natural camera control
* Reliable interaction behavior

### Why It Matters

Behavior cloning significantly reduces:

* RL instability
* Exploration difficulty
* Training time

---

## Stage 3 — Reinforcement Learning

### Goal

Transition from imitation to autonomous learning.

### Algorithm

Recommended:

* PPO (Proximal Policy Optimization)

Possible future alternatives:

* SAC
* Dreamer
* IMPALA

### Chamber Curriculum

#### Level 1

* Empty room
* Reach visible goal

#### Level 2

* Simple maze navigation

#### Level 3

* Button opens door

#### Level 4

* Carry cube to button

### Reward Shaping

Rewards:

* Goal completion
* Progress toward objective
* Successful interactions

Penalties:

* Wasted actions
* Excessive time
* Failure states

### Success Criteria

Agent reliably solves training chambers without demonstrations.

---

## Stage 4 — Portal Mechanics Learning

### Goal

Introduce Portal-specific reasoning.

### Curriculum

#### Level 5

Single portal usage

#### Level 6

Dual portal usage

#### Level 7

Momentum puzzles

#### Level 8

Multi-step chamber solving

### Major Challenges

* Sparse rewards
* Long planning horizons
* Portal placement reasoning
* Momentum prediction

---

## Stage 5 — Memory and Planning

### Goal

Enable long-term reasoning.

### Architecture Upgrade

Current:

```text
CNN → Policy Head
```

Upgraded:

```text
CNN → LSTM / Transformer → Policy Head
```

### Required Memory

The agent must remember:

* Portal locations
* Room layout
* Previous interactions
* Puzzle state
* Long-term objectives

### Expected Result

Improved multi-step puzzle solving and strategic behavior.

---

## Stage 6 — Curriculum Learning

### Philosophy

Do not train "Portal AI" directly.

Train individual skills first.

### Skill Progression

1. Movement
2. Navigation
3. Interaction
4. Cube mechanics
5. Single portal usage
6. Dual portal usage
7. Momentum puzzles
8. Full chamber solving

### Rule

Never skip curriculum stages.

Each stage becomes the foundation for the next.

---

# 🏗️ System Architecture

## 1. Perception Layer

Visual processing system:

* Screen capture (30–60 FPS)
* Resize and normalization
* Frame stacking
* Observation encoding

Possible future upgrades:

* Vision Transformers
* Self-supervised visual representations

---

## 2. Optical Neural Network (ONN)

Core reasoning module.

Responsibilities:

* Spatial understanding
* Object recognition
* Portal recognition
* Navigation awareness

Potential architectures:

* CNN
* CNN + LSTM
* Vision Transformer
* Hybrid models

---

## 3. Decision Policy

Outputs actions directly to the game.

Action space:

### Movement

* W
* A
* S
* D

### Mouse

* Horizontal movement
* Vertical movement

### Interaction

* Jump
* Use
* Fire portal
* Pick up objects

---

## 4. Learning System

### Imitation Learning

Learn from human demonstrations.

### Reinforcement Learning

Learn through rewards and self-play.

### Curriculum Learning

Progressively unlock harder challenges.

---

# 🧪 Training Pipeline

## Phase 1 — Human Demonstrations

Train:

* Movement
* Camera control
* Navigation

Result:

Stable baseline behavior.

---

## Phase 2 — Autonomous Learning

Train:

* Exploration
* Interaction
* Puzzle solving

Result:

Independent chamber completion.

---

## Phase 3 — Curriculum Scaling

Introduce:

* Buttons
* Cubes
* Doors
* Portals
* Momentum mechanics
* Multi-step puzzles

Result:

General-purpose Portal reasoning.

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
│   ├── screen_capture/
│   ├── input_control/
│   └── portal_wrapper/
│
├── models/
│   ├── imitation/
│   ├── rl/
│   └── memory/
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

# 🚀 Long-Term Vision

SHIONN aims to become a fully autonomous Portal test subject capable of:

* Navigating unfamiliar chambers
* Understanding puzzle mechanics
* Using portals strategically
* Solving multi-step challenges
* Generalizing beyond its training environment

The ultimate objective is not merely to create a Portal-playing bot, but to investigate how visual learning, memory, planning, and reinforcement learning can combine to produce a capable autonomous problem-solving agent.
