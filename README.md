# Project-SIONN

---

# SIONN — Self-learning Independent Optical Neural Network

SIONN is an experimental AI agent designed for the *Portal 2* environment, built to operate as a fully autonomous test subject capable of learning directly from visual input and interacting with puzzle-based physics systems.

The goal of this project is to explore reinforcement learning + imitation learning in a structured physics puzzle environment using only pixel-based perception and keyboard/mouse action control.

---

## 🧠 Concept

SIONN (**Self-learning Independent Optical Neural Network**) is a single-player autonomous robot designed for iterative problem solving inside Portal-style test chambers.

Unlike scripted bots or rule-based agents, SIONN:
- Observes the environment through raw screen pixels
- Builds an internal model of spatial relationships
- Learns from trial-and-error interactions
- Continuously improves through reinforcement feedback

---

## 🎯 Project Goals

- Train an AI agent to complete Portal-style puzzle chambers
- Use **visual-only input (no game state API)**
- Output real-time keyboard + mouse actions
- Support generalization across custom test chambers
- Combine:
  - Imitation learning (human gameplay)
  - Reinforcement learning (self-improvement)
  - Curriculum learning (progressive chamber difficulty)

---

## 🧩 Environment

Target game:
- *Portal 2* (custom maps / test chambers)

Optional simulation layer:
- Gym-like wrapper around game window
- Frame capture (OpenCV or similar)
- Input control (keyboard + mouse automation)

---

## 🏗️ System Architecture

### 1. Perception Layer
- Optical input (screen capture at 30–60 FPS)
- Frame preprocessing (resize, normalization, frame stacking)

### 2. Optical Neural Network (ONN)
- Convolutional neural network for spatial reasoning
- Optional vision transformer extension
- Temporal memory (LSTM / GRU or frame stacking)

### 3. Decision Policy
- Reinforcement learning policy network
- Outputs discrete + continuous actions:
  - Movement (WASD)
  - Look direction (mouse delta)
  - Interaction (E / click)
  - Jump

### 4. Learning System
- Reward function based on:
  - Puzzle progression
  - Level completion
  - Efficiency (time / steps)
- Training methods:
  - PPO / SAC (RL)
  - Behavior cloning (imitation learning)
  - Curriculum progression

---

## 🧪 Training Pipeline

### Phase 1 — Imitation Learning
- Record human gameplay in custom chambers
- Train model to replicate actions from visual input

### Phase 2 — Reinforcement Learning
- Allow agent to attempt chambers autonomously
- Reward successful puzzle progression
- Penalize inefficient or failed actions

### Phase 3 — Curriculum Scaling
- Gradually increase chamber complexity
- Introduce new mechanics:
  - Portals
  - Cubes & buttons
  - Turrets
  - Timing puzzles

---

## ⚙️ Requirements

### Hardware
- NVIDIA GPU (recommended RTX 3060+)
- 16–32 GB RAM
- CPU capable of real-time frame processing

### Software
- Python 3.10+
- PyTorch
- OpenCV
- Gym-like environment wrapper (custom)
- Input automation library (e.g. PyAutoGUI or equivalent)

---

## 📦 Project Structure
