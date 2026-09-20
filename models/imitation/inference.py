"""Live frame-by-frame inference for a trained SHIONN behavior-cloning policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import BINARY_ACTION_COLUMNS
from .mouse_bins import MouseBins
from .network import BINNED_ARCHITECTURE_VERSION, BinnedImitationPolicy, ImitationPolicy, LegacyImitationPolicy
from .preprocess import PREPROCESSING_CONFIG, preprocess_bgr_frame


class PolicyInference:
    """Maintains the required four-frame history and predicts one factored action."""
    def __init__(
        self, checkpoint_path: Path, device: str = "auto", jump_threshold: float | None = None,
        move_w_threshold: float | None = None,
    ):
        if jump_threshold is not None and not 0 < jump_threshold < 1:
            raise ValueError("jump threshold must be between 0 and 1")
        if move_w_threshold is not None and not 0 < move_w_threshold < 1:
            raise ValueError("move-w threshold must be between 0 and 1")
        self.jump_threshold = jump_threshold
        self.move_w_threshold = move_w_threshold
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable")
        checkpoint = load_checkpoint(Path(checkpoint_path), device=self.device)
        config = checkpoint["config"]
        architecture = config.get("architecture")
        if architecture not in (
            "shionn_imitation_v1", "shionn_imitation_v2", "shionn_imitation_v3",
            BINNED_ARCHITECTURE_VERSION,
        ):
            raise ValueError(f"Unsupported architecture: {config.get('architecture')!r}")
        if config.get("preprocessing") != PREPROCESSING_CONFIG:
            raise ValueError("Checkpoint preprocessing differs from this inference implementation")
        self.config = config
        target_processing = config.get("target_processing", {})
        self.mouse_bins = (
            MouseBins(target_processing["mouse_bins"])
            if architecture == BINNED_ARCHITECTURE_VERSION else None
        )
        self.mouse_scale = None
        if self.mouse_bins is not None:
            self.policy = BinnedImitationPolicy(*self.mouse_bins.class_counts()).to(self.device)
        else:
            self.mouse_scale = np.asarray(target_processing.get("mouse_scale", [1.0, 1.0]), dtype=np.float32)
            if self.mouse_scale.shape != (2,) or np.any(self.mouse_scale <= 0):
                raise ValueError(f"Invalid checkpoint mouse scale: {self.mouse_scale}")
            policy_class = ImitationPolicy if architecture == "shionn_imitation_v3" else LegacyImitationPolicy
            self.policy = policy_class().to(self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        self.frames: deque[np.ndarray] = deque(maxlen=PREPROCESSING_CONFIG["frame_stack"])

    def reset(self) -> None:
        self.frames.clear()

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        """Add one raw BGR observation from the existing Wayland capture path."""
        self.frames.append(preprocess_bgr_frame(frame_bgr))

    def predict(self, frame_bgr: np.ndarray | None = None) -> dict[str, int]:
        action, _ = self.predict_with_diagnostics(frame_bgr)
        return action

    def predict_with_diagnostics(
        self, frame_bgr: np.ndarray | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        """Predict an action and expose JSON-serializable policy diagnostics."""
        if frame_bgr is not None:
            self.add_frame(frame_bgr)
        if not self.frames:
            raise RuntimeError("Call add_frame() or pass a frame to predict() before inference")
        stacked = list(self.frames)
        while len(stacked) < PREPROCESSING_CONFIG["frame_stack"]:
            stacked.insert(0, stacked[0])
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(stacked).transpose(0, 3, 1, 2).reshape(12, 180, 320))).unsqueeze(0)
        with torch.no_grad():
            output = self.policy(tensor.to(self.device, dtype=torch.float32).div_(255.0))
        action = {name: int(output[name].argmax(dim=1).item()) for name in BINARY_ACTION_COLUMNS}
        probabilities = {
            name: float(torch.softmax(output[name], dim=1)[0, 1].item())
            for name in BINARY_ACTION_COLUMNS
        }
        if self.jump_threshold is not None:
            action["jump"] = int(probabilities["jump"] >= self.jump_threshold)
        if self.move_w_threshold is not None:
            action["move_w"] = int(probabilities["move_w"] >= self.move_w_threshold)
        diagnostics = {
            "binary_probabilities": probabilities,
            "history_frames": len(self.frames),
        }
        if self.mouse_bins is not None:
            mouse_probabilities = {}
            for axis_index, (name, key) in enumerate((
                ("dx", "mouse_dx_logits"), ("dy", "mouse_dy_logits")
            )):
                axis_probabilities = torch.softmax(output[key], dim=1)[0].cpu().tolist()
                class_index = int(output[key].argmax(dim=1).item())
                action[f"mouse_{name}"] = self.mouse_bins.decode(axis_index, class_index)
                mouse_probabilities[name] = axis_probabilities
            diagnostics["mouse_bin_probabilities"] = mouse_probabilities
        else:
            normalized_mouse = output["mouse_mean"].squeeze(0).cpu().numpy()
            mouse = normalized_mouse * self.mouse_scale
            mouse_std = (
                output["mouse_log_std"].squeeze(0).exp().cpu().numpy()
                * self.mouse_scale
            )
            action["mouse_dx"] = int(np.rint(mouse[0]))
            action["mouse_dy"] = int(np.rint(mouse[1]))
            diagnostics["mouse_mean"] = [float(mouse[0]), float(mouse[1])]
            diagnostics["mouse_std"] = [float(mouse_std[0]), float(mouse_std[1])]
        return action, diagnostics

    def apply(self, controller: Any, frame_bgr: np.ndarray | None = None) -> dict[str, int]:
        """Predict and send state transitions through Portal2Controller.apply_action()."""
        action = self.predict(frame_bgr)
        controller.apply_action(action)
        return action
