"""Live frame-by-frame inference for a trained SHIONN behavior-cloning policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import BINARY_ACTION_COLUMNS
from .network import ImitationPolicy, LegacyImitationPolicy
from .preprocess import PREPROCESSING_CONFIG, preprocess_bgr_frame


class PolicyInference:
    """Maintains the required four-frame history and predicts one factored action."""
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable")
        checkpoint = load_checkpoint(Path(checkpoint_path), device=self.device)
        config = checkpoint["config"]
        if config.get("architecture") not in (
            "shionn_imitation_v1", "shionn_imitation_v2", "shionn_imitation_v3"
        ):
            raise ValueError(f"Unsupported architecture: {config.get('architecture')!r}")
        if config.get("preprocessing") != PREPROCESSING_CONFIG:
            raise ValueError("Checkpoint preprocessing differs from this inference implementation")
        self.config = config
        target_processing = config.get("target_processing", {})
        self.mouse_scale = np.asarray(target_processing.get("mouse_scale", [1.0, 1.0]), dtype=np.float32)
        if self.mouse_scale.shape != (2,) or np.any(self.mouse_scale <= 0):
            raise ValueError(f"Invalid checkpoint mouse scale: {self.mouse_scale}")
        policy_class = (
            ImitationPolicy
            if config.get("architecture") == "shionn_imitation_v3"
            else LegacyImitationPolicy
        )
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
        mouse = output["mouse_mean"].squeeze(0).cpu().numpy() * self.mouse_scale
        action["mouse_dx"] = int(np.rint(mouse[0]))
        action["mouse_dy"] = int(np.rint(mouse[1]))
        return action

    def apply(self, controller: Any, frame_bgr: np.ndarray | None = None) -> dict[str, int]:
        """Predict and send state transitions through Portal2Controller.apply_action()."""
        action = self.predict(frame_bgr)
        controller.apply_action(action)
        return action
