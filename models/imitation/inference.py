"""Live frame-by-frame inference for a trained SHIONN behavior-cloning policy."""

from __future__ import annotations

# Python standard library: frame history, paths, and type annotations.
from collections import deque
from pathlib import Path
from typing import Any

# Installed dependencies: frame arrays and PyTorch inference.
import numpy as np
import torch

# Sibling modules in models.imitation (leading dot = this package).
from .checkpoint import load_checkpoint
from .dataset import BINARY_ACTION_COLUMNS
from .goals import GoalCondition
from .mouse_bins import MouseBins
from .network import (
    ARCHITECTURE_VERSION,
    BINNED_ARCHITECTURE_VERSION,
    GATED_BINNED_ARCHITECTURE_VERSION,
    V3_ARCHITECTURE_VERSION,
    V4_BINNED_ARCHITECTURE_VERSION,
    BinnedImitationPolicy,
    GatedBinnedImitationPolicy,
    ImitationPolicy,
    LegacyImitationPolicy,
    V3ImitationPolicy,
    V4BinnedImitationPolicy,
)
from .preprocess import PREPROCESSING_CONFIG, preprocess_bgr_frame
from .recurrent import (
    GOAL_CONDITIONED_ARCHITECTURE_VERSION,
    RECURRENT_ARCHITECTURE_VERSION,
    GoalConditionedRecurrentPolicy,
    RecurrentBinnedPolicy,
)
from .visual_goal import (
    VISUAL_GOAL_ARCHITECTURE_VERSION,
    VISUAL_GOAL_CONFIG,
    VisualGoalRecurrentPolicy,
)
from .visual_goal_dataset import preprocess_goal_rgb


def decode_mouse_probabilities(
    probabilities: np.ndarray,
    representatives: list[float] | np.ndarray,
    *,
    mode: str = "argmax",
    move_threshold: float = 0.5,
) -> np.ndarray:
    """Decode batched per-axis mouse distributions without changing the policy."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    representatives = np.asarray(representatives, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(representatives):
        raise ValueError("Mouse probabilities must have shape [batch, number of bins]")
    if len(representatives) < 3 or len(representatives) % 2 != 1:
        raise ValueError("Mouse decoding requires an odd number of at least three bins")
    if (
        not np.all(np.isfinite(representatives))
        or representatives[len(representatives) // 2] != 0
    ):
        raise ValueError(
            "Mouse bins must have finite representatives and a centered zero class"
        )
    if mode not in ("argmax", "hierarchical", "expected"):
        raise ValueError(f"Unsupported mouse decoder: {mode!r}")
    if not 0.0 < move_threshold < 1.0:
        raise ValueError("Mouse move threshold must be between 0 and 1")
    if mode != "hierarchical" and move_threshold != 0.5:
        raise ValueError("Mouse move threshold only applies to hierarchical decoding")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Mouse probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("Each mouse probability row must sum to one")

    if mode == "argmax":
        return np.rint(representatives[probabilities.argmax(axis=1)]).astype(np.int64)
    if mode == "expected":
        return np.rint(probabilities @ representatives).astype(np.int64)

    zero_index = len(representatives) // 2
    decoded = np.zeros(len(probabilities), dtype=np.int64)
    moving_rows = np.flatnonzero(probabilities[:, zero_index] < move_threshold)
    for row_index in moving_rows:
        row = probabilities[row_index]
        left_mass = row[:zero_index].sum()
        right_mass = row[zero_index + 1 :].sum()
        if right_mass > left_mass:
            side_indices = np.arange(zero_index + 1, len(representatives))
        else:
            # Exact direction ties choose left, matching the stable class order.
            side_indices = np.arange(zero_index)
        class_index = side_indices[np.argmax(row[side_indices])]
        decoded[row_index] = int(np.rint(representatives[class_index]))
    return decoded


class PolicyInference:
    """Maintains the required four-frame history and predicts one factored action."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        jump_threshold: float | None = None,
        move_w_threshold: float | None = None,
        mouse_decode: str = "argmax",
        mouse_move_threshold: float = 0.5,
    ):
        if jump_threshold is not None and not 0 < jump_threshold < 1:
            raise ValueError("jump threshold must be between 0 and 1")
        if move_w_threshold is not None and not 0 < move_w_threshold < 1:
            raise ValueError("move-w threshold must be between 0 and 1")
        if mouse_decode not in ("argmax", "hierarchical", "expected"):
            raise ValueError(f"Unsupported mouse decoder: {mouse_decode!r}")
        if not 0 < mouse_move_threshold < 1:
            raise ValueError("Mouse move threshold must be between 0 and 1")
        if mouse_decode != "hierarchical" and mouse_move_threshold != 0.5:
            raise ValueError(
                "Mouse move threshold only applies to hierarchical decoding"
            )
        self.jump_threshold = jump_threshold
        self.move_w_threshold = move_w_threshold
        self.mouse_decode = mouse_decode
        self.mouse_move_threshold = mouse_move_threshold
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable")
        checkpoint = load_checkpoint(Path(checkpoint_path), device=self.device)
        config = checkpoint["config"]
        architecture = config.get("architecture")
        if architecture not in (
            RECURRENT_ARCHITECTURE_VERSION,
            VISUAL_GOAL_ARCHITECTURE_VERSION,
            GOAL_CONDITIONED_ARCHITECTURE_VERSION,
            GATED_BINNED_ARCHITECTURE_VERSION,
            "shionn_imitation_v1",
            "shionn_imitation_v2",
            V3_ARCHITECTURE_VERSION,
            V4_BINNED_ARCHITECTURE_VERSION,
            ARCHITECTURE_VERSION,
            BINNED_ARCHITECTURE_VERSION,
        ):
            raise ValueError(
                f"Unsupported architecture: {config.get('architecture')!r}"
            )
        if config.get("preprocessing") != PREPROCESSING_CONFIG:
            raise ValueError(
                "Checkpoint preprocessing differs from this inference implementation"
            )
        self.config = config
        target_processing = config.get("target_processing", {})
        self.mouse_bins = (
            MouseBins(target_processing["mouse_bins"])
            if architecture
            in (
                RECURRENT_ARCHITECTURE_VERSION,
                GOAL_CONDITIONED_ARCHITECTURE_VERSION,
                VISUAL_GOAL_ARCHITECTURE_VERSION,
                GATED_BINNED_ARCHITECTURE_VERSION,
                V4_BINNED_ARCHITECTURE_VERSION,
                BINNED_ARCHITECTURE_VERSION,
            )
            else None
        )
        if self.mouse_bins is None and mouse_decode != "argmax":
            raise ValueError(
                "Non-argmax mouse decoding requires a binned-mouse checkpoint"
            )
        self.mouse_scale = None
        if self.mouse_bins is not None:
            policy_class = (
                GatedBinnedImitationPolicy
                if architecture == GATED_BINNED_ARCHITECTURE_VERSION
                else BinnedImitationPolicy
                if architecture == BINNED_ARCHITECTURE_VERSION
                else V4BinnedImitationPolicy
            )
            if architecture == RECURRENT_ARCHITECTURE_VERSION:
                memory = config["memory"]
                if memory.get("type") != "gru" or not isinstance(
                    memory.get("feature_gate"), bool
                ):
                    raise ValueError("Invalid recurrent checkpoint configuration")
                self.policy = RecurrentBinnedPolicy(
                    *self.mouse_bins.class_counts(),
                    memory["hidden_size"],
                    memory["feature_gate"],
                ).to(self.device)
            elif architecture == VISUAL_GOAL_ARCHITECTURE_VERSION:
                memory = config.get("memory", {})
                goal_config = config.get("visual_goal", {})
                if memory.get("type") != "gru" or not isinstance(
                    memory.get("feature_gate"), bool
                ):
                    raise ValueError("Invalid visual-goal recurrent configuration")
                if goal_config.get("preprocessing") != VISUAL_GOAL_CONFIG:
                    raise ValueError("Visual crop preprocessing contract mismatch")
                self.policy = VisualGoalRecurrentPolicy(
                    *self.mouse_bins.class_counts(),
                    memory["hidden_size"],
                    memory["feature_gate"],
                    goal_config["embedding_size"],
                    goal_config["adapter_size"],
                ).to(self.device)
            elif architecture == GOAL_CONDITIONED_ARCHITECTURE_VERSION:
                memory = config.get("memory", {})
                goal_config = config.get("goal_conditioning", {})
                if memory.get("type") != "gru" or not isinstance(
                    memory.get("feature_gate"), bool
                ):
                    raise ValueError(
                        "Invalid goal-conditioned recurrent checkpoint configuration"
                    )
                if goal_config.get("schema") != "shionn_goal_condition_v1":
                    raise ValueError(
                        "Unsupported goal-conditioning checkpoint contract"
                    )
                self.policy = GoalConditionedRecurrentPolicy(
                    *self.mouse_bins.class_counts(),
                    memory["hidden_size"],
                    memory["feature_gate"],
                    goal_config.get("adapter_size", 128),
                ).to(self.device)
            else:
                self.policy = policy_class(*self.mouse_bins.class_counts()).to(
                    self.device
                )
        else:
            self.mouse_scale = np.asarray(
                target_processing.get("mouse_scale", [1.0, 1.0]), dtype=np.float32
            )
            if self.mouse_scale.shape != (2,) or np.any(self.mouse_scale <= 0):
                raise ValueError(f"Invalid checkpoint mouse scale: {self.mouse_scale}")
            if architecture == ARCHITECTURE_VERSION:
                policy_class = ImitationPolicy
            elif architecture == V3_ARCHITECTURE_VERSION:
                policy_class = V3ImitationPolicy
            else:
                policy_class = LegacyImitationPolicy
            self.policy = policy_class().to(self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if self.device.type == "cuda":
            self.policy = self.policy.to(memory_format=torch.channels_last)
        self.policy.eval()
        self.hidden = None
        self.current_goal = None
        self.visual_goal_embedding = None
        self.visual_goal_id = None
        self.frames: deque[np.ndarray] = deque(
            maxlen=PREPROCESSING_CONFIG["frame_stack"]
        )

    def reset(self) -> None:
        """Reset an episode and clear its selected objective."""
        self.frames.clear()
        self.hidden = None
        if hasattr(self, "current_goal"):
            self.current_goal = None
        self.visual_goal_embedding = None
        self.visual_goal_id = None

    def reset_memory(self) -> None:
        """Reset visual/recurrent state while preserving a persistent goal."""
        self.frames.clear()
        self.hidden = None

    def set_goal(self, goal: GoalCondition) -> None:
        """Supply/update the persistent objective; frame ingestion never resets it."""
        if not isinstance(self.policy, GoalConditionedRecurrentPolicy):
            raise RuntimeError("This checkpoint does not accept goal conditioning")
        if not isinstance(goal, GoalCondition):
            raise TypeError("goal must be a validated GoalCondition")
        self.current_goal = goal

    def set_visual_goal(self, crop_rgb: np.ndarray, *, goal_id: str) -> None:
        """Encode a visually selected uint8 RGB crop once; no engine features.

        Goal replacement retains scene memory. reset_memory retains the goal;
        episode reset clears it. RGB is explicit (game screenshots are BGR).
        """
        if not isinstance(self.policy, VisualGoalRecurrentPolicy):
            raise RuntimeError("This checkpoint does not accept visual targets")
        if not isinstance(goal_id, str) or not goal_id.strip():
            raise ValueError("Visual target needs a nonempty bookkeeping ID")
        crop = torch.from_numpy(preprocess_goal_rgb(crop_rgb)).unsqueeze(0)
        with torch.no_grad():
            embedding = self.policy.encode_goal(
                crop.to(self.device, dtype=torch.float32) / 255.0
            )
        self.visual_goal_embedding = embedding.detach().clone()
        self.visual_goal_id = goal_id

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
            raise RuntimeError(
                "Call add_frame() or pass a frame to predict() before inference"
            )
        stacked = list(self.frames)
        while len(stacked) < PREPROCESSING_CONFIG["frame_stack"]:
            stacked.insert(0, stacked[0])
        tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.stack(stacked).transpose(0, 3, 1, 2).reshape(12, 180, 320)
            )
        ).unsqueeze(0)
        return self.predict_stacked_tensor(tensor, history_frames=len(self.frames))

    def predict_stacked_tensor(
        self, stacked: np.ndarray | torch.Tensor, *, history_frames: int | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        """Predict from one cached RGB frame stack in live inference format.

        ``stacked`` is a uint8 or 0..255 tensor shaped ``[12, 180, 320]`` (or
        ``[1, 12, 180, 320]``). Recurrent checkpoints advance exactly one
        observation and retain their hidden state, allowing offline evaluators
        to process an episode in order with explicit resets between episodes.
        """
        tensor = torch.as_tensor(stacked)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (1, 12, 180, 320):
            raise ValueError("Expected one cached frame stack shaped [12, 180, 320]")
        input_tensor = tensor.to(self.device, dtype=torch.float32)
        if self.device.type == "cuda":
            input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            if isinstance(self.policy, VisualGoalRecurrentPolicy):
                if self.visual_goal_embedding is None:
                    raise RuntimeError("Set a persistent visual crop before inference")
                output, self.hidden = self.policy.step(
                    input_tensor.div_(255.0), self.visual_goal_embedding, self.hidden
                )
                self.hidden = self.hidden.detach()
            elif isinstance(self.policy, RecurrentBinnedPolicy):
                output, self.hidden = self.policy.step(
                    input_tensor.div_(255.0), self.hidden
                )
                self.hidden = self.hidden.detach()
            elif isinstance(self.policy, GoalConditionedRecurrentPolicy):
                if self.current_goal is None:
                    raise RuntimeError(
                        "Set a persistent goal before goal-conditioned inference"
                    )
                goal_tensor = self.current_goal.as_tensor(device=self.device)
                output, self.hidden = self.policy.step(
                    input_tensor.div_(255.0),
                    goal_tensor,
                    self.hidden,
                )
                self.hidden = self.hidden.detach()
            else:
                output = self.policy(input_tensor.div_(255.0))
        predicted, binary_probabilities, mouse_probabilities = self.decode_outputs(
            output
        )
        action = {
            name: int(predicted[0, index])
            for index, name in enumerate(
                (*BINARY_ACTION_COLUMNS, "mouse_dx", "mouse_dy")
            )
        }
        probabilities = {
            name: float(binary_probabilities[0, index])
            for index, name in enumerate(BINARY_ACTION_COLUMNS)
        }
        diagnostics = {
            "binary_probabilities": probabilities,
            "history_frames": len(self.frames)
            if history_frames is None
            else history_frames,
        }
        if isinstance(self.policy, VisualGoalRecurrentPolicy):
            diagnostics["visual_goal"] = {
                "goal_id": self.visual_goal_id,
                "persistent": True,
            }
        if isinstance(self.policy, GoalConditionedRecurrentPolicy):
            diagnostics["goal"] = {
                "goal_id": self.current_goal.goal_id,
                "target_kind": self.current_goal.target_kind,
                "visible": self.current_goal.visible,
                "x_norm": self.current_goal.x_norm,
                "y_norm": self.current_goal.y_norm,
            }
        if self.mouse_bins is not None:
            diagnostics["mouse_bin_probabilities"] = {
                axis: mouse_probabilities[index][0].tolist()
                for index, axis in enumerate(("dx", "dy"))
            }
            diagnostics["mouse_decode"] = self.mouse_decode
            diagnostics["mouse_move_threshold"] = (
                self.mouse_move_threshold
                if self.mouse_decode == "hierarchical"
                else None
            )
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

    def decode_outputs(self, output: dict[str, torch.Tensor]):
        """Apply the shared live decoder to one or more policy output rows."""
        predicted_binary = np.stack(
            [
                output[name].argmax(dim=1).cpu().numpy()
                for name in BINARY_ACTION_COLUMNS
            ],
            axis=1,
        )
        binary_probabilities = np.stack(
            [
                torch.softmax(output[name], dim=1)[:, 1].cpu().numpy()
                for name in BINARY_ACTION_COLUMNS
            ],
            axis=1,
        )
        for name, threshold in (
            ("jump", self.jump_threshold),
            ("move_w", self.move_w_threshold),
        ):
            if threshold is not None:
                index = BINARY_ACTION_COLUMNS.index(name)
                predicted_binary[:, index] = binary_probabilities[:, index] >= threshold

        mouse_probabilities = None
        if self.mouse_bins is not None:
            mouse_probabilities = [
                torch.softmax(output[key], dim=1).cpu().numpy()
                for key in ("mouse_dx_logits", "mouse_dy_logits")
            ]
            predicted_mouse = np.stack(
                [
                    decode_mouse_probabilities(
                        mouse_probabilities[axis_index],
                        self.mouse_bins.representatives(axis_index),
                        mode=self.mouse_decode,
                        move_threshold=self.mouse_move_threshold,
                    )
                    for axis_index in range(2)
                ],
                axis=1,
            )
        else:
            predicted_mouse = np.rint(
                output["mouse_mean"].cpu().numpy() * self.mouse_scale
            )
        return (
            np.concatenate((predicted_binary, predicted_mouse), axis=1),
            binary_probabilities,
            mouse_probabilities,
        )

    def apply(
        self, controller: Any, frame_bgr: np.ndarray | None = None
    ) -> dict[str, int]:
        """Predict and send state transitions through Portal2Controller.apply_action()."""
        action = self.predict(frame_bgr)
        controller.apply_action(action)
        return action
