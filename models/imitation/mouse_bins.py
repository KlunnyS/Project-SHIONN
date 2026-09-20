"""Foveated, zero-centered mouse bins fitted on training actions only."""

from __future__ import annotations

import numpy as np
import torch


class MouseBins:
    """Encode signed mouse deltas and decode concrete per-axis bin actions."""

    def __init__(self, config: dict):
        self.config = config
        self.axes = []
        for name in ("dx", "dy"):
            axis = config[name]
            edges = np.asarray(axis["magnitude_edges"], dtype=np.float32)
            representatives = np.asarray(axis["representatives"], dtype=np.float32)
            magnitude_bins = len(edges) + 1
            if (
                edges.ndim != 1
                or not np.all(np.isfinite(edges))
                or np.any(edges <= 0)
                or np.any(np.diff(edges) <= 0)
                or representatives.shape != (2 * magnitude_bins + 1,)
                or not np.all(np.isfinite(representatives))
                or representatives[magnitude_bins] != 0
            ):
                raise ValueError(f"Invalid {name} mouse bin configuration")
            self.axes.append((edges, representatives))

    @classmethod
    def fit(cls, values: np.ndarray, bins_per_axis: int = 15) -> "MouseBins":
        """Fit symmetric magnitude quantiles, preserving an exact zero class."""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Mouse bin fitting needs finite [frames, 2] deltas")
        if bins_per_axis < 3 or bins_per_axis % 2 != 1:
            raise ValueError("Mouse bin count must be odd and at least 3")
        requested_magnitude_bins = (bins_per_axis - 1) // 2
        config = {"scheme": "symmetric_nonzero_magnitude_quantiles", "requested_bins": bins_per_axis}
        for axis_index, name in enumerate(("dx", "dy")):
            magnitude = np.abs(values[:, axis_index])
            nonzero = magnitude[magnitude > 0]
            if len(nonzero) == 0:
                config[name] = {"magnitude_edges": [], "representatives": [0.0, 0.0, 0.0]}
                continue
            levels = np.arange(1, requested_magnitude_bins) / requested_magnitude_bins
            edges = np.unique(np.quantile(nonzero, levels, method="higher"))
            # A boundary at the maximum would create an empty tail bin.
            edges = edges[edges < nonzero.max()]
            magnitude_class = np.searchsorted(edges, nonzero, side="left")
            magnitudes = [float(np.median(nonzero[magnitude_class == index]))
                          for index in range(len(edges) + 1)]
            config[name] = {
                "magnitude_edges": [float(edge) for edge in edges],
                "representatives": [-value for value in reversed(magnitudes)]
                + [0.0] + magnitudes,
            }
        return cls(config)

    def class_counts(self) -> tuple[int, int]:
        return tuple(len(representatives) for _, representatives in self.axes)

    def encode(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if deltas.ndim != 2 or deltas.shape[1] != 2:
            raise ValueError("Mouse bin encoding needs [batch, 2] deltas")
        indices = []
        for axis_index, (edges, _) in enumerate(self.axes):
            values = deltas[:, axis_index]
            magnitude_bin_count = len(edges) + 1
            boundaries = torch.as_tensor(edges, device=values.device, dtype=values.dtype)
            magnitude_index = torch.bucketize(values.abs().contiguous(), boundaries, right=False)
            encoded = torch.where(
                values < 0,
                magnitude_bin_count - 1 - magnitude_index,
                magnitude_bin_count + 1 + magnitude_index,
            )
            encoded = torch.where(values == 0, magnitude_bin_count, encoded)
            indices.append(encoded.long())
        return indices[0], indices[1]

    def decode(self, axis_index: int, class_index: int) -> int:
        _, representatives = self.axes[axis_index]
        return int(np.rint(representatives[class_index]))

    def representatives(self, axis_index: int) -> list[float]:
        return self.axes[axis_index][1].tolist()
