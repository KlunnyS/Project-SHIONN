"""Read-only paired-goal action sensitivity, never a navigation success claim."""

from __future__ import annotations

import torch


@torch.no_grad()
def compare_visual_goals(
    model,
    frames,
    goal_a,
    goal_b,
    *,
    hidden=None,
    valid=None,
    dx_representatives=None,
    decision_mask=None,
):
    """Same observation/history/state, separate branches, no caller-state mutation.

    Returns per-frame distribution shifts and signed expected dx (if supplied).
    Sensitivity alone is not expert-correctness, reaching or generalization.
    """
    if decision_mask is not None and (
        decision_mask.shape != frames.shape[:2] or not decision_mask.any()
    ):
        raise ValueError("Decision mask must select at least one observation")
    prior_mode = model.training
    model.eval()
    try:
        initial = None if hidden is None else hidden.detach().clone()
        a, _ = model(
            frames,
            goal_a,
            hidden=None if initial is None else initial.clone(),
            valid=valid,
        )
        b, _ = model(
            frames,
            goal_b,
            hidden=None if initial is None else initial.clone(),
            valid=valid,
        )
        metrics = {}
        for key in a:
            pa, pb = a[key].softmax(-1), b[key].softmax(-1)
            if not torch.isfinite(pa).all() or not torch.isfinite(pb).all():
                raise FloatingPointError("Nonfinite counterfactual probabilities")
            tv = 0.5 * (pa - pb).abs().sum(-1)
            selected = tv if decision_mask is None else tv[decision_mask]
            entry = {
                "mean_total_variation": float(selected.mean()),
                "argmax_change_rate": float(
                    (
                        (pa.argmax(-1) != pb.argmax(-1)).float()
                        if decision_mask is None
                        else (pa.argmax(-1) != pb.argmax(-1))[decision_mask].float()
                    ).mean()
                ),
            }
            if key == "mouse_dx_logits" and dx_representatives is not None:
                reps = torch.as_tensor(
                    dx_representatives, device=pa.device, dtype=pa.dtype
                )
                if reps.shape != (pa.shape[-1],) or not torch.isfinite(reps).all():
                    raise ValueError("Mouse representatives do not match policy")
                entry["goal_a_expected_dx"] = (pa @ reps).cpu().tolist()
                entry["goal_b_expected_dx"] = (pb @ reps).cpu().tolist()
            metrics[key] = entry
        return {
            "interpretation": "goal sensitivity only; no correctness or gameplay score",
            "metrics": metrics,
        }
    finally:
        model.train(prior_mode)
