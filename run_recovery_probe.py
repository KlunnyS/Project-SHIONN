"""Plan, or explicitly execute, a small paired recovery-initiation fine-tune.

Default operation only reads metadata/labels and prints the plan. No game APIs
are used. Execution has a shared soft deadline; use the documented systemd
RuntimeMaxSec limit to also bound blocked device/I/O calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from benchmark_expert import ActionMetrics
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import (
    ACTION_COLUMNS, BINARY_ACTION_COLUMNS, BehaviorCloningDataset,
    discover_cached_episodes,
)
from models.imitation.frozen_split import load_frozen_split
from models.imitation.mouse_bins import MouseBins
from models.imitation.network import BINNED_ARCHITECTURE_VERSION, BinnedImitationPolicy
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.train_bc import (
    augment_batch, behavior_cloning_loss, partition_frozen_training,
    training_sampling_weights,
)

SOURCE = Path("models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt")
OUTPUT = Path("models/imitation/checkpoints/runs/recovery_onset_probe_v1")
ASSESSMENT = Path("model_attempts/recovery_assessment_20261101/cache")
GROUP_FRACTIONS = {"recovery": 0.05, "proactive_turn": 0.05}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + seconds

    def check(self):
        if self.clock() >= self.deadline:
            raise BudgetExceeded("Shared probe time budget exhausted")


class CheckedLoader:
    """Check validation as well as training, without silently truncating either."""
    def __init__(self, loader, budget):
        self.loader, self.budget = loader, budget

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.budget.check()
        for batch in self.loader:
            self.budget.check()
            yield batch
            self.budget.check()


def recovery_opening_weights(dataset, groups, *, window_frames=24, boost=1.0):
    if window_frames < 1 or not math.isfinite(boost) or boost < 1:
        raise ValueError("Recovery opening window must be positive and boost finite, >= 1")
    weights = training_sampling_weights(
        dataset, "chamber-balanced", correction_episode_names=set(groups),
        correction_episode_groups=groups, correction_group_fractions=GROUP_FRACTIONS,
    )
    recovery = np.array([groups.get(dataset.episode_names[e]) == "recovery"
                         for e, _ in dataset.index])
    opening = np.array([is_recovery and frame - dataset.first_action_frames[e] < window_frames
                       for is_recovery, (e, frame) in zip(recovery, dataset.index)])
    if not opening.any():
        raise ValueError("No usable recovery-opening frames")
    mass = weights[recovery].sum()
    with np.errstate(over="ignore"):
        weights[opening] *= boost
        boosted_mass = weights[recovery].sum()
    if not np.all(np.isfinite(weights)) or not math.isfinite(boosted_mass) or boosted_mass <= 0:
        raise ValueError("Opening boost overflowed the sampling weights")
    weights[recovery] *= mass / boosted_mass
    return weights, opening


def validate_source(config):
    if config.get("architecture") != BINNED_ARCHITECTURE_VERSION:
        raise ValueError("Probe supports the existing V5 binned architecture only")
    if config.get("preprocessing") != PREPROCESSING_CONFIG or config.get("action_columns") != list(ACTION_COLUMNS):
        raise ValueError("Source preprocessing/action contract differs")
    target = config["target_processing"]
    MouseBins(target["mouse_bins"])
    weights = np.asarray(target["binary_class_weights"])
    if weights.shape != (8, 2) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Invalid source binary loss weights")


def load_data(args, config, budget=None):
    split = load_frozen_split(args.split_manifest)
    if config["training"].get("split_sha256") != split.digest:
        raise ValueError("Source and probe must use the identical frozen split")
    base = discover_cached_episodes(args.base_cache)
    split.verify_cache(base, "base")
    train, validation = partition_frozen_training(base, split)
    groups = {}
    known = {m.name for m in base}
    for group, path in (("recovery", args.recovery_cache), ("proactive_turn", args.proactive_cache)):
        extra = discover_cached_episodes(path)
        split.verify_cache(extra, group)
        if any(split.role(m, group) != "train" or m.name in known for m in extra):
            raise ValueError("Correction cache contains duplicate/non-training episodes")
        known.update(m.name for m in extra)
        groups.update({m.name: group for m in extra})
        train.extend(extra)
        if budget:
            budget.check()
    assessment = discover_cached_episodes(args.assessment_cache)
    if len(assessment) != 6 or len({m.name for m in assessment}) != 6:
        raise ValueError("This probe requires exactly the six independent assessment episodes")
    if {m.name for m in assessment} & (known | {name for _, name in split.episodes}):
        raise ValueError("Assessment episodes overlap frozen data")
    if {m.name for m in train} != set(config["training"]["training_episodes"]):
        raise ValueError("Probe training membership differs from the source checkpoint")
    if {m.name for m in validation} != set(config["training"]["validation_episodes"]):
        raise ValueError("Probe validation membership differs from the source checkpoint")
    result = [BehaviorCloningDataset(ms) for ms in (train, validation, assessment)]
    if budget:
        budget.check()
    return (*result, groups, split)


def plan_probe(args, config, data):
    train, validation, assessment, groups, split = data
    plans = {}
    for arm, boost in (("control", 1.0), ("boosted", args.boost)):
        weights, opening = recovery_opening_weights(train, groups, window_frames=args.window_frames, boost=boost)
        probability = float(weights[opening].sum() / weights.sum())
        plans[arm] = {"recovery_opening_boost": boost, "optimizer_steps": args.steps,
                      "sample_draws": args.steps * args.batch_size,
                      "expected_recovery_opening_share": probability,
                      "expected_recovery_opening_draws": probability * args.steps * args.batch_size}
    return {
        "source": str(args.source.resolve()), "output": str(args.output.resolve()),
        "mode": "execute" if args.execute else "plan_only_no_training",
        "split_id": split.split_id, "split_sha256": split.digest,
        "train_episodes": len(train.episode_names), "validation_episodes": len(validation.episode_names),
        "assessment_episodes": len(assessment.episode_names),
        "cache_paths": {name: str(path.resolve()) for name, path in (
            ("base", args.base_cache), ("recovery", args.recovery_cache),
            ("proactive_turn", args.proactive_cache), ("assessment", args.assessment_cache))},
        "source_targets_preserved": True, "fresh_optimizer_each_arm": True,
        "training_precision": "float32_no_amp_skipped_updates",
        "group_fractions": {"base": 0.9, **GROUP_FRACTIONS},
        "window_frames": args.window_frames, "learning_rate": args.learning_rate,
        "seed": args.seed, "batch_size": args.batch_size, "arms": plans,
        "validation_batch_size": args.validation_batch_size,
        "shared_soft_budget_seconds": args.time_budget_seconds,
        "external_hard_cap_required": "Use documented systemd RuntimeMaxSec=20min command",
        "rough_total_runtime_minutes": [12, 18],
        "estimate_note": "Based on prior workstation runs; may time out before both arms finish. No automatic continuation.",
        "selection": "Fixed endpoints only; no selection on assessment, no automatic promotion",
    }


def evaluate(model, dataset, device, target, budget, batch_size, label_smoothing=0.01):
    bins = MouseBins(target["mouse_bins"])
    loss_weights = torch.tensor(target["binary_class_weights"], dtype=torch.float32, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = {name: ActionMetrics() for name in ("all", "first8", "first24")}
    idle = {name: [0, 0] for name in metrics}
    first = []
    total_loss = 0.0
    offset = 0
    model.eval()
    with torch.no_grad():
        for frames, targets in CheckedLoader(loader, budget):
            x = frames.to(device=device, dtype=torch.float32).div_(255.0)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            predictions = model(x)
            loss, _ = behavior_cloning_loss(predictions, targets.to(device), loss_weights,
                                             mouse_bins=bins, label_smoothing=label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            total_loss += loss.item() * len(targets)
            binary = np.stack([predictions[k].argmax(1).cpu().numpy() for k in BINARY_ACTION_COLUMNS], axis=1)
            mouse = np.stack([np.rint(np.asarray(bins.representatives(axis))[predictions[key].argmax(1).cpu().numpy()])
                              for axis, key in enumerate(("mouse_dx_logits", "mouse_dy_logits"))], axis=1)
            actions = np.concatenate([binary, mouse], axis=1)
            expected = targets.numpy()
            for j, (e, p) in enumerate(zip(expected, actions)):
                episode, frame = dataset.index[offset + j]
                age = frame - dataset.first_action_frames[episode]
                for segment in metrics:
                    if segment == "first8" and age >= 8 or segment == "first24" and age >= 24:
                        continue
                    metrics[segment].add(e, p)
                    idle[segment][0] += int(not np.any(e))
                    idle[segment][1] += int(not np.any(p))
                if age == 0:
                    first.append({"episode": dataset.episode_names[episode], "frame_idx": frame,
                                  "expected": e.tolist(), "predicted": p.tolist(), "predicted_idle": not bool(np.any(p))})
            offset += len(targets)
            if offset % (200 * batch_size) == 0:
                print(f"  evaluated {offset:,}/{len(dataset):,} frames", flush=True)
    budget.check()
    if offset != len(dataset):
        raise RuntimeError("Incomplete evaluation cannot be reported as complete")
    return {"complete": True, "frames": offset, "loss": total_loss / offset,
            "precision": "float32", "segments": {
                name: {**m.report(), "expert_idle_frames": idle[name][0], "predicted_idle_frames": idle[name][1]}
                for name, m in metrics.items()}, "first_frames": first}


def fresh_model(state, config, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = BinnedImitationPolicy(*MouseBins(config["target_processing"]["mouse_bins"]).class_counts())
    model.load_state_dict(state)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def train_probe_steps(model, loader, optimizer, device, config, budget, progress):
    """Float32 updates avoid differing AMP skip counts in the small paired budget."""
    model.train()
    settings, target = config["training"], config["target_processing"]
    bins = MouseBins(target["mouse_bins"])
    weights = torch.tensor(target["binary_class_weights"], dtype=torch.float32, device=device)
    loss_sum = 0.0
    samples = 0
    for frames, targets in CheckedLoader(loader, budget):
        frames = frames.to(device=device, dtype=torch.float32).div_(255.0)
        if device.type == "cuda":
            frames = frames.contiguous(memory_format=torch.channels_last)
        aug = settings["augmentation"]
        frames, targets = augment_batch(
            frames, targets.to(device), brightness=aug["brightness"], contrast=aug["contrast"],
            horizontal_flip_probability=aug["horizontal_flip_probability"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = behavior_cloning_loss(model(frames), targets, weights, mouse_bins=bins,
                                        label_smoothing=settings["label_smoothing"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite probe training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"] or float("inf"),
                                       error_if_nonfinite=True)
        optimizer.step()
        progress["steps_completed"] += 1
        loss_sum += loss.item() * len(targets)
        samples += len(targets)
        if progress["steps_completed"] % 100 == 0:
            print(f"{progress['arm']}: {progress['steps_completed']} updates", flush=True)
        budget.check()
    return {"loss": loss_sum / samples, "samples": samples, "precision": "float32_no_amp_skips"}


def endpoint_config(source_config, plan, train, validation, groups, arm, source_hash):
    config = copy.deepcopy(source_config)
    config["training"].update({
        "learning_rate": plan["learning_rate"], "seed": plan["seed"], "batch_size": plan["batch_size"],
        "sampling": "chamber-balanced", "start_sampling_boost": 1.0,
        "validation_batch_size": plan["validation_batch_size"],
        "correction_sampling_fraction": 0.0, "correction_group_fractions": GROUP_FRACTIONS.copy(),
        "correction_episode_groups": groups.copy(),
        "training_episodes": train.episode_names.copy(), "validation_episodes": validation.episode_names.copy(),
        "train_extra_cache_dirs": [plan["cache_paths"][g] for g in ("recovery", "proactive_turn")],
        "split_id": plan["split_id"], "split_sha256": plan["split_sha256"],
        "lr_scheduler": {"name": "none"},
        "recovery_probe": {"arm": arm, "source": plan["source"], "source_sha256": source_hash,
                           "window_frames": plan["window_frames"], **plan["arms"][arm],
                           "cache_paths": plan["cache_paths"],
                           "fresh_optimizer": True, "generic_resume_supported": False,
                           "assessment_used_for_gradient_or_selection": False},
    })
    return config


def execute_probe(args, config, data, plan, budget):
    train, validation, assessment, groups, _ = data
    if args.output.exists():
        raise FileExistsError("Output must be a new directory; existing checkpoints will not be overwritten")
    with args.source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = load_checkpoint(args.source, device="cpu")
    if checkpoint["config"] != config:
        raise ValueError("Source sidecar changed during preparation")
    state = checkpoint["model_state_dict"]
    source_progress = {"epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"]}
    del checkpoint
    budget.check()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback training will start")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"plan": plan, "source_sha256": source_hash, "source_progress": source_progress,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "running", "paired_complete": False, "arms": {}}

    def write_report():
        report["elapsed_seconds"] = round(budget.clock() - budget.started, 3)
        temporary = args.output / "report.tmp.json"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "report.json")

    write_report()
    model = None
    try:
        model = fresh_model(state, config, device, args.seed)
        print("Evaluating the unchanged source checkpoint before either arm...", flush=True)
        smoothing = config["training"]["label_smoothing"]
        report["source_baseline"] = {
            "validation": evaluate(model, validation, device, config["target_processing"], budget, args.validation_batch_size, smoothing),
            "assessment": evaluate(model, assessment, device, config["target_processing"], budget, args.validation_batch_size, smoothing),
        }
        del model
        model = None
        write_report()
        for arm in ("control", "boosted"):
            budget.check()
            progress = {"arm": arm, "status": "training", "steps_completed": 0}
            report["arms"][arm] = progress
            write_report()
            model = fresh_model(state, config, device, args.seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                          weight_decay=config["training"]["weight_decay"], fused=device.type == "cuda")
            weights, _ = recovery_opening_weights(train, groups, window_frames=args.window_frames,
                                                  boost=plan["arms"][arm]["recovery_opening_boost"])
            generator = torch.Generator().manual_seed(args.seed)
            sampler = WeightedRandomSampler(torch.from_numpy(weights), args.steps * args.batch_size,
                                             replacement=True, generator=generator)
            draws = np.asarray(list(sampler), dtype=np.int64)
            labels = np.asarray([groups.get(train.episode_names[e], "base") for e, _ in train.index])
            _, opening_mask = recovery_opening_weights(train, groups, window_frames=args.window_frames)
            loader = DataLoader(Subset(train, draws.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            target = config["target_processing"]
            arm_config = endpoint_config(config, plan, train, validation, groups, arm, source_hash)

            def save_endpoint(name, status, loss):
                # epoch=0 explicitly avoids claiming a completed full dataset epoch.
                save_checkpoint(args.output / arm / name, model=model, optimizer=optimizer,
                                epoch=0, step_in_epoch=progress["steps_completed"],
                                global_step=progress["steps_completed"], best_val_loss=loss,
                                config=arm_config, training_state={"probe_status": status})

            def record_consumed_draws():
                consumed = draws[:progress["steps_completed"] * args.batch_size]
                progress["consumed_draw_counts"] = {
                    name: int(np.sum(labels[consumed] == name)) for name in ("base", "recovery", "proactive_turn")
                }
                progress["consumed_recovery_opening_draws"] = int(opening_mask[consumed].sum())
                progress["draw_sequence_sha256"] = hashlib.sha256(draws.tobytes()).hexdigest()

            try:
                training = train_probe_steps(model, loader, optimizer, device, config, budget, progress)
                record_consumed_draws()
                if progress["steps_completed"] != args.steps:
                    raise RuntimeError("Optimizer step budget did not complete")
                progress["training"] = training
                save_endpoint("endpoint.pt", "pending_evaluation", report["source_baseline"]["validation"]["loss"])
                progress["status"] = "evaluating"
                print(f"{arm}: updates complete; evaluating full validation and assessment", flush=True)
                write_report()
                progress["validation"] = evaluate(model, validation, device, target, budget, args.validation_batch_size, smoothing)
                progress["assessment"] = evaluate(model, assessment, device, target, budget, args.validation_batch_size, smoothing)
                save_endpoint("endpoint.pt", "complete_fixed_endpoint", progress["validation"]["loss"])
                budget.check()
                progress["status"] = "complete"
                write_report()
            except BudgetExceeded:
                record_consumed_draws()
                progress["status"] = "budget_exhausted_incomplete"
                if progress["steps_completed"]:
                    save_endpoint("partial.pt", "incomplete_not_for_comparison", report["source_baseline"]["validation"]["loss"])
                raise
            del model, optimizer
            model = None
        report["paired_complete"] = True
        report["status"] = "complete_no_automatic_promotion"
    except BudgetExceeded as error:
        report["status"] = "budget_exhausted_incomplete"
        report["reason"] = str(error)
    except BaseException as error:
        report["status"] = "interrupted_or_failed_incomplete"
        report["reason"] = f"{type(error).__name__}: {error}"
        write_report()
        raise
    finally:
        write_report()
    print(f"Probe status: {report['status']}; report: {args.output / 'report.json'}", flush=True)
    return 0 if report["paired_complete"] else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly run the capped two-arm probe; otherwise plan only")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/frozen_v3.json"))
    parser.add_argument("--base-cache", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--recovery-cache", type=Path, default=Path("data/datasets/recovery_cached_frames"))
    parser.add_argument("--proactive-cache", type=Path, default=Path("data/datasets/proactive_turn_cached_frames"))
    parser.add_argument("--assessment-cache", type=Path, default=ASSESSMENT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--time-budget-seconds", type=float, default=1080)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--window-frames", type=int, default=24)
    parser.add_argument("--boost", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1000 or not math.isfinite(args.time_budget_seconds) or not 0 < args.time_budget_seconds <= 1080:
        parser.error("Budget must be 1–1000 steps/arm and at most1080 seconds total")
    if args.batch_size < 1 or args.validation_batch_size < 1 or args.window_frames < 1:
        parser.error("Batch sizes and opening window must be positive")
    if not math.isfinite(args.boost) or args.boost <= 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Boost must be finite >1; learning rate must be finite >0")
    return args


def main(argv=None):
    args = parse_args(argv)
    budget = Budget(args.time_budget_seconds) if args.execute else None
    config = json.loads(args.source.with_suffix(".json").read_text())["config"]
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    validate_source(config)
    data = load_data(args, config, budget)
    plan = plan_probe(args, config, data)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    return execute_probe(args, config, data, plan, budget)


if __name__ == "__main__":
    raise SystemExit(main())
