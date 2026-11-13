"""Plan a single history-augmentation arm against a verified, completed control.

Nothing trains unless --execute is supplied. The previous probe helper stays
unchanged so its code hash and sampling/optimizer/evaluation protocol can be
checked before reusing its control. No game APIs are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import run_recovery_probe as base
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import discover_cached_episodes

REFERENCE = Path("models/imitation/checkpoints/runs/recovery_onset_probe_v1/report.json")
OUTPUT = Path("models/imitation/checkpoints/runs/recovery_history_probe_v1")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repetition_mask(draws, labels, probability, seed):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Repeat probability must be in [0, 1]")
    # Private RNG: do not alter the Torch RNG used by existing augmentation/dropout.
    rng = np.random.default_rng((seed + 20261102) % (2 ** 64))
    return (labels[draws] == "recovery") & (rng.random(len(draws)) < probability)


class RepeatedHistoryDataset(Dataset):
    def __init__(self, dataset, draws, mask):
        if len(draws) != len(mask):
            raise ValueError("Each draw position needs an augmentation flag")
        self.dataset, self.draws, self.mask = dataset, draws, mask

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, position):
        frames, target = self.dataset[int(self.draws[position])]
        if self.mask[position]:
            frames = np.tile(frames[-3:], (4, 1, 1))
        return frames, target


def validate_reference(reference, helper_hash):
    if reference.get("status") != "complete_no_automatic_promotion" or reference.get("paired_complete") is not True:
        raise ValueError("Reference must be a fully completed paired probe")
    if reference.get("script_sha256") != helper_hash:
        raise ValueError("Original helper changed; control cannot be silently reused")
    plan, control = reference["plan"], reference["arms"]["control"]
    if plan["training_precision"] != "float32_no_amp_skipped_updates":
        raise ValueError("Reference training precision differs")
    if plan["group_fractions"] != {"base": .9, **base.GROUP_FRACTIONS}:
        raise ValueError("Reference group recipe differs")
    steps = plan["arms"]["control"]["optimizer_steps"]
    if not 1 <= steps <= 1000 or plan["arms"]["control"]["recovery_opening_boost"] != 1:
        raise ValueError("Reference control must be unboosted and bounded")
    if control.get("status") != "complete" or control.get("steps_completed") != steps:
        raise ValueError("Reference control update count/status differs")
    if control["training"].get("samples") != steps * plan["batch_size"]:
        raise ValueError("Reference sample budget differs")
    if control["training"].get("precision") != "float32_no_amp_skips":
        raise ValueError("Reference optimizer-update precision differs")
    for cohort in ("validation", "assessment"):
        if control.get(cohort, {}).get("complete") is not True:
            raise ValueError("Reference evaluation is incomplete")
    return plan, control


def cache_fingerprint(paths, cutoff_ns):
    """Conservative historical mtime guard; record current labels/metadata hashes.

    v1 did not hash historical pixels. Preserve that uncertainty rather than
    claiming a retroactively proved bitwise-identical dataset.
    """
    digest = hashlib.sha256()
    files = set()
    for path in paths.values():
        root = Path(path)
        files.update(p.resolve() for p in root.glob("*.json"))
        for manifest in discover_cached_episodes(root):
            files.update((manifest.frames_path.resolve(), manifest.actions_path.resolve()))
    for path in sorted(files):
        stat = path.stat()
        if stat.st_mtime_ns > cutoff_ns:
            raise ValueError(f"Cache changed after reference began: {path.name}")
        digest.update(str(path).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        if path.suffix == ".json" or path.name.endswith(".actions.npy"):
            digest.update(file_hash(path).encode())
    return {"files": len(files), "current_metadata_actions_and_frame_stats_sha256": digest.hexdigest(),
            "frame_pixel_contents_hashed": False,
            "historical_full_content_hashes_available": False}


def prepare(args, budget=None):
    reference = json.loads(args.reference.read_text())
    plan, control = validate_reference(reference, file_hash(base.__file__))
    common = base.parse_args([])
    common.source, common.output = Path(plan["source"]), args.output
    common.execute, common.device = args.execute, args.device
    common.split_manifest = args.split_manifest
    for name in ("seed", "batch_size", "learning_rate", "window_frames", "validation_batch_size"):
        setattr(common, name, plan[name])
    common.steps = control["steps_completed"]
    common.time_budget_seconds = args.time_budget_seconds
    common.base_cache = Path(plan["cache_paths"]["base"])
    common.recovery_cache = Path(plan["cache_paths"]["recovery"])
    common.proactive_cache = Path(plan["cache_paths"]["proactive_turn"])
    common.assessment_cache = Path(plan["cache_paths"]["assessment"])
    if file_hash(common.source) != reference["source_sha256"]:
        raise ValueError("Parent checkpoint changed")
    config = json.loads(common.source.with_suffix(".json").read_text())["config"]
    base.validate_source(config)
    data = base.load_data(common, config, budget)
    train, validation, assessment, groups, split = data
    if split.digest != plan["split_sha256"]:
        raise ValueError("Reference split changed")
    for cohort, ds in (("validation", validation), ("assessment", assessment)):
        score = control[cohort]
        if score["frames"] != len(ds) or {x["episode"] for x in score["first_frames"]} != set(ds.episode_names):
            raise ValueError(f"Reference {cohort} cohort differs")
    endpoint = args.reference.parent / "control/endpoint.pt"
    sidecar = json.loads(endpoint.with_suffix(".json").read_text())
    expected_config = base.endpoint_config(config, plan, train, validation, groups, "control", reference["source_sha256"])
    if sidecar["config"] != expected_config:
        raise ValueError("Reference control configuration differs")
    if (sidecar["epoch"], sidecar["global_step"], sidecar["step_in_epoch"]) != (0, common.steps, common.steps):
        raise ValueError("Reference endpoint step metadata differs")
    if sidecar.get("training_state", {}).get("probe_status") != "complete_fixed_endpoint":
        raise ValueError("Reference endpoint is incomplete")
    cutoff = args.reference.stat().st_mtime_ns - int(reference["elapsed_seconds"] * 1e9)
    fingerprint = cache_fingerprint(plan["cache_paths"], cutoff)
    if any(args.output.resolve().is_relative_to(Path(p).resolve()) for p in plan["cache_paths"].values()):
        raise ValueError("Output must stay outside all data caches")
    weights, _ = base.recovery_opening_weights(train, groups, window_frames=common.window_frames)
    generator = torch.Generator().manual_seed(common.seed)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), common.steps * common.batch_size,
                                    replacement=True, generator=generator)
    draws = np.asarray(list(sampler), dtype=np.int64)
    draw_hash = hashlib.sha256(draws.tobytes()).hexdigest()
    labels = np.asarray([groups.get(train.episode_names[e], "base") for e, _ in train.index])
    counts = {group: int((labels[draws] == group).sum()) for group in ("base", "recovery", "proactive_turn")}
    if draw_hash != control["draw_sequence_sha256"] or counts != control["consumed_draw_counts"]:
        raise ValueError("Reconstructed control draws do not match; reuse refused")
    mask = repetition_mask(draws, labels, args.repeat_probability, common.seed)
    if budget:
        budget.check()
    result = {
        "mode": "execute" if args.execute else "plan_only_no_training",
        "source": str(common.source.resolve()), "output": str(args.output.resolve()),
        "reference_report": str(args.reference.resolve()), "reference_report_sha256": file_hash(args.reference),
        "reference_control_checkpoint": str(endpoint.resolve()), "reference_control_sha256": file_hash(endpoint),
        "helper_sha256": reference["script_sha256"], "parent_sha256": reference["source_sha256"],
        "control_retrained": False, "steps": common.steps, "batch_size": common.batch_size,
        "seed": common.seed, "learning_rate": common.learning_rate, "draw_sequence_sha256": draw_hash,
        "draw_counts": counts, "repeat_probability": args.repeat_probability,
        "repeat_scope": "recovery_only_all_frames", "repeated_draws": int(mask.sum()),
        "repeat_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "unchanged_labels_and_other_groups": True, "assessment_episodes": len(assessment.episode_names),
        "shared_soft_budget_seconds": args.time_budget_seconds, "rough_total_runtime_minutes": [4, 8],
        "external_hard_cap": "Use documented RuntimeMaxSec=12min command",
        "cache_assurance": fingerprint,
        "reuse_limitations": "v1 did not record library versions or full pixel hashes. Code/source/config/draws/cohorts and file-time guards match; historical bitwise identity cannot be proved.",
        "current_libraries": {"torch": str(torch.__version__), "numpy": np.__version__},
        "no_automatic_promotion": True,
    }
    return SimpleNamespace(args=args, common=common, config=config, data=data, reference=reference,
                           draws=draws, labels=labels, mask=mask, plan=result, endpoint=endpoint)


def execute(prepared, budget):
    args, common, cfg, plan = prepared.args, prepared.common, prepared.config, prepared.plan
    train, validation, assessment, groups, _ = prepared.data
    if args.output.exists():
        raise FileExistsError("Use a fresh output; prior runs will not be overwritten")
    for path, expected in ((args.reference, plan["reference_report_sha256"]),
                           (prepared.endpoint, plan["reference_control_sha256"]),
                           (base.__file__, plan["helper_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("Reference/helper changed during preparation; reuse refused")
    reference_endpoint = load_checkpoint(prepared.endpoint, device="cpu")
    if reference_endpoint["config"] != base.endpoint_config(
            cfg, prepared.reference["plan"], train, validation, groups, "control", plan["parent_sha256"]):
        raise ValueError("Reference binary and expected control differ")
    if tuple(reference_endpoint[name] for name in ("epoch", "global_step", "step_in_epoch")) != (0, common.steps, common.steps):
        raise ValueError("Reference binary progress differs")
    del reference_endpoint
    source = load_checkpoint(common.source, device="cpu")
    if source["config"] != cfg or file_hash(common.source) != plan["parent_sha256"]:
        raise ValueError("Parent changed during preparation")
    state = source["model_state_dict"]
    del source
    budget.check()
    device = torch.device(common.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    progress = {"arm": "history_repeat", "status": "training", "steps_completed": 0}
    report = {"plan": plan, "status": "running", "comparison_complete": False, "new_arm": progress,
              "reused_control": copy.deepcopy(prepared.reference["arms"]["control"]),
              "source_baseline": copy.deepcopy(prepared.reference["source_baseline"]),
              "script_sha256": file_hash(__file__)}

    def write_report():
        report["elapsed_seconds"] = round(budget.clock() - budget.started, 3)
        p = args.output / "report.tmp.json"
        p.write_text(json.dumps(report, indent=2) + "\n")
        p.replace(args.output / "report.json")

    write_report()
    model = optimizer = None
    try:
        # Parent, never the trained control endpoint; no optimizer state is loaded.
        model = base.fresh_model(state, cfg, device, common.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=common.learning_rate,
                                      weight_decay=cfg["training"]["weight_decay"], fused=device.type == "cuda")
        dataset = RepeatedHistoryDataset(train, prepared.draws, prepared.mask)
        loader = DataLoader(dataset, batch_size=common.batch_size, shuffle=False, num_workers=0)
        endpoint_plan = copy.deepcopy(prepared.reference["plan"])
        endpoint_plan["arms"]["history_repeat"] = copy.deepcopy(endpoint_plan["arms"]["control"])
        endpoint_cfg = base.endpoint_config(cfg, endpoint_plan, train, validation, groups, "history_repeat", plan["parent_sha256"])
        endpoint_cfg["training"]["recovery_probe"]["history_repetition"] = {
            "probability": args.repeat_probability, "scope": plan["repeat_scope"],
            "mask_sha256": plan["repeat_mask_sha256"], "reference_report_sha256": plan["reference_report_sha256"],
            "reference_control_sha256": plan["reference_control_sha256"],
        }

        def record_consumed():
            n = progress["steps_completed"] * common.batch_size
            progress["consumed_draw_counts"] = {
                name: int((prepared.labels[prepared.draws[:n]] == name).sum())
                for name in ("base", "recovery", "proactive_turn")}
            progress["consumed_repeated_draws"] = int(prepared.mask[:n].sum())

        def save_endpoint(filename, status, loss):
            save_checkpoint(args.output / filename, model=model, optimizer=optimizer, epoch=0,
                            step_in_epoch=progress["steps_completed"], global_step=progress["steps_completed"],
                            best_val_loss=loss, config=endpoint_cfg, training_state={"probe_status": status})

        try:
            progress["training"] = base.train_probe_steps(model, loader, optimizer, device, cfg, budget, progress)
            record_consumed()
            if progress["steps_completed"] != common.steps or progress["consumed_draw_counts"] != plan["draw_counts"]:
                raise RuntimeError("Completed updates/draws do not match reference")
            save_endpoint("endpoint.pt", "pending_evaluation", report["reused_control"]["validation"]["loss"])
            progress["status"] = "evaluating"
            write_report()
            print("Updates complete; evaluating full validation and six assessment clips...", flush=True)
            smoothing = cfg["training"]["label_smoothing"]
            progress["validation"] = base.evaluate(model, validation, device, cfg["target_processing"], budget,
                                                   common.validation_batch_size, smoothing)
            progress["assessment"] = base.evaluate(model, assessment, device, cfg["target_processing"], budget,
                                                   common.validation_batch_size, smoothing)
            save_endpoint("endpoint.pt", "complete_fixed_endpoint", progress["validation"]["loss"])
            budget.check()
            progress["status"] = "complete"
            report["status"] = "complete_no_automatic_promotion"
            report["comparison_complete"] = True
        except base.BudgetExceeded:
            record_consumed()
            progress["status"] = "budget_exhausted_incomplete"
            if progress["steps_completed"]:
                save_endpoint("partial.pt", "incomplete_not_for_comparison", report["reused_control"]["validation"]["loss"])
            raise
    except base.BudgetExceeded as error:
        report["status"] = "budget_exhausted_incomplete"
        report["reason"] = str(error)
    except BaseException as error:
        report["status"] = "interrupted_or_failed_incomplete"
        report["reason"] = f"{type(error).__name__}: {error}"
        write_report()
        raise
    finally:
        write_report()
    print(f"History probe: {report['status']}; report: {args.output / 'report.json'}", flush=True)
    return 0 if report["comparison_complete"] else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/frozen_v3.json"))
    parser.add_argument("--repeat-probability", type=float, default=.5)
    parser.add_argument("--time-budget-seconds", type=float, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if not math.isfinite(args.repeat_probability) or not 0 < args.repeat_probability <= 1:
        parser.error("Repeat probability must be finite and in (0, 1]")
    if not math.isfinite(args.time_budget_seconds) or not 0 < args.time_budget_seconds <= 600:
        parser.error("Shared budget must be at most 600 seconds")
    return args


def main(argv=None):
    args = parse_args(argv)
    budget = base.Budget(args.time_budget_seconds) if args.execute else None
    prepared = prepare(args, budget)
    print(json.dumps(prepared.plan, indent=2), flush=True)
    if not args.execute:
        return 0
    return execute(prepared, budget)


if __name__ == "__main__":
    raise SystemExit(main())
