"""Summarize binned mouse uncertainty on annotated wall-facing rollout frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def parse_window(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = float(start_text), float(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Wall windows must be START:END seconds") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise argparse.ArgumentTypeError("Wall windows need 0 <= START < END")
    return start, end


def axis_distribution(probabilities: list[float], representatives: list[float]) -> dict:
    if len(probabilities) != len(representatives) or len(probabilities) < 2:
        raise ValueError("Mouse probability and bin representative lengths differ")
    total = sum(probabilities)
    if not math.isfinite(total) or total <= 0 or any(
        not math.isfinite(value) or value < 0 for value in probabilities
    ):
        raise ValueError("Invalid mouse bin probabilities")
    normalized = [value / total for value in probabilities]
    ordered = sorted(range(len(normalized)), key=normalized.__getitem__, reverse=True)
    first, second = ordered[:2]
    positive_mass = sum(value for value, representative in zip(normalized, representatives) if representative > 0)
    negative_mass = sum(value for value, representative in zip(normalized, representatives) if representative < 0)
    zero_mass = sum(value for value, representative in zip(normalized, representatives) if representative == 0)
    entropy = -sum(value * math.log(value) for value in normalized if value > 0)
    first_direction = math.copysign(1, representatives[first]) if representatives[first] else 0
    second_direction = math.copysign(1, representatives[second]) if representatives[second] else 0
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(len(normalized)) if len(normalized) > 1 else 0.0,
        "peak_gap": normalized[first] - normalized[second],
        "top_bin": first,
        "top_probability": normalized[first],
        "second_bin": second,
        "second_probability": normalized[second],
        "opposing_top_bins": first_direction * second_direction < 0,
        "negative_mass": negative_mass,
        "zero_mass": zero_mass,
        "positive_mass": positive_mass,
    }


def analyze_attempt(path: Path, windows: list[tuple[float, float]]) -> tuple[dict, list[dict]]:
    metadata = None
    selected = []
    all_ticks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "metadata":
                metadata = record
            elif record.get("type") == "tick":
                all_ticks.append(record)
                if any(start <= record.get("elapsed_seconds", -1) <= end for start, end in windows):
                    selected.append(record)
    if not metadata or not metadata.get("mouse_bins"):
        raise ValueError("Attempt log has no binned mouse checkpoint metadata")
    bins = metadata["mouse_bins"]
    if not selected:
        raise ValueError("No ticks fall inside the annotated wall windows")
    rows = []
    for tick in selected:
        probabilities = (tick.get("policy") or {}).get("mouse_bin_probabilities")
        if not probabilities:
            raise ValueError(f"Tick {tick.get('tick')} has no binned mouse probabilities")
        row = {"tick": tick["tick"], "elapsed_seconds": tick["elapsed_seconds"]}
        for axis in ("dx", "dy"):
            metrics = axis_distribution(probabilities[axis], bins[axis]["representatives"])
            row.update({f"{axis}_{name}": value for name, value in metrics.items()})
        row["jump_probability"] = tick["policy"]["binary_probabilities"]["jump"]
        row["mouse_dx"] = tick["action"]["mouse_dx"]
        row["mouse_dy"] = tick["action"]["mouse_dy"]
        rows.append(row)
    selected_ticks = {tick["tick"] for tick in selected}
    other_ticks = [tick for tick in all_ticks if tick["tick"] not in selected_ticks]
    summary = {
        "attempt_log": str(path.resolve()),
        "map": metadata.get("map"),
        "wall_windows": windows,
        "wall_frames": len(rows),
        "metrics": {
            axis: {
                name: mean(float(row[f"{axis}_{name}"]) for row in rows)
                for name in (
                    "normalized_entropy", "peak_gap", "opposing_top_bins",
                    "negative_mass", "zero_mass", "positive_mass",
                )
            }
            for axis in ("dx", "dy")
        },
        "mean_jump_probability": mean(row["jump_probability"] for row in rows),
        "jump_action_rate_wall": mean(bool(tick["action"]["jump"]) for tick in selected),
        "jump_action_rate_other": (
            mean(bool(tick["action"]["jump"]) for tick in other_ticks)
            if other_ticks else None
        ),
    }
    return summary, rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_log", type=Path)
    parser.add_argument("--wall-window", type=parse_window, action="append", required=True,
                        metavar="START:END", help="Seconds identified as wall-facing in the attempt video")
    parser.add_argument("--output-prefix", type=Path, help="Defaults beside the attempt log")
    args = parser.parse_args(argv)
    summary, rows = analyze_attempt(args.attempt_log, args.wall_window)
    prefix = args.output_prefix or args.attempt_log.with_name(args.attempt_log.stem + "_wall_mouse")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path = prefix.with_suffix(".json")
    rows_path = prefix.with_suffix(".csv")
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wall frames: {len(rows)}")
    for axis in ("dx", "dy"):
        metrics = summary["metrics"][axis]
        print(
            f"{axis}: entropy={metrics['normalized_entropy']:.3f} "
            f"peak gap={metrics['peak_gap']:.3f} "
            f"opposing top bins={metrics['opposing_top_bins']:.1%}"
        )
    print(f"Report: {report_path}")
    print(f"Per-frame: {rows_path}")


if __name__ == "__main__":
    main()
