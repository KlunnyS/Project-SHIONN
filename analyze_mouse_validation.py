"""Measure mouse-bin uncertainty and jump errors on annotated validation frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from analyze_mouse_rollout import axis_distribution
from benchmark_jump import score_jump_rows


def read_wall_ranges(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Read inclusive frame ranges labeled wall-facing in expert episodes."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"episode", "start_frame", "end_frame"}.issubset(reader.fieldnames or []):
            raise ValueError("Wall ranges CSV needs episode,start_frame,end_frame columns")
        for row in reader:
            episode = row["episode"].strip()
            start, end = int(row["start_frame"]), int(row["end_frame"])
            if not episode or start < 0 or end < start:
                raise ValueError(f"Invalid wall frame range: {row}")
            ranges.setdefault(episode, []).append((start, end))
    if not ranges:
        raise ValueError("Wall ranges CSV contains no ranges")
    return ranges


def analyze_validation(
    comparison_path: Path, summary_path: Path, ranges_path: Path,
    jump_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    if not 0 < jump_threshold < 1:
        raise ValueError("Jump threshold must be between 0 and 1")
    benchmark = json.loads(summary_path.read_text(encoding="utf-8"))
    bins = benchmark.get("mouse_bins")
    if not bins:
        raise ValueError("Expert report must come from a binned mouse checkpoint")
    wall_ranges = read_wall_ranges(ranges_path)
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wall_rows, other_rows, details = [], [], []
    for row in rows:
        frame_idx = int(row["frame_idx"])
        is_wall = any(
            start <= frame_idx <= end
            for start, end in wall_ranges.get(row["episode"], [])
        )
        if not is_wall:
            other_rows.append(row)
            continue
        wall_rows.append(row)
        detail = {
            "episode": row["episode"], "map": row["map"], "frame_idx": frame_idx,
            "expected_jump": int(row["expected_jump"]),
            "jump_probability": float(row["prob_jump"]),
            "jump_false_positive": (
                int(row["expected_jump"]) == 0
                and float(row["prob_jump"]) >= jump_threshold
            ),
        }
        for axis in ("dx", "dy"):
            distribution = json.loads(row[f"mouse_{axis}_bin_probabilities"])
            metrics = axis_distribution(distribution, bins[axis]["representatives"])
            detail.update({f"{axis}_{name}": value for name, value in metrics.items()})
            detail[f"expected_mouse_{axis}"] = float(row[f"expected_mouse_{axis}"])
            detail[f"predicted_mouse_{axis}"] = int(row[f"predicted_mouse_{axis}"])
        details.append(detail)
    if not wall_rows:
        raise ValueError("No validation frames match the annotated wall ranges")

    def jump_stats(selected: list[dict]) -> dict | None:
        if not selected:
            return None
        overall = next(
            score for score in score_jump_rows(selected, (jump_threshold,))
            if score["map"] == "overall"
        )
        negatives = overall["false_positive"] + overall["true_negative"]
        overall["false_positive_rate"] = (
            overall["false_positive"] / negatives if negatives else None
        )
        overall["frames"] = len(selected)
        return overall

    report = {
        "comparison": str(comparison_path.resolve()),
        "wall_ranges": str(ranges_path.resolve()),
        "jump_threshold": jump_threshold,
        "wall_frames": len(wall_rows),
        "other_frames": len(other_rows),
        "mouse": {
            axis: {
                name: mean(float(row[f"{axis}_{name}"]) for row in details)
                for name in (
                    "normalized_entropy", "peak_gap", "opposing_top_bins",
                    "negative_mass", "zero_mass", "positive_mass",
                )
            }
            for axis in ("dx", "dy")
        },
        "jump": {"wall": jump_stats(wall_rows), "other": jump_stats(other_rows)},
    }
    return report, details


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action_comparison", type=Path)
    parser.add_argument("--wall-ranges", type=Path, required=True,
                        help="CSV with episode,start_frame,end_frame columns")
    parser.add_argument("--summary", type=Path,
                        help="Defaults to summary.json beside action_comparison.csv")
    parser.add_argument("--jump-threshold", type=float, default=0.5)
    parser.add_argument("--output-prefix", type=Path,
                        help="Defaults to wall_validation beside action_comparison.csv")
    args = parser.parse_args(argv)
    summary_path = args.summary or args.action_comparison.with_name("summary.json")
    report, rows = analyze_validation(
        args.action_comparison, summary_path, args.wall_ranges, args.jump_threshold
    )
    prefix = args.output_prefix or args.action_comparison.with_name("wall_validation")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path, rows_path = prefix.with_suffix(".json"), prefix.with_suffix(".csv")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wall validation frames: {len(rows)}")
    for axis in ("dx", "dy"):
        metrics = report["mouse"][axis]
        print(
            f"{axis}: entropy={metrics['normalized_entropy']:.3f} "
            f"peak gap={metrics['peak_gap']:.3f} "
            f"opposing top bins={metrics['opposing_top_bins']:.1%}"
        )
    print(f"Report: {report_path}")
    print(f"Per-frame: {rows_path}")


if __name__ == "__main__":
    main()
