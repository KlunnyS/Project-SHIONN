"""Sweep jump decision thresholds on held-out expert action predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def score_jump_rows(rows: list[dict], thresholds: tuple[float, ...]) -> list[dict]:
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map"]].append(row)
    by_map["overall"] = rows
    scores = []
    for map_name, selected in sorted(by_map.items()):
        for threshold in thresholds:
            tp = fp = fn = tn = 0
            for row in selected:
                expected = int(row["expected_jump"]) == 1
                predicted = float(row["prob_jump"]) >= threshold
                tp += expected and predicted
                fp += not expected and predicted
                fn += expected and not predicted
                tn += not expected and not predicted
            scores.append({
                "map": map_name,
                "threshold": threshold,
                "positives": tp + fn,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            })
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action_comparison", type=Path,
                        help="action_comparison.csv from benchmark_expert.py")
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds",
                        help="Repeat to replace the default 0.1–0.9 sweep")
    parser.add_argument("--output", type=Path, help="Defaults beside the input CSV")
    args = parser.parse_args(argv)
    thresholds = tuple(args.thresholds or DEFAULT_THRESHOLDS)
    if any(not 0 < value < 1 for value in thresholds):
        parser.error("thresholds must be between 0 and 1")
    with args.action_comparison.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Action comparison CSV has no rows")
    scores = score_jump_rows(rows, thresholds)
    output = args.output or args.action_comparison.with_name("jump_thresholds.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source": str(args.action_comparison.resolve()), "scores": scores,
    }, indent=2) + "\n", encoding="utf-8")
    for row in scores:
        if row["map"] == "overall":
            precision = f"{row['precision']:.3f}" if row["precision"] is not None else "n/a"
            recall = f"{row['recall']:.3f}" if row["recall"] is not None else "n/a"
            print(
                f"threshold {row['threshold']:.2f}: precision={precision} "
                f"recall={recall} TP={row['true_positive']} FP={row['false_positive']}"
            )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
