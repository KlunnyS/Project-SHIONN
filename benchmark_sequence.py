"""Summarize autonomous chamber attempts from run_model_sequence.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


COMPLETED_OUTCOMES = {"goal_reached", "episode_failed", "time_limit"}


def read_attempt_log(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    metadata = {}
    summary = {}
    last_tick_elapsed = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    metadata = record
                elif record.get("type") == "tick" and isinstance(record.get("elapsed_seconds"), (int, float)):
                    last_tick_elapsed = record["elapsed_seconds"]
                elif record.get("type") == "summary":
                    summary = record
    except (OSError, json.JSONDecodeError):
        return {}
    fps = metadata.get("capture", {}).get("fps")
    ticks = summary.get("ticks")
    duration = last_tick_elapsed
    if duration is None and isinstance(ticks, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
        duration = ticks / fps
    return {
        "duration_seconds": duration,
        "focus_losses": summary.get("focus_losses"),
        "dry_run": metadata.get("dry_run", False),
        "has_summary": bool(summary),
    }


def summarize_sequence(sequence_path: Path) -> dict:
    sequence_path = sequence_path / "sequence.jsonl" if sequence_path.is_dir() else sequence_path
    if not sequence_path.is_file():
        raise FileNotFoundError(f"Sequence manifest not found: {sequence_path}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    attempts = []
    with sequence_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "result":
                continue
            log_path = Path(record["log"]) if record.get("log") else None
            if log_path is not None and not log_path.is_absolute():
                log_path = sequence_path.parent / log_path
            details = read_attempt_log(log_path)
            outcome = record.get("stop_reason", "incomplete")
            valid = (
                record.get("returncode") == 0
                and outcome in COMPLETED_OUTCOMES
                and details.get("has_summary", False)
                and not details.get("dry_run", False)
            )
            attempt = {
                "job": record.get("job"), "model": record["model"],
                "map": record["map"], "repeat": record.get("repeat"),
                "outcome": outcome, "valid": valid,
                "duration_seconds": details.get("duration_seconds"),
                "focus_losses": details.get("focus_losses"),
                "log": record.get("log"), "video": record.get("video"),
            }
            attempts.append(attempt)
            groups[(attempt["model"], attempt["map"])].append(attempt)
    if not attempts:
        raise ValueError(f"No completed sequence jobs found in {sequence_path}")

    summary = []
    for (model, map_name), rows in sorted(groups.items()):
        valid_rows = [row for row in rows if row["valid"]]
        successes = [row for row in valid_rows if row["outcome"] == "goal_reached"]
        success_times = [
            row["duration_seconds"] for row in successes
            if row["duration_seconds"] is not None
        ]
        summary.append({
            "model": model, "map": map_name,
            "scheduled_attempts": len(rows), "valid_attempts": len(valid_rows),
            "invalid_attempts": len(rows) - len(valid_rows),
            "successes": len(successes),
            "success_rate": len(successes) / len(valid_rows) if valid_rows else None,
            "median_success_seconds": median(success_times) if success_times else None,
            "outcomes": dict(sorted(Counter(row["outcome"] for row in valid_rows).items())),
            "focus_losses": sum(int(row["focus_losses"] or 0) for row in valid_rows),
        })
    return {
        "sequence": str(sequence_path.resolve()),
        "summary": summary,
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Report autonomous success by model and chamber.")
    parser.add_argument("sequence", type=Path, help="Sequence directory or sequence.jsonl")
    parser.add_argument("--output", type=Path, help="Defaults to benchmark_summary.json beside sequence.jsonl")
    args = parser.parse_args(argv)
    report = summarize_sequence(args.sequence)
    output = args.output or Path(report["sequence"]).with_name("benchmark_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["summary"]:
        rate = f"{row['success_rate']:.0%}" if row["success_rate"] is not None else "n/a"
        duration = (
            f"{row['median_success_seconds']:.1f}s"
            if row["median_success_seconds"] is not None else "n/a"
        )
        print(
            f"{row['model']} on {row['map']}: {row['successes']}/{row['valid_attempts']} "
            f"success ({rate}), median success {duration}, "
            f"invalid attempts {row['invalid_attempts']}"
        )
    print(f"Benchmark report: {output}")
    return report


if __name__ == "__main__":
    main()
