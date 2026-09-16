"""Report completed demonstration episodes by outcome category and date."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EPISODE_DATE_PATTERN = re.compile(r"^episode_(\d{4})(\d{2})(\d{2})(?:_|$)")
UNCATEGORIZED = "uncategorized"


@dataclass
class DatasetCount:
    episodes: int = 0
    action_rows: int = 0

    def add(self, action_rows: int) -> None:
        self.episodes += 1
        self.action_rows += action_rows


@dataclass
class DatasetStats:
    categories: dict[str, dict[str, DatasetCount]]
    skipped_incomplete: int = 0

    @property
    def total(self) -> DatasetCount:
        result = DatasetCount()
        for dates in self.categories.values():
            for count in dates.values():
                result.episodes += count.episodes
                result.action_rows += count.action_rows
        return result


def episode_date(episode_dir: Path) -> str:
    """Read the local recording date from the standard episode directory name."""
    match = EPISODE_DATE_PATTERN.match(episode_dir.name)
    if match:
        compact_date = "".join(match.groups())
        try:
            return datetime.strptime(compact_date, "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(episode_dir.stat().st_mtime).date().isoformat()


def count_action_rows(actions_path: Path) -> int:
    with actions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if row)


def collect_dataset_stats(root: Path) -> DatasetStats:
    """Count complete episodes and aligned action rows by category and date."""
    root = root.resolve()
    categories: dict[str, dict[str, DatasetCount]] = defaultdict(
        lambda: defaultdict(DatasetCount)
    )
    skipped_incomplete = 0
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    for actions_path in sorted(root.rglob("actions.csv")):
        episode_dir = actions_path.parent
        relative_parts = episode_dir.relative_to(root).parts
        if not relative_parts or any(part.startswith(".") for part in relative_parts):
            continue
        if not (episode_dir / "video.mp4").is_file():
            skipped_incomplete += 1
            continue
        category = relative_parts[0] if len(relative_parts) > 1 else UNCATEGORIZED
        date = episode_date(episode_dir)
        categories[category][date].add(count_action_rows(actions_path))

    return DatasetStats(
        categories={
            category: dict(dates) for category, dates in categories.items()
        },
        skipped_incomplete=skipped_incomplete,
    )


def format_count(count: DatasetCount) -> str:
    episode_word = "episode" if count.episodes == 1 else "episodes"
    row_word = "action row" if count.action_rows == 1 else "action rows"
    return (
        f"{count.episodes:,} {episode_word}, "
        f"{count.action_rows:,} {row_word}"
    )


def format_report(root: Path, stats: DatasetStats) -> str:
    lines = [f"Dataset: {root.resolve()}"]
    if not stats.categories:
        lines.append("No completed episodes found.")
    for category in sorted(stats.categories):
        dates = stats.categories[category]
        category_total = DatasetCount(
            episodes=sum(count.episodes for count in dates.values()),
            action_rows=sum(count.action_rows for count in dates.values()),
        )
        lines.append(f"{category}: {format_count(category_total)}")
        for date in sorted(dates):
            lines.append(f"  {date}: {format_count(dates[date])}")
    lines.append(f"TOTAL: {format_count(stats.total)}")
    if stats.skipped_incomplete:
        lines.append(
            f"Skipped incomplete entries: {stats.skipped_incomplete:,} "
            "(missing video.mp4)"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count completed dataset episodes and action rows by category and date."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=Path("episodes"),
        help="Episode dataset directory (default: episodes)",
    )
    args = parser.parse_args()
    stats = collect_dataset_stats(args.dataset_dir)
    print(format_report(args.dataset_dir, stats))


if __name__ == "__main__":
    main()
