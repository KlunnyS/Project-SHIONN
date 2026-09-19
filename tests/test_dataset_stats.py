import json
import tempfile
import unittest
from pathlib import Path

from dataset_stats import collect_dataset_stats, format_report


class DatasetStatsTest(unittest.TestCase):
    @staticmethod
    def make_episode(
        root: Path, category: str, name: str, rows: int, chamber: str | None = None
    ) -> None:
        episode = root / category / name
        episode.mkdir(parents=True)
        (episode / "video.mp4").touch()
        content = ["frame_idx,move_w", *[f"{index},0" for index in range(rows)]]
        (episode / "actions.csv").write_text("\n".join(content) + "\n")
        if chamber is not None:
            (episode / "metadata.json").write_text(json.dumps({"map": chamber}) + "\n")

    def test_counts_completed_episodes_by_category_and_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episodes"
            self.make_episode(
                root, "goal_reached", "episode_20260915_120000_000001", 3
            )
            self.make_episode(
                root, "goal_reached", "episode_20260915_130000_000002", 2
            )
            self.make_episode(
                root, "goal_reached", "episode_20260916_120000_000003", 4
            )
            self.make_episode(
                root, "timeout", "episode_20260916_120000_000004", 5
            )

            stats = collect_dataset_stats(root)

            self.assertEqual(stats.categories["goal_reached"]["2026-09-15"].episodes, 2)
            self.assertEqual(stats.categories["goal_reached"]["2026-09-15"].action_rows, 5)
            self.assertEqual(stats.categories["goal_reached"]["2026-09-16"].episodes, 1)
            self.assertEqual(stats.categories["timeout"]["2026-09-16"].action_rows, 5)
            self.assertEqual(stats.total.episodes, 4)
            self.assertEqual(stats.total.action_rows, 14)

    def test_excludes_in_progress_and_reports_incomplete_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episodes"
            self.make_episode(
                root, ".in_progress", "episode_20260916_120000_000001", 3
            )
            incomplete = root / "timeout" / "episode_20260916_120000_000002"
            incomplete.mkdir(parents=True)
            (incomplete / "actions.csv").write_text("frame_idx,move_w\n0,0\n")

            stats = collect_dataset_stats(root)
            report = format_report(root, stats)

            self.assertEqual(stats.total.episodes, 0)
            self.assertEqual(stats.skipped_incomplete, 1)
            self.assertIn("No completed episodes found.", report)
            self.assertIn("Skipped incomplete entries: 1", report)

    def test_reports_chambers_within_dates_and_overall(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episodes"
            self.make_episode(
                root, "goal_reached", "episode_20260919_120000_000001", 3,
                "dataset_test1",
            )
            self.make_episode(
                root, "goal_reached", "episode_20260919_130000_000002", 2,
                "dataset_test2",
            )
            self.make_episode(
                root, "timeout", "episode_20260919_140000_000003", 4,
                "dataset_test2",
            )
            self.make_episode(
                root, "timeout", "episode_20260919_150000_000004", 1
            )

            stats = collect_dataset_stats(root)
            report = format_report(root, stats)

            self.assertEqual(stats.total.episodes, 4)
            self.assertEqual(stats.chambers["dataset_test2"].episodes, 2)
            self.assertEqual(stats.chambers["dataset_test2"].action_rows, 6)
            self.assertIn("    dataset_test1: 1 episode, 3 action rows", report)
            self.assertIn("    dataset_test2: 1 episode, 2 action rows", report)
            self.assertIn("  dataset_test2: 2 episodes, 6 action rows", report)
            self.assertIn("  unknown: 1 episode, 1 action row", report)


if __name__ == "__main__":
    unittest.main()
