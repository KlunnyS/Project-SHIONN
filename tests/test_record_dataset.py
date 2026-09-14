import unittest

from record_dataset import EpisodeEvent, EpisodeEventStream, event_outcome


class EpisodeEventStreamTest(unittest.TestCase):
    def test_parses_events_split_across_netconsole_chunks(self):
        stream = EpisodeEventStream()
        stream.feed("console noise\nEVT|chamber_")
        self.assertIsNone(stream.pop())
        stream.feed("ready|123.5\nmore noise\n")
        self.assertEqual(stream.pop(), EpisodeEvent("chamber_ready", "123.5"))
        self.assertIsNone(stream.pop())

    def test_parses_multiple_events_and_outcomes(self):
        stream = EpisodeEventStream()
        stream.feed("EVT|goal_reached|1\nEVT|episode_failed|out_of_bounds\n")
        goal = stream.pop()
        failure = stream.pop()
        self.assertEqual(event_outcome(goal), "goal_reached")
        self.assertEqual(event_outcome(failure), "out_of_bounds")

    def test_sanitizes_failure_reason_for_episode_directory(self):
        event = EpisodeEvent("episode_failed", "bad reason/with spaces")
        self.assertEqual(event_outcome(event), "bad_reason_with_spaces")


if __name__ == "__main__":
    unittest.main()
