"""Waypoint events remain separate from chamber success and policy inputs."""

import unittest
from unittest.mock import Mock

from episode_events import EpisodeEventStream
from record_dataset import wait_for_terminal_event
from record_recovery import parse_args


class WaypointRecordingTests(unittest.TestCase):
    def outcome(self, text, requested):
        controller = Mock()
        controller.read_console.return_value = text
        events = []
        result = wait_for_terminal_event(
            controller,
            EpisodeEventStream(),
            1,
            requested_waypoint=requested,
            waypoint_events=events,
        )
        return result, events

    def test_correct_and_wrong_waypoints_are_distinct(self):
        for requested in ("A", "B"):
            for reached in ("A", "B"):
                with self.subTest(requested=requested, reached=reached):
                    outcome, events = self.outcome(
                        f"EVT|waypoint_reached|{reached}\n", requested
                    )
                    self.assertEqual(
                        outcome,
                        "waypoint_reached"
                        if requested == reached
                        else "wrong_waypoint",
                    )
                    self.assertEqual(events[0]["value"], reached)
                    self.assertGreater(events[0]["received_monotonic"], 0)

    def test_exit_is_not_waypoint_success(self):
        self.assertEqual(
            self.outcome("EVT|goal_reached|1\n", "A")[0], "wrong_destination"
        )

    def test_timeout_and_invalid_payload(self):
        self.assertEqual(
            self.outcome("EVT|episode_failed|timeout\n", "B")[0], "timeout"
        )
        self.assertEqual(
            self.outcome("EVT|waypoint_reached|C\n", "B")[0], "invalid_waypoint_event"
        )

    def test_normal_recording_ignores_waypoint_event(self):
        self.assertEqual(
            self.outcome("EVT|waypoint_reached|A\nEVT|goal_reached|1\n", None)[0],
            "goal_reached",
        )

    def test_chunked_events_preserve_order(self):
        stream = EpisodeEventStream()
        stream.feed("EVT|waypoint_")
        self.assertIsNone(stream.pop())
        stream.feed("reached|B\nEVT|goal_reached|1\n")
        self.assertEqual(stream.pop().value, "B")
        self.assertEqual(stream.pop().name, "goal_reached")

    def test_cli_is_opt_in(self):
        self.assertIsNone(parse_args(["--map", "goal_choice_v1"]).requested_waypoint)
        self.assertEqual(
            parse_args(
                ["--map", "goal_choice_v1", "--requested-waypoint", "A"]
            ).requested_waypoint,
            "A",
        )

    def test_map_loading_is_opt_in(self):
        self.assertFalse(parse_args(["--map", "goal_choice_v1"]).load_map)
        self.assertTrue(parse_args(["--map", "goal_choice_v1", "--load-map"]).load_map)
