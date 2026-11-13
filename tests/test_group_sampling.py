"""Expected group exposure, legacy behavior and resume contracts."""
import unittest
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from torch.utils.data import WeightedRandomSampler

from models.imitation.dataset import BehaviorCloningDataset
from models.imitation.train_bc import (
    checkpoint_config, correction_cache_group, make_training_loader, parse_correction_group_fractions,
    training_sampling_weights, validate_resume_group_sampling,
)


class GroupSamplingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = BehaviorCloningDataset.__new__(BehaviorCloningDataset)
        self.dataset.episode_names = ["base_a", "base_b", "turn", "recover"]
        self.dataset.episode_maps = ["a", "b", "a", "b"]
        self.dataset.first_action_frames = [0, 0, 0, 0]
        self.dataset.index = [(episode, frame) for episode, count in enumerate((5, 12, 3, 7))
                              for frame in range(count)]
        self.groups = {"turn": "proactive_turn", "recover": "recovery"}
        self.fractions = {"proactive_turn": .05, "recovery": .05}

    def weights(self, **kwargs):
        options = dict(correction_episode_names=set(self.groups),
                       correction_episode_groups=self.groups,
                       correction_group_fractions=self.fractions)
        options.update(kwargs)
        return training_sampling_weights(self.dataset, "chamber-balanced", **options)

    def test_exact_exposure_and_within_group_weights(self):
        old = training_sampling_weights(self.dataset, "chamber-balanced")
        new = self.weights()
        self.assertAlmostEqual(new.sum(), len(self.dataset))
        for names, share in (({"base_a", "base_b"}, .9), ({"turn"}, .05), ({"recover"}, .05)):
            mask = np.array([self.dataset.episode_names[e] in names for e, _ in self.dataset.index])
            self.assertAlmostEqual(new[mask].sum() / new.sum(), share)
            np.testing.assert_allclose(new[mask] / new[mask].sum(), old[mask] / old[mask].sum())
        self.assertTrue(np.all(new > 0))

    def test_opening_boost_does_not_change_group_targets(self):
        weights = self.weights(start_window_frames=2, start_sampling_boost=4)
        for episode, fraction in ((2, .05), (3, .05)):
            mask = np.array([e == episode for e, _ in self.dataset.index])
            self.assertAlmostEqual(weights[mask].sum() / weights.sum(), fraction)
        self.assertAlmostEqual(weights[0] / weights[2], 4)

    def test_legacy_is_identical_with_inactive_group_metadata(self):
        for fraction in (0, .2):
            old = training_sampling_weights(self.dataset, "chamber-balanced",
                correction_episode_names=set(self.groups), correction_sampling_fraction=fraction)
            new = self.weights(correction_group_fractions={}, correction_sampling_fraction=fraction)
            np.testing.assert_array_equal(old, new)

    def test_uniform_loader_uses_groups_and_repeats_seed(self):
        args = SimpleNamespace(sampling="uniform", seed=0, batch_size=2, workers=0,
                               correction_group_fractions=self.fractions)
        def loader():
            return make_training_loader(self.dataset, args, False, 1, set(self.groups), self.groups)
        first, second = loader(), loader()
        self.assertIsInstance(first.sampler, WeightedRandomSampler)
        self.assertEqual(list(first.sampler), list(second.sampler))
        self.assertEqual(len(first.sampler), len(self.dataset))
        self.assertAlmostEqual(float(first.sampler.weights[-7:].sum() / first.sampler.weights.sum()), .05)

    def test_fraction_parser_rejects_invalid_or_duplicate_recipes(self):
        for values in (["base=.9"], ["recovery=nan"], ["recovery=inf"], ["recovery=0"],
                       ["recovery=-.1"], ["recovery=1"], ["=0.1"], ["broken"],
                       ["recovery=.1", "recovery=.2"], ["a=.5", "b=.5"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_correction_group_fractions(values)
        self.assertEqual(parse_correction_group_fractions(["proactive_turn=.05", "recovery=.05"]), self.fractions)

    def test_bad_membership_and_fraction_recipes_rejected(self):
        bad = [dict(correction_group_fractions={"recovery": .1}),
               dict(correction_group_fractions={"recovery": .1, "unknown": .1}),
               dict(correction_episode_groups={"turn": "proactive_turn"}),
               dict(correction_sampling_fraction=.1),
               dict(correction_group_fractions={"proactive_turn": float("nan"), "recovery": .1}),
               dict(correction_group_fractions={"proactive_turn": .5, "recovery": .5}),
               dict(correction_group_fractions={"proactive_turn": 0, "recovery": .1})]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.weights(**options)

    def test_empty_usable_group_and_missing_base_rejected(self):
        self.dataset.index = [(e, f) for e, f in self.dataset.index if e != 2]
        with self.assertRaisesRegex(ValueError, "no usable frames"):
            self.weights()
        self.dataset.index = [(2, 0), (3, 0)]
        with self.assertRaisesRegex(ValueError, "base frames"):
            self.weights()

    def test_frozen_group_identity_survives_moved_cache(self):
        split = SimpleNamespace(episodes={("proactive_turn", "turn"): {}})
        additions = [SimpleNamespace(name="turn")]
        self.assertEqual(correction_cache_group(additions, Path("/tmp/moved-cache"), split), "proactive_turn")
        split.episodes[("recovery", "recover")] = {}
        with self.assertRaises(ValueError):
            correction_cache_group(additions + [SimpleNamespace(name="recover")], Path("/tmp/mixed"), split)
        with self.assertRaises(ValueError):
            correction_cache_group([SimpleNamespace(name="unknown")], Path("/tmp/unknown"), split)

    def test_resume_rejects_recipe_or_membership_changes(self):
        current = {"correction_group_fractions": self.fractions, "correction_episode_groups": self.groups}
        validate_resume_group_sampling(current, current)
        validate_resume_group_sampling({}, {"correction_group_fractions": {}, "correction_episode_groups": {}})
        for previous in ({}, {**current, "correction_group_fractions": {"proactive_turn": .1, "recovery": .1}},
                         {**current, "correction_episode_groups": {"turn": "recovery", "recover": "proactive_turn"}}):
            with self.assertRaises(ValueError):
                validate_resume_group_sampling(previous, current)

    def test_checkpoint_config_serializes_recipe_and_membership(self):
        args = SimpleNamespace(
            batch_size=8, validation_batch_size=32, learning_rate=.0002,
            weight_decay=.0001, max_grad_norm=1, label_smoothing=.01,
            validation_fraction=.2, holdout_maps=["test11", "test12"],
            split_manifest=Path("frozen.json"), unfrozen_split=False,
            seed=0, sampling="chamber-balanced", binary_class_weighting="none",
            jump_positive_weight=3, mouse_head="gaussian", maps=[],
            train_extra_cache_dirs=[Path("proactive"), Path("recovery")],
            start_window_frames=8, start_sampling_boost=1,
            correction_sampling_fraction=0, correction_group_fractions=self.fractions,
            correction_episode_groups=self.groups, brightness_augmentation=.05,
            contrast_augmentation=.1, horizontal_flip_probability=0,
            lr_scheduler="none", lr_reduction_factor=.25,
            lr_reduction_patience=0, min_learning_rate=.000001,
        )
        config = checkpoint_config(args, np.ones((8, 2)), np.ones(2))
        saved = json.loads(json.dumps(config))["training"]
        self.assertEqual(saved["correction_group_fractions"], self.fractions)
        self.assertEqual(saved["correction_episode_groups"], self.groups)
        args.correction_group_fractions = {}
        legacy = checkpoint_config(args, np.ones((8, 2)), np.ones(2))["training"]
        self.assertEqual(legacy["correction_episode_groups"], {})
        validate_resume_group_sampling({}, legacy)


if __name__ == "__main__":
    unittest.main()
