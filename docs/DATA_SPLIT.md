# Frozen data split (v1)

The episode-level source of truth is [`data/splits/frozen_v1.json`](../data/splits/frozen_v1.json), frozen on 2026-10-06. It lists every currently cached episode's ID, chamber, role, recording date, and cache group. Large frames and recordings remain outside Git; the manifest is committed.

| Role | Chambers | Current recordings | Use |
| --- | --- | ---: | --- |
| Training | `dataset_test1`–`dataset_test10` | 1,100 base + 1 recovery | Fit weights; recovery cache is optional |
| Validation | `dataset_test11`, `dataset_test12` | 200 base | Select checkpoints and compare candidates |
| Held-out live evaluation | `evaluation1`, `evaluation2` | 0 cached | Test generalization; never add to training |

This is a whole-chamber split, not a random frame split. The two evaluation chambers have been used in prior live tests, but their recordings were not part of training. Report that history when presenting results; they remain useful held-out evaluations. A completely new chamber is optional for a stricter never-consulted benchmark.

Training reads this manifest by default. It stops if the base cache gains or loses an episode, if a map label changes, or if a requested extra cache contains unregistered or non-training data. Explicit `--holdout-map` values must match the frozen validation chambers. The checkpoint stores the split ID and SHA-256 of the manifest. `--unfrozen-split` is an explicit escape hatch for experiments, not for reported frozen-split results.

To train using the frozen split, omit the old holdout arguments (they are applied automatically):

```bash
.venv/bin/python -m models.imitation.train_bc \
  --checkpoint-dir models/imitation/checkpoints/runs/NEW_RUN
```

The v1 manifest registers one recovery episode. As of 2026-10-18, the separate
recovery cache contains that episode plus 30 additional corrections from the
2026-10-18 session. These 30 are not registered in v1; passing the expanded
cache with `--train-extra-cache-dir` and the default frozen manifest will fail
validation. Before training with the new corrections, generate and review a
new split revision. The 2026-10-13 pilot remains outside the cache and should
be considered separately. Do not edit the frozen v1 manifest in place.
Evaluation recordings, if ever captured, belong in a separate evaluation
location and must never be added as a training cache.

## Recovery-inclusive revision (v2)

After the 30 corrections from 2026-10-18 were reviewed and cached, a new
manifest was generated at [`data/splits/frozen_v2.json`](../data/splits/frozen_v2.json)
with split ID `frozen-2026-10-18-v2`. It contains 1,131 training episodes
(1,100 unchanged base episodes plus 31 recovery episodes), the same 200
validation episodes, and no cached evaluation episodes. The original
`frozen_v1.json` remains unchanged. The 2026-10-13 pilot is not in the recovery
cache or v2. Use `--split-manifest data/splits/frozen_v2.json` when a planned
candidate is ready to train; the manifest alone does not mean training has
started.

When new training corrections are recorded, keep them on training chambers only. To create a new split revision, generate a **new** manifest filename and split ID, review the diff and per-role counts, then point `--split-manifest` at it. Do not silently rewrite v1:

```bash
.venv/bin/python -m scripts.generate_frozen_split \
  --split-id frozen-YYYY-MM-DD-v2 \
  --output data/splits/frozen_v2.json
```

The generator includes the base, recovery, and proactive-turn caches. Keep
`evaluation1`, `evaluation2`, `dataset_test11`, and `dataset_test12` out of all
training correction caches while their roles stay frozen.

The current manifest has now been generated as `frozen_v3.json`; the paragraph
below describes the verified result, not a future procedure.

## Proactive-turn revision (v3)

The approved 2026-10-18 pilot contains 20 goal-reached clips: 10 left and 10
right turns across `dataset_test1`–`dataset_test5`. They are pre-wall
demonstrations and remain separate from the recovery recordings. Cache them
separately, then generate a new revision so their role is explicit:

```bash
.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes/corrections/proactive_turns/goal_reached \
  --cache-dir data/datasets/proactive_turn_cached_frames
.venv/bin/python -m scripts.generate_frozen_split \
  --split-id frozen-2026-10-18-v3 \
  --output data/splits/frozen_v3.json
```

The generator registers them as `proactive_turn`, with the same training-only
role as the recovery cache. Expected totals are 1,151 train (1,100 base, 31
recovery, 20 proactive-turn), 200 validation, and 0 evaluation episodes. The
v3 manifest is the candidate for subsequent training; v1 and v2 remain
unchanged. Training can pass both correction caches with repeated
`--train-extra-cache-dir` options.
