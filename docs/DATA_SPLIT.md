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

To include the registered recovery episode, add `--train-extra-cache-dir data/datasets/recovery_cached_frames`. Evaluation recordings, if ever captured, belong in a separate evaluation location and must never be added as a training cache.

When new training corrections are recorded, keep them on training chambers only. To create a new split revision, generate a **new** manifest filename and split ID, review the diff and per-role counts, then point `--split-manifest` at it. Do not silently rewrite v1:

```bash
.venv/bin/python -m scripts.generate_frozen_split \
  --split-id frozen-YYYY-MM-DD-v2 \
  --output data/splits/frozen_v2.json
```

The generator currently includes the base cache and the recovery cache. If corrections use another cache group, extend the generator and register that group before training. Keep `evaluation1`, `evaluation2`, `dataset_test11`, and `dataset_test12` out of all training correction caches while their roles stay frozen.
