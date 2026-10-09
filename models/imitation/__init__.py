"""Stage 1–2 behavior-cloning pipeline.

Re-export the small dataset API used by callers importing ``models.imitation``.
Training, preprocessing, and inference remain in their own modules.
"""

# Sibling module: the leading dot means ``models.imitation.dataset``.
from .dataset import ACTION_COLUMNS, BehaviorCloningDataset, split_episode_manifests

__all__ = ["ACTION_COLUMNS", "BehaviorCloningDataset", "split_episode_manifests"]
