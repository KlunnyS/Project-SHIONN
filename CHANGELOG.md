# Changelog

All notable changes to Project SHIONN are documented in this file.

## Unreleased

### Added

- Add the 13.2M-parameter residual v5 imitation policy with GroupNorm, SiLU,
  squeeze/excitation, dropout, and a spatial 3x5 visual bottleneck.
- Add whole-chamber validation through repeatable `--holdout-map` arguments.
- Add temporally consistent brightness and contrast augmentation, optional
  label-corrected horizontal flips, label smoothing, gradient clipping, and
  configurable AdamW weight decay.
- Add tests for whole-chamber split isolation and mirrored action labels.

### Changed

- Make the binned mouse head, unweighted binary losses, and a 3x
  jump-positive weight the behavior-cloning defaults.
- Use CUDA channels-last tensors, fused AdamW, persistent data workers, and
  cuDNN benchmarking where supported.
- Strengthen checkpoint resume validation for model, optimizer, dataset,
  holdout, sampling, and augmentation settings.
- Reduce default early-stopping patience from five unimproved epochs to three.
- Update the documented dataset status to 850 demonstrations across 12
  chambers and describe the v5 training and evaluation workflow.

### Compatibility

- Keep exact v1/v2, v3, and v4-binned network implementations available so
  previously trained checkpoints remain loadable for inference.
