# Repository maintenance guidance

- Keep `CHANGELOG.md` and the relevant files under `docs/` synchronized with
  changes to behavior, defaults, commands, data contracts, or model versions.
- Treat `docs/TRAINING_RUNBOOK.md` as the operational source of truth for the
  current training run. Update its dated status whenever a run is started,
  stopped, resumed, completed, or superseded.
- Do not commit recordings, cached datasets, checkpoints, model-attempt media,
  or other generated runtime artifacts. They are intentionally ignored.
- Run `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` before
  preparing a commit that changes Python behavior.
