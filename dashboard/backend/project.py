"""SHIONN status and safe subprocess controls for the local Dash interface."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent
SPLIT_PATH = ROOT / "data/splits/frozen_v3.json"
TRAINING_MAPS = [f"dataset_test{index}" for index in range(1, 11)]
LIVE_BENCHMARK_MAPS = ["dataset_test11", "dataset_test12", "evaluation1", "evaluation2"]
DEVICE_CHOICES = {"auto", "cpu", "cuda"}

_lock = threading.RLock()
_job: dict | None = None
_process: subprocess.Popen | None = None


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def checkpoints() -> list[dict[str, str]]:
    """Find usable SHIONN best checkpoints and their configuration contracts."""
    candidates = set()
    patterns = (
        "models/imitation/checkpoints/runs/*/best.pt",
        "models/imitation/checkpoints_v*/runs/*/best.pt",
        "models/imitation/checkpoints_v*/best.pt",
        "models/imitation/checkpoints_*/runs/*/best.pt",
        "models/imitation/checkpoints_*/best.pt",
    )
    for pattern in patterns:
        candidates.update(ROOT.glob(pattern))
    options = []
    for path in sorted(candidates):
        contract = path.with_suffix(".json")
        if not path.is_file() or not contract.is_file():
            continue
        relative = _relative(path)
        options.append({"label": relative, "value": relative})
    return options


def training_run_options() -> list[dict[str, str]]:
    root = ROOT / "models/imitation/checkpoints/runs"
    if not root.is_dir():
        return []
    options = []
    for run in sorted(root.iterdir(), reverse=True):
        metrics = run / "metrics.jsonl"
        if run.is_dir() and metrics.is_file():
            options.append({"label": run.name, "value": run.name})
    return options


def training_run_data(run_name: str) -> tuple[dict, list[dict]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_name):
        raise ValueError("Invalid run name")
    run = (ROOT / "models/imitation/checkpoints/runs" / run_name).resolve()
    parent = (ROOT / "models/imitation/checkpoints/runs").resolve()
    if not run.is_relative_to(parent):
        raise ValueError("Run path is outside the SHIONN checkpoint directory")
    metadata_path = run / "best.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metrics_path = run / "metrics.jsonl"
    metrics = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metadata, metrics


def benchmark_history(limit: int = 10) -> list[dict]:
    root = ROOT / "model_attempts/sequences"
    if not root.is_dir():
        return []
    results = []
    for summary_path in sorted(root.glob("sequence_*/benchmark_summary.json"), reverse=True):
        try:
            results.append({"path": _relative(summary_path.parent), "summary": json.loads(summary_path.read_text(encoding="utf-8"))})
        except (OSError, json.JSONDecodeError):
            continue
        if len(results) >= limit:
            break
    return results


def project_status() -> dict:
    """Summarize frozen-v3 counts, caches, checkpoints, and active job state."""
    result = {
        "split_id": "missing",
        "train": 0,
        "validation": 0,
        "evaluation": 0,
        "groups": {},
        "split_ready": False,
        "checkpoints": checkpoints(),
    }
    if SPLIT_PATH.is_file():
        try:
            manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
            result["split_id"] = manifest.get("split_id", "unknown")
            episodes = manifest.get("episodes", [])
            result.update(Counter(item.get("role", "unknown") for item in episodes))
            result["groups"] = dict(Counter(item.get("cache_group", "unknown") for item in episodes))
            result["split_ready"] = True
            for group, cache in (
                ("base", ROOT / "data/datasets/cached_frames"),
                ("recovery", ROOT / "data/datasets/recovery_cached_frames"),
                ("proactive_turn", ROOT / "data/datasets/proactive_turn_cached_frames"),
            ):
                result.setdefault("cache_dirs", {})[group] = cache.is_dir()
        except (OSError, json.JSONDecodeError):
            result["split_error"] = "Could not read frozen_v3.json"
    with _lock:
        result["job"] = _snapshot_locked()
    return result


def _snapshot_locked() -> dict | None:
    if _job is None:
        return None
    snapshot = dict(_job)
    snapshot.pop("process", None)
    if _process is not None:
        code = _process.poll()
        if code is not None and snapshot["state"] in {"Starting", "Running", "Stopping"}:
            snapshot["state"] = "Stopped" if snapshot["state"] == "Stopping" else "Completed" if code == 0 else "Failed"
            snapshot["returncode"] = code
            snapshot["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _job.update(state=snapshot["state"], returncode=code, finished_at=snapshot["finished_at"])
    try:
        log_path = Path(snapshot["log_path"])
        with log_path.open("rb") as log_file:
            size = log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, size - 12_000))
            text = log_file.read().decode("utf-8", errors="replace")
        snapshot["output"] = text[-12_000:]
        snapshot["last_line"] = next((line for line in reversed(text.splitlines()) if line.strip()), "")
        epoch_lines = re.findall(r"epoch\s+(\d{1,3})\s+train=", text, flags=re.IGNORECASE)
        snapshot["epoch"] = int(epoch_lines[-1]) if epoch_lines else 0
    except OSError:
        snapshot["output"] = ""
        snapshot["last_line"] = ""
    return snapshot


def active_job() -> dict | None:
    """Return a current task, updating its status from the child process."""
    with _lock:
        snapshot = _snapshot_locked()
        if snapshot and snapshot["state"] in {"Completed", "Failed", "Stopped"}:
            if _process is None or _process.poll() is not None:
                return snapshot
        return snapshot


def _require_idle() -> None:
    global _job, _process
    current = _snapshot_locked()
    if current and current["state"] in {"Starting", "Running", "Stopping"}:
        raise RuntimeError(f"Another task is already active: {current['label']} ({current['state']}).")
    _job = None
    _process = None


def _validated_checkpoint(value: str) -> str:
    allowed = {item["value"] for item in checkpoints()}
    if value not in allowed:
        raise ValueError("Choose an existing SHIONN checkpoint from the list.")
    path = ROOT / value
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError("Choose an existing SHIONN checkpoint from the list.")
    if not path.with_suffix(".json").is_file():
        raise ValueError("Checkpoint JSON contract is missing.")
    return _relative(path)


def build_command(task: str, config: dict) -> tuple[list[str], str]:
    """Build an allow-listed CLI command; no user-entered shell is evaluated."""
    python = sys.executable
    if task == "record":
        mode = config.get("mode", "demonstration")
        maps = config.get("maps") or []
        if not maps or any(item not in TRAINING_MAPS for item in maps):
            raise ValueError("Recording is restricted to training maps dataset_test1–dataset_test10.")
        if mode in {"demonstration", "proactive_turn"}:
            command = [python, str(ROOT / "record_dataset.py")]
            for map_name in maps:
                command.extend(("--map", map_name))
            episodes = int(config.get("episodes", 1))
            duration = float(config.get("duration", 60))
            if episodes < 1 or not 0 < duration <= 600:
                raise ValueError("Use 1+ successful episodes and a duration from 1 to 600 seconds.")
            command.extend(("--episodes", str(episodes), "--duration", str(duration)))
            if mode == "proactive_turn":
                direction = config.get("direction")
                if direction not in {"left", "right"}:
                    raise ValueError("Choose a left or right turn direction.")
                command.extend(("--proactive-turn", "--turn-direction", direction))
            return command, "Dataset recording"
        if mode == "recovery":
            checkpoint = _validated_checkpoint(config.get("checkpoint", ""))
            target = int(config.get("target_corrections", 1))
            if target < 1 or target > 1000:
                raise ValueError("Correction target must be between 1 and 1,000.")
            command = [python, str(ROOT / "run_recovery_session.py"), "--checkpoint", checkpoint,
                       "--target-corrections", str(target), "--manual-focus"]
            for map_name in maps:
                command.extend(("--map", map_name))
            if config.get("test_mode"):
                command.append("--test-mode")
            return command, "Supervised recovery recording"
        raise ValueError("Unknown recording mode.")

    if task == "benchmark":
        checkpoint = _validated_checkpoint(config.get("checkpoint", ""))
        mode = config.get("mode", "live")
        maps = config.get("maps") or []
        device = config.get("device", "auto")
        if device not in DEVICE_CHOICES:
            raise ValueError("Choose auto, CPU, or CUDA.")
        if mode == "expert":
            allowed = {"dataset_test11", "dataset_test12"}
            if not maps or any(item not in allowed for item in maps):
                raise ValueError("Expert benchmark maps must be validation chambers dataset_test11/12.")
            command = [python, str(ROOT / "benchmark_expert.py"), "--checkpoint", checkpoint,
                       "--subset", "validation", "--device", device]
            for map_name in maps:
                command.extend(("--map", map_name))
            return command, "Offline validation benchmark"
        if mode == "live":
            if not maps or any(item not in LIVE_BENCHMARK_MAPS for item in maps):
                raise ValueError("Live benchmark is limited to validation/evaluation chambers.")
            repeats = int(config.get("repeats", 1))
            seconds = float(config.get("max_seconds", 60))
            if not 1 <= repeats <= 20 or not 1 <= seconds <= 600:
                raise ValueError("Repeats must be 1–20 and attempt time 1–600 seconds.")
            command = [python, str(ROOT / "run_model_sequence.py"), "--checkpoint", checkpoint,
                       "--repeats", str(repeats), "--max-seconds", str(seconds),
                       "--device", device, "--keep-focused", "--no-launch"]
            for map_name in maps:
                command.extend(("--map", map_name))
            if config.get("dry_run"):
                command.append("--dry-run")
            return command, "Live benchmark (uses Portal 2 controls)"
        raise ValueError("Choose an offline expert or live benchmark.")

    if task == "train":
        epochs = int(config.get("epochs", 20))
        device = config.get("device", "auto")
        if not 1 <= epochs <= 200 or device not in DEVICE_CHOICES:
            raise ValueError("Epochs must be 1–200 and device must be auto, CPU, or CUDA.")
        if not SPLIT_PATH.is_file():
            raise ValueError("Frozen v3 split is missing; refusing to train without the approved split.")
        command = [python, "-m", "models.imitation.train_bc", "--split-manifest",
                   "data/splits/frozen_v3.json", "--epochs", str(epochs), "--device", device]
        extras = config.get("extra_caches", ["recovery", "proactive_turn"])
        cache_paths = {
            "recovery": "data/datasets/recovery_cached_frames",
            "proactive_turn": "data/datasets/proactive_turn_cached_frames",
        }
        if set(extras) != set(cache_paths):
            raise ValueError("The frozen-v3 candidate must include both recovery and proactive-turn caches.")
        if not (ROOT / "data/datasets/cached_frames").is_dir():
            raise ValueError("The frozen-v3 base cache is missing.")
        for extra in extras:
            path = ROOT / cache_paths[extra]
            if not path.is_dir():
                raise ValueError(f"Selected correction cache is missing: {cache_paths[extra]}")
            command.extend(("--train-extra-cache-dir", cache_paths[extra]))
        correction_fraction = float(config.get("correction_fraction", 0.10))
        if extras and not 0 < correction_fraction <= 0.5:
            raise ValueError("Correction sampling share must be between 0 and 0.5.")
        if extras:
            command.extend(("--correction-sampling-fraction", str(correction_fraction)))
        run_name = config.get("run_name", "v5_recovery_proactive_v3").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_name):
            raise ValueError("Run name may contain only letters, numbers, underscores, and hyphens.")
        output = ROOT / "models/imitation/checkpoints/runs" / run_name
        if output.exists():
            raise ValueError(f"Checkpoint output already exists: {_relative(output)}. Choose a new run name.")
        command.extend(("--checkpoint-dir", _relative(output)))
        return command, "Frozen-v3 behavior-cloning training"

    raise ValueError("Unknown task type.")


def start_task(task: str, config: dict) -> dict:
    """Launch one explicitly requested allow-listed project task."""
    global _job, _process
    command, label = build_command(task, config)
    with _lock:
        _require_idle()
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        task_dir = ROOT / "model_attempts/interface_jobs" / job_id
        task_dir.mkdir(parents=True, exist_ok=False)
        log_path = task_dir / "output.log"
        (task_dir / "command.json").write_text(
            json.dumps({"label": label, "command": command, "started_at": datetime.now().isoformat()}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        try:
            _process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=(os.name == "posix"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception:
            log_handle.close()
            raise
        log_handle.close()
        _job = {
            "id": job_id,
            "label": label,
            "task": task,
            "state": "Running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": command,
            "log_path": str(log_path),
            "task_dir": _relative(task_dir),
            "returncode": None,
        }
        return _snapshot_locked() or {}


def stop_task() -> dict | None:
    """Ask a task to stop gracefully (SIGINT); never force-kill silently."""
    global _job
    with _lock:
        current = _snapshot_locked()
        if _process is None or _process.poll() is not None:
            return current
        if _job is not None:
            _job["state"] = "Stopping"
        try:
            if os.name == "posix":
                os.killpg(_process.pid, signal.SIGINT)
            else:
                _process.send_signal(signal.CTRL_BREAK_EVENT)
        except ProcessLookupError:
            _snapshot_locked()
        return _snapshot_locked()
