"""Run a matrix of imitation checkpoints and Portal 2 chambers in order."""

from __future__ import annotations

# Python standard library: build and run the checkpoint-by-chamber job plan.
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

# Project benchmark module at the repository root.
from benchmark_sequence import main as write_benchmark_summary
from focus_guard import FOCUS_GUARD_EXIT_CODE, FocusGuardViolation


ROOT = Path(__file__).resolve().parent


@contextmanager
def persistent_virtual_mouse(enabled):
    """Keep the input device registered while attempts reset their other state."""
    if not enabled:
        yield None
        return
    from wrapper import Portal2Controller

    owner = Portal2Controller(log_file=None)
    try:
        if not owner.init_virtual_mouse():
            raise RuntimeError("Could not create the sequence's persistent virtual mouse")
        # Allow device registration before map loading/focus acquisition.
        time.sleep(0.5)
        yield owner.ui
    finally:
        owner.disconnect()


def run_persistent_attempt(command, mouse):
    """Call the same runner in-process, retaining only the owned mouse device."""
    from run_imitation import main as run_attempt

    original_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        run_attempt(command[2:], virtual_mouse=mouse)
    except FocusGuardViolation:
        return FOCUS_GUARD_EXIT_CODE
    except Exception as error:
        print(f"Runner error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        os.chdir(original_directory)
    return 0


@dataclass(frozen=True)
class Model:
    label: str
    checkpoint: Path


@dataclass(frozen=True)
class Job:
    number: int
    model: Model
    map_name: str
    repeat: int


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "item"


def parse_model(value: str) -> Model:
    if "=" in value:
        label, checkpoint = value.split("=", 1)
        if not label or not checkpoint:
            raise ValueError("--checkpoint must be PATH or LABEL=PATH")
    else:
        checkpoint = value
        path = Path(checkpoint)
        label = f"{path.parent.name}_{path.stem}"
    return Model(label, Path(checkpoint))


def build_jobs(models: list[Model], maps: list[str], repeats: int, order: str) -> list[Job]:
    jobs = []
    for repeat in range(1, repeats + 1):
        pairs = (
            ((model, map_name) for map_name in maps for model in models)
            if order == "map-first"
            else ((model, map_name) for model in models for map_name in maps)
        )
        for model, map_name in pairs:
            jobs.append(Job(len(jobs) + 1, model, map_name, repeat))
    return jobs


def job_directory(sequence_dir: Path, job: Job) -> Path:
    return sequence_dir / (
        f"{job.number:03d}_{safe_name(job.model.label)}_"
        f"{safe_name(job.map_name)}_r{job.repeat}"
    )


def runner_command(job: Job, run_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(ROOT / "run_imitation.py"),
        "--checkpoint", str(job.model.checkpoint),
        "--map", job.map_name,
        "--recording-dir", str(run_dir),
        "--log-actions",
        "--device", args.device,
        "--port", str(args.port),
        "--fps", str(args.fps),
        "--width", str(args.width),
        "--height", str(args.height),
        "--countdown", str(args.countdown),
        "--max-seconds", str(args.max_seconds),
        "--status-every", str(args.status_every),
        "--telemetry-hz", str(args.telemetry_hz),
        "--hyprland-instance", args.hyprland_instance,
    ]
    if not args.no_video:
        command.append("--record-video")
    if args.output:
        command.extend(("--output", args.output))
    if args.jump_threshold is not None:
        command.extend(("--jump-threshold", str(args.jump_threshold)))
    if args.move_w_threshold is not None:
        command.extend(("--move-w-threshold", str(args.move_w_threshold)))
    if args.mouse_max_abs is not None:
        command.extend(("--mouse-max-abs", str(args.mouse_max_abs)))
    if args.no_launch:
        command.append("--no-launch")
    if args.keep_focused:
        command.append("--keep-focused")
    if args.hyprland_focus_guard:
        command.append("--hyprland-focus-guard")
    if args.hyprland_confine_pointer:
        command.append("--hyprland-confine-pointer")
    if args.mouse_preflight and not args.dry_run:
        command.append("--mouse-preflight")
    if args.dry_run:
        command.append("--dry-run")
    if args.verbose or args.dry_run:
        command.append("--verbose")
    return command


def inspect_attempt(run_dir: Path) -> tuple[str, Path | None, str | None]:
    """Read the completed runner log; missing logs mean startup was cancelled."""
    logs = sorted(run_dir.glob("attempt_*.jsonl"))
    if len(logs) != 1:
        return "cancelled" if not logs else "incomplete", None, None
    stop_reason = "incomplete"
    video = None
    try:
        with logs[0].open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    video = record.get("video")
                elif record.get("type") == "summary":
                    stop_reason = record.get("stop_reason") or "time_limit"
    except (OSError, json.JSONDecodeError):
        return "incomplete", logs[0], video
    return stop_reason, logs[0], video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each checkpoint on each chamber, with separate attempt logs."
    )
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH|LABEL=PATH",
                        help="Repeat for each model; an optional label names its result folders")
    parser.add_argument("--map", dest="maps", action="append", required=True, metavar="NAME",
                        help="Repeat for each chamber")
    parser.add_argument("--repeats", type=int, default=1, help="Attempts per checkpoint/chamber pair")
    parser.add_argument("--order", choices=("map-first", "model-first"), default="map-first")
    parser.add_argument("--recording-root", type=Path, default=Path("model_attempts/sequences"))
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Delay between attempts")
    parser.add_argument("--output", help="Wayland output name, such as DP-1")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--jump-threshold", type=float,
                        help="Override the binary jump decision threshold for every live attempt")
    parser.add_argument("--move-w-threshold", type=float,
                        help="Override the forward decision threshold for every live attempt")
    parser.add_argument("--mouse-max-abs", type=int,
                        help="Clamp applied mouse dx/dy for every live attempt")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--status-every", type=float, default=1.0)
    parser.add_argument("--telemetry-hz", type=float, default=15.0)
    parser.add_argument("--mouse-preflight", action=argparse.BooleanOptionalAction, default=True,
                        help="Require a bidirectional capture check and map reload for each live attempt")
    parser.add_argument("--hyprland-instance", default="auto")
    parser.add_argument("--hyprland-focus-guard", action="store_true",
                        help="Disable hover focus temporarily and stop the sequence if guarded focus changes")
    parser.add_argument("--hyprland-confine-pointer", action="store_true",
                        help="Temporarily confine the game pointer; implies focus guard and requires live preflight")
    parser.add_argument("--no-video", action="store_true", help="Save JSONL diagnostics only")
    parser.add_argument("--no-launch", action="store_true", help="Require Portal 2 to be running")
    parser.add_argument("--keep-focused", action=argparse.BooleanOptionalAction, default=True,
                        help="Restore Portal 2 focus through the active compositor")
    parser.add_argument("--dry-run", action="store_true", help="Predict without applying controls")
    parser.add_argument("--verbose", action="store_true", help="Print periodic runner status")
    parser.add_argument("--continue-on-error", action="store_true", help="Try later jobs after a runner error")
    parser.add_argument("--plan-only", action="store_true", help="Print jobs without opening Portal 2")
    parser.add_argument("--persistent-mouse", action="store_true",
                        help="Run attempts in one Python process with one virtual mouse for the whole sequence")
    args = parser.parse_args(argv)
    if args.hyprland_confine_pointer:
        args.hyprland_focus_guard = True
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    for name in ("max_seconds", "countdown", "pause_seconds"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be zero or greater")
    for name in ("fps", "width", "height", "status_every", "port"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.jump_threshold is not None and not 0 < args.jump_threshold < 1:
        parser.error("--jump-threshold must be between 0 and 1")
    if args.move_w_threshold is not None and not 0 < args.move_w_threshold < 1:
        parser.error("--move-w-threshold must be between 0 and 1")
    if args.mouse_max_abs is not None and args.mouse_max_abs <= 0:
        parser.error("--mouse-max-abs must be greater than zero")
    if not 1 <= args.telemetry_hz <= 20:
        parser.error("--telemetry-hz must be between 1 and 20")
    if args.hyprland_focus_guard and not args.keep_focused:
        parser.error("--hyprland-focus-guard requires --keep-focused")
    if args.hyprland_confine_pointer and (not args.mouse_preflight or args.dry_run):
        parser.error("--hyprland-confine-pointer requires mouse preflight without --dry-run")
    if args.mouse_preflight and not args.dry_run and not args.keep_focused:
        parser.error("mouse preflight requires --keep-focused; use --no-mouse-preflight for diagnostic runs")
    if not all(args.maps) or len(set(args.maps)) != len(args.maps):
        parser.error("--map values must be nonempty and unique")
    try:
        args.models = [parse_model(value) for value in args.checkpoint]
    except ValueError as error:
        parser.error(str(error))
    if len({model.label for model in args.models}) != len(args.models):
        parser.error("checkpoint labels must be unique; use LABEL=PATH")
    for model in args.models:
        path = model.checkpoint if model.checkpoint.is_absolute() else ROOT / model.checkpoint
        if not path.is_file():
            parser.error(f"checkpoint not found: {model.checkpoint}")
        if not path.with_suffix(".json").is_file():
            parser.error(f"checkpoint configuration not found: {path.with_suffix('.json')}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args.models, args.maps, args.repeats, args.order)
    print(f"Planned attempts: {len(jobs)}")
    for job in jobs:
        print(f"  {job.number:03d}  {job.model.label}  {job.map_name}  repeat {job.repeat}")
    if args.plan_only:
        return 0

    recording_root = args.recording_root if args.recording_root.is_absolute() else ROOT / args.recording_root
    sequence_dir = recording_root / datetime.now().strftime("sequence_%Y%m%d_%H%M%S_%f")
    sequence_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = sequence_dir / "sequence.jsonl"
    print(f"Sequence results: {sequence_dir}")
    had_error = False
    with manifest_path.open("w", encoding="utf-8", buffering=1) as manifest, persistent_virtual_mouse(args.persistent_mouse) as mouse:
        manifest.write(json.dumps({
            "type": "sequence", "time_utc": datetime.now(timezone.utc).isoformat(),
            "jobs": len(jobs), "order": args.order,
            "execution_mode": "persistent_mouse_in_process" if args.persistent_mouse else "subprocess_per_attempt",
        }) + "\n")
        for index, job in enumerate(jobs):
            run_dir = job_directory(sequence_dir, job)
            command = runner_command(job, run_dir, args)
            print(f"\n[{job.number}/{len(jobs)}] {job.model.label} on {job.map_name} (repeat {job.repeat})", flush=True)
            print(shlex.join(command), flush=True)
            started = time.monotonic()
            try:
                if args.persistent_mouse:
                    print("Running in-process with the sequence's persistent virtual mouse.", flush=True)
                    returncode = run_persistent_attempt(command, mouse)
                else:
                    returncode = subprocess.run(command, cwd=ROOT, check=False).returncode
            except KeyboardInterrupt:
                print("\nSequence interrupted; remaining jobs were skipped.")
                return 130
            stop_reason, log_path, video_path = inspect_attempt(run_dir)
            if returncode != 0:
                if returncode == FOCUS_GUARD_EXIT_CODE and stop_reason not in ("focus_guard_violation", "focus_guard_restore_failed"):
                    stop_reason = "focus_guard_violation"
                elif stop_reason not in ("mouse_capture_failed", "focus_guard_violation", "focus_guard_restore_failed"):
                    stop_reason = "runner_error"
                had_error = True
            record = {
                "type": "result", "job": job.number, "model": job.model.label,
                "checkpoint": str(job.model.checkpoint), "map": job.map_name,
                "repeat": job.repeat, "stop_reason": stop_reason,
                "returncode": returncode,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "log": str(log_path) if log_path else None, "video": video_path,
            }
            manifest.write(json.dumps(record) + "\n")
            print(f"Result: {stop_reason}; log: {log_path or 'none'}")
            if stop_reason in ("mouse_capture_failed", "focus_guard_violation", "focus_guard_restore_failed"):
                print(f"Input/focus check failed ({stop_reason}); remaining jobs were skipped.")
                return 1
            if stop_reason in ("escape_key", "keyboard_interrupt", "cancelled", "incomplete"):
                print("Sequence stopped; remaining jobs were skipped.")
                return 130
            if had_error and not args.continue_on_error:
                print("Runner error; remaining jobs were skipped.")
                return 1
            if index + 1 < len(jobs) and args.pause_seconds:
                try:
                    time.sleep(args.pause_seconds)
                except KeyboardInterrupt:
                    print("\nSequence interrupted; remaining jobs were skipped.")
                    return 130
    write_benchmark_summary([str(manifest_path)])
    print(f"\nSequence complete. Manifest: {manifest_path}")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
