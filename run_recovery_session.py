"""Cycle model attempts and record human corrections after a P-key handoff."""

from __future__ import annotations

# Python standard library: session arguments, subprocess handoff, and audit log.
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project runner: read the completed model attempt's structured stop reason.
from run_model_sequence import inspect_attempt


ROOT = Path(__file__).resolve().parent
TRAINING_MAPS = tuple(f"dataset_test{number}" for number in range(1, 11))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--map", dest="maps", action="append", choices=TRAINING_MAPS,
                        help="Training chamber to cycle; defaults to all ten")
    parser.add_argument("--target-corrections", type=int, default=30)
    parser.add_argument("--max-model-attempts", type=int, default=100)
    parser.add_argument("--model-seconds", type=float, default=60.0)
    parser.add_argument("--recovery-seconds", type=float, default=0.0,
                        help="Human correction limit; 0 means no code timer (default)")
    parser.add_argument("--focus-delay", type=float, default=5.0)
    parser.add_argument("--manual-focus", action="store_true",
                        help="After takeover, check focus and press P to start recovery manually")
    parser.add_argument("--test-mode", action="store_true",
                        help="Exercise handoff and controls without saving model videos or recovery clips")
    parser.add_argument("--no-model-video", action="store_true",
                        help="Save real corrections without encoding the preceding model attempt MP4")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", help="Wayland output name for both runners")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--episodes-root", type=Path, default=Path("episodes/recovery"))
    parser.add_argument("--session-root", type=Path)
    args = parser.parse_args(argv)
    if args.session_root is None:
        args.session_root = Path(
            "model_attempts/recovery_tests" if args.test_mode
            else "model_attempts/recovery_sessions"
        )
    args.maps = args.maps or list(TRAINING_MAPS)
    if len(set(args.maps)) != len(args.maps):
        parser.error("--map values must be unique")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.target_corrections < 1 or args.max_model_attempts < 1:
        parser.error("targets and maximum attempts must be positive")
    if args.model_seconds <= 0 or args.recovery_seconds < 0 or args.fps <= 0:
        parser.error("model seconds and FPS must be positive; recovery seconds cannot be negative")
    if args.focus_delay < 2 or args.countdown < 0:
        parser.error("focus delay must be at least 2 seconds for mouse verification; countdown cannot be negative")
    if args.port <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("port and frame dimensions must be positive")
    return args


def model_command(args: argparse.Namespace, map_name: str, attempt_dir: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "run_imitation.py"),
        "--checkpoint", str(args.checkpoint), "--map", map_name,
        "--device", args.device, "--port", str(args.port),
        "--fps", str(args.fps), "--width", str(args.width), "--height", str(args.height),
        "--countdown", str(args.countdown), "--max-seconds", str(args.model_seconds),
        "--recording-dir", str(attempt_dir), "--log-actions",
        "--keep-focused", "--recovery-handoff", "--disable-chamber-timeout",
    ]
    if not args.test_mode and not args.no_model_video:
        command.append("--record-video")
    if args.output:
        command.extend(("--output", args.output))
    return command


def recovery_command(
    args: argparse.Namespace, map_name: str, attempt_log: Path, result_file: Path
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "record_recovery.py"),
        "--map", map_name, "--source-attempt", str(attempt_log),
        "--result-file", str(result_file), "--episodes-root", str(args.episodes_root),
        "--max-seconds", str(args.recovery_seconds),
        "--focus-delay", str(args.focus_delay), "--port", str(args.port),
        "--fps", str(args.fps), "--width", str(args.width),
        "--height", str(args.height), "--keep-focused", "--disable-chamber-timeout",
        "--session-controls",
    ]
    if args.manual_focus:
        command.append("--manual-focus")
    if args.test_mode:
        command.append("--test-mode")
    if args.output:
        command.extend(("--output", args.output))
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session_root = args.session_root.resolve()
    session_dir = session_root / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = session_dir / "session.jsonl"
    corrections = 0
    map_index = 0
    print(f"Recovery session: {session_dir}")
    if args.test_mode:
        print("TEST MODE: no model MP4s or recovery MP4/CSV episodes will be saved; diagnostics stay in this test session folder.")
    print("P hands off for recovery; R restarts the current chamber; K stops the session. Esc is emergency stop.")
    print("P is read from the physical keyboard and may also reach Portal 2; ensure it is unbound there. R is temporarily unbound in-game.")
    if args.manual_focus:
        print("Manual focus: after takeover, check confinement and re-align the view; press P again to start the recovery clip.")
    with manifest_path.open("w", encoding="utf-8", buffering=1) as manifest:
        manifest.write(json.dumps({
            "type": "session", "time_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()), "maps": args.maps,
            "target_corrections": args.target_corrections,
            "max_model_attempts": args.max_model_attempts,
            "test_mode": args.test_mode,
            "model_video": not (args.test_mode or args.no_model_video),
        }) + "\n")
        try:
            for number in range(1, args.max_model_attempts + 1):
                map_name = args.maps[map_index]
                attempt_dir = session_dir / f"{number:03d}_{map_name}"
                command = model_command(args, map_name, attempt_dir)
                print(f"\n[{number}/{args.max_model_attempts}] Model on {map_name} ({corrections}/{args.target_corrections} corrections)", flush=True)
                print(shlex.join(command), flush=True)
                result = subprocess.run(command, cwd=ROOT, check=False)
                reason, log_path, video_path = inspect_attempt(attempt_dir)
                manifest.write(json.dumps({
                    "type": "model_attempt", "number": number, "map": map_name,
                    "stop_reason": reason, "returncode": result.returncode,
                    "log": str(log_path) if log_path else None, "video": video_path,
                }) + "\n")
                if result.returncode != 0 or reason in ("escape_key", "keyboard_interrupt", "cancelled", "incomplete", "session_stop"):
                    print(f"Session stopped after model attempt: {reason} (exit {result.returncode}).")
                    return 1 if result.returncode != 0 else 0 if reason == "session_stop" else 130
                if reason in ("restart_requested", "time_limit", "goal_reached"):
                    print(f"Model {reason}; retrying {map_name} from the beginning.")
                    continue
                if reason == "recovery_handoff":
                    if log_path is None:
                        print("Handoff log is missing; cannot attribute a recovery.")
                        return 1
                    result_file = attempt_dir / "recovery_result.json"
                    command = recovery_command(args, map_name, log_path, result_file)
                    print(shlex.join(command), flush=True)
                    recovery = subprocess.run(command, cwd=ROOT, check=False)
                    if recovery.returncode != 0 or not result_file.is_file():
                        print("Recovery did not finish cleanly; session stopped without advancing.")
                        return 1
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    manifest.write(json.dumps({
                        "type": "recovery", "number": number, "map": map_name,
                        "source_attempt": str(log_path), **data,
                    }) + "\n")
                    if data["outcome"] == "restart_requested":
                        print(f"Correction restarted or handoff was invalid; retrying model on {map_name}.")
                        continue
                    if data["outcome"] == "model_chamber_finished":
                        print(f"Model finished {map_name} before recovery; restarting the same chamber without counting a correction.")
                        continue
                    if data["outcome"] != "goal_reached":
                        print(f"Recovery ended as {data['outcome']}; session stopped for review.")
                        return 0 if data["outcome"] == "interrupted" else 130
                    corrections += 1
                    if corrections >= args.target_corrections:
                        print(f"Target met: {corrections} successful corrections. Manifest: {manifest_path}")
                        return 0
                elif reason != "episode_failed":
                    print(f"Unexpected model stop reason {reason}; session stopped for review.")
                    return 1
                map_index = (map_index + 1) % len(args.maps)
        except KeyboardInterrupt:
            print("\nSession interrupted; no further chamber will be loaded.")
            return 130
    print(f"Attempt limit reached with {corrections}/{args.target_corrections} corrections. Manifest: {manifest_path}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
