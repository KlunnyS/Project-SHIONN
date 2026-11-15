"""Capture aligned Portal 2 screen frames and physical-input episodes.

Also provides the Wayland camera and input-device selection used by live runs.
"""

# Python standard library: paths and interpreter setup for direct script use.
import os
import sys
import glob

# Ensure local .venv packages are accessible even when run outside active venv or via sudo
_repo_dir = os.path.dirname(os.path.abspath(__file__))
_venv_sites = glob.glob(os.path.join(_repo_dir, ".venv", "lib", "python*", "site-packages"))
if _venv_sites and _venv_sites[0] not in sys.path:
    sys.path.insert(0, _venv_sites[0])

# Python standard library: episode files, processes, timing, and threads.
import csv
import time
import json
import threading
import subprocess
import re
from collections import deque
from datetime import datetime
from typing import NamedTuple

# Installed dependencies: NumPy frames and Linux evdev input devices.
import numpy as np
import evdev

# Project module at the repository root: Portal 2 console/game control.
from episode_events import EpisodeEventStream
from wrapper import Portal2Controller, is_game_running

# --- Device Discovery ---
def _matches_device_selector(device, selector):
    selector = selector.casefold()
    return selector == device.path.casefold() or selector in device.name.casefold()


def _pointer_score(device):
    """Prefer a real mouse/touchpad over receiver and keyboard side channels."""
    name = device.name.casefold()
    keys = device.capabilities().get(evdev.ecodes.EV_KEY, [])
    score = 50 if evdev.ecodes.BTN_LEFT in keys else 0
    if any(token in name for token in ("mouse", "touchpad", "trackpad")):
        score += 100
    if any(token in name for token in ("basilisk", "deathadder", "razer", "logitech")):
        score += 75
    if any(token in name for token in ("keyboard", "dongle", "receiver")):
        score -= 25
    if any(token in name for token in ("virtual", "uinput", "portal2-wrapper")):
        score -= 200
    return score


def _is_pointer_device(device):
    """Return whether the recorder can turn this device into mouse deltas."""
    capabilities = device.capabilities()
    relative_axes = capabilities.get(evdev.ecodes.EV_REL, [])
    if evdev.ecodes.REL_X in relative_axes and evdev.ecodes.REL_Y in relative_axes:
        return True

    absolute_axes = capabilities.get(evdev.ecodes.EV_ABS, [])
    keys = capabilities.get(evdev.ecodes.EV_KEY, [])
    name = device.name.casefold()
    return (
        evdev.ecodes.ABS_X in absolute_axes
        and evdev.ecodes.ABS_Y in absolute_axes
        and (
            "touchpad" in name
            or "trackpad" in name
            or evdev.ecodes.BTN_TOUCH in keys
        )
    )


def _is_keyboard_device(device):
    """Mirror the keyboard filtering used by the recorder."""
    keys = device.capabilities().get(evdev.ecodes.EV_KEY, [])
    return (
        evdev.ecodes.KEY_W in keys
        and evdev.ecodes.KEY_A in keys
        and (
            "keyboard" in device.name.casefold()
            or "K57" in device.name
        )
    )


def find_input_devices(mouse_selector=None):
    print("Scanning for keyboard and mouse...")
    devices = []
    for path in evdev.list_devices():
        try:
            devices.append(evdev.InputDevice(path))
        except OSError as error:
            print(f"Skipping unreadable input device: {path} ({error})")
    keyboards = []
    mouse_candidates = []
    
    for device in devices:
        if _is_pointer_device(device):
            mouse_candidates.append(device)
            print(f"Found pointer candidate: {device.name} ({device.path})")

        if _is_keyboard_device(device):
            keyboards.append(device)
            print(f"Found keyboard: {device.name} ({device.path})")
    
    if mouse_selector:
        mice = [
            device for device in mouse_candidates
            if _matches_device_selector(device, mouse_selector)
        ]
        if not mice:
            available = ", ".join(
                f"{device.path} ({device.name})" for device in mouse_candidates
            ) or "none"
            raise RuntimeError(
                f"Mouse selector '{mouse_selector}' matched no pointer device. "
                f"Available pointer devices: {available}"
            )
    elif mouse_candidates:
        mice = [max(mouse_candidates, key=_pointer_score)]
    else:
        mice = []

    for device in mice:
        print(f"Selected pointer: {device.name} ({device.path})")

    selected_paths = {device.path for device in (*keyboards, *mice)}
    for device in devices:
        if device.path not in selected_paths:
            device.close()
    return keyboards, mice

# --- Input Tracker ---
class InputTracker:
    def __init__(self, kbd_devs, mouse_devs, restart_enabled=False, start_enabled=False):
        self.kbds = kbd_devs
        self.mice = mouse_devs
        self.restart_enabled = restart_enabled
        self.start_enabled = start_enabled
        
        self.keys_held = {
            'W': False, 'A': False, 'S': False, 'D': False,
            'SPACE': False, 'E': False, 'CROUCH': False,
            'BTN_LEFT': False, 'BTN_RIGHT': False
        }
        
        # Latches for keys pressed and released entirely within a single tick
        self.keys_pressed_this_tick = {k: False for k in self.keys_held}
        # Track physical sources separately because several keycodes/devices
        # can map to one action (for example Ctrl and C both mean crouch).
        self._physical_keys_down = {k: set() for k in self.keys_held}
        self.key_lock = threading.Lock()
        
        self.mouse_dx = 0
        self.mouse_dy = 0
        # Cumulative physical motion lets handoff checks inspect input without
        # consuming the deltas later written to an episode's actions.csv.
        self.mouse_total_left = 0
        self.mouse_total_right = 0
        self.mouse_total_up = 0
        self.mouse_total_down = 0
        self.mouse_lock = threading.Lock()
        self.mouse_event_counts = {device.path: 0 for device in self.mice}
        self.mouse_errors = []
        self._health_lock = threading.Lock()
        self._device_errors = []
        # Control keys never become training labels; the main recorder loop consumes them.
        self.control_lock = threading.Lock()
        self.control_events = deque()
        self.control_keys_down = set()
        self.last_control_press = {}
        
        self.running = True
        self.threads = []
        for kbd in self.kbds:
            self.threads.append(threading.Thread(target=self._kbd_loop, args=(kbd,), daemon=True))
        for mouse in self.mice:
            self.threads.append(threading.Thread(target=self._mouse_loop, args=(mouse,), daemon=True))
        
    def start(self):
        for t in self.threads:
            t.start()
        
    def _kbd_loop(self, kbd):
        key_map = {
            evdev.ecodes.KEY_W: 'W',
            evdev.ecodes.KEY_A: 'A',
            evdev.ecodes.KEY_S: 'S',
            evdev.ecodes.KEY_D: 'D',
            evdev.ecodes.KEY_SPACE: 'SPACE',
            evdev.ecodes.KEY_E: 'E',
            evdev.ecodes.KEY_LEFTCTRL: 'CROUCH',
            evdev.ecodes.KEY_C: 'CROUCH'
        }
        try:
            for event in kbd.read_loop():
                if not self.running: break
                if event.type == evdev.ecodes.EV_KEY:
                    if event.code in (evdev.ecodes.KEY_ESC, evdev.ecodes.KEY_K) or (
                        self.restart_enabled and event.code == evdev.ecodes.KEY_R
                    ) or (self.start_enabled and event.code == evdev.ecodes.KEY_P):
                        self._track_control_key(kbd, event)
                    if event.code in key_map:
                        self._update_physical_key(
                            key_map[event.code], kbd, event.code, event.value
                        )
            if self.running:
                self._record_device_error("keyboard", kbd, "read loop ended unexpectedly")
        except Exception as e:
            self._record_device_error("keyboard", kbd, e)

    def _record_device_error(self, kind, device, error):
        reason = f"{kind} device disconnected or failed: {device.path} ({device.name}): {error}"
        with self._health_lock:
            self._device_errors.append(reason)
            if kind == "mouse":
                self.mouse_errors.append(reason)
        print(f"Input {kind} loop error ({device.name}): {error}")

    def capture_errors(self):
        """Return a stable snapshot of required-device read failures."""
        with self._health_lock:
            return tuple(self._device_errors)

    def _update_physical_key(self, action, device, code, value):
        """Update one logical action from a specific physical key source."""
        source = (getattr(device, "path", id(device)), code)
        with self.key_lock:
            down = self._physical_keys_down[action]
            if value in (1, 2):
                down.add(source)
                self.keys_pressed_this_tick[action] = True
            elif value == 0:
                down.discard(source)
            self.keys_held[action] = bool(down)

    def _track_control_key(self, kbd, event):
        """Queue one action per physical press, ignoring repeats and duplicate interfaces."""
        key = (getattr(kbd, "path", id(kbd)), event.code)
        with self.control_lock:
            if event.value == 0:
                self.control_keys_down.discard(key)
                return
            if event.value != 1 or key in self.control_keys_down:
                return
            self.control_keys_down.add(key)
            now = time.monotonic()
            if now - self.last_control_press.get(event.code, float("-inf")) < 0.2:
                return
            self.last_control_press[event.code] = now
            action = (
                "pause" if event.code == evdev.ecodes.KEY_ESC
                else "restart" if event.code == evdev.ecodes.KEY_R
                else "start" if event.code == evdev.ecodes.KEY_P else "stop"
            )
            self.control_events.append(action)

    def pop_control_events(self):
        """Atomically take pending Escape/K events for the recorder's main loop."""
        with self.control_lock:
            events = list(self.control_events)
            self.control_events.clear()
        return events

    def pause_is_pending(self):
        """Let the frame thread stop as soon as Escape is seen by evdev."""
        with self.control_lock:
            return "pause" in self.control_events
                        
    def _mouse_loop(self, mouse):
        last_abs_x = None
        last_abs_y = None
        try:
            for event in mouse.read_loop():
                if not self.running: break
                if event.type == evdev.ecodes.EV_REL:
                    with self.mouse_lock:
                        if event.code == evdev.ecodes.REL_X:
                            self.mouse_dx += event.value
                            if event.value < 0:
                                self.mouse_total_left -= event.value
                            else:
                                self.mouse_total_right += event.value
                            self.mouse_event_counts[mouse.path] += 1
                        elif event.code == evdev.ecodes.REL_Y:
                            self.mouse_dy += event.value
                            if event.value < 0:
                                self.mouse_total_up -= event.value
                            else:
                                self.mouse_total_down += event.value
                            self.mouse_event_counts[mouse.path] += 1
                elif event.type == evdev.ecodes.EV_ABS:
                    with self.mouse_lock:
                        if event.code == evdev.ecodes.ABS_X:
                            if last_abs_x is not None:
                                delta = event.value - last_abs_x
                                self.mouse_dx += delta
                                if delta < 0:
                                    self.mouse_total_left -= delta
                                else:
                                    self.mouse_total_right += delta
                                self.mouse_event_counts[mouse.path] += 1
                            last_abs_x = event.value
                        elif event.code == evdev.ecodes.ABS_Y:
                            if last_abs_y is not None:
                                delta = event.value - last_abs_y
                                self.mouse_dy += delta
                                if delta < 0:
                                    self.mouse_total_up -= delta
                                else:
                                    self.mouse_total_down += delta
                                self.mouse_event_counts[mouse.path] += 1
                            last_abs_y = event.value
                elif event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_TOUCH and event.value == 0:
                        # Reset tracking points when finger is lifted to prevent jumps
                        last_abs_x = None
                        last_abs_y = None
                    if event.code in (evdev.ecodes.BTN_LEFT, evdev.ecodes.BTN_TOUCH):
                        self._update_physical_key(
                            'BTN_LEFT', mouse, event.code, event.value
                        )
                    elif event.code == evdev.ecodes.BTN_RIGHT:
                        self._update_physical_key(
                            'BTN_RIGHT', mouse, event.code, event.value
                        )
            if self.running:
                self._record_device_error("mouse", mouse, "read loop ended unexpectedly")
        except Exception as e:
            self._record_device_error("mouse", mouse, e)

    def get_mouse_delta_and_reset(self):
        with self.mouse_lock:
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0
        return dx, dy

    def mouse_counters_snapshot(self):
        """Read physical motion totals without clearing recording input."""
        with self.mouse_lock:
            return {
                "left": self.mouse_total_left,
                "right": self.mouse_total_right,
                "up": self.mouse_total_up,
                "down": self.mouse_total_down,
            }

    def get_snapshot_and_reset(self):
        dx, dy = self.get_mouse_delta_and_reset()
            
        # A key is considered active this tick if it's CURRENTLY held down,
        # OR if it was pressed and released entirely within the tick duration.
        with self.key_lock:
            def is_active(k):
                active = self.keys_held[k] or self.keys_pressed_this_tick[k]
                self.keys_pressed_this_tick[k] = False # clear the latch
                return active

            keys = {key: is_active(key) for key in self.keys_held}
            
        return {
            'move_w': int(keys['W']),
            'move_a': int(keys['A']),
            'move_s': int(keys['S']),
            'move_d': int(keys['D']),
            'jump': int(keys['SPACE']),
            'crouch': int(keys['CROUCH']),
            'use': int(keys['E']),
            'fire_left': int(keys['BTN_LEFT']),
            'fire_right': int(keys['BTN_RIGHT']),
            'mouse_dx': dx,
            'mouse_dy': dy
        }
        
    def stop(self):
        self.running = False


# --- Camera ---
def get_default_output():
    try:
        out = subprocess.check_output(["wf-recorder", "-L"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r'Name:\s+(\S+)', out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

class FrameSample(NamedTuple):
    """A complete pipe frame and its local receive time, not exposure time."""
    frame: np.ndarray | None
    frame_id: int | None
    received_monotonic: float | None


class WaylandCamera:
    def __init__(self, width=1920, height=1080, fps=24, output_name=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_name = output_name
        self.frame_size = width * height * 3 
        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self._latest_sample = FrameSample(None, None, None)
        self._next_frame_id = 0
        self.running = False
        self.proc = None
        self.thread = None
        self.stderr_thread = None
        self.stderr_lines = []
        self._state_lock = threading.Lock()
        self._stop_requested = False
        self._capture_error = None

    def start(self):
        cmd = [
            "wf-recorder",
            "-c", "rawvideo",
            "-m", "rawvideo",
            "-x", "bgr24",
            "-D",
            "-r", str(self.fps),
            "-f", "pipe:1",
            "-y",
            "-F", f"scale={self.width}:{self.height}"
        ]
        if self.output_name:
            cmd.insert(1, "-o")
            cmd.insert(2, self.output_name)
            
        self.proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 2
        )
        self.running = True
        self._stop_requested = False
        self._capture_error = None
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _read_stderr(self):
        """Keep recent wf-recorder diagnostics without risking a full pipe."""
        for raw_line in self.proc.stderr:
            line = raw_line.decode(errors="replace").strip()
            if line:
                self.stderr_lines.append(line)
                del self.stderr_lines[:-20]

    def error_summary(self):
        poll = getattr(self.proc, "poll", None)
        if callable(poll) and poll() is not None and self.stderr_thread:
            self.stderr_thread.join(timeout=0.5)
        return "\n".join(self.stderr_lines)

    @property
    def capture_error(self):
        with self._state_lock:
            return self._capture_error

    def _fail_capture(self, reason):
        with self._state_lock:
            if self._capture_error is None:
                self._capture_error = reason

    def _read_exact_frame(self):
        """Read one raw frame despite pipe chunking; return bytes and EOF state."""
        chunks = bytearray()
        while len(chunks) < self.frame_size:
            raw = self.proc.stdout.read(self.frame_size - len(chunks))
            if not raw:
                return bytes(chunks), True
            chunks.extend(raw)
        return bytes(chunks), False

    def _update(self):
        try:
            while self.running:
                raw, eof = self._read_exact_frame()
                if len(raw) == self.frame_size:
                    received = time.monotonic()
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                    frame = frame.copy()
                    with self._frame_lock:
                        self._next_frame_id += 1
                        self._latest_sample = FrameSample(frame, self._next_frame_id, received)
                        self.latest_frame = frame
                else:
                    with self._state_lock:
                        requested_stop = self._stop_requested
                    if not requested_stop:
                        detail = self.error_summary()
                        reason = "camera stream ended unexpectedly" if not raw else (
                            f"camera stream ended with a partial frame ({len(raw)}/{self.frame_size} bytes)"
                        )
                        if detail:
                            reason += f": {detail}"
                        self._fail_capture(reason)
                    self.running = False
                    break
        except Exception as error:
            self._fail_capture(f"camera reader failed: {type(error).__name__}: {error}")
            self.running = False

    def get_latest_frame(self):
        return self.get_latest_frame_sample().frame

    def get_latest_frame_sample(self):
        """Read coherent image/id/time without waiting for a new frame."""
        with self._frame_lock:
            return self._latest_sample

    def stop(self):
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is not None:
            with state_lock:
                self._stop_requested = True
        else:
            self._stop_requested = True
        self.running = False
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3.0)
        if self.stderr_thread:
            self.stderr_thread.join(timeout=1.0)


class FFmpegVideoWriter:
    """Stream BGR frames to an H.264 MP4 without retaining an episode in RAM."""
    def __init__(self, output_path, width, height, fps, crf=20):
        self.width = width
        self.height = height
        self.frame_shape = (height, width, 3)
        self.output_path = output_path
        command = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
        ]
        try:
            self.proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError as error:
            raise RuntimeError("ffmpeg with libx264 is required to record H.264 video") from error

    def write(self, frame):
        if frame.shape != self.frame_shape:
            raise ValueError(f"Expected BGR frame {self.frame_shape}, received {frame.shape}")
        if self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited unexpectedly while writing {self.output_path}")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg stopped accepting frames for {self.output_path}") from error

    def release(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            return_code = self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait()
            raise RuntimeError(f"ffmpeg did not finalize {self.output_path} within 30 seconds")
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while finalizing {self.output_path} (exit {return_code})")


# --- Main Recorder App ---
class EpisodeRecorder:
    """Records aligned FHD frames and actions for behavior cloning.

    Resolution and tick rate remain arguments for diagnostics and legacy data,
    but new recordings default to the 1920x1080 / 24 Hz dataset contract.
    """
    def __init__(
        self,
        fps=24,
        width=1920,
        height=1080,
        video_crf=20,
        controller=None,
        output_name=None,
        mouse_device=None,
        episodes_root="episodes",
        restart_enabled=False,
        start_enabled=False,
        prepare_output=True,
        resume_delay_seconds=0.0,
    ):
        if resume_delay_seconds < 0:
            raise ValueError("resume_delay_seconds cannot be negative")
        self.fps = fps
        self.kbds, self.mice = find_input_devices(mouse_selector=mouse_device)
        if not self.kbds or not self.mice:
            raise RuntimeError(
                "Could not find both keyboard and mouse. Configure /dev/input "
                "permissions for your desktop user; do not run the recorder with sudo."
            )
            
        self.tracker = InputTracker(
            self.kbds, self.mice, restart_enabled=restart_enabled,
            start_enabled=start_enabled,
        )
        
        out_name = output_name if output_name is not None else get_default_output()
        self.width = width
        self.height = height
        self.video_crf = video_crf
        self.camera = WaylandCamera(width, height, fps, output_name=out_name)
        
        self.controller = controller if controller else Portal2Controller(8020, log_file=None)
        
        self.recording = False
        self.paused = False
        self.resume_delay_seconds = resume_delay_seconds
        self.resume_deadline = None
        self.ep_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        
        self.frame_idx = 0
        self._capture_failure_reason = None
        self._capture_metadata = None
        
        self.episodes_root = os.fspath(episodes_root)
        self.in_progress_root = os.path.join(self.episodes_root, ".in_progress")
        if prepare_output:
            os.makedirs(self.in_progress_root, exist_ok=True)
        
    def start_recording(self, map_name=None, metadata_extra=None):
        print("\n--- STARTING RECORDING ---")
        # Include microseconds so rapid automatic map resets cannot reuse an
        # episode directory created earlier in the same second.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.episode_name = f"episode_{timestamp}"
        self.ep_dir = os.path.join(self.in_progress_root, self.episode_name)
        os.makedirs(self.ep_dir, exist_ok=True)
        self._capture_metadata = {"map": map_name, **(metadata_extra or {}), "frame_timing": {
                "version": 1, "clock": "time.monotonic",
                "frame_timestamp": "complete_raw_frame_received_in_python",
                "input_timestamp": "snapshot_call_brackets_not_individual_event_times",
            }, "capture_status": "in_progress", "capture_valid": None,
            "capture_failure_reason": None}
        self._write_capture_metadata()
        
        # Open CSV
        self.csv_path = os.path.join(self.ep_dir, "actions.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'frame_idx', 'move_w', 'move_a', 'move_s', 'move_d',
            'jump', 'crouch', 'use', 'fire_left', 'fire_right', 'mouse_dx', 'mouse_dy',
            'input_snapshot_started_monotonic', 'input_snapshot_finished_monotonic',
            'frame_id',
            'frame_received_monotonic', 'frame_sampled_monotonic',
            'frame_age_seconds', 'frame_reused', 'input_discard_count'
        ])
        
        # Stream directly to H.264; CRF 20 is visually near-lossless while manageable on disk.
        vid_path = os.path.join(self.ep_dir, "video.mp4")
        self.video_writer = FFmpegVideoWriter(vid_path, self.width, self.height, self.fps, self.video_crf)
        
        self.frame_idx = 0
        self.input_discard_count = 0
        self.paused = False
        self.resume_deadline = None
        self.sync_lock = threading.Lock()
        self.recording = True
        
        # Start the sync loop thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

    def toggle_pause(self):
        """Pause or resume the current episode without splitting its MP4/CSV."""
        if not self.recording:
            return False
        # Wait for any in-flight frame/CSV tick before changing pause state.
        with self.sync_lock:
            if not self.paused:
                self.paused = True
                self.resume_deadline = None
                message = "Recording paused. Press Esc to resume; K to stop."
            elif self.resume_deadline is not None:
                # Another Escape during the grace period reopens the game menu.
                self.resume_deadline = None
                message = "Resume cancelled; recording remains paused."
            elif getattr(self, "resume_delay_seconds", 0.0):
                self.resume_deadline = time.monotonic() + self.resume_delay_seconds
                message = f"Game resumed; recording restarts in {self.resume_delay_seconds:g} seconds."
            else:
                self.paused = False
                message = "Recording resumed."
            self._discard_pending_input()
        print(message)
        return self.paused

    def _finish_resume_delay(self, now):
        """Resume after game menus have disappeared, dropping grace-period input."""
        deadline = getattr(self, "resume_deadline", None)
        if self.paused and deadline is not None and now >= deadline:
            self.resume_deadline = None
            self.paused = False
            self._discard_pending_input()
            print("Recording resumed.")

    def _discard_pending_input(self):
        """Mark input-window discontinuities caused by menus/resume grace."""
        self.tracker.get_snapshot_and_reset()
        self.input_discard_count = getattr(self, "input_discard_count", 0) + 1
        
    def stop_recording(self, outcome="unknown"):
        if self.ep_dir is None:
            return
            
        print(f"\n--- STOPPING RECORDING (Outcome: {outcome}) ---")
        self.recording = False
        self.paused = False
        self.resume_deadline = None
        self.sync_thread.join(timeout=2.0)
        if getattr(self.sync_thread, "is_alive", lambda: False)() is True:
            self._mark_capture_failure("capture synchronization thread did not stop")

        try:
            if self.video_writer:
                self.video_writer.release()
        except Exception as error:
            self._mark_capture_failure(f"video writer failed to finalize: {error}")
        finally:
            if self.csv_file:
                self.csv_file.close()

        tracker = getattr(self, "tracker", None)
        tracker_errors = tracker.capture_errors() if tracker is not None and hasattr(tracker, "capture_errors") else ()
        if not isinstance(tracker_errors, (tuple, list)):
            tracker_errors = ()
        if tracker_errors:
            self._mark_capture_failure("; ".join(tracker_errors))
        camera_error = getattr(getattr(self, "camera", None), "capture_error", None)
        if isinstance(camera_error, str) and camera_error:
            self._mark_capture_failure(camera_error)

        invalid_outcomes = (
            "interrupted", "focus_lost", "oracle_paused", "map_reloaded",
            "restart_requested", "discarded", "capture_error", "unknown", "invalid_waypoint_event"
        )
        if outcome in invalid_outcomes:
            self._mark_capture_failure(
                f"recording ended without a valid terminal result: {outcome}"
            )
        valid = getattr(self, "_capture_failure_reason", None) is None
        if not getattr(self, "_capture_metadata", None):
            metadata_path = os.path.join(self.ep_dir, "metadata.json")
            try:
                with open(metadata_path, encoding="utf-8") as metadata_file:
                    self._capture_metadata = json.load(metadata_file)
            except (OSError, ValueError):
                self._capture_metadata = {}
        self._capture_metadata.update({
            "capture_status": "complete" if valid else "invalid",
            "capture_valid": valid,
            "capture_outcome": outcome,
            "capture_failure_reason": getattr(self, "_capture_failure_reason", None),
        })
        self._write_capture_metadata()
            
        # Publish the completed recording under its outcome. Keeping active
        # captures in .in_progress prevents preprocessors from seeing partial
        # MP4/CSV pairs if recording is interrupted unexpectedly.
        safe_outcome = re.sub(r"[^a-zA-Z0-9_-]+", "_", outcome).strip("_") or "unknown"
        outcome_dir = os.path.join(self.episodes_root, safe_outcome if valid else "invalid")
        os.makedirs(outcome_dir, exist_ok=True)
        completed_dir = os.path.join(outcome_dir, self.episode_name)
        os.rename(self.ep_dir, completed_dir)
        self.ep_dir = completed_dir
        print(f"Episode saved to: {completed_dir}")
        if not valid:
            print(f"Recording is invalid for training: {getattr(self, '_capture_failure_reason', None) or 'session ended before a scored terminal state'}")
        return outcome if valid else "invalid"

    def _write_capture_metadata(self):
        if not self.ep_dir or not self._capture_metadata:
            return
        path = os.path.join(self.ep_dir, "metadata.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as metadata_file:
            json.dump(self._capture_metadata, metadata_file, indent=2)
            metadata_file.write("\n")
        os.replace(temporary, path)

    def _mark_capture_failure(self, reason):
        if getattr(self, "_capture_failure_reason", None) is None:
            self._capture_failure_reason = str(reason)

    @property
    def capture_failure_reason(self):
        return getattr(self, "_capture_failure_reason", None)

    def handle_console_events(self, event_stream, output):
        """Apply complete episode events while retaining fragmented TCP lines."""
        event_stream.feed(output)
        while True:
            event = event_stream.pop()
            if event is None:
                return
            if event.name == "chamber_ready":
                if not self.recording:
                    self.start_recording()
            elif event.name == "goal_reached":
                if self.recording:
                    self.stop_recording("goal_reached")
            elif event.name == "episode_failed":
                if self.recording:
                    self.stop_recording("episode_failed")

    def _sync_loop(self):
        tick_duration = 1.0 / self.fps
        next_tick = time.monotonic()
        previous_frame_id = None
        try:
            while self.recording:
                with self.sync_lock:
                    if not self.recording:
                        break
                    input_errors = self.tracker.capture_errors() if hasattr(self.tracker, "capture_errors") else ()
                    if isinstance(input_errors, (tuple, list)) and input_errors:
                        self._mark_capture_failure("; ".join(input_errors))
                        break
                    camera_error = getattr(getattr(self, "camera", None), "capture_error", None)
                    if isinstance(camera_error, str) and camera_error:
                        self._mark_capture_failure(camera_error)
                        break
                    self._finish_resume_delay(time.monotonic())
                    paused = self.paused or self.tracker.pause_is_pending()
                    if paused:
                        # Drop menu input and reset the tick clock; no catch-up frames.
                        self._discard_pending_input()
                        next_tick = time.monotonic()
                    else:
                        input_started = time.monotonic()
                        actions = self.tracker.get_snapshot_and_reset()
                        input_finished = time.monotonic()
                        sample = self.camera.get_latest_frame_sample()
                        sampled = time.monotonic()
                        frame = sample.frame
                        if frame is None or sample.frame_id is None or sample.received_monotonic is None:
                            self._mark_capture_failure("camera has not provided a complete frame")
                            break
                        max_age = max(0.1, 2.5 / self.fps)
                        frame_age = sampled - sample.received_monotonic
                        if frame_age < 0 or frame_age > max_age:
                            self._mark_capture_failure(
                                f"camera frame is stale or has an invalid timestamp (age {frame_age:.3f}s; limit {max_age:.3f}s)"
                            )
                            break
                        if previous_frame_id is not None and sample.frame_id > previous_frame_id + 1:
                            self._mark_capture_failure(
                                f"camera frame gap detected ({previous_frame_id} to {sample.frame_id})"
                            )
                            break
                        if previous_frame_id is not None and sample.frame_id < previous_frame_id:
                            self._mark_capture_failure("camera frame ID moved backwards")
                            break
                        self.video_writer.write(frame)
                        self.csv_writer.writerow([
                            time.time(), self.frame_idx,
                            actions['move_w'], actions['move_a'], actions['move_s'], actions['move_d'],
                            actions['jump'], actions['crouch'], actions['use'],
                            actions['fire_left'], actions['fire_right'],
                            actions['mouse_dx'], actions['mouse_dy'],
                            input_started, input_finished,
                            sample.frame_id, sample.received_monotonic, sampled,
                            frame_age,
                            sample.frame_id == previous_frame_id if previous_frame_id is not None else None,
                            getattr(self, 'input_discard_count', 0)
                        ])
                        self.csv_file.flush()
                        self.frame_idx += 1
                        previous_frame_id = sample.frame_id
                        next_tick += tick_duration

                    if getattr(self, "_capture_failure_reason", None):
                        break

                sleep_time = (min(0.05, tick_duration) if paused
                              else next_tick - time.monotonic())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as error:
            self._mark_capture_failure(f"capture synchronization failed: {type(error).__name__}: {error}")

    def run(self):
        print("Starting Tracker and Camera...")
        self.tracker.start()
        self.camera.start()
        
        print("Waiting for game connection...")
        while not self.controller.connect():
            time.sleep(2)
            
        print("Connected to game console. Listening for EVT signals...")
        event_stream = EpisodeEventStream()
        
        try:
            while True:
                if self.capture_failure_reason and self.recording:
                    self.stop_recording("capture_error")
                    return
                for control in self.tracker.pop_control_events():
                    if control == "stop":
                        print("K pressed; ending recording session.")
                        return
                    if control == "pause":
                        self.toggle_pause()
                out = self.controller.read_console()
                self.handle_console_events(event_stream, out)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.recording:
                self.stop_recording("interrupted")
            self.tracker.stop()
            self.camera.stop()
            self.controller.disconnect()


if __name__ == "__main__":
    recorder = EpisodeRecorder()
    recorder.run()
