import os
import sys
import glob

# Ensure local .venv packages are accessible even when run outside active venv or via sudo
_repo_dir = os.path.dirname(os.path.abspath(__file__))
_venv_sites = glob.glob(os.path.join(_repo_dir, ".venv", "lib", "python*", "site-packages"))
if _venv_sites and _venv_sites[0] not in sys.path:
    sys.path.insert(0, _venv_sites[0])

import csv
import time
import json
import threading
import subprocess
import numpy as np
import re
import evdev
from datetime import datetime
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
    def __init__(self, kbd_devs, mouse_devs):
        self.kbds = kbd_devs
        self.mice = mouse_devs
        
        self.keys_held = {
            'W': False, 'A': False, 'S': False, 'D': False,
            'SPACE': False, 'E': False, 'CROUCH': False,
            'BTN_LEFT': False, 'BTN_RIGHT': False
        }
        
        # Latches for keys pressed and released entirely within a single tick
        self.keys_pressed_this_tick = {k: False for k in self.keys_held}
        
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.mouse_lock = threading.Lock()
        self.mouse_event_counts = {device.path: 0 for device in self.mice}
        self.mouse_errors = []
        
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
                    if event.code in key_map:
                        if event.value in (1, 2):
                            self.keys_held[key_map[event.code]] = True
                            self.keys_pressed_this_tick[key_map[event.code]] = True
                        elif event.value == 0:
                            self.keys_held[key_map[event.code]] = False
        except Exception as e:
            print(f"Keyboard loop error ({kbd.name}): {e}")
                        
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
                            self.mouse_event_counts[mouse.path] += 1
                        elif event.code == evdev.ecodes.REL_Y:
                            self.mouse_dy += event.value
                            self.mouse_event_counts[mouse.path] += 1
                elif event.type == evdev.ecodes.EV_ABS:
                    with self.mouse_lock:
                        if event.code == evdev.ecodes.ABS_X:
                            if last_abs_x is not None:
                                self.mouse_dx += (event.value - last_abs_x)
                                self.mouse_event_counts[mouse.path] += 1
                            last_abs_x = event.value
                        elif event.code == evdev.ecodes.ABS_Y:
                            if last_abs_y is not None:
                                self.mouse_dy += (event.value - last_abs_y)
                                self.mouse_event_counts[mouse.path] += 1
                            last_abs_y = event.value
                elif event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_TOUCH and event.value == 0:
                        # Reset tracking points when finger is lifted to prevent jumps
                        last_abs_x = None
                        last_abs_y = None
                    if event.code in (evdev.ecodes.BTN_LEFT, evdev.ecodes.BTN_TOUCH):
                        is_down = event.value in (1, 2)
                        self.keys_held['BTN_LEFT'] = is_down
                        if is_down: self.keys_pressed_this_tick['BTN_LEFT'] = True
                    elif event.code == evdev.ecodes.BTN_RIGHT:
                        is_down = event.value in (1, 2)
                        self.keys_held['BTN_RIGHT'] = is_down
                        if is_down: self.keys_pressed_this_tick['BTN_RIGHT'] = True
        except Exception as e:
            self.mouse_errors.append(f"{mouse.path} ({mouse.name}): {e}")
            print(f"Mouse/touchpad loop error ({mouse.name}): {e}")

    def get_mouse_delta_and_reset(self):
        with self.mouse_lock:
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0
        return dx, dy

    def get_snapshot_and_reset(self):
        dx, dy = self.get_mouse_delta_and_reset()
            
        # A key is considered active this tick if it's CURRENTLY held down,
        # OR if it was pressed and released entirely within the tick duration.
        def is_active(k):
            active = self.keys_held[k] or self.keys_pressed_this_tick[k]
            self.keys_pressed_this_tick[k] = False # clear the latch
            return active
            
        return {
            'move_w': int(is_active('W')),
            'move_a': int(is_active('A')),
            'move_s': int(is_active('S')),
            'move_d': int(is_active('D')),
            'jump': int(is_active('SPACE')),
            'crouch': int(is_active('CROUCH')),
            'use': int(is_active('E')),
            'fire_left': int(is_active('BTN_LEFT')),
            'fire_right': int(is_active('BTN_RIGHT')),
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

class WaylandCamera:
    def __init__(self, width=1920, height=1080, fps=24, output_name=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_name = output_name
        self.frame_size = width * height * 3 
        self.latest_frame = None
        self.running = False
        self.proc = None
        self.thread = None
        self.stderr_thread = None
        self.stderr_lines = []

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
        if self.proc and self.proc.poll() is not None and self.stderr_thread:
            self.stderr_thread.join(timeout=0.5)
        return "\n".join(self.stderr_lines)

    def _update(self):
        while self.running:
            raw = self.proc.stdout.read(self.frame_size)
            if len(raw) == self.frame_size:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                self.latest_frame = frame.copy()
            elif len(raw) == 0:
                break 

    def get_latest_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
            self.proc.wait()
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
    ):
        self.fps = fps
        self.kbds, self.mice = find_input_devices(mouse_selector=mouse_device)
        if not self.kbds or not self.mice:
            raise RuntimeError(
                "Could not find both keyboard and mouse. Configure /dev/input "
                "permissions for your desktop user; do not run the recorder with sudo."
            )
            
        self.tracker = InputTracker(self.kbds, self.mice)
        
        out_name = output_name if output_name is not None else get_default_output()
        self.width = width
        self.height = height
        self.video_crf = video_crf
        self.camera = WaylandCamera(width, height, fps, output_name=out_name)
        
        self.controller = controller if controller else Portal2Controller(8020, log_file=None)
        
        self.recording = False
        self.ep_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        
        self.frame_idx = 0
        
        self.episodes_root = "episodes"
        self.in_progress_root = os.path.join(self.episodes_root, ".in_progress")
        os.makedirs(self.in_progress_root, exist_ok=True)
        
    def start_recording(self, map_name=None):
        print("\n--- STARTING RECORDING ---")
        # Include microseconds so rapid automatic map resets cannot reuse an
        # episode directory created earlier in the same second.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.episode_name = f"episode_{timestamp}"
        self.ep_dir = os.path.join(self.in_progress_root, self.episode_name)
        os.makedirs(self.ep_dir, exist_ok=True)
        with open(os.path.join(self.ep_dir, "metadata.json"), "w", encoding="utf-8") as metadata_file:
            json.dump({"map": map_name}, metadata_file, indent=2)
            metadata_file.write("\n")
        
        # Open CSV
        self.csv_path = os.path.join(self.ep_dir, "actions.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'frame_idx', 'move_w', 'move_a', 'move_s', 'move_d',
            'jump', 'crouch', 'use', 'fire_left', 'fire_right', 'mouse_dx', 'mouse_dy'
        ])
        
        # Stream directly to H.264; CRF 20 is visually near-lossless while manageable on disk.
        vid_path = os.path.join(self.ep_dir, "video.mp4")
        self.video_writer = FFmpegVideoWriter(vid_path, self.width, self.height, self.fps, self.video_crf)
        
        self.frame_idx = 0
        self.recording = True
        
        # Start the sync loop thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
    def stop_recording(self, outcome="unknown"):
        if not self.recording:
            return
            
        print(f"\n--- STOPPING RECORDING (Outcome: {outcome}) ---")
        self.recording = False
        self.sync_thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
        if self.csv_file:
            self.csv_file.close()
            
        # Publish the completed recording under its outcome. Keeping active
        # captures in .in_progress prevents preprocessors from seeing partial
        # MP4/CSV pairs if recording is interrupted unexpectedly.
        safe_outcome = re.sub(r"[^a-zA-Z0-9_-]+", "_", outcome).strip("_") or "unknown"
        outcome_dir = os.path.join(self.episodes_root, safe_outcome)
        os.makedirs(outcome_dir, exist_ok=True)
        completed_dir = os.path.join(outcome_dir, self.episode_name)
        os.rename(self.ep_dir, completed_dir)
        self.ep_dir = completed_dir
        print(f"Episode saved to: {completed_dir}")

    def _sync_loop(self):
        tick_duration = 1.0 / self.fps
        next_tick = time.time()
        
        while self.recording:
            # 1. Get Action Snapshot
            actions = self.tracker.get_snapshot_and_reset()
            
            # 2. Get Frame
            frame = self.camera.get_latest_frame()
            if frame is not None:
                self.video_writer.write(frame)
            else:
                # If no frame yet, write a black frame to keep sync
                self.video_writer.write(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                
            # 3. Write CSV Row
            self.csv_writer.writerow([
                time.time(), self.frame_idx,
                actions['move_w'], actions['move_a'], actions['move_s'], actions['move_d'],
                actions['jump'], actions['crouch'], actions['use'],
                actions['fire_left'], actions['fire_right'],
                actions['mouse_dx'], actions['mouse_dy']
            ])
            self.csv_file.flush()
            
            self.frame_idx += 1
            
            # Sleep until next tick
            next_tick += tick_duration
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        print("Starting Tracker and Camera...")
        self.tracker.start()
        self.camera.start()
        
        print("Waiting for game connection...")
        while not self.controller.connect():
            time.sleep(2)
            
        print("Connected to game console. Listening for EVT signals...")
        
        try:
            while True:
                out = self.controller.read_console()
                if out:
                    # Parse out for events
                    for line in out.splitlines():
                        if "EVT|chamber_ready" in line:
                            if not self.recording:
                                self.start_recording()
                        elif "EVT|goal_reached" in line:
                            if self.recording:
                                self.stop_recording("goal_reached")
                        elif "EVT|episode_failed" in line:
                            if self.recording:
                                self.stop_recording("episode_failed")
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
