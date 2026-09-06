import os
import cv2
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
def find_input_devices():
    print("Scanning for keyboard and mouse...")
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    keyboards = []
    mice = []
    
    for device in devices:
        cap = device.capabilities()
        # Look for mouse (EV_REL with REL_X and REL_Y)
        if evdev.ecodes.EV_REL in cap:
            rels = cap[evdev.ecodes.EV_REL]
            if evdev.ecodes.REL_X in rels and evdev.ecodes.REL_Y in rels:
                mice.append(device)
                print(f"Found mouse: {device.name} ({device.path})")
                    
        # Look for keyboard (EV_KEY with W, A, S, D)
        if evdev.ecodes.EV_KEY in cap:
            keys = cap[evdev.ecodes.EV_KEY]
            if evdev.ecodes.KEY_W in keys and evdev.ecodes.KEY_A in keys:
                # Ignore the mouse pretending to be a keyboard if we can
                if "Keyboard" in device.name or "keyboard" in device.name or "K57" in device.name:
                    keyboards.append(device)
                    print(f"Found keyboard: {device.name} ({device.path})")
    
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
        try:
            for event in mouse.read_loop():
                if not self.running: break
                if event.type == evdev.ecodes.EV_REL:
                    with self.mouse_lock:
                        if event.code == evdev.ecodes.REL_X:
                            self.mouse_dx += event.value
                        elif event.code == evdev.ecodes.REL_Y:
                            self.mouse_dy += event.value
                elif event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_LEFT:
                        is_down = event.value in (1, 2)
                        self.keys_held['BTN_LEFT'] = is_down
                        if is_down: self.keys_pressed_this_tick['BTN_LEFT'] = True
                    elif event.code == evdev.ecodes.BTN_RIGHT:
                        is_down = event.value in (1, 2)
                        self.keys_held['BTN_RIGHT'] = is_down
                        if is_down: self.keys_pressed_this_tick['BTN_RIGHT'] = True
        except Exception as e:
            print(f"Mouse loop error ({mouse.name}): {e}")
                    
    def get_snapshot_and_reset(self):
        with self.mouse_lock:
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0
            
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
    def __init__(self, width=1280, height=720, fps=20, output_name=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_name = output_name
        self.frame_size = width * height * 3 
        self.latest_frame = None
        self.running = False
        self.proc = None
        self.thread = None

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
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 2
        )
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

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
            self.proc.terminate()
            self.proc.wait()


# --- Main Recorder App ---
class EpisodeRecorder:
    def __init__(self, fps=20, controller=None):
        self.fps = fps
        self.kbds, self.mice = find_input_devices()
        if not self.kbds or not self.mice:
            print("Error: Could not find both keyboard and mouse. Ensure you run with sudo!")
            exit(1)
            
        self.tracker = InputTracker(self.kbds, self.mice)
        
        out_name = get_default_output()
        self.camera = WaylandCamera(1280, 720, fps, output_name=out_name)
        
        self.controller = controller if controller else Portal2Controller(8020, log_file=None)
        
        self.recording = False
        self.ep_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        
        self.frame_idx = 0
        
        # Ensure episodes directory exists
        os.makedirs("episodes", exist_ok=True)
        
    def start_recording(self):
        print("\n--- STARTING RECORDING ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_dir = os.path.join("episodes", f"episode_{timestamp}")
        os.makedirs(self.ep_dir, exist_ok=True)
        
        # Open CSV
        self.csv_path = os.path.join(self.ep_dir, "actions.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'frame_idx', 'move_w', 'move_a', 'move_s', 'move_d',
            'jump', 'crouch', 'use', 'fire_left', 'fire_right', 'mouse_dx', 'mouse_dy'
        ])
        
        # Open VideoWriter
        vid_path = os.path.join(self.ep_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(vid_path, fourcc, self.fps, (1280, 720))
        
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
            
        # Tag the episode folder
        tagged_dir = f"{self.ep_dir}_{outcome}"
        os.rename(self.ep_dir, tagged_dir)
        print(f"Episode saved to: {tagged_dir}")

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
                self.video_writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
                
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
