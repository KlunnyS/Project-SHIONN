import threading
import subprocess
import numpy as np
import time
import cv2
import re

def get_default_output():
    """Detect the first available display output from wf-recorder."""
    try:
        out = subprocess.check_output(["wf-recorder", "-L"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r'Name:\s+(\S+)', out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

class WaylandCamera:
    """A background threaded camera reading raw frames from wf-recorder."""
    def __init__(self, width=1280, height=720, fps=20, output_name=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_name = output_name
        # 3 bytes per pixel (B, G, R)
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
            "-x", "bgr24", # OpenCV uses BGR natively
            "-D",          # Disable damage tracking (forces constant frame output)
            "-r", str(self.fps),
            "-f", "pipe:1",# Output directly to stdout
            "-y",          # Overwrite output without prompting
            "-F", f"scale={self.width}:{self.height}" # Scale to consistent resolution
        ]
        
        if self.output_name:
            # Insert output parameter early in the command
            cmd.insert(1, "-o")
            cmd.insert(2, self.output_name)
            
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=None, # Allow stderr to print to terminal for debugging
            bufsize=self.frame_size * 2
        )
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        frames_read = 0
        while self.running:
            raw = self.proc.stdout.read(self.frame_size)
            if len(raw) == self.frame_size:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                self.latest_frame = frame.copy()
                if frames_read == 0:
                    print(f"DEBUG: First frame successfully read from wf-recorder! (Size: {len(raw)} bytes)")
                frames_read += 1
            elif len(raw) == 0:
                print(f"DEBUG: _update thread hit EOF. wf-recorder process likely died. Return code: {self.proc.poll()}")
                break 
            else:
                print(f"DEBUG: Read partial frame: {len(raw)} bytes (expected {self.frame_size})")

    def get_latest_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


def test_recording():
    print("Detecting displays...")
    output_name = get_default_output()
    if output_name:
        print(f"Found display: {output_name}")
    else:
        print("Could not detect display automatically. wf-recorder will try to guess.")
        
    # We use a fixed internal resolution (1280x720) so the script works on ANY monitor!
    INTERNAL_WIDTH = 1280
    INTERNAL_HEIGHT = 720
    
    print(f"Starting wf-recorder capture thread (Scaled to {INTERNAL_WIDTH}x{INTERNAL_HEIGHT})...")
    camera = WaylandCamera(INTERNAL_WIDTH, INTERNAL_HEIGHT, output_name=output_name)
    camera.start()

    print("Capturing baseline frame. Forcing screen updates...")
    print(">>> (Wayland compositors won't send frames without screen damage) <<<")
    # Forcing screen damage by printing repeatedly to the terminal
    for countdown in range(30, 0, -1):
        print(f"Force-updating screen... {countdown}/30", flush=True)
        time.sleep(0.1)
    
    baseline_frame = None
    
    print("DEBUG: Waiting for the first frame to arrive...")
    for i in range(20):
        baseline_frame = camera.get_latest_frame()
        if baseline_frame is not None:
            print("DEBUG: Baseline frame captured.")
            break
        print(f"DEBUG: Waiting... (attempt {i+1}/20)")
        time.sleep(0.1)
        
    if baseline_frame is None:
        print("ERROR: Failed to capture screen. Check the wf-recorder output above for errors.")
        print(f"DEBUG: wf-recorder status code: {camera.proc.poll()}")
        camera.stop()
        return

    gray_baseline = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    
    print("Capturing moving frame in 0.5 seconds...")
    time.sleep(0.5)
    
    moving_frame = camera.get_latest_frame()
    gray_moving = cv2.cvtColor(moving_frame, cv2.COLOR_BGR2GRAY)
    
    camera.stop()
    
    # Calculate difference
    diff = cv2.absdiff(gray_baseline, gray_moving)
    mean_diff = np.mean(diff)
    
    print(f"\n--- Analysis ---")
    print(f"Mean pixel difference: {mean_diff:.2f} (out of 255)")
    
    if mean_diff > 3.0:
        print("SUCCESS: Visually confirmed screen changes using continuous streaming!")
    else:
        print("WARNING: No significant movement detected.")

if __name__ == "__main__":
    test_recording()
