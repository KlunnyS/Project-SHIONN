import subprocess
import time
import sys

cmd = [
    "wf-recorder",
    "-o", "DP-1",
    "-c", "rawvideo",
    "-m", "rawvideo",
    "-x", "bgr24",
    "-D",
    "-r", "20",
    "-f", "-",
    "-y",
    "-F", "scale=1280:720"
]

print(f"Starting command: {' '.join(cmd)}")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

total_read = 0
start_time = time.time()
try:
    while time.time() - start_time < 5:
        # Read 1024 bytes at a time so we don't block waiting for a huge chunk
        chunk = proc.stdout.read1(1024) # read1 reads up to 1024 bytes without looping
        if not chunk:
            print("\nEOF reached.")
            break
        total_read += len(chunk)
        sys.stdout.write(f"\rTotal bytes read: {total_read}")
        sys.stdout.flush()
except KeyboardInterrupt:
    pass

print(f"\nFinal total bytes read: {total_read}")
proc.terminate()
