import time
import sys
import os
import glob

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_sites = glob.glob(os.path.join(_repo_dir, ".venv", "lib", "python*", "site-packages"))
if _venv_sites and _venv_sites[0] not in sys.path:
    sys.path.insert(0, _venv_sites[0])

from evdev import UInput, ecodes as e

def move():
    print("Initializing kernel-level virtual keyboard...")
    try:
        ui = UInput()
    except Exception as ex:
        print(f"ERROR: Could not create virtual keyboard. Did you run this with sudo? ({ex})")
        return

    print("Starting in 3 seconds. PLEASE FOCUS THE PORTAL 2 WINDOW NOW!")
    time.sleep(3)

    print("Sending 'W' keypress (moving forward)...")
    ui.write(e.EV_KEY, e.KEY_W, 1) # 1 = Key down
    ui.syn() 
    
    # Wait a moment for Portal 2 to move Chell
    time.sleep(0.5)
    
    # Release the key
    ui.write(e.EV_KEY, e.KEY_W, 0) # 0 = Key up
    ui.syn()
    print("Released 'W'.")
    
    ui.close()

if __name__ == "__main__":
    move()
