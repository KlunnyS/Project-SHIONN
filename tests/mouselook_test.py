import time
import sys
import os
import glob

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_sites = glob.glob(os.path.join(_repo_dir, ".venv", "lib", "python*", "site-packages"))
if _venv_sites and _venv_sites[0] not in sys.path:
    sys.path.insert(0, _venv_sites[0])

from evdev import UInput, ecodes as e

capabilities = {
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
    e.EV_REL: [e.REL_X, e.REL_Y],
}

ui = UInput(capabilities, name="portal2-test-mouse")

time.sleep(3)  # tab into the Portal 2 window during this delay

print("Sending mouse movement...")
for _ in range(50):
    ui.write(e.EV_REL, e.REL_X, 5)   # small rightward nudge each iteration
    ui.write(e.EV_REL, e.REL_Y, 0)
    ui.syn()
    time.sleep(0.01)

ui.close()