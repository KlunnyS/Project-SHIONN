#!/usr/bin/env bash
# ==============================================================================
# Project-SHIONN Dependency Checker
# ==============================================================================

set -u

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
BOLD="\033[1m"
NC="\033[0m"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

print_header() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}"
}

status_ok() {
    echo -e "  [${GREEN}OK${NC}] $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

status_warn() {
    echo -e "  [${YELLOW}WARN${NC}] $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

status_fail() {
    echo -e "  [${RED}FAIL${NC}] $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}Project-SHIONN: Dependency & Environment Check${NC}"
echo "=================================================="

# 1. Python Environment Check
print_header "1. Python Environment"

if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
    status_ok "Python 3 is installed (version $PY_VER)"
else
    status_fail "Python 3 is not installed or not in PATH"
fi

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN=""

if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
    status_ok "Local virtual environment found at .venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_BIN="$(which python)"
    status_ok "Active virtual environment detected ($VIRTUAL_ENV)"
else
    status_warn "No .venv found. Using system Python ($(which python3 2>/dev/null || echo 'none'))"
    PYTHON_BIN="$(command -v python3 || echo '')"
fi

# 2. Python Packages Check
print_header "2. Python Packages (requirements.txt)"

check_python_module() {
    local mod_name="$1"
    local import_name="$2"
    local extra_info="${3:-}"

    if [ -z "$PYTHON_BIN" ]; then
        status_fail "$mod_name: No valid Python interpreter found"
        return
    fi

    if "$PYTHON_BIN" -c "import $import_name" >/dev/null 2>&1; then
        local ver
        ver=$("$PYTHON_BIN" -c "import $import_name; print(getattr($import_name, '__version__', 'available'))" 2>/dev/null || echo "available")
        status_ok "$mod_name ($ver) is installed"
    else
        status_fail "$mod_name is NOT installed. Run: $PYTHON_BIN -m pip install $mod_name"
    fi
}

check_python_module "evdev" "evdev"
check_python_module "numpy" "numpy"
check_python_module "opencv-python" "cv2"
check_python_module "torch" "torch"

# 3. System Utilities Check
print_header "3. System Tools & Binaries"

# wf-recorder
if command -v wf-recorder >/dev/null 2>&1; then
    WF_VER=$(wf-recorder -v 2>&1 | head -n 1 || echo "installed")
    status_ok "wf-recorder found ($WF_VER)"
else
    status_warn "wf-recorder is NOT installed (Required for Wayland screen recording in recorder.py)"
    if [ -f /etc/arch-release ]; then
        echo -e "         ${BLUE}-> Install on Arch:${NC} sudo pacman -S wf-recorder"
    elif [ -f /etc/debian_version ]; then
        echo -e "         ${BLUE}-> Install on Debian/Ubuntu:${NC} sudo apt install wf-recorder"
    elif [ -f /etc/fedora-release ]; then
        echo -e "         ${BLUE}-> Install on Fedora:${NC} sudo dnf install wf-recorder"
    fi
fi

# pgrep
if command -v pgrep >/dev/null 2>&1; then
    status_ok "pgrep found (used to check if Portal 2 is running)"
else
    status_fail "pgrep (procps) is NOT installed"
fi

# steam
if command -v steam >/dev/null 2>&1; then
    status_ok "Steam client found in PATH"
else
    status_warn "Steam client not found in PATH (launch_game() wrapper may fail if steam is not accessible)"
fi

# 4. Device Permissions (evdev / uinput)
print_header "4. Hardware & Device Permissions (/dev/uinput & /dev/input)"

# Check /dev/uinput
if [ -e /dev/uinput ]; then
    if [ -w /dev/uinput ]; then
        status_ok "/dev/uinput exists and is writable by current user (Virtual mouse input enabled)"
    else
        status_warn "/dev/uinput exists but is NOT writable by current user ($(whoami))"
        echo -e "         ${BLUE}-> Fix:${NC} Configure udev rules (do not run the recorder itself with sudo):"
        echo -e "             echo 'KERNEL==\"uinput\", MODE=\"0660\", GROUP=\"input\", OPTIONS+=\"static_node=uinput\"' | sudo tee /etc/udev/rules.d/99-uinput.rules"
        echo -e "             sudo usermod -aG input \$USER && sudo udevadm control --reload-rules && sudo udevadm trigger"
    fi
else
    status_warn "/dev/uinput does not exist. (Kernel module 'uinput' may need loading: sudo modprobe uinput)"
fi

# Check /dev/input permissions
INPUT_READABLE=0
EVENT_DEV_COUNT=0
if [ -d /dev/input ]; then
    for dev in /dev/input/event*; do
        if [ -e "$dev" ]; then
            EVENT_DEV_COUNT=$((EVENT_DEV_COUNT + 1))
            if [ -r "$dev" ]; then
                INPUT_READABLE=$((INPUT_READABLE + 1))
            fi
        fi
    done
fi

if [ "$EVENT_DEV_COUNT" -eq 0 ]; then
    status_warn "No /dev/input/event* devices found."
elif [ "$INPUT_READABLE" -eq "$EVENT_DEV_COUNT" ]; then
    status_ok "Read access granted to all detected event devices ($INPUT_READABLE/$EVENT_DEV_COUNT) (Hardware tracking enabled)"
elif [ "$INPUT_READABLE" -gt 0 ]; then
    status_warn "Partial read access to event devices ($INPUT_READABLE/$EVENT_DEV_COUNT readable)."
else
    status_warn "Cannot read /dev/input/event* devices without root privileges."
    echo -e "         ${BLUE}-> Fix:${NC} Add user to the 'input' group: sudo usermod -aG input \$USER (then log out and back in)"
    echo -e "         ${BLUE}-> Then:${NC} Log out and back in before running the recorder without sudo."
fi

# Check user group membership
if groups "$USER" 2>/dev/null | grep -q '\binput\b'; then
    status_ok "User '$USER' is a member of the 'input' group"
else
    status_warn "User '$USER' is NOT currently in the 'input' group"
fi

# Show every event device and apply the recorder's own selection rules. This
# makes multi-interface mice (pointer, keyboard, receiver, etc.) distinguishable
# without requiring the recorder or the game to be running.
print_header "5. Recorder Input Device Discovery"
echo "  The AUTO-SELECTED labels show what a default recorder run would capture."
if [ -z "$PYTHON_BIN" ]; then
    echo "    unavailable: no Python interpreter was found"
elif ! "$PYTHON_BIN" -c "import evdev" >/dev/null 2>&1; then
    echo "    unavailable: the evdev Python package is not installed"
else
    "$PYTHON_BIN" - <<'PY'
import glob
from pathlib import Path

import evdev

from recorder import _is_keyboard_device, _is_pointer_device, _pointer_score


def sysfs_name(path):
    name_file = Path("/sys/class/input") / Path(path).name / "device/name"
    try:
        return name_file.read_text().strip()
    except OSError:
        return "unknown device"


def relevant_capabilities(device):
    capabilities = device.capabilities()
    labels = []
    checks = (
        (evdev.ecodes.EV_REL, evdev.ecodes.REL_X, "REL_X"),
        (evdev.ecodes.EV_REL, evdev.ecodes.REL_Y, "REL_Y"),
        (evdev.ecodes.EV_ABS, evdev.ecodes.ABS_X, "ABS_X"),
        (evdev.ecodes.EV_ABS, evdev.ecodes.ABS_Y, "ABS_Y"),
        (evdev.ecodes.EV_KEY, evdev.ecodes.BTN_LEFT, "BTN_LEFT"),
        (evdev.ecodes.EV_KEY, evdev.ecodes.BTN_RIGHT, "BTN_RIGHT"),
        (evdev.ecodes.EV_KEY, evdev.ecodes.KEY_W, "KEY_W"),
        (evdev.ecodes.EV_KEY, evdev.ecodes.KEY_A, "KEY_A"),
    )
    for event_type, code, label in checks:
        if code in capabilities.get(event_type, []):
            labels.append(label)
    return ", ".join(labels) or "no recorder-relevant axes/keys"


device_rows = []
opened_devices = []
for path in sorted(glob.glob("/dev/input/event*")):
    try:
        device = evdev.InputDevice(path)
    except OSError as error:
        device_rows.append((path, sysfs_name(path), None, str(error)))
        continue

    opened_devices.append(device)
    device_rows.append((path, device.name or "unnamed device", device, None))

pointer_candidates = [
    device for device in opened_devices if _is_pointer_device(device)
]
captured_pointer = (
    max(pointer_candidates, key=_pointer_score) if pointer_candidates else None
)
captured_keyboards = {
    device.path for device in opened_devices if _is_keyboard_device(device)
}

for path, name, device, error in device_rows:
    print(f"    {path}: {name}")
    if error is not None:
        print(f"      UNREADABLE - not captured ({error})")
        continue

    roles = []
    if _is_pointer_device(device):
        roles.append("pointer candidate")
    if _is_keyboard_device(device):
        roles.append("keyboard candidate")
    if captured_pointer is not None and path == captured_pointer.path:
        roles.append("AUTO-SELECTED POINTER")
    if path in captured_keyboards:
        roles.append("AUTO-SELECTED KEYBOARD")
    if not roles:
        roles.append("ignored by recorder")
    print(f"      {', '.join(roles)}; {relevant_capabilities(device)}")

print("\n    Recorder auto-selection:")
if captured_pointer is None:
    print("      Pointer: NONE")
else:
    print(f"      Pointer: {captured_pointer.path} ({captured_pointer.name})")
    print(
        "      Override: "
        f"--mouse-device {captured_pointer.path} (or a unique name fragment)"
    )
if captured_keyboards:
    for path in sorted(captured_keyboards):
        keyboard = next(device for device in opened_devices if device.path == path)
        print(f"      Keyboard: {path} ({keyboard.name})")
else:
    print("      Keyboard: NONE")

for device in opened_devices:
    device.close()
PY
fi

# Summary
print_header "Summary"
echo -e "Passed:   ${GREEN}$PASS_COUNT${NC}"
echo -e "Warnings: ${YELLOW}$WARN_COUNT${NC}"
echo -e "Failures: ${RED}$FAIL_COUNT${NC}"
echo "=================================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "${RED}${BOLD}Some required dependencies are missing.${NC} Run './install_dependencies.sh' to set them up."
    exit 1
elif [ "$WARN_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}${BOLD}Core Python requirements met, but some system tools or permissions have warnings.${NC}"
    exit 0
else
    echo -e "${GREEN}${BOLD}All dependencies and environment checks passed!${NC}"
    exit 0
fi
