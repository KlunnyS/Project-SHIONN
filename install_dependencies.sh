#!/usr/bin/env bash
# ==============================================================================
# Project-SHIONN Dependency Installer (Python packages, optional system setup)
# ==============================================================================

set -euo pipefail

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
BOLD="\033[1m"
NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_SYSTEM=false
SETUP_PERMS=false
current_user="$(id -un)"

confirm() {
    # A prompted system change proceeds only after an explicit yes.
    local response
    read -r -p "$1 [y/N] " response
    [[ "$response" =~ ^([yY]|[yY][eE][sS])$ ]]
}

install_wf_recorder() {
    # Keep package-manager selection in one place for prompted and flag-driven installs.
    if [ -f /etc/arch-release ]; then
        sudo pacman -S --needed --noconfirm wf-recorder
    elif [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y wf-recorder
    elif [ -f /etc/fedora-release ]; then
        sudo dnf install -y wf-recorder
    else
        echo -e "${RED}Unsupported package manager; install wf-recorder manually.${NC}" >&2
        return 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)
            INSTALL_SYSTEM=true
            shift
            ;;
        --permissions)
            SETUP_PERMS=true
            shift
            ;;
        --all)
            INSTALL_SYSTEM=true
            SETUP_PERMS=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./install_dependencies.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --system       Install system packages (wf-recorder) using system package manager (requires sudo)"
            echo "  --permissions  Configure udev rules & add user to 'input' group for hardware access (requires sudo)"
            echo "  --all          Install Python packages, system packages, and configure permissions"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${BLUE}==================================================${NC}"
echo -e "${BOLD}Project-SHIONN: Dependency Installation${NC}"
echo -e "${BOLD}${BLUE}==================================================${NC}"

# 1. Verify Python 3
echo -e "\n${BOLD}[Step 1/4] Checking Python 3...${NC}"
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: python3 is not installed or not in PATH.${NC}"
    exit 1
fi
echo -e "${GREEN}Python 3 found: $(python3 --version)${NC}"

# 2. Setup Virtual Environment
echo -e "\n${BOLD}[Step 2/4] Setting up Python virtual environment (.venv)...${NC}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}Existing virtual environment found at $VENV_DIR.${NC}"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip --quiet

echo "Installing requirements from requirements.txt..."
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo -e "${RED}requirements.txt is missing; refusing an incomplete installation.${NC}" >&2
    exit 1
fi
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
echo -e "${GREEN}Python packages successfully installed into .venv!${NC}"

# 3. System Packages (wf-recorder)
echo -e "\n${BOLD}[Step 3/4] Checking system packages (wf-recorder)...${NC}"
if command -v wf-recorder >/dev/null 2>&1; then
    echo -e "${GREEN}wf-recorder is already installed.${NC}"
else
    if [ "$INSTALL_SYSTEM" = true ]; then
        echo "Installing wf-recorder via system package manager..."
        install_wf_recorder
    else
        echo -e "${YELLOW}wf-recorder is not installed.${NC}"
        echo "It is required for screen recording on Wayland."
        if [ -t 0 ]; then
            if confirm "Would you like to install wf-recorder now?"; then
                install_wf_recorder
            else
                echo "Skipping wf-recorder installation. You can install it later or run with --system."
            fi
        else
            echo "To install automatically, rerun with: ./install_dependencies.sh --system"
        fi
    fi
fi

# 4. Permissions Setup (/dev/uinput and /dev/input)
echo -e "\n${BOLD}[Step 4/4] Hardware permissions & udev rules...${NC}"
NEEDS_PERMS=false
if ! id -nG | tr ' ' '\n' | grep -Fxq input; then
    NEEDS_PERMS=true
fi
if [ ! -w /dev/uinput 2>/dev/null ]; then
    NEEDS_PERMS=true
fi

apply_permissions() {
    # Persistent udev/group changes need sudo; never run the recorder itself as root.
    echo "Configuring /etc/udev/rules.d/99-uinput.rules and group membership..."
    sudo usermod -aG input "$current_user"
    echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.rules >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    # Also ensure module is loaded
    sudo modprobe uinput 2>/dev/null || true
    echo -e "${GREEN}Permissions configured!${NC}"
    echo -e "${YELLOW}Note: You may need to log out and log back in (or run 'newgrp input') for group changes to take effect.${NC}"
}

if [ "$SETUP_PERMS" = true ]; then
    apply_permissions
elif [ "$NEEDS_PERMS" = true ]; then
    echo -e "${YELLOW}Notice: Current user lacks permission to access /dev/input devices without sudo.${NC}"
    if [ -t 0 ]; then
        if confirm "Would you like to configure udev rules and add '$current_user' to the 'input' group?"; then
            apply_permissions
        else
            echo "Skipping permission configuration. Do not run the recorder with sudo; wf-recorder needs your desktop Wayland session."
        fi
    else
        echo "To configure permissions automatically, rerun with: ./install_dependencies.sh --permissions"
    fi
else
    echo -e "${GREEN}Hardware permissions appear to be properly configured.${NC}"
fi

# Run verification check
echo -e "\n${BOLD}${BLUE}==================================================${NC}"
echo -e "${BOLD}Running Verification Check...${NC}"
echo -e "${BOLD}${BLUE}==================================================${NC}"
"$SCRIPT_DIR/check_dependencies.sh"
