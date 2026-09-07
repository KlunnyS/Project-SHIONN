#!/usr/bin/env bash
# ==============================================================================
# Project-SHIONN Dependency Installer
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

# Activate or target venv pip
PIP_BIN="$VENV_DIR/bin/pip"
PYTHON_BIN="$VENV_DIR/bin/python"

echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip --quiet

echo "Installing requirements from requirements.txt..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    "$PIP_BIN" install -r "$SCRIPT_DIR/requirements.txt"
    echo -e "${GREEN}Python packages successfully installed into .venv!${NC}"
else
    echo -e "${YELLOW}requirements.txt not found. Installing base packages directly...${NC}"
    "$PIP_BIN" install evdev numpy opencv-python
fi

# 3. System Packages (wf-recorder)
echo -e "\n${BOLD}[Step 3/4] Checking system packages (wf-recorder)...${NC}"
if command -v wf-recorder >/dev/null 2>&1; then
    echo -e "${GREEN}wf-recorder is already installed.${NC}"
else
    if [ "$INSTALL_SYSTEM" = true ]; then
        echo "Installing wf-recorder via system package manager..."
        if [ -f /etc/arch-release ]; then
            sudo pacman -S --needed --noconfirm wf-recorder
        elif [ -f /etc/debian_version ]; then
            sudo apt-get update && sudo apt-get install -y wf-recorder
        elif [ -f /etc/fedora-release ]; then
            sudo dnf install -y wf-recorder
        else
            echo -e "${YELLOW}Unsupported package manager. Please install wf-recorder manually.${NC}"
        fi
    else
        echo -e "${YELLOW}wf-recorder is not installed.${NC}"
        echo "It is required for screen recording on Wayland."
        if [ -t 0 ]; then
            read -r -p "Would you like to install wf-recorder now? [y/N] " response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                if [ -f /etc/arch-release ]; then
                    sudo pacman -S --needed wf-recorder
                elif [ -f /etc/debian_version ]; then
                    sudo apt-get update && sudo apt-get install -y wf-recorder
                elif [ -f /etc/fedora-release ]; then
                    sudo dnf install -y wf-recorder
                fi
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
if ! groups "$USER" 2>/dev/null | grep -q '\binput\b'; then
    NEEDS_PERMS=true
fi
if [ ! -w /dev/uinput 2>/dev/null ]; then
    NEEDS_PERMS=true
fi

apply_permissions() {
    echo "Configuring /etc/udev/rules.d/99-uinput.rules and group membership..."
    sudo usermod -aG input "$USER"
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
        read -r -p "Would you like to configure udev rules and add '$USER' to the 'input' group? [y/N] " response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            apply_permissions
        else
            echo "Skipping permission configuration. (You can run recorder.py with sudo instead)."
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
