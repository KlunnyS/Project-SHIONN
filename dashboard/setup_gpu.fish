#!/usr/bin/env fish

set project_dir (dirname (status --current-filename))
cd $project_dir

set gpu_mode auto
if test (count $argv) -gt 1
    echo "Usage: ./setup_gpu.fish [auto|legacy|modern]"
    exit 2
else if test (count $argv) -eq 1
    set gpu_mode $argv[1]
end

if test "$gpu_mode" = auto
    if not type -q nvidia-smi
        echo "nvidia-smi was not found. Specify a wheel family: legacy or modern."
        exit 1
    end
    set capabilities (nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader | string trim)
    if test (count $capabilities) -eq 0
        echo "Could not read GPU compute capability. Specify legacy or modern explicitly."
        exit 1
    end
    set gpu_mode modern
    for capability in $capabilities
        set parts (string split . -- $capability)
        if test (count $parts) -lt 2
            echo "Could not parse GPU compute capability '$capability'."
            exit 1
        end
        if test $parts[1] -lt 7; or test $parts[1] -eq 7 -a $parts[2] -lt 5
            set gpu_mode legacy
            break
        end
    end
end

if not contains -- $gpu_mode legacy modern
    echo "Unknown GPU wheel family '$gpu_mode'. Use auto, legacy, or modern."
    exit 2
end

if not type -q python3.11
    echo "Python 3.11 is required but was not found."
    exit 1
end

echo "Creating isolated GPU environment .venv-gpu..."
python3.11 -m venv .venv-gpu
or exit 1

set python_path .venv-gpu/bin/python

echo "Updating pip..."
$python_path -m pip install --upgrade pip
or exit 1

if test "$gpu_mode" = legacy
    echo "Installing PyTorch 2.14.0 / torchvision 0.29.0 with CUDA 12.6 for GTX 10-series and newer..."
    $python_path -m pip install \
        torch==2.14.0 \
        torchvision==0.29.0 \
        --index-url https://download.pytorch.org/whl/cu126
else
    echo "Installing PyTorch 2.14.1 / torchvision 0.29.1 with CUDA 13.0 for RTX 20-series through RTX 50-series..."
    $python_path -m pip install \
        torch==2.14.1 \
        torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cu130
end
or exit 1

echo "Installing SHIONN runtime dependencies..."
$python_path -m pip install -r ../requirements.txt
or exit 1

echo "Installing dashboard dependencies..."
$python_path -m pip install -r requirements-gpu.txt
or exit 1

echo
echo "GPU environment is ready. Validate the GPU and start the app with:"
echo "  ./run_gpu.fish"
