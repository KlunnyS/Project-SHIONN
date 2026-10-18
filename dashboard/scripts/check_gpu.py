import sys

import torch
import torchvision

expected = {
    "12.6": ("2.14.0", "0.29.0"),
    "13.0": ("2.14.1", "0.29.1"),
}
installed = (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0])

print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")

if expected.get(torch.version.cuda) != installed:
    print("Reinstall with setup_gpu using the legacy or modern wheel.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA is unavailable. Check the NVIDIA driver and GPU.")
    sys.exit(1)

from backend.devices import device_report

report = device_report()
if not report.get("supported"):
    print(report.get("message", "This GPU is not supported by the installed wheel."))
    sys.exit(1)

print(f"GPU: {report['name']} (sm_{report['capability'][0]}{report['capability'][1]})")

try:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 32 * 32, 10),
    ).cuda()
    batch = torch.randn(2, 3, 32, 32, device="cuda")
    model(batch).square().mean().backward()
    torch.cuda.synchronize()
except Exception as exc:
    print(f"CUDA model operation failed: {exc}")
    sys.exit(1)

print("CUDA forward and backward passes succeeded.")
