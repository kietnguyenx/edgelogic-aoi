"""Environment verification script for EdgeLogic-AOI.

Checks:
1. Python executable & version (ensures .venv is active)
2. PyTorch + CUDA + RTX GPU compute smoke test
3. Intel OpenVINO runtime & available hardware devices
4. Core ML / Edge / MLOps dependencies
"""

from __future__ import annotations

import importlib
import sys
import time


def check_package(import_name: str, display_name: str | None = None) -> tuple[bool, str]:
    label = display_name or import_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "installed (no __version__)")
        return True, f"[OK]   {label:<15} : {version}"
    except Exception as exc:
        return False, f"[FAIL] {label:<15} : {exc}"


def main() -> None:
    print("=" * 60)
    print("EdgeLogic-AOI — Environment Health Check")
    print("=" * 60)

    # 1. Python info
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {sys.executable}")
    print("-" * 60)

    # 2. PyTorch & CUDA check
    ok_torch, msg_torch = check_package("torch", "PyTorch")
    print(msg_torch)
    if ok_torch:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_ok}")
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Device        : {gpu_name} ({vram_gb:.2f} GB VRAM)")
            print(f"CUDA Runtime      : {torch.version.cuda}")

            # Quick GPU matmul smoke test
            t0 = time.perf_counter()
            a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
            b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
            _ = a @ b
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"CUDA Smoke Test   : PASS (1024x1024 FP16 matmul in {elapsed_ms:.2f} ms)")
        else:
            print("[WARN] PyTorch is installed WITHOUT CUDA! Reinstall with --index-url .../cu124")

    print("-" * 60)

    # 3. Core libraries check
    packages = [
        ("torchvision", "torchvision"),
        ("anomalib", "anomalib"),
        ("openvino", "OpenVINO"),
        ("nncf", "Intel NNCF"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("mlflow", "MLflow"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
    ]

    all_ok = ok_torch
    for mod_name, label in packages:
        ok, msg = check_package(mod_name, label)
        print(msg)
        all_ok = all_ok and ok

    # 4. OpenVINO devices check
    print("-" * 60)
    try:
        from openvino import Core

        devices = Core().available_devices
        print(f"OpenVINO Devices  : {devices}")
    except Exception as exc:
        print(f"[FAIL] OpenVINO Core : {exc}")
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("STATUS: READY FOR PHASE 1 (Data & Baseline)!")
    else:
        print("STATUS: SOME DEPENDENCIES FAILED — Check [FAIL] lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
