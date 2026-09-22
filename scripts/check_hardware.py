#!/usr/bin/env python3
"""Inventory WSL resources and validate a small PyTorch CUDA calculation.

Run with the project's isolated Python. No datasets or models are downloaded.
Raw machine paths belong in an ignored report, not in version control.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_meminfo() -> dict[str, int]:
    """Linux /proc/meminfo uses kB to mean KiB; report bytes explicitly."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(value.split()[0]) * 1024
    return values


def cgroup_memory() -> dict:
    """Read the current cgroup and visible ancestors, including parent limits."""
    unified = next(
        (line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
         if line.startswith("0::")),
        None,
    )
    if unified is None:
        return {"available": False, "reason": "No cgroup v2 entry is visible."}
    mount = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] == "cgroup2":
            fields = before.split()
            mount = (Path(fields[3]), Path(fields[4]))
            break
    if mount is None:
        return {"available": False, "reason": "No cgroup v2 mount is visible."}
    mount_root, mount_point = mount
    try:
        relative = Path(unified).relative_to(mount_root)
    except ValueError:
        return {"available": False, "reason": "Cannot map the current cgroup to its mount."}
    current = mount_point / relative
    scopes, headrooms = [], []
    while True:
        scope = {"path": str(current), "values_bytes": {}}
        for name in ("memory.max", "memory.high", "memory.current",
                     "memory.swap.max", "memory.swap.current"):
            try:
                raw = (current / name).read_text().strip()
                scope["values_bytes"][name] = raw if raw == "max" else int(raw)
            except FileNotFoundError:
                continue
            except OSError as exc:
                scope.setdefault("read_errors", {})[name] = str(exc)
        maximum = scope["values_bytes"].get("memory.max")
        usage = scope["values_bytes"].get("memory.current")
        if isinstance(maximum, int) and isinstance(usage, int):
            remaining = max(0, maximum - usage)
            scope["hard_limit_headroom_bytes"] = remaining
            headrooms.append(remaining)
        scopes.append(scope)
        if current == mount_point:
            break
        current = current.parent
    return {
        "available": True,
        "current_cgroup": unified,
        "visible_scopes": scopes,
        "minimum_hard_limit_headroom_bytes": min(headrooms) if headrooms else None,
        "note": "Snapshot of visible limits; memory.high is a throttle, not a hard limit.",
    }


def python_environment() -> dict:
    config = Path(sys.prefix) / "pyvenv.cfg"
    settings = {}
    if config.is_file():
        for line in config.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_venv": sys.prefix != sys.base_prefix,
        "include_system_site_packages": settings.get("include-system-site-packages"),
        "sys_path": sys.path,
        "pythonpath_environment": os.environ.get("PYTHONPATH"),
    }


def nvidia_inventory() -> dict:
    wsl_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    command = str(wsl_smi) if wsl_smi.is_file() else shutil.which("nvidia-smi")
    if command is None:
        return {"available": False, "reason": "nvidia-smi was not found."}
    fields = ["index", "name", "memory.total", "memory.free", "memory.used", "driver_version"]
    try:
        result = subprocess.run(
            [command, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=20,
        )
        devices = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            device = dict(zip(fields, (item.strip() for item in row), strict=True))
            for name in ("memory.total", "memory.free", "memory.used"):
                raw = device.pop(name)
                device[name.replace(".", "_") + "_bytes"] = (
                    int(float(raw) * 1024**2) if raw.replace(".", "", 1).isdigit() else raw
                )
            devices.append(device)
        return {"available": True, "command": command, "devices": devices}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}


def inventory() -> dict:
    cpu_model = next(
        (line.split(":", 1)[1].strip()
         for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")),
        platform.processor(),
    )
    memory = read_meminfo()
    cgroup = cgroup_memory()
    headroom = cgroup.get("minimum_hard_limit_headroom_bytes")
    effective_available = memory["MemAvailable"]
    if headroom is not None:
        effective_available = min(effective_available, headroom)
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "kernel_release": platform.release(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "cpu": {
            "model": cpu_model,
            "logical_count": os.cpu_count(),
            "affinity_count": len(os.sched_getaffinity(0)),
        },
        "memory_bytes": memory,
        "cgroup_v2": cgroup,
        "estimated_available_ram_bytes": effective_available,
        "ram_note": "Minimum of WSL MemAvailable and visible cgroup hard-limit headroom; not a reservation.",
        "disk": {"path": str(PROJECT_ROOT), "total_bytes": disk.total,
                 "used_bytes": disk.used, "free_bytes": disk.free},
        "python": python_environment(),
        "nvidia_smi": nvidia_inventory(),
    }


def validate_cuda(report: dict) -> None:
    environment = report["python"]
    if not environment["is_venv"] or environment["include_system_site_packages"] != "false":
        raise RuntimeError(
            "Use .venv/bin/python with include-system-site-packages = false; "
            "the diagnostic refuses a shared Python environment."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the project's pinned PyTorch dependencies in .venv first.") from exc
    torch_path = Path(torch.__file__).resolve()
    if not torch_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"PyTorch was imported from outside this virtual environment: {torch_path}")
    result = {
        "status": "running",
        "torch_version": torch.__version__,
        "torch_path": str(torch_path),
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["cuda_check"] = result
    if not result["cuda_available"]:
        raise RuntimeError(
            "PyTorch cannot access CUDA. Check the CUDA-enabled wheel, Windows NVIDIA driver, "
            "WSL GPU passthrough and CUDA_VISIBLE_DEVICES; there is no CPU fallback."
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    result.update({
        "device": str(device),
        "device_name": properties.name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_total_memory_bytes": properties.total_memory,
        "visible_device_count": torch.cuda.device_count(),
    })
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    size = 512
    generator = torch.Generator(device="cpu").manual_seed(2026)
    left = (torch.randn(size, size, generator=generator) / math.sqrt(size)).requires_grad_()
    right = (torch.randn(size, size, generator=generator) / math.sqrt(size)).requires_grad_()
    reference = left @ right
    reference.square().sum().div(size).backward()
    free_before, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_left = left.detach().to(device).requires_grad_()
    gpu_right = right.detach().to(device).requires_grad_()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = gpu_left @ gpu_right
    loss = output.square().sum().div(size)
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    errors = {}
    for name, actual, expected, atol in (
        ("matmul", output, reference, 1e-5),
        ("left_gradient", gpu_left.grad, left.grad, 1e-7),
        ("right_gradient", gpu_right.grad, right.grad, 1e-7),
    ):
        if actual is None or expected is None:
            raise RuntimeError(f"Missing gradient for {name}.")
        actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach()
        if not torch.isfinite(actual_cpu).all().item():
            raise RuntimeError(f"Non-finite values in {name}.")
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=1e-4, atol=atol)
        errors[name] = {"max_absolute_error": (actual_cpu - expected_cpu).abs().max().item(),
                        "rtol": 1e-4, "atol": atol}
    free_after, _ = torch.cuda.mem_get_info(device)
    result.update({
        "status": "passed",
        "shape": [size, size],
        "dtype": "float32",
        "seed": 2026,
        "cpu_threads": torch.get_num_threads(),
        "tf32_enabled": False,
        "loss": loss.item(),
        "comparisons": errors,
        "synchronized_forward_backward_seconds": elapsed,
        "timing_note": "Single cold diagnostic calculation, excluding transfers and comparisons; not a benchmark.",
        "memory_bytes": {
            "global_free_before": free_before,
            "global_free_after": free_after,
            "global_total": total,
            "pytorch_peak_allocated": torch.cuda.max_memory_allocated(device),
            "pytorch_peak_reserved": torch.cuda.max_memory_reserved(device),
        },
        "memory_note": "PyTorch allocator peaks exclude driver/context and other processes; global free is shared.",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file.")
    parser.add_argument("--inventory-only", action="store_true", help="Use only the standard library; skip CUDA math.")
    args = parser.parse_args()
    report = {}
    exit_code = 0
    try:
        report = inventory()
        if args.inventory_only:
            report["cuda_check"] = {"status": "not_run", "reason": "--inventory-only"}
        else:
            validate_cuda(report)
    except Exception as exc:
        exit_code = 1
        report["error"] = f"{type(exc).__name__}: {exc}"
        report.setdefault("cuda_check", {})["status"] = "failed"
        print(f"Hardware check failed: {report['error']}", file=sys.stderr)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        except OSError as exc:
            print(f"Cannot write report {args.output}: {exc}", file=sys.stderr)
            exit_code = 1
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
