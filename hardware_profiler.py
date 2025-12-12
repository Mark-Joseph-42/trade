import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NodeRole(Enum):
    GPU_LEARNER = "gpu_learner"
    CPU_ACTOR = "cpu_actor"


@dataclass
class HardwareProfile:
    compute_score: int
    role: NodeRole
    num_cpu_cores: int
    has_cuda: bool
    gpu_name: Optional[str]
    total_vram_gb: Optional[float]


def _detect_gpu_with_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        name = props.name
        total_vram_gb = round(props.total_memory / (1024 ** 3), 2)
        multi_proc = getattr(props, "multi_processor_count", 0)
        compute_score = int(total_vram_gb * 8 + multi_proc * 1.5)
        return name, total_vram_gb, compute_score
    except Exception:
        return None


def _detect_gpu_with_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            return None
        name = parts[0]
        vram_gb = float(parts[1]) / 1024.0
        total_vram_gb = round(vram_gb, 2)
        compute_score = int(total_vram_gb * 10)
        return name, total_vram_gb, compute_score
    except Exception:
        return None


def get_hardware_profile() -> HardwareProfile:
    cpu_cores = os.cpu_count() or 1
    gpu_info = _detect_gpu_with_torch()
    if gpu_info is None:
        gpu_info = _detect_gpu_with_nvidia_smi()
    if gpu_info is not None:
        gpu_name, total_vram_gb, gpu_score = gpu_info
        compute_score = max(gpu_score, int(cpu_cores * 5))
        role = NodeRole.GPU_LEARNER if compute_score >= 100 else NodeRole.CPU_ACTOR
        return HardwareProfile(
            compute_score=compute_score,
            role=role,
            num_cpu_cores=cpu_cores,
            has_cuda=True,
            gpu_name=gpu_name,
            total_vram_gb=total_vram_gb,
        )
    cpu_score = int(cpu_cores * 5)
    return HardwareProfile(
        compute_score=cpu_score,
        role=NodeRole.CPU_ACTOR,
        num_cpu_cores=cpu_cores,
        has_cuda=False,
        gpu_name=None,
        total_vram_gb=None,
    )
