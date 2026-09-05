"""Cross-process GPU lease for this project's single-GPU training/evaluation."""
from contextlib import contextmanager
from pathlib import Path


def set_gpu_budget(gib: float) -> float:
    """Limit this process's PyTorch allocator, not other processes or all CUDA allocations."""
    import math
    import torch

    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("GPU budget must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    fraction = gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    if fraction >= 1:
        raise ValueError("GPU budget must leave device memory headroom")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


@contextmanager
def gpu_lease(path=Path("artifacts/scheduler/gpu-compute.lock")):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
