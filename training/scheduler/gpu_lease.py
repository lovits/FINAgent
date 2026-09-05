"""Cross-process GPU lease for this project's single-GPU training/evaluation."""
from contextlib import contextmanager
from pathlib import Path


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
