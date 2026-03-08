from __future__ import annotations
import os, psutil
from typing import Tuple

# Environment knob names for BLAS backends
_BACKEND_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)

def pin_blas_threads(n: int = 1) -> None:
    """Pin BLAS/NumExpr thread counts via env. Call in parent and each worker."""
    n = max(1, int(n))
    for k in _BACKEND_VARS:
        os.environ[k] = str(n)
    # Optional: try vendor-specific APIs (safe no-ops if missing)
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n)
    except Exception:
        pass
    try:
        import numexpr as ne  # type: ignore
        ne.set_num_threads(n)
    except Exception:
        pass


def human_hw_summary() -> str:
    """Return HW summary incl. GPU details if CuPy can see them."""
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = float('nan')

    g = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_part = "GPU: none"
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            names = []
            for i in range(n):
                props = cp.cuda.runtime.getDeviceProperties(i)
                names.append(props.get('name', f'GPU{i}'))
            free_b, total_b = cp.cuda.Device(0).mem_info
            gpu_part = (f"GPU: {n}x " + ", ".join(names)
                        + f" mem={total_b/1024**3:.1f}GB (free {free_b/1024**3:.1f}GB)")
    except Exception:
        pass

    return (f"CPU threads={os.cpu_count()} RAM={ram_gb:.1f} GB "
            f"CUDA_VISIBLE_DEVICES='{g}' {gpu_part}")


def choose_engine(requested: str) -> str:
    """Return effective engine from requested ('auto'|'numpy'|'cupy'|'numba').
    When 'auto', prefer CuPy only if at least one device is visible.
    """
    req = (requested or 'auto').lower()
    if req != 'auto':
        return req
    try:
        import cupy as cp  # noqa: F401
        n = cp.cuda.runtime.getDeviceCount()
        return 'cupy' if n and n > 0 else 'numpy'
    except Exception:
        return 'numpy'


def max_batch_and_sample(max_ram_gb: float,
                         *,
                         default_batch: int = 256,
                         default_sample_cap: int = 4000) -> Tuple[int, int]:
    """Heuristic for batch size & sampling cap based on available RAM.
    - If max_ram_gb <= 0: return defaults.
    - Otherwise, scale modestly. This avoids over-aggressive memory growth.
    """
    if not max_ram_gb or max_ram_gb <= 0:
        return int(default_batch), int(default_sample_cap)

    # Batch heuristic
    if   max_ram_gb < 4:   batch = min(default_batch, 128)
    elif max_ram_gb < 8:   batch = min(default_batch, 192)
    elif max_ram_gb < 16:  batch = min(default_batch, 256)
    elif max_ram_gb < 32:  batch = max(256, min(default_batch, 320))
    else:                  batch = max(256, default_batch)

    # Sample cap heuristic
    if   max_ram_gb < 4:   cap = min(default_sample_cap, 2000)
    elif max_ram_gb < 8:   cap = min(default_sample_cap, 3000)
    elif max_ram_gb < 16:  cap = default_sample_cap
    elif max_ram_gb < 32:  cap = max(default_sample_cap, 5000)
    else:                  cap = max(default_sample_cap, 8000)

    return int(batch), int(cap)


def worker_initializer() -> None:
    """Initializer for ProcessPoolExecutor workers (Windows 'spawn')."""
    pin_blas_threads(1)
