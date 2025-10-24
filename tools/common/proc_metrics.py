# tools/common/proc_metrics.py
import os
import time
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple


def run_and_measure(
    cmd: Iterable[str] | str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    poll_interval: float = 0.02,
    use_uss: bool = False,
) -> Tuple[float, Optional[int], str, str, int]:
    """
    Run *cmd* while measuring wall time and peak memory (RSS or USS),
    enforcing a deterministic, single-threaded environment unless overridden.

    Parameters
    ----------
    cmd : sequence[str] | str
        Command to execute (list/tuple preferred; string executes via shell rules of Popen).
    cwd : str | None
        Working directory for the subprocess.
    env : dict[str, str] | None
        Extra environment variables (merged on top of safe defaults below).
    poll_interval : float
        How often (seconds) to poll process memory while running.
    use_uss : bool
        If True, report USS (unique set size). Otherwise report RSS.

    Returns
    -------
    elapsed_s : float
        Wall-clock seconds.
    peak_bytes : int | None
        Peak memory in bytes across the process and its children (None if psutil unavailable).
    stdout : str
        Captured standard output (UTF-8, errors ignored).
    stderr : str
        Captured standard error (UTF-8, errors ignored).
    returncode : int
        Process return code.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore

    # --- Deterministic, single-threaded defaults (caller may override) ---
    ENV_DEFAULTS = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "GDAL_NUM_THREADS": "1",
        # Some OpenJPEG/JP2OpenJPEG builds:
        "OPENJPEG_NUM_THREADS": "1",
        "OPJ_NUM_THREADS": "1",
    }
    env_final = os.environ.copy()
    for k, v in ENV_DEFAULTS.items():
        env_final.setdefault(k, v)
    if env:  # caller overrides
        env_final.update(env)

    # --- Launch subprocess ---
    p = subprocess.Popen(
        cmd, cwd=cwd, env=env_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    start = time.perf_counter()
    peak = 0
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(p.pid)
        except Exception:
            proc = None

    def _mem_of(pr: Any) -> int:
        if use_uss:
            # memory_full_info().uss may not exist on all platforms; fall back to rss
            fi = getattr(pr, "memory_full_info", None)
            if fi is not None:
                try:
                    return getattr(fi(), "uss", pr.memory_info().rss)
                except Exception:
                    return pr.memory_info().rss
        return pr.memory_info().rss

    # Poll until process exits, tracking peak memory of the tree
    while True:
        if p.poll() is not None:
            break
        if proc is not None:
            try:
                mem = _mem_of(proc)
                for c in proc.children(recursive=True):
                    mem += _mem_of(c)
                if mem > peak:
                    peak = mem
            except Exception:
                pass
        time.sleep(poll_interval)

    out_b, err_b = p.communicate()
    elapsed = time.perf_counter() - start
    stdout = (out_b or b"").decode("utf-8", errors="ignore")
    stderr = (err_b or b"").decode("utf-8", errors="ignore")

    return elapsed, (peak or None), stdout, stderr, p.returncode


def bytes_to_mib(nbytes: Optional[int]) -> Optional[float]:
    """Convert bytes to MiB rounded to two decimals (None → None)."""
    if nbytes is None:
        return None
    return round(nbytes / (1024 * 1024), 2)
