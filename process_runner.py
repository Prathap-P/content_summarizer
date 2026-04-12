import multiprocessing
import threading
from typing import Any, Callable

# Only one model subprocess runs at a time — prevents simultaneous model loads crashing RAM.
_model_lock = threading.Semaphore(1)


def _worker(fn: Callable, args: tuple, kwargs: dict, q) -> None:
    # Module-level so spawn can pickle it.
    try:
        result = fn(*args, **kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put(("error", str(e)))


def run_in_subprocess(fn: Callable, *args, **kwargs) -> Any:
    """Run fn(*args, **kwargs) in a fresh spawn subprocess and return its result.

    Acquires a global semaphore so only one model subprocess is in memory at a
    time. Blocks indefinitely — no timeout — because audio/ASR jobs on long
    videos can legitimately take 20+ minutes. When the subprocess exits the OS
    reclaims all its memory (MPS buffers, PyTorch allocator, MLX Metal pool).
    Raises RuntimeError on subprocess failure.
    """
    ctx = multiprocessing.get_context("spawn")
    _model_lock.acquire()
    try:
        q = ctx.Queue()
        p = ctx.Process(target=_worker, args=(fn, args, kwargs, q))
        p.start()
        p.join()
        status, payload = q.get()
        if status == "error":
            raise RuntimeError(payload)
        return payload
    finally:
        _model_lock.release()
