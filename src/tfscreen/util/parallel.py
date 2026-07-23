"""
Shared helpers for CPU-parallel work dispatch across CLIs.
"""

import os


def resolve_workers(num_workers):
    """
    Resolve a joblib-style worker count to a concrete number of processes.

    Parameters
    ----------
    num_workers : int or None
        ``None`` or ``1`` -> serial (returns 1); ``-1`` (or any negative) ->
        ``os.cpu_count() - 1`` (at least 1); ``N`` -> ``N``.

    Returns
    -------
    int
        Concrete, positive worker count (>= 1).
    """
    if num_workers is None or int(num_workers) == 1:
        return 1
    if int(num_workers) < 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return int(num_workers)
