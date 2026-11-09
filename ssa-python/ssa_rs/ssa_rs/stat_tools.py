"""Statistical tools (same API as ``py_ssa_lib.stat_tools``), computed in Rust."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _core


def bootstrap_prediction_intervals(
    residuals: NDArray[np.float64],
    forecasts: NDArray[np.float64],
    alpha: float = 0.05,
    N_sims: int = 1000,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Model-free bootstrap prediction intervals (Hyndman & Athanasopoulos, FPP3).

    Same arguments as py-ssa-lib plus an optional ``seed`` (Python draws from
    the global numpy RNG, so individual draws differ between the libraries).
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float).ravel()
    return _core.bootstrap_prediction_intervals(
        residuals.tolist(), forecasts.tolist(), float(alpha), int(N_sims), seed
    )
