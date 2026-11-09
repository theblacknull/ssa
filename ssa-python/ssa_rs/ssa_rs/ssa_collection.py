"""Drop-in, Rust-accelerated replacement of ``py_ssa_lib.ssa_collection``.

Same classes (``BaseSSA``, ``SSA``, ``MSSA``), constructor arguments,
attributes and method names as py-ssa-lib 2.0.2; every numerical routine
runs in the ``ssa_rs._core`` extension module (crate ``ssa_rs``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import _core
from .svd_strategies import SVDStrategy, FullSVD
from .weights_strategies import WeightingStrategy, UniformWeights


def _normalize_indices(idx: Any, d: int) -> list[int]:
    """numpy-style component selection -> list of non-negative ints.

    Accepts lists/arrays of ints (negative values count from the end,
    duplicates are kept) or a boolean mask of length ``d``.  A scalar is
    treated as a one-element list.
    """
    arr = np.asarray(idx)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return []
    if arr.dtype == bool:
        if arr.shape != (d,):
            raise IndexError(
                f"boolean index did not match indexed array along axis 0; size of axis is {d} but size of corresponding boolean axis is {arr.shape[0]}"
            )
        return [int(i) for i in np.flatnonzero(arr)]
    if not np.issubdtype(arr.dtype, np.integer):
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) "
            "and integer or boolean arrays are valid indices"
        )
    out = []
    for i in arr.ravel().tolist():
        j = i + d if i < 0 else i
        if not 0 <= j < d:
            raise IndexError(f"index {i} is out of bounds for axis 0 with size {d}")
        out.append(j)
    return out


class BaseSSA(ABC):
    """Base class for each SSA implementation."""

    DEFAULT_DECOMPOSITION = FullSVD
    DEFAULT_WEIGHTING = UniformWeights

    def __init__(
        self,
        decomposition_strategy: SVDStrategy | None = None,
        weighting_strategy: WeightingStrategy | None = None,
    ) -> None:
        self.decomposition_strategy = decomposition_strategy or self.DEFAULT_DECOMPOSITION()
        self.weighting_strategy = weighting_strategy or self.DEFAULT_WEIGHTING()

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        data: pd.Series | pd.DataFrame | NDArray,
        L: int | None = None,
        **weighting_kwargs: Any,
    ) -> "BaseSSA":
        self.X_s = self._prepare_data(data)
        self.N, self.S = self.X_s.shape

        max_L = (self.N + 1) // 2
        self.L = max_L if L is None else L
        if not 2 <= self.L <= max_L:
            raise ValueError("L must satisfy 2 <= L <= floor((N + 1) / 2).")
        self.K = self.N - self.L + 1

        self.weights = np.asarray(
            self.weighting_strategy.compute_weights(self.X_s, **weighting_kwargs), dtype=float
        )

        X = np.ascontiguousarray(self.X_s, dtype=float)
        weights = [float(w) for w in np.ravel(self.weights)]
        spec = getattr(self.decomposition_strategy, "_rust_spec", lambda: None)()
        if spec is None:
            # Custom (pure Python) strategy: decompose in Python, rest in Rust.
            X_ss = _core.trajectory_matrix(X, int(self.L), weights) if self.S else np.empty((self.L, 0))
            U, Sigma, Vt = self.decomposition_strategy.decompose(X_ss)
            self._model = _core.fit(
                X, int(self.L), weights, method="given",
                u=np.ascontiguousarray(U, dtype=float),
                sigma=np.ascontiguousarray(Sigma, dtype=float),
                vt=np.ascontiguousarray(Vt, dtype=float),
            )
        else:
            self._model = _core.fit(X, int(self.L), weights, **spec)

        m = self._model
        self.U = m.u
        self.Sigma = m.sigma
        self.V = m.v
        self.Vt = self.V.T
        self.d = m.d
        self.rel_contribution = m.rel_contribution
        self.cumsum_contr = m.cumsum_contr
        self.__dict__.pop("_X_elem", None)
        self.__dict__.pop("_X_ss", None)
        return self

    @abstractmethod
    def _prepare_data(self, data) -> NDArray[np.float64]:
        pass

    # --------------------------------------------------------- structures
    @property
    def X_ss(self) -> NDArray[np.float64]:
        """Weighted trajectory matrix (L, K*S); built lazily on first access."""
        if "_X_ss" not in self.__dict__:
            self.__dict__["_X_ss"] = self._model.x_ss
        return self.__dict__["_X_ss"]

    @property
    def X_elem(self) -> NDArray[np.float64]:
        """All elementary matrices, shape (d, L, K*S); built lazily on first access."""
        if "_X_elem" not in self.__dict__:
            self.__dict__["_X_elem"] = self._model.elementary_matrices()
        return self.__dict__["_X_elem"]

    def construct_trajectory_matrix(self) -> NDArray[np.float64]:
        return _core.trajectory_matrix(
            np.ascontiguousarray(self.X_s, dtype=float), int(self.L), [float(w) for w in self.weights]
        )

    def elementary_matrix(self) -> NDArray[np.float64]:
        return self._model.elementary_matrices()

    @staticmethod
    def diagonal_averaging(X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("diag requires an array of at least two dimensions")
        return _core.diagonal_averaging(np.ascontiguousarray(X))

    def construct_hankel_weights(self) -> NDArray[np.int_]:
        return self._model.hankel_weights()

    def _reconstruct_elementary_components(self) -> NDArray[np.float64]:
        return self._model.elementary_components()

    def compute_weighted_correlation_matrix(self) -> NDArray[np.float64]:
        return self._model.wcorr()

    def reconstruct_components(self, idx_components) -> NDArray[np.float64]:
        return self._model.reconstruct(_normalize_indices(idx_components, self.d))

    # ------------------------------------------------------ estimation
    def estimate_ESPRIT(self, idx_components):
        re, im, rho, omega = self._model.esprit(_normalize_indices(idx_components, self.d))
        return re + 1j * im, rho, omega

    def estimate_LRR(self, idx_components) -> NDArray[np.float64]:
        return self._model.lrr(_normalize_indices(idx_components, self.d))

    @staticmethod
    def _steps(forecast_steps) -> int:
        if isinstance(forecast_steps, (bool, np.bool_)) or int(forecast_steps) != forecast_steps:
            raise ValueError("forecast_steps must be an integer.")
        if forecast_steps < 0:
            raise ValueError("forecast_steps must be a non-negative integer.")
        return int(forecast_steps)

    def L_forecast(self, forecast_steps: int, idx_components, return_full: bool = True) -> NDArray[np.float64]:
        return self._model.l_forecast(self._steps(forecast_steps), _normalize_indices(idx_components, self.d), bool(return_full))

    def K_forecast(self, forecast_steps: int, idx_components, return_full: bool = True) -> NDArray[np.float64]:
        return self._model.k_forecast(self._steps(forecast_steps), _normalize_indices(idx_components, self.d), bool(return_full))


    # ------------------------------------------- Rssa-compatible extras
    # (not in py-ssa-lib; same results as the R package Rssa)

    def wcor(self, groups=None) -> NDArray[np.float64]:
        """Signed weighted correlations between reconstructed groups
        (one group per component when ``groups`` is None)."""
        return self._model.wcor_signed(None if groups is None else [list(map(int, g)) for g in groups])

    def wnorm(self) -> float:
        return self._model.wnorm()

    def contributions(self) -> NDArray[np.float64]:
        return self._model.contributions()

    def frobenius_cor(self, groups) -> NDArray[np.float64]:
        return self._model.frobenius_cor([list(map(int, g)) for g in groups])

    def residuals(self, idx_components) -> NDArray[np.float64]:
        return self._model.residuals(_normalize_indices(idx_components, self.d))

    def grouping_auto_wcor(self, components, nclust: int) -> list[list[int]]:
        return self._model.grouping_auto_wcor(_normalize_indices(components, self.d), int(nclust))

    def grouping_auto_pgram(self, components, bins=None, threshold=None, base="series",
                            method="constant", drop_empty=True):
        """Periodogram-based grouping: ``(groups, contributions, splitting)``."""
        return self._model.grouping_auto_pgram(
            _normalize_indices(components, self.d), bins, threshold, base, method, drop_empty)

    def parestimate(self, idx_components, subspace="column", method="ls", normalize=False) -> dict:
        """ESPRIT (LS / TLS): dict of ``roots``, ``periods``, ``frequencies``, ``moduli``, ``rates``."""
        return self._model.parestimate(_normalize_indices(idx_components, self.d), subspace, method, normalize)

    def rforecast(self, idx_components, steps, base="reconstructed", direction="column", reverse=False):
        """Recurrent forecast, only the new values, shape ``(steps, S)``."""
        return self._model.rforecast(_normalize_indices(idx_components, self.d), int(steps), base, direction, bool(reverse))

    def vforecast(self, idx_components, steps, direction="column"):
        return self._model.vforecast(_normalize_indices(idx_components, self.d), int(steps), direction)

    def bforecast(self, idx_components, steps, replicates=100, level=0.95, vector=False, prediction=False, seed=0):
        """Bootstrap forecast of one series: ``(mean, lower, upper)``."""
        return self._model.bforecast(_normalize_indices(idx_components, self.d), int(steps), int(replicates),
                                     float(level), bool(vector), bool(prediction), int(seed))

    def forecast(self, idx_components, steps, vector=False, interval="none", level=0.95, replicates=100, seed=0) -> dict:
        """Rssa ``forecast`` / ``predict`` with optional confidence / prediction intervals."""
        return self._model.forecast(_normalize_indices(idx_components, self.d), int(steps), bool(vector),
                                    interval, float(level), int(replicates), int(seed))

    def cadzow(self, rank, correct=True, tol=1e-6, maxiter=0):
        return self._model.cadzow(int(rank), bool(correct), float(tol), int(maxiter))

    def _wrap_ossa(self, res):
        m = object.__new__(type(self))
        m.__dict__.update(self.__dict__)
        m._model = res[0]
        m.U, m.Sigma, m.V = m._model.u, m._model.sigma, m._model.v
        m.Vt = m.V.T
        m.rel_contribution, m.cumsum_contr = m._model.rel_contribution, m._model.cumsum_contr
        m.__dict__.pop("_X_elem", None)
        m.__dict__.pop("_X_ss", None)
        return (m, *res[1:])

    def fossa(self, nested_groups, filter=None, gamma=float("inf"), normalize=True):
        """Returns ``(model, groups, set)``."""
        return self._wrap_ossa(self._model.fossa([list(map(int, g)) for g in nested_groups], filter, gamma, normalize))

    def iossa(self, nested_groups, tol=1e-5, kappa=2.0, maxiter=100, kappa_balance=0.5):
        """Returns ``(model, groups, set, iterations, converged)``."""
        return self._wrap_ossa(self._model.iossa([list(map(int, g)) for g in nested_groups], tol, kappa, maxiter, kappa_balance))

    def eossa(self, nested_groups, k, method="ls"):
        """Returns ``(model, groups, set)``."""
        return self._wrap_ossa(self._model.eossa([list(map(int, g)) for g in nested_groups], int(k), method))

    def owcor(self, groups, set):
        return self._model.owcor([list(map(int, g)) for g in groups], list(map(int, set)))


class SSA(BaseSSA):
    """SSA for univariate time series."""

    def _prepare_data(self, data) -> NDArray[np.float64]:
        if isinstance(data, pd.Series):
            self.time_index = data.index
            self.series_name = data.name
            X = data.to_numpy(dtype=float)
        else:
            X = np.asarray(data, dtype=float)
            self.time_index = None
            self.series_name = None
        if X.ndim != 1:
            raise ValueError("SSA expects a single time series.")
        return X[:, None]


class MSSA(BaseSSA):
    """MSSA for multivariate time series arranged as (N, S)."""

    def _prepare_data(self, data) -> NDArray[np.float64]:
        if isinstance(data, pd.DataFrame):
            self.time_index = data.index
            self.series_names = list(data.columns)
            X = data.to_numpy(dtype=float)
        else:
            X = np.asarray(data, dtype=float)
            self.time_index = None
            self.series_names = None
        if X.ndim != 2:
            raise ValueError("MSSA expects multiple time series with shape (n_timesteps, n_series).")
        return X
