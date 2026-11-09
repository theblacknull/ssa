"""SVD strategies (same API as ``py_ssa_lib.svd_strategies``).

``FullSVD``, ``RandomizedSVD`` and ``EigenSVD`` run in Rust.  Any other
subclass of ``SVDStrategy`` (or a built-in one given extra keyword arguments
that only numpy/sklearn understand) is executed in Python and its result is
handed to the Rust core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SVDStrategy(ABC):
    """Base class for performing SVD."""

    @abstractmethod
    def decompose(self, X: NDArray[np.float64]):
        """Return ``(U, sigma, Vt)``."""

    def _rust_spec(self) -> dict | None:
        """Keyword arguments for ``_core.fit`` or None to run ``decompose`` in Python."""
        return None


class FullSVD(SVDStrategy):
    """Exact thin SVD (Rust/faer; numpy when extra numpy kwargs are given)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def _rust_spec(self):
        return None if self.kwargs else {"method": "full"}

    def decompose(self, X):
        # Only reached when called directly or with numpy-specific kwargs;
        # inside ``fit`` the Rust SVD is used.
        return np.linalg.svd(X, full_matrices=False, **self.kwargs)


class EigenSVD(SVDStrategy):
    """Exact SVD via the eigen-decomposition of ``X X^T`` (Rssa "eigen").

    Not part of py-ssa-lib: fastest exact option when ``L << K*S``.
    """

    def _rust_spec(self):
        return {"method": "eigen"}

    def decompose(self, X):
        X = np.asarray(X, dtype=float)
        lam, U = np.linalg.eigh(X @ X.T)
        U = U[:, ::-1]
        XtU = X.T @ U
        s = np.linalg.norm(XtU, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(s > 0, XtU / s, 0.0)
        order = np.argsort(-s, kind="stable")
        return U[:, order], s[order], V[:, order].T


class RandomizedSVD(SVDStrategy):
    """Approximate truncated SVD (Halko et al.), same parameters as sklearn."""

    def __init__(
        self,
        n_components: int,
        n_oversamples: int = 10,
        n_iter: int | str = "auto",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state
        self.kwargs = kwargs

    def _rust_spec(self):
        rs = self.random_state
        rust_seed = rs is None or (isinstance(rs, (int, np.integer)) and not isinstance(rs, bool) and rs >= 0)
        rust_iter = self.n_iter == "auto" or (isinstance(self.n_iter, (int, np.integer)) and self.n_iter >= 0)
        if self.kwargs or not rust_seed or not rust_iter:
            return None
        return {
            "method": "randomized",
            "n_components": int(self.n_components),
            "n_oversamples": int(self.n_oversamples),
            "n_iter": None if self.n_iter == "auto" else int(self.n_iter),
            "random_state": None if rs is None else int(rs),
        }

    def decompose(self, X):
        from sklearn.utils.extmath import randomized_svd

        return randomized_svd(
            X,
            n_components=self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=self.random_state,
            **self.kwargs,
        )


class LanczosSVD(SVDStrategy):
    """Leading ``n_components`` eigentriples by Lanczos on the FFT trajectory
    operator (Rssa ``nutrlan``); the trajectory matrix is never formed."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def _rust_spec(self):
        return {"method": "lanczos", "n_components": int(self.n_components)}

    def decompose(self, X):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        k = self.n_components
        return U[:, :k], s[:k], Vt[:k]


class ToeplitzSVD(SVDStrategy):
    """Toeplitz SSA (Rssa ``kind = "toeplitz-ssa"``), univariate series only."""

    def __init__(self, neig: int | None = None) -> None:
        self.neig = neig

    def _rust_spec(self):
        return {"method": "toeplitz", "n_components": None if self.neig is None else int(self.neig)}

    def decompose(self, X):
        raise NotImplementedError("Toeplitz SSA needs the series; use it through SSA.fit")
