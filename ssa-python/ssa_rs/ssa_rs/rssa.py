"""Rssa-compatible SSA in Python, computed by the Rust core.

Everything the R package Rssa offers beyond py-ssa-lib is here, with the
same results as Rssa (checked against it) and much faster:

    from ssa_rs.rssa import NdSsa, Cssa, MssaList, ShapedSsa

``NdSsa.fit`` covers 1-D, 2-D and nD fields, shaped windows (``circle`` /
``triangle`` masks), fields with missing values, circular dimensions,
Toeplitz SSA, Projection SSA (``column_projector`` / ``row_projector``) and
weighted-oblique SSA (``column_oblique`` / ``row_oblique``).

Arrays use NumPy's C order: an ``(n0, n1)`` field is returned as
``(n0, n1)``.  Groups are lists of 0-based component indices.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._core import (  # noqa: F401  (re-exported)
    Cssa as _Cssa,
    MssaList as _MssaList,
    NdSsa as _NdSsa,
    ShapedSsa as _ShapedSsa,
    ball_mask,
    capabilities,
    complete_lag_fraction,
    hmatr,
    igapfill,
    simplex_mask,
    summarize_gaps,
)

__all__ = [
    "NdSsa", "Cssa", "MssaList", "ShapedSsa",
    "circle", "triangle", "ball_mask", "simplex_mask",
    "hmatr", "igapfill", "summarize_gaps", "complete_lag_fraction",
    "capabilities", "wossa_pssa",
]


def _groups(g):
    """Accepts one group (flat list) or a list of groups."""
    g = list(g)
    if g and np.ndim(g[0]) == 0:
        return [[int(i) for i in g]]
    return [[int(i) for i in x] for x in g]


def _group(g):
    return [int(i) for i in np.ravel(g)]


def circle(radius: int) -> np.ndarray:
    """Circular window mask of the given radius (Rssa ``circle``)."""
    return ball_mask(int(radius), 2)


def triangle(side: int) -> np.ndarray:
    """Triangular window mask (Rssa ``triangle``)."""
    return simplex_mask(int(side), 2)


class _Wrapped:
    """Delegates unknown attributes to the Rust object."""

    _impl = None

    def __getattr__(self, name):
        impl = object.__getattribute__(self, "_impl")
        return getattr(impl, name)

    def __repr__(self):
        return f"<{type(self).__name__} d={self._impl.d}>"


class NdSsa(_Wrapped):
    """Fitted nD / shaped / circular / projection / oblique SSA."""

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def fit(
        cls,
        field,
        window=None,
        *,
        wmask=None,
        mask=None,
        circular=None,
        column_projector=None,
        row_projector=None,
        column_oblique=None,
        row_oblique=None,
        toeplitz: bool = False,
        svd_method: str = "auto",
        neig: int = 50,
    ) -> "NdSsa":
        """Decomposes ``field`` (NaN = missing).

        ``svd_method``: ``"auto"`` (Rssa default: exact for small windows,
        otherwise Lanczos with ``neig`` triples), ``"full"`` (exact, all
        triples) or ``"lanczos"``.
        """
        f = np.ascontiguousarray(field, dtype=float)
        if circular is not None:
            circular = [bool(c) for c in np.broadcast_to(circular, (f.ndim,))]
        return cls(_NdSsa.fit(
            f,
            None if window is None else [int(w) for w in np.broadcast_to(window, (f.ndim,))],
            None if wmask is None else np.ascontiguousarray(wmask, dtype=bool),
            None if mask is None else np.ascontiguousarray(mask, dtype=bool),
            circular, column_projector, row_projector,
            None if column_oblique is None else [float(w) for w in column_oblique],
            None if row_oblique is None else [float(w) for w in row_oblique],
            toeplitz, svd_method, int(neig),
        ))

    # ---- decomposition-level results
    def reconstruct(self, group):
        return self._impl.reconstruct(_group(group))

    def residuals(self, group):
        return self._impl.residuals(_group(group))

    def wcor(self, groups):
        return self._impl.wcor(_groups(groups))

    def parestimate(self, group, subspace="column", dimensions=None, method="ls", pairing="diag", normalize=None):
        return self._impl.parestimate(_group(group), subspace, dimensions, method, pairing, normalize)

    def igapfill(self, groups, fill=None, tol=1e-6, maxiter=0):
        return self._impl.igapfill(_groups(groups), fill, tol, maxiter)

    def rforecast(self, group, steps, reconstructed_base=True, reverse=False):
        return self._impl.rforecast(_group(group), int(steps), reconstructed_base, reverse)

    def vforecast(self, group, steps):
        return self._impl.vforecast(_group(group), int(steps))

    def lrr(self, group, reverse=False):
        return self._impl.lrr(_group(group), reverse)

    def _ossa(self, d):
        d = dict(d)
        d["model"] = NdSsa(d["model"])
        return d

    def fossa(self, nested_groups, filter=None, gamma=float("inf"), normalize=True):
        return self._ossa(self._impl.fossa(_groups(nested_groups), filter, gamma, normalize))

    def iossa(self, nested_groups, tol=1e-5, kappa=2.0, maxiter=100, kappa_balance=0.5):
        return self._ossa(self._impl.iossa(_groups(nested_groups), tol, kappa, maxiter, kappa_balance))

    def eossa(self, nested_groups, k, method="ls", dimensions=None):
        return self._ossa(self._impl.eossa(_groups(nested_groups), int(k), method, dimensions))

    def owcor(self, groups, set):
        return self._impl.owcor(_groups(groups), _group(set))


class Cssa(_Wrapped):
    """Complex SSA of a complex-valued series (NaN in either part = gap)."""

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def fit(cls, z, window: int) -> "Cssa":
        return cls(_Cssa.fit(np.ascontiguousarray(z, dtype=complex), int(window)))

    def reconstruct(self, group):
        return self._impl.reconstruct(_group(group))

    def wcor(self, groups):
        return self._impl.wcor(_groups(groups))

    def parestimate(self, group, tls=False):
        return self._impl.parestimate(_group(group), tls)

    def rforecast(self, group, steps):
        return self._impl.rforecast(_group(group), int(steps))

    def vforecast(self, group, steps):
        return self._impl.vforecast(_group(group), int(steps))

    def gapfill(self, group, reconstructed_base=False, simultaneous=False):
        return self._impl.gapfill(_group(group), reconstructed_base, simultaneous)

    def igapfill(self, groups, fill=None, tol=1e-6, maxiter=0):
        fill = None if fill is None else (complex(fill).real, complex(fill).imag)
        return self._impl.igapfill(_groups(groups), fill, tol, maxiter)

    def _ossa(self, d):
        d = dict(d)
        d["model"] = Cssa(d["model"])
        return d

    def fossa(self, nested_groups, filter=None, gamma=float("inf"), normalize=True):
        return self._ossa(self._impl.fossa(_groups(nested_groups), filter, gamma, normalize))

    def iossa(self, nested_groups, tol=1e-5, kappa=2.0, maxiter=100, kappa_balance=0.5):
        return self._ossa(self._impl.iossa(_groups(nested_groups), tol, kappa, maxiter, kappa_balance))

    def eossa(self, nested_groups, k):
        return self._ossa(self._impl.eossa(_groups(nested_groups), int(k)))

    def owcor(self, groups, set):
        return self._impl.owcor(_groups(groups), _group(set))


class MssaList(_Wrapped):
    """MSSA of series with different lengths (leading / trailing NaN allowed)."""

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def fit(cls, series, window=None, svd_method="auto", neig=50) -> "MssaList":
        return cls(_MssaList.fit([list(map(float, s)) for s in series], window, svd_method, int(neig)))

    def reconstruct(self, group):
        return [np.asarray(s) for s in self._impl.reconstruct(_group(group))]

    def wcor(self, groups):
        return self._impl.wcor(_groups(groups))

    def rforecast(self, group, steps, base="reconstructed", direction="column"):
        return [np.asarray(s) for s in self._impl.rforecast(_group(group), int(steps), base, direction)]

    def vforecast(self, group, steps, direction="column"):
        return [np.asarray(s) for s in self._impl.vforecast(_group(group), int(steps), direction)]

    def gapfill(self, group, reconstructed_base=False):
        return [np.asarray(s) for s in self._impl.gapfill(_group(group), reconstructed_base)]

    def igapfill(self, groups, fill=None, tol=1e-6, maxiter=0):
        return [[np.asarray(s) for s in g] for g in self._impl.igapfill(_groups(groups), fill, tol, maxiter)]


class ShapedSsa(_Wrapped):
    """1-D SSA of a series with missing values."""

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def fit(cls, x, window: int) -> "ShapedSsa":
        return cls(_ShapedSsa.fit(list(map(float, x)), int(window)))

    def reconstruct(self, group):
        return self._impl.reconstruct(_group(group))

    def wcor(self, groups):
        return self._impl.wcor(_groups(groups))

    def gapfill(self, group, reconstructed_base=False, simultaneous=False):
        return self._impl.gapfill(_group(group), reconstructed_base, simultaneous)


def wossa_pssa(x, window, column_projector=None, row_projector=None, column_oblique=None, row_oblique=None) -> NdSsa:
    """Weighted-oblique SSA combined with Projection SSA (1-D)."""
    return NdSsa.fit(
        np.asarray(x, dtype=float), [window],
        column_projector=column_projector, row_projector=row_projector,
        column_oblique=column_oblique, row_oblique=row_oblique, svd_method="full",
    )
