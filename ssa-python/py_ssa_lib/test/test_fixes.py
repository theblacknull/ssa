"""Regression tests for bugs fixed in py_ssa_lib (input validation and edge cases)."""

import numpy as np
import pytest

from py_ssa_lib.ssa_collection import SSA, MSSA
from py_ssa_lib.stat_tools import bootstrap_prediction_intervals
from py_ssa_lib.weights_strategies import ClusterAwareWeights, DistanceWeights, PrecomputedWeights

RNG = np.random.default_rng(0)
X = np.sin(np.arange(100) / 5) + 0.1 * RNG.normal(size=100)


def test_zero_forecast_steps_returns_no_future_values():
    m = SSA().fit(X, L=30)
    assert m.L_forecast(0, [0, 1], return_full=False).shape == (0, 1)
    assert m.K_forecast(0, [0, 1], return_full=False).shape == (0, 1)
    assert m.L_forecast(0, [0, 1]).shape == (100, 1)


@pytest.mark.parametrize("steps", [-1, 2.5, True])
def test_bad_forecast_steps(steps):
    with pytest.raises(ValueError):
        SSA().fit(X, L=30).L_forecast(steps, [0, 1])


def test_nan_and_inf_are_rejected_clearly():
    for bad in (np.nan, np.inf):
        x = X.copy()
        x[10] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            SSA().fit(x, L=30)


def test_window_must_be_integer_valued():
    assert SSA().fit(X, L=30.0).L == 30
    with pytest.raises(ValueError):
        SSA().fit(X, L=30.5)


def test_weights_are_validated():
    def fit(w):
        return MSSA(weighting_strategy=PrecomputedWeights()).fit(np.c_[X, X], L=30, weights=w)

    with pytest.raises(ValueError, match="Expected 2 weights"):
        fit([1.0])
    for w in ([1.0, 0.0], [1.0, -2.0], [1.0, np.nan]):
        with pytest.raises(ValueError, match="strictly positive"):
            fit(w)
    with pytest.raises(ValueError):
        MSSA(weighting_strategy=DistanceWeights()).fit(np.c_[X, X], L=30, distances=[0, 1], bandwidth=0)
    with pytest.raises(ValueError):
        MSSA(weighting_strategy=ClusterAwareWeights()).fit(
            np.c_[X, X], L=30, distances=[0, 1], focal_cluster=0, neighbour_clusters=[0])


def test_component_indices_are_checked():
    m = SSA().fit(X, L=30)
    with pytest.raises(IndexError):
        m.reconstruct_components([0, 999])
    with pytest.raises(IndexError):
        m.reconstruct_components([-31])
    with pytest.raises(IndexError):
        m.reconstruct_components([0.5])
    assert np.allclose(m.reconstruct_components([-1]), m.reconstruct_components([29]))
    mask = np.zeros(30, bool)
    mask[:2] = True
    assert np.allclose(m.reconstruct_components(mask), m.reconstruct_components([0, 1]))
    with pytest.raises(ValueError):
        m.estimate_LRR([])


def test_zero_series_is_rejected():
    with pytest.raises(ValueError, match="identically zero"):
        SSA().fit(np.zeros(40), L=10)


def test_wcorr_has_no_nan_for_a_zero_norm_component():
    m = SSA().fit(np.ones(40), L=10)   # rank one: the other components have zero norm
    assert np.isfinite(m.compute_weighted_correlation_matrix()).all()


def test_lrr_requires_non_vertical_subspace():
    m = SSA().fit(X, L=30)
    m.U[:, 0] = 0.0
    m.U[-1, 0] = 1.0          # e_L in the subspace: no recurrence exists
    with pytest.raises(ValueError, match="Verticality"):
        m.estimate_LRR([0])


def test_bootstrap_is_reproducible_and_validated():
    res, fc = RNG.normal(size=50), np.zeros(5)
    a = bootstrap_prediction_intervals(res, fc, N_sims=200, random_state=3)
    b = bootstrap_prediction_intervals(res, fc, N_sims=200, random_state=3)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert np.all(a[0] <= a[1])
    assert bootstrap_prediction_intervals(res, fc[:, None], N_sims=10)[0].shape == (5,)
    for bad in (dict(alpha=2.0), dict(alpha=-0.1), dict(N_sims=0)):
        with pytest.raises(ValueError):
            bootstrap_prediction_intervals(res, fc, **bad)
    with pytest.raises(ValueError):
        bootstrap_prediction_intervals(np.array([]), fc)
    with pytest.raises(ValueError):
        bootstrap_prediction_intervals(np.array([1.0, np.nan]), fc)
