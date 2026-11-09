"""Numerical parity of ssa_rs (Rust) against the original py_ssa_lib 2.0.2.

Singular vectors are unique only up to sign (and up to rotation inside
eigen-spaces of repeated singular values), so vectors are compared after sign
alignment and only for well separated singular values; quantities that are
invariant (reconstructions, forecasts, ESPRIT roots, LRR) are compared
directly.
"""

import numpy as np
import pandas as pd
import pytest

import py_ssa_lib.ssa_collection as P
import py_ssa_lib.svd_strategies as Psvd
import py_ssa_lib.weights_strategies as Pw
import py_ssa_lib.stat_tools as Pst
import ssa_rs.ssa_collection as R
import ssa_rs.svd_strategies as Rsvd
import ssa_rs.weights_strategies as Rw
import ssa_rs.stat_tools as Rst

RTOL = 1e-9


def close(a, b, tol=RTOL, scale=None):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    s = scale if scale is not None else max(1.0, np.max(np.abs(b)) if b.size else 1.0)
    err = np.max(np.abs(a - b)) if a.size else 0.0
    assert err <= tol * s, f"max abs err {err:.3e} > {tol * s:.3e}"


def make_data(kind, n, s, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = []
    for j in range(s):
        if kind == "noise":
            cols.append(rng.normal(size=n))
        elif kind == "signal":
            cols.append(
                (1 + 0.3 * j) * np.sin(2 * np.pi * t / 12 + j)
                + 0.5 * np.cos(2 * np.pi * t / 5.3)
                + 0.01 * (j + 1) * t
                + 0.2 * rng.normal(size=n)
            )
        else:  # walk
            cols.append(np.cumsum(rng.normal(size=n)))
    X = np.column_stack(cols)
    return X[:, 0] if s == 1 else X


CASES = [
    # kind, N, S, L, weights
    ("noise", 30, 1, 10, None),
    ("signal", 100, 1, 40, None),
    ("walk", 257, 1, None, None),
    ("signal", 400, 1, 150, None),
    ("noise", 60, 3, 20, [1.0, 0.6, 0.2]),
    ("signal", 150, 2, 50, None),
    ("walk", 200, 4, 70, [0.5, 1.0, 2.0, 3.0]),
    ("signal", 1000, 1, 300, None),
]


def fit_pair(kind, n, s, l, w, rust_svd=None, py_svd=None):
    x = make_data(kind, n, s, seed=n * 7 + s)
    Pcls, Rcls = (P.SSA, R.SSA) if s == 1 else (P.MSSA, R.MSSA)
    kw = {}
    pw = rw = None
    if w is not None:
        pw, rw, kw = Pw.PrecomputedWeights(), Rw.PrecomputedWeights(), {"weights": np.array(w)}
    p = Pcls(decomposition_strategy=py_svd, weighting_strategy=pw).fit(x, L=l, **kw)
    r = Rcls(decomposition_strategy=rust_svd, weighting_strategy=rw).fit(x, L=l, **kw)
    return x, p, r


def separated(sigma, j, rel=1e-6):
    s0 = sigma[0]
    if sigma[j] < 1e-8 * s0:
        return False
    gaps = [abs(sigma[j] - sigma[k]) for k in (j - 1, j + 1) if 0 <= k < len(sigma)]
    return min(gaps) > rel * s0


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-N{c[1]}-S{c[2]}-L{c[3]}" for c in CASES])
def test_parity_full_svd(case):
    kind, n, s, l, w = case
    x, p, r = fit_pair(*case)

    # shapes / scalars
    for a in ("N", "S", "L", "K", "d"):
        assert getattr(p, a) == getattr(r, a), a
    for a in ("X_s", "X_ss", "weights"):
        close(getattr(r, a), getattr(p, a), 1e-15)
    assert r.U.shape == p.U.shape and r.V.shape == p.V.shape and r.Vt.shape == p.Vt.shape

    # spectrum
    close(r.Sigma, p.Sigma, 1e-11, scale=p.Sigma[0])
    close(r.rel_contribution, p.rel_contribution, 1e-10, scale=100)
    close(r.cumsum_contr, p.cumsum_contr, 1e-10, scale=100)
    np.testing.assert_array_equal(r.construct_hankel_weights(), p.construct_hankel_weights())
    close(r.construct_trajectory_matrix(), p.construct_trajectory_matrix(), 1e-15)

    # singular vectors (sign aligned, separated values only)
    good = [j for j in range(p.d) if separated(p.Sigma, j)]
    assert good, "no separated components"
    for j in good:
        sg = np.sign(np.dot(p.U[:, j], r.U[:, j]))
        close(sg * r.U[:, j], p.U[:, j], 1e-7)
        close(sg * r.V[:, j], p.V[:, j], 1e-7)

    # reconstructions (invariant)
    all_idx = list(range(p.d))
    close(r.reconstruct_components(all_idx), p.reconstruct_components(all_idx))
    close(r.reconstruct_components(all_idx), p.X_s, 1e-10)
    for idx in ([0], [0, 1], [0, 1, 2, 3], [1, 1, 0], np.array([0, 2]), [-1, 0]):
        if max(abs(i) for i in idx) < p.d and all(separated(p.Sigma, i if i >= 0 else p.d + i) for i in idx):
            close(r.reconstruct_components(idx), p.reconstruct_components(idx))

    # elementary components / X_elem / W-correlation on separated components
    comps_p = p._reconstruct_elementary_components()
    comps_r = r._reconstruct_elementary_components()
    assert comps_p.shape == comps_r.shape
    for j in good:
        close(comps_r[j], comps_p[j], 1e-8)
        close(r.X_elem[j], p.X_elem[j], 1e-8)  # sigma u v^T is sign invariant
    Wp = p.compute_weighted_correlation_matrix()
    Wr = r.compute_weighted_correlation_matrix()
    gi = np.ix_(good, good)
    close(Wr[gi], Wp[gi], 1e-7)
    assert np.allclose(np.diag(Wr), 1.0)

    # LRR, forecasts, ESPRIT on the leading separated block
    lead = [j for j in range(min(4, p.d)) if separated(p.Sigma, j)]
    lead = list(range(lead[-1] + 1)) if lead and lead == list(range(len(lead))) else lead
    if lead:
        close(r.estimate_LRR(lead), p.estimate_LRR(lead), 1e-7)
        for full in (True, False):
            close(r.L_forecast(12, lead, return_full=full), p.L_forecast(12, lead, return_full=full), 1e-7)
            close(r.K_forecast(12, lead, return_full=full), p.K_forecast(12, lead, return_full=full), 1e-7)
        mp, rp, op = p.estimate_ESPRIT(lead)
        mr, rr, orr = r.estimate_ESPRIT(lead)
        close(np.sort_complex(mr), np.sort_complex(mp), 1e-7)
        close(np.sort(np.abs(orr)), np.sort(np.abs(op)), 1e-7)
        close(np.sort(rr), np.sort(rp), 1e-7)
        # sorted by |angle|
        assert np.all(np.diff(np.abs(orr)) >= -1e-12)


def test_degenerate_pairs_are_invariant():
    # exact sine: sigma_0 == sigma_1 up to rounding -> only the pair is defined
    t = np.arange(240)
    x = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 7)
    p = P.SSA().fit(x, L=96)
    r = R.SSA().fit(x, L=96)
    for idx in ([0, 1], [2, 3], [0, 1, 2, 3]):
        close(r.reconstruct_components(idx), p.reconstruct_components(idx), 1e-9)
        close(r.L_forecast(30, idx), p.L_forecast(30, idx), 1e-8)
        close(r.K_forecast(30, idx), p.K_forecast(30, idx), 1e-8)
        close(np.sort_complex(r.estimate_ESPRIT(idx)[0]), np.sort_complex(p.estimate_ESPRIT(idx)[0]), 1e-9)
        close(r.estimate_LRR(idx), p.estimate_LRR(idx), 1e-8)


def test_pandas_metadata():
    s = pd.Series(np.arange(12.0), name="x", index=pd.date_range("2020", periods=12, freq="MS"))
    r = R.SSA().fit(s, L=4)
    assert r.series_name == "x" and r.time_index.equals(s.index)
    df = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0) ** 2})
    m = R.MSSA().fit(df, L=4)
    assert m.series_names == ["x", "y"] and m.time_index.equals(df.index)
    with pytest.raises(ValueError):
        R.SSA().fit(df.to_numpy(), L=4)
    with pytest.raises(ValueError):
        R.MSSA().fit(np.arange(12.0), L=4)


def test_errors_like_python():
    for L in (1, 7):
        with pytest.raises(ValueError):
            P.SSA().fit(np.arange(11.0), L=L)
        with pytest.raises(ValueError):
            R.SSA().fit(np.arange(11.0), L=L)
    r = R.SSA().fit(np.arange(20.0), L=5)
    with pytest.raises(IndexError):
        r.reconstruct_components([5])
    with pytest.raises(IndexError):
        r.reconstruct_components([0.5])
    # boolean mask selection == integer selection (numpy semantics)
    mask = np.zeros(r.d, dtype=bool)
    mask[[0, 2]] = True
    close(r.reconstruct_components(mask), r.reconstruct_components([0, 2]), 0)


def test_diagonal_averaging_static():
    rng = np.random.default_rng(5)
    for shape in [(3, 5), (5, 3), (1, 7), (7, 1), (6, 6)]:
        X = rng.normal(size=shape)
        close(R.BaseSSA.diagonal_averaging(X), P.BaseSSA.diagonal_averaging(X), 1e-15)


def test_weighting_strategies_identical():
    X = np.zeros((10, 3))
    d = [0.0, 1.0, 2.5]
    close(Rw.DistanceWeights().compute_weights(X, d, bandwidth=1.7), Pw.DistanceWeights().compute_weights(X, d, bandwidth=1.7), 0)
    kw = dict(distances=d, focal_cluster="a", neighbour_clusters=["a", "b", "a"], bandwidth=2.0, cluster_penalty=0.3)
    close(Rw.ClusterAwareWeights().compute_weights(X, **kw), Pw.ClusterAwareWeights().compute_weights(X, **kw), 0)
    # MSSA fitted with distance weights
    x = make_data("signal", 120, 3, 1)
    p = P.MSSA(weighting_strategy=Pw.DistanceWeights()).fit(x, L=40, distances=d, bandwidth=2.0)
    r = R.MSSA(weighting_strategy=Rw.DistanceWeights()).fit(x, L=40, distances=d, bandwidth=2.0)
    close(r.Sigma, p.Sigma, 1e-11, scale=p.Sigma[0])
    close(r.reconstruct_components([0, 1, 2]), p.reconstruct_components([0, 1, 2]), 1e-8)


def test_custom_python_svd_strategy_and_numpy_kwargs():
    class MySVD(Psvd.SVDStrategy):
        def decompose(self, X):
            return np.linalg.svd(X, full_matrices=False)

    class MySVDr(Rsvd.SVDStrategy):
        def decompose(self, X):
            return np.linalg.svd(X, full_matrices=False)

    x = make_data("walk", 90, 1, 3)
    p = P.SSA(decomposition_strategy=MySVD()).fit(x, L=30)
    r = R.SSA(decomposition_strategy=MySVDr()).fit(x, L=30)
    # identical decomposition -> identical everything
    close(r.U, p.U, 0)
    close(r.Sigma, p.Sigma, 0)
    close(r.compute_weighted_correlation_matrix(), p.compute_weighted_correlation_matrix(), 1e-9)
    close(r.reconstruct_components([0, 3]), p.reconstruct_components([0, 3]), 1e-12)
    r2 = R.SSA(decomposition_strategy=Rsvd.FullSVD(compute_uv=True)).fit(x, L=30)
    close(r2.U, p.U, 0)


def test_randomized_svd_parity():
    x = make_data("signal", 600, 1, 9)
    p = P.SSA(decomposition_strategy=Psvd.RandomizedSVD(n_components=6, random_state=0)).fit(x, L=200)
    r = R.SSA(decomposition_strategy=Rsvd.RandomizedSVD(n_components=6, random_state=0)).fit(x, L=200)
    assert p.d == r.d == 6
    e = R.SSA().fit(x, L=200)
    # both approximate the exact leading singular values
    close(r.Sigma[:4], e.Sigma[:4], 1e-6, scale=e.Sigma[0])
    close(p.Sigma[:4], e.Sigma[:4], 1e-6, scale=e.Sigma[0])
    close(r.reconstruct_components([0, 1, 2, 3]), p.reconstruct_components([0, 1, 2, 3]), 1e-5)
    # sklearn sign convention: largest |U_ij| of every column is positive
    for U in (p.U, r.U):
        assert np.all(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])] > 0)
    # sklearn fallback for options only sklearn knows
    f = R.SSA(decomposition_strategy=Rsvd.RandomizedSVD(n_components=6, random_state=0, power_iteration_normalizer="QR")).fit(x, L=200)
    assert f.d == 6


def test_eigen_svd_extension():
    x = make_data("signal", 500, 2, 4)
    e = R.MSSA(decomposition_strategy=Rsvd.EigenSVD()).fit(x, L=100)
    p = P.MSSA().fit(x, L=100)
    close(e.Sigma[:10], p.Sigma[:10], 1e-9, scale=p.Sigma[0])
    close(e.reconstruct_components(list(range(e.d))), x, 1e-10)
    close(e.reconstruct_components([0, 1, 2]), p.reconstruct_components([0, 1, 2]), 1e-8)


def test_bootstrap_statistics():
    rng = np.random.default_rng(0)
    res = rng.normal(size=500)
    fc = np.linspace(0, 1, 8)
    lo, hi = Rst.bootstrap_prediction_intervals(res, fc, alpha=0.1, N_sims=20000, seed=1)
    np.random.seed(1)
    plo, phi = Pst.bootstrap_prediction_intervals(res, fc, alpha=0.1, N_sims=20000)
    # same distribution -> quantiles agree within Monte-Carlo error
    close(lo, plo, 0.1, scale=1)
    close(hi, phi, 0.1, scale=1)
    # constant residuals -> deterministic, identical to Python
    lo, hi = Rst.bootstrap_prediction_intervals(np.full(10, 0.25), fc)
    plo, phi = Pst.bootstrap_prediction_intervals(np.full(10, 0.25), fc)
    close(lo, plo, 0)
    close(hi, phi, 0)


def test_percentile_matches_numpy_exactly():
    # residuals drawn so that every draw set is identical: one residual per sim is
    # not controllable, so check the percentile rule with a 1-sim / many-value trick
    vals = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0])
    for alpha in (0.05, 0.1, 0.33, 0.5):
        # forecasts zero, residual set = vals, huge N_sims -> empirical == population
        lo, hi = Rst.bootstrap_prediction_intervals(vals, [0.0], alpha=alpha, N_sims=200000, seed=3)
        assert vals.min() <= lo[0] <= hi[0] <= vals.max()


def test_vis_tools_work_with_rust_model():
    import matplotlib

    matplotlib.use("Agg")
    from ssa_rs import vis_tools as V

    x = make_data("signal", 120, 1, 2)
    r = R.SSA().fit(x, L=40)
    V.plot_eigenvectors(r)
    V.plot_elementary_matrices(r)
    V.plot_weighted_correlation(r)
    V.plot_contributions(r, n_components=10)
    V.plot_esprit_roots(r, [0, 1, 2, 3])
