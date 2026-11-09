"""Python API of the Rssa-compatible classes, checked against references
computed by the R package Rssa (same JSON files as the Rust tests)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ssa_rs import SSA, MSSA, rssa
from ssa_rs.rssa import Cssa, MssaList, NdSsa, ShapedSsa

DATA = Path(__file__).resolve().parents[3] / "ssa-rust" / "tests" / "data"
REF = json.loads((DATA / "rssa_ref_ext.json").read_text())
REF2 = json.loads((DATA / "rssa_ref_ext2.json").read_text())


def arr(v):
    if isinstance(v, dict) and "re" in v:
        return np.array(v["re"]) + 1j * np.array(v["im"])
    return np.array([np.nan if e in ("NA", None) else float(e) for e in np.ravel(v)])


def fvec(a):
    return np.asarray(a).ravel(order="F")


def csort(z):
    z = np.asarray(z)
    return z[np.lexsort((np.round(z.imag, 4), np.round(z.real, 4)))]


def close(a, b, tol=1e-7):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def test_circular_1d():
    x = arr(REF["x"])
    s = NdSsa.fit(x, [50], circular=[True], svd_method="full")
    close(s.sigma[:10], arr(REF["circ"]["sigma"]))
    close(s.reconstruct([0, 1, 2]), arr(REF["circ"]["rec"]))
    close(s.rforecast([0, 1, 2], 8), arr(REF["circ"]["rf"]))


def test_shaped_window_1d():
    x = arr(REF["x"])
    wm = np.array(REF["wm1"]["wmask"], dtype=bool)
    s = NdSsa.fit(x, wmask=wm, svd_method="full")
    close(s.sigma[:10], arr(REF["wm1"]["sigma"]))
    close(s.reconstruct([0, 1, 2, 3]), arr(REF["wm1"]["rec"]))


def test_2d_circle_window():
    f = arr(REF["c2"]["f"]).reshape(24, 30, order="F")
    s = NdSsa.fit(f, wmask=rssa.circle(4), svd_method="full")
    close(s.sigma[:8], arr(REF["c2"]["sigma"]))
    close(fvec(s.reconstruct([0, 1, 2, 3])), arr(REF["c2"]["rec"]))


def test_2d_circular_na_projection_oblique():
    f = arr(REF["c2"]["f"]).reshape(24, 30, order="F")
    s = NdSsa.fit(f, [6, 7], circular=[True, False], svd_method="full")
    close(fvec(s.reconstruct([0, 1, 2, 3])), arr(REF["c2circ"]["rec"]))

    fna = f.copy()
    fna[9:12, 13:16] = np.nan
    fna[0, 0] = np.nan
    s = NdSsa.fit(fna, [6, 7], svd_method="full")
    close(s.sigma[:8], arr(REF["c2na"]["sigma"]))
    close(fvec(s.reconstruct([0, 1, 2, 3])), arr(REF["c2na"]["rec"]))
    close(fvec(s.igapfill([list(range(6))], maxiter=10)[0]), arr(REF["c2na"]["igap"]), 1e-6)

    s = NdSsa.fit(f, [6, 7], column_projector=1, row_projector=1, svd_method="full")
    close(s.sigma[:8], arr(REF["c2p"]["sigma"]))
    close(fvec(s.reconstruct(list(range(5)))), arr(REF["c2p"]["rec"]))
    assert s.nspecial == 2

    s = NdSsa.fit(f, [6, 7], column_oblique=REF["c2w"]["cw"], row_oblique=REF["c2w"]["rw"], svd_method="full")
    close(s.sigma[:8], arr(REF["c2w"]["sigma"]))
    close(fvec(s.reconstruct([0, 1, 2, 3])), arr(REF["c2w"]["rec"]))


def test_2d_esprit_and_oblique_methods():
    f = arr(REF["c2"]["f"]).reshape(24, 30, order="F")
    s = NdSsa.fit(f, [6, 7], svd_method="full")
    for pairing, keys in (("diag", ("d1", "d2")), ("memp", ("m1", "m2"))):
        got = s.parestimate([0, 1, 2, 3], pairing=pairing)
        for g, k in zip(got, keys):
            close(csort(g["roots"]), csort(arr(REF["c2e"][k])), 1e-6)
    got = s.parestimate([0, 1, 2, 3], subspace="row")
    for g, k in zip(got, ("r1", "r2")):
        close(csort(g["roots"]), csort(arr(REF["c2e"][k])), 1e-6)

    fo = s.fossa([[0, 1, 2, 3]])
    close(fo["model"].sigma[:4], arr(REF["c2o"]["fsigma"]), 1e-6)
    io = s.iossa([[0, 1], [2, 3]], tol=1e-8, maxiter=30)
    assert io["groups"] == [list(map(int, g)) for g in [np.array(g) - 1 for g in REF["c2o"]["igroups"]]]
    for g, e in zip(io["groups"], REF["c2o"]["irec"].values()):
        close(fvec(io["model"].reconstruct(g)), arr(e), 1e-5)
    eo = s.eossa([[0, 1, 2, 3]], 2)
    for g, e in zip(eo["groups"], REF["c2o"]["erec"].values()):
        close(fvec(eo["model"].reconstruct(g)), arr(e), 1e-5)
    close(fvec(s.cadzow(4, correct=True, maxiter=5)), arr(REF["c2o"]["cadzow"]), 1e-6)


def test_3d_memp():
    f = arr(REF["c3"]["f"]).reshape(10, 9, 8, order="F")
    s = NdSsa.fit(f, [4, 4, 4], svd_method="full")
    close(fvec(s.reconstruct([0, 1])), arr(REF["c3"]["rec"]))
    got = s.parestimate([0, 1], pairing="memp")
    for g, e in zip(got, REF["c3"]["pm"].values()):
        close(csort(g["roots"]), csort(arr(e)), 1e-6)


def test_mssa_list_and_gaps():
    a, b = arr(REF["ml"]["a"]), arr(REF["ml"]["b"])
    m = MssaList.fit([a, b], 40, svd_method="full")
    close(m.sigma[:8], arr(REF["ml"]["sigma"]))
    for got, e in zip(m.reconstruct([0, 1, 2]), REF["ml"]["rec"]):
        close(got, arr(e))
    for key, direction, fn in (("rfr", "row", "rforecast"), ("rfc", "column", "rforecast"), ("vfc", "column", "vforecast")):
        out = getattr(m, fn)([0, 1, 2], 7, direction=direction)
        for got, e in zip(out, REF["ml"][key]):
            close(got, arr(e), 1e-6)


def test_cssa_gapfill_matches_real_rssa():
    xg = arr(REF2["xg"])
    s = Cssa.fit(xg.astype(complex), 24)
    group = list(range(5))
    close(s.gapfill(group).real, arr(REF2["gap_seq"]))
    close(s.gapfill(group, simultaneous=True).real, arr(REF2["gap_sim"]))
    close(s.igapfill([group], maxiter=10)[0].real, arr(REF2["igap"]), 1e-6)
    z = arr(REF2["z0"])
    c = Cssa.fit(z, 24)
    assert c.wnorm() == pytest.approx(REF2["wnorm"] if np.isscalar(REF2["wnorm"]) else REF2["wnorm"][0], rel=1e-9)
    close(c.contributions([0, 1, 2, 3]), arr(REF2["contrib"]), 1e-8)


def test_cssa_oblique_keeps_imaginary_part():
    t = np.arange(200)
    z = np.exp(2j * np.pi * t / 10) + 0.98 * np.exp(2j * np.pi * t / 10.6)
    c = Cssa.fit(z, 50)
    e = c.eossa([[0, 1]], 2)
    assert e["model"].reconstruct(e["groups"][0]).imag.any()
    truth = np.exp(2j * np.pi * t / 10)
    errs = [np.sqrt(np.mean(abs(e["model"].reconstruct(g) - truth) ** 2)) for g in e["groups"]]
    assert min(errs) < 1e-6


def test_wossa_with_projection_runs_and_sums_to_series():
    x = arr(REF2["x"])
    m = rssa.wossa_pssa(x, 30, column_projector=1, row_projector=2,
                        column_oblique=REF2["cw"], row_oblique=REF2["rw"])
    assert m.nspecial == 3
    close(m.reconstruct(list(range(m.d))), x, 1e-8)


def test_shaped_ssa_and_gap_helpers():
    xg = arr(REF2["xg"])
    s = ShapedSsa.fit(xg, 24)
    close(s.gapfill(list(range(5))), arr(REF2["gap_seq"]))
    close(s.gapfill(list(range(5)), simultaneous=True), arr(REF2["gap_sim"]))
    gaps = rssa.summarize_gaps(xg, 24)
    assert len(gaps) == 2 and gaps[0][2] in ("internal", "left", "right")
    assert rssa.complete_lag_fraction(xg)[0][0] == 2


def test_model_extras_on_plain_ssa():
    x = arr(REF["x"])
    m = SSA().fit(x, L=50)
    g = [0, 1, 2]
    assert m.rforecast(g, 8).shape == (8, 1)
    assert m.vforecast(g, 8).shape == (8, 1)
    close(m.rforecast(g, 8)[:, 0], m.L_forecast(8, g, return_full=False)[:, 0], 1e-8)
    f = m.forecast(g, 8, interval="prediction", replicates=50, seed=1)
    assert f["lower"].shape == (8,) and np.all(f["lower"] <= f["upper"])
    p = m.parestimate([0, 1])
    assert set(p) == {"roots", "periods", "frequencies", "moduli", "rates"}
    w = m.wcor([[0, 1], [2], [3, 4]])
    assert w.shape == (3, 3)
    assert m.cadzow(2, maxiter=3).shape == (200, 1)
    res = m.iossa([[0, 1], [2, 3]], maxiter=20)
    assert isinstance(res[0], SSA)
    assert m.contributions().sum() == pytest.approx(1.0, rel=1e-9)
    assert rssa.hmatr(x, window=20).ndim == 2
    assert "rforecast" in rssa.capabilities("1d-ssa")


def test_errors_are_value_errors():
    with pytest.raises(ValueError):
        NdSsa.fit(np.ones(10), [20])
    with pytest.raises(ValueError):
        NdSsa.fit(np.arange(30.0), [10], svd_method="nope")
    with pytest.raises(ValueError):
        Cssa.fit(np.ones(10, dtype=complex), 20)
