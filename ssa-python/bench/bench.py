"""Benchmark py_ssa_lib (Python) vs ssa_rs (Rust) on identical workloads.

Every (library, case) pair runs in a fresh subprocess so that the peak
memory of the process can be measured.  Usage:

    python ssa-python/bench/bench.py            # multi-threaded run -> ssa-python/bench/results_mt.json
    python ssa-python/bench/bench.py --single        # 1 thread for BLAS and Rust -> results_st.json
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (name, N, S, L, run_python)
CASES = [
    ("SSA N=200 L=100", 200, 1, 100, True),
    ("SSA N=500 L=250", 500, 1, 250, True),
    ("SSA N=1000 L=500", 1000, 1, 500, True),
    ("SSA N=2000 L=200", 2000, 1, 200, True),
    ("SSA N=5000 L=100", 5000, 1, 100, True),
    ("SSA N=10000 L=100", 10000, 1, 100, True),
    ("MSSA N=300 S=10 L=100", 300, 10, 100, True),
    ("MSSA N=500 S=5 L=200", 500, 5, 200, True),
    ("SSA N=4000 L=2000", 4000, 1, 2000, False),
    ("SSA N=20000 L=1000", 20000, 1, 1000, False),
    ("MSSA N=2000 S=8 L=500", 2000, 8, 500, False),
]
R_COMP = 10  # number of components used for reconstruction / forecasts / ESPRIT
STEPS = 50


def peak_rss_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return float("nan")
    return pmc.PeakWorkingSetSize / 2**20


def make_data(n, s):
    import numpy as np

    rng = np.random.default_rng(42)
    t = np.arange(n)
    cols = [
        (1 + 0.2 * j) * np.sin(2 * np.pi * t / 50 + j) + 0.5 * np.cos(2 * np.pi * t / 7.3)
        + 0.001 * t + 0.3 * rng.normal(size=n)
        for j in range(s)
    ]
    x = np.column_stack(cols)
    return x[:, 0] if s == 1 else x


def timeit(fn, budget):
    """Median wall time of fn() over up to 7 runs within a time budget (s)."""
    times = []
    t_end = time.perf_counter() + budget
    while len(times) < 7:
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() > t_end and len(times) >= 1:
            break
    return statistics.median(times), out


def worker(lib: str, n: int, s: int, l: int) -> dict:
    import numpy as np

    if lib == "python":
        from py_ssa_lib.ssa_collection import SSA, MSSA
        from py_ssa_lib.svd_strategies import FullSVD as SVD
    else:
        from ssa_rs.ssa_collection import SSA, MSSA
        from ssa_rs.svd_strategies import FullSVD, EigenSVD

        SVD = EigenSVD if lib == "rust-eigen" else FullSVD
    x = make_data(n, s)
    cls = SSA if s == 1 else MSSA
    idx = list(range(R_COMP))
    budget = 6.0
    res = {}
    res["fit"], m = timeit(lambda: cls(decomposition_strategy=SVD()).fit(x, L=l), budget)
    res["reconstruct"], rec = timeit(lambda: m.reconstruct_components(idx), budget)
    res["wcorr"], w = timeit(lambda: m.compute_weighted_correlation_matrix(), budget)
    res["L_forecast"], lf = timeit(lambda: m.L_forecast(STEPS, idx, return_full=False), budget)
    res["K_forecast"], kf = timeit(lambda: m.K_forecast(STEPS, idx, return_full=False), budget)
    res["ESPRIT"], es = timeit(lambda: m.estimate_ESPRIT(idx), budget)
    res["total"] = sum(res[k] for k in ("fit", "reconstruct", "wcorr", "L_forecast", "K_forecast", "ESPRIT"))
    if lib == "python":
        # context: cost of the SVD alone (numpy/LAPACK) inside the Python fit
        res["np_svd_only"], _ = timeit(lambda: np.linalg.svd(m.X_ss, full_matrices=False), budget)
    res["peak_mb"] = peak_rss_mb()
    # fingerprints to check that both libraries computed the same thing
    res["check"] = {
        "sigma0": float(m.Sigma[0]),
        "rec_norm": float(np.linalg.norm(rec)),
        "lf_norm": float(np.linalg.norm(lf)),
        "kf_norm": float(np.linalg.norm(kf)),
        "wcorr_01": float(w[0, 1]),
    }
    return res


def main():
    single = "--single" in sys.argv
    env = dict(os.environ)
    if single:
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[k] = "1"
    cases = CASES if not single else [c for c in CASES if c[4]]
    out = []
    for name, n, s, l, run_py in cases:
        row = {"case": name, "N": n, "S": s, "L": l, "K": n - l + 1}
        libs = (["python"] if run_py else []) + ["rust", "rust-eigen"]
        for lib in libs:
            cmd = [sys.executable, __file__, "--worker", lib, str(n), str(s), str(l)]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
            if p.returncode != 0:
                row[lib] = {"error": p.stderr.strip().splitlines()[-1] if p.stderr else "failed"}
            else:
                row[lib] = json.loads(p.stdout.strip().splitlines()[-1])
            print(name, lib, json.dumps(row[lib])[:160], flush=True)
        out.append(row)
    path = HERE / ("results_st.json" if single else "results_mt.json")
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        lib, n, s, l = sys.argv[2], *map(int, sys.argv[3:6])
        print(json.dumps(worker(lib, n, s, l)))
    else:
        main()
