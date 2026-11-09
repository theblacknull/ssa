"""Turn ssa-python/bench/results_*.json into Markdown tables (printed to stdout)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = ["fit", "reconstruct", "wcorr", "L_forecast", "K_forecast", "ESPRIT", "total"]


def ms(v):
    v *= 1000
    return f"{v:.3f}" if v < 1 else (f"{v:.2f}" if v < 100 else f"{v:,.0f}")


def speed(p, r):
    x = p / r
    return f"**{x:,.1f}×**" if x >= 1 else f"{x:.2f}×"


def table(path, title):
    rows = json.loads(Path(path).read_text())
    print(f"\n### {title}\n")
    print("| Trường hợp | Thao tác | Python (ms) | Rust FullSVD (ms) | Tăng tốc | Rust EigenSVD (ms) | Tăng tốc |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        p, r, e = row.get("python"), row.get("rust"), row.get("rust-eigen")
        for op in OPS:
            cells = [row["case"] if op == "fit" else "", op if op != "total" else "**tổng**"]
            if p and "error" not in p:
                cells += [ms(p[op]), ms(r[op]), speed(p[op], r[op]), ms(e[op]), speed(p[op], e[op])]
            else:
                perr = p["error"][:40] if p else "không chạy (thiếu RAM)"
                cells += [perr, ms(r[op]), "—", ms(e[op]), "—"]
            print("| " + " | ".join(cells) + " |")
    print("\n| Trường hợp | Bộ nhớ đỉnh Python (MB) | Rust (MB) | Giảm | SVD numpy riêng (ms) | Rust fit (ms) | Kiểm tra σ₀ / ‖forecast‖ khớp |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        p, r = row.get("python"), row.get("rust")
        if not p or "error" in p:
            print(f"| {row['case']} | — | {r['peak_mb']:,.0f} | — | — | {ms(r['fit'])} | — |")
            continue
        ok = all(
            abs(p["check"][k] - r["check"][k]) <= 1e-6 * max(1.0, abs(p["check"][k]))
            for k in ("sigma0", "rec_norm", "lf_norm", "kf_norm", "wcorr_01")
        )
        print(
            f"| {row['case']} | {p['peak_mb']:,.0f} | {r['peak_mb']:,.0f} | {p['peak_mb'] / r['peak_mb']:.1f}× | "
            f"{ms(p['np_svd_only'])} | {ms(r['fit'])} | {'✅' if ok else '❌'} |"
        )


if __name__ == "__main__":
    for name, title in [("results_mt.json", "Đa luồng (16 luồng)"), ("results_st.json", "Đơn luồng (1 luồng)")]:
        if (HERE / name).exists():
            table(HERE / name, title)
