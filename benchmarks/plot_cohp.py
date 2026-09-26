"""-pCOHP curves: LOBSTER 5.1.1 (CPU, grey reference) vs gpucohp (GPU) on the
six LOBSTER example systems.

    python plot_cohp.py RESULTS_DIR OUT_STEM

RESULTS_DIR/<system>/<run>/COHPCAR.lobster, runs as listed in PANELS.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\Daniil\.claude\skills\sci-figures\scripts")
from sciplot import apply_style, SERIES, REF_KW, finish_axes, stats_box, save_figure

# (system dir, LOBSTER run, gpucohp run, title, what is plotted)
PANELS = [
    ("diamond", "cpu", "gpu_tet", "diamond, C–C", "pair"),
    ("gaas", "cpu", "gpu_tet", "GaAs, Ga–As", "pair"),
    ("cnt", "cpu", "gpu", "C nanotube, mean of 48 C–C", "avg"),
    ("c60", "cpu", "gpu", "C$_{60}$, mean of 90 C–C", "avg"),
    ("ti", "cpu", "gpu_pbe_tet", "hcp Ti, Ti–Ti", "pair"),
    ("feni3", "cpu", "gpu_pbe", "FeNi$_3$, Fe–Ni", "pair"),
]
NOTES = {"diamond": "tetrahedra, Bunge", "gaas": "tetrahedra, Bunge",
         "cnt": "Gaussian, Bunge", "c60": "Gaussian, Bunge",
         "ti": "tetrahedra, pbeVaspFit2015 (imported)",
         "feni3": "Gaussian, pbeVaspFit2015 (imported)"}


def read_cohpcar(path):
    """(E, {spin: (avg, pair1)}) from a COHPCAR.lobster."""
    lines = open(path).read().splitlines()
    nspin = int(lines[1].split()[1])
    rows = []
    for ln in lines[2:]:
        f = ln.split()
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            continue
    ncol = max(len(r) for r in rows)
    A = np.array([r for r in rows if len(r) == ncol])
    per_spin = (ncol - 1) // nspin
    out = {}
    for s in range(nspin):
        o = 1 + s * per_spin
        out[s] = (A[:, o], A[:, o + 2] if per_spin > 2 else A[:, o])
    return A[:, 0], out


def main(res, stem):
    apply_style("paper")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for ax, (sysd, cpu, gpu, title, what) in zip(axes.ravel(), PANELS):
        Ec, cc = read_cohpcar(f"{res}/{sysd}/{cpu}/COHPCAR.lobster")
        Eg, gg = read_cohpcar(f"{res}/{sysd}/{gpu}/COHPCAR.lobster")
        k = 0 if what == "avg" else 1
        dmax, peak = 0.0, 0.0
        for s in cc:
            yc, yg = -cc[s][k], -gg[s][k]
            lab = "gpucohp (GPU)" if len(cc) == 1 else f"gpucohp, spin {'↑↓'[s]}"
            ax.plot(Eg, yg, color=SERIES[s][0], lw=1.8, label=lab, zorder=4)
            ax.plot(Ec, yc, ls="--", color="0.35", lw=1.3, zorder=5,
                    label="LOBSTER 5.1.1 (CPU)" if s == 0 else None)
            yi = np.interp(Ec, Eg, yg)
            dmax, peak = max(dmax, np.abs(yi - yc).max()), max(peak, np.abs(yc).max())
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(Ec.min(), Ec.max())
        ax.set_title(title)
        ax.set_xlabel("$E - E_\\mathrm{F}$ (eV)")
        ax.set_ylabel("$-$pCOHP (eV)")
        stats_box(ax, "\n".join([f"max |Δ| = {dmax:.4f} eV", f"peak = {peak:.2f} eV", NOTES[sysd]]),
                  loc="lower left", fontsize=9)
        finish_axes(ax)
        if sysd in ("diamond", "feni3"):
            ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
