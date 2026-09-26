"""hcp Ti, Ti-Ti bond: open Koga basis (gpucohp default) vs LOBSTER's
pbeVaspFit2015 (user-imported), each computed with LOBSTER 5.1.1 (grey) and
gpucohp (colour), on the same WAVECAR.

    python plot_basis.py RESULTS_DIR OUT_STEM
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\Daniil\.claude\skills\sci-figures\scripts")
from sciplot import apply_style, SERIES, finish_axes, stats_box, save_figure
from plot_cohp import read_cohpcar

# label, LOBSTER run, gpucohp run, colour
CASES = [("pbeVaspFit2015 4s 3d 4p (imported)", "cpu", "gpu_pbe_tet", SERIES[0][0]),
         ("Koga 4s 3d (gpucohp default)", "cpu_koga", "gpu_koga_tet", SERIES[1][0])]


def read_cols(path):
    E, d = read_cohpcar(path)
    lines = open(path).read().splitlines()
    rows = [[float(x) for x in l.split()] for l in lines[2:] if l.split() and l.split()[0][0] in "-0123456789."
            and len(l.split()) > 2]
    A = np.array(rows)
    return A[:, 0], -A[:, 3], -A[:, 4]          # E, -pCOHP, -ICOHP of the one pair


def spill(path):
    import re
    return float(re.findall(r"abs\. charge spilling:\s+([\d.]+)%", open(path).read())[0])


def main(res, stem):
    apply_style("paper")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.9))
    lines = []
    for lab, cpu, gpu, col in CASES:
        Ec, cc, ci = read_cols(f"{res}/ti/{cpu}/COHPCAR.lobster")
        Eg, gc, gi = read_cols(f"{res}/ti/{gpu}/COHPCAR.lobster")
        a1.plot(Eg, gc, color=col, lw=1.8, label=lab, zorder=4)
        a1.plot(Ec, cc, ls="--", color="0.35", lw=1.2, zorder=5)
        a2.plot(Eg, gi, color=col, lw=1.8, zorder=4)
        a2.plot(Ec, ci, ls="--", color="0.35", lw=1.2, zorder=5)
        i0g, i0c = np.argmin(np.abs(Eg)), np.argmin(np.abs(Ec))
        lines.append(f"{lab.split(' (')[0]}:\n  −ICOHP(E$_F$) {gi[i0g]:.2f} / {ci[i0c]:.2f} eV,"
                     f" spilling {spill(f'{res}/ti/{gpu}/lobsterout'):.1f} / {spill(f'{res}/ti/{cpu}/lobsterout'):.1f} %")
    a1.plot([], [], ls="--", color="0.35", lw=1.2, label="LOBSTER 5.1.1, same basis")
    for ax in (a1, a2):
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(-8, 6)
        ax.set_xlabel("$E - E_\\mathrm{F}$ (eV)")
        finish_axes(ax)
    a1.set_ylabel("$-$pCOHP, Ti–Ti (eV)")
    a2.set_ylabel("$-$ICOHP up to $E$, Ti–Ti (eV)")
    a1.set_title("hcp Ti, 2.86 Å bond: $-$pCOHP")
    a2.set_title("integrated: gpucohp / LOBSTER")
    a1.legend(loc="lower left", fontsize=9)
    stats_box(a2, "\n".join(lines), loc="upper left", fontsize=9)
    a2.set_ylim(top=a2.get_ylim()[1] * 1.45)
    fig.tight_layout()
    save_figure(fig, stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
