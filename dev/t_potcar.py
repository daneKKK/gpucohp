import sys, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp")
from gpucohp.potcar import read_potcar
from scipy.special import spherical_jn
sp = read_potcar(r"D:\work\lobster\ref_TiB2_sber\POTCAR")
for s in sp:
    print(s)
    print(" rmax per channel (A):", [c["rmax"] for c in s.channels], " rcore", round(s.rcore, 4),
          "rdep", round(s.rdep, 4), "rmax_proj", round(s.rmax_proj, 4))
    print(" grid: r0=%.3e rN=%.4f A  n=%d" % (s.rgrid[0], s.rgrid[-1], len(s.rgrid)))
    q = s.qij_from_waves()
    print(" Qij file:\n", s.qij.round(4))
    print(" Qij waves:\n", q.round(4))
    for c in s.channels:
        r = np.linspace(0, c["rmax"], 100)
        for ip in range(c["nproj"]):
            pr = c["pr"][ip]
            qq = s.qgrid
            pq_num = np.array([4 * np.pi * np.trapz(r ** 2 * pr * spherical_jn(c["l"], qv * r), r) for qv in qq])
            sel = slice(1, 20)
            ratio = c["pq"][ip][sel] / np.where(np.abs(pq_num[sel]) > 1e-8, pq_num[sel], np.nan)
            print(f"  l={c['l']} ip={ip}: pq[0..3]={c['pq'][ip][:4].round(4)} num[0..3]={pq_num[:4].round(4)} "
                  f"ratio~{np.nanmean(ratio):.5f} +- {np.nanstd(ratio):.1e}")
    for i, (ic, l, ii) in enumerate(s.proj_table):
        print(f"  wf {i} l={l}: ae(rN)={s.ae_wf[i][-1]:.4f} ps(rN)={s.ps_wf[i][-1]:.4f} "
              f"|ae-ps|(rN)={abs(s.ae_wf[i][-1]-s.ps_wf[i][-1]):.2e}")
