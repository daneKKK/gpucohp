"""Diagnostic: does the B-B ICOHP difference come from LOBSTER's overlap matrices?
Run projection on the k subset available in the LOBSTER dump with (a) our S(k), (b) LOBSTER's S(k)."""
import sys, time, math, numpy as np, torch
sys.path.insert(0, r"D:\work\lobster\gpucohp"); sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.paw import PawSetup
from gpucohp.augment import Augmentation
from gpucohp.twocenter import RadialFTTable, TwoCenterOverlap
from gpucohp.engine import ProjectionEngine, _hermitian_power
from gpucohp.lobsterin import LobsterIn
from gpucohp.analysis import build_pairs, EnergyGrid, Analysis
from refmat import read_matrices
REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR"); st = Structure.from_poscar(REF + r"\CONTCAR"); sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
aug = Augmentation(basis, st, paw)
table = RadialFTTable(basis.radials); tc = TwoCenterOverlap(basis, st, table, rmax_pair=17.4)
eng = ProjectionEngine(w, st, basis, paw, aug, tc, device="cpu")
refS = read_matrices(REF + r"\overlapMatrices.lobster")
nks = len(refS); print("k-points in dump:", nks)
li = LobsterIn.read(REF + r"\lobsterin"); pairs = build_pairs(st, li)
sel = [p for p in pairs if not (p.iA == p.iB)]  # Ti-B and B-B
for p in pairs: p.orbitalwise = False
eg = EnergyGrid(li.estart, li.eend, li.esteps, w.efermi)
def run(use_ref):
    eng.results = [[None] * w.nkpts]
    for ik in range(nks):
        r = eng.project_k(0, ik)
        if use_ref:
            # redo the post-processing with LOBSTER's S
            S = torch.tensor(refS[ik]["mat"]); S = 0.5*(S+S.mH)
            # need T: reconstruct from our S and C1? Use T = S_ours C_n; we stored only C1. Recompute T:
            T = r["_T"]
            Cn = torch.linalg.solve(S, T); O = Cn.mH @ S @ Cn; C1 = Cn @ _hermitian_power(O, -0.5)
            Sh = _hermitian_power(S, 0.5); C2 = Sh @ C1; E = r["eig"]
            H2 = (C2 * E.to(C2.dtype)) @ C2.mH; H2 = 0.5*(H2+H2.mH)
            r.update(dict(C1=C1, C2=C2, S=S, H2=H2))
        eng.results[0][ik] = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in r.items()}
    # restrict analysis to the k subset with equal weights
    eng.kweights = np.zeros(w.nkpts); eng.kweights[:nks] = 1.0 / nks
    an = Analysis(eng, st, basis, pairs, eg, li.sigma, device="cpu")
    an.wav = w
    # monkeypatch nkpts to subset
    class W: pass
    wsub = W(); wsub.nkpts = nks; wsub.kpoints = w.kpoints[:nks]; wsub.eig = w.eig[:, :nks]; wsub.occ = w.occ[:, :nks]; wsub.efermi = w.efermi; wsub.nspin = 1
    an.wav = wsub; an.w = eng.kweights[:nks]
    res = an.bonding("cohp", 0)
    return res["at_ef"]
# expose T from project_k
import gpucohp.engine as E_
orig = E_.ProjectionEngine.project_k
def pk(self, s, ik):
    r = orig(self, s, ik)
    # recompute T cheaply: T = S C_n where C_n = C1 O^{1/2}; simpler: store from internals -> re-derive via S C1 (O^{1/2}) unknown; so recompute directly
    return r
# simplest: re-implement project_k partially to capture T
def project_k_T(self, ispin, ik):
    wav = self.wav; dev, rd, cd = self.device, self.rdtype, self.cdtype
    G = torch.tensor(wav.gvectors(ik), device=dev, dtype=rd); kfrac = torch.tensor(wav.kpoints[ik], device=dev, dtype=rd)
    qcart = (G + kfrac) @ self.rec; C = torch.tensor(wav.coeffs(ispin, ik)[: self.Nb], device=dev).to(cd)
    cb = self._atom_centered_pw(qcart, "basis"); cp = self._atom_centered_pw(qcart, "proj")
    P = cp.conj() @ C.T; T = cb.conj() @ C.T + self.Ak(kfrac) @ P
    r = orig(self, ispin, ik); r["_T"] = T
    return r
E_.ProjectionEngine.project_k = project_k_T
a = run(False); b = run(True)
for p, x, y in zip(pairs, a, b):
    if p.index in (1, 2, 9, 15, 33, 34, 36):
        print(f"pair {p.index:2d} {st.symbol_of_atom[p.iA]}{p.iA+1}-{st.symbol_of_atom[p.iB]}{p.iB+1} {p.cell} d={p.dist:.3f}: ICOHP ours-S {x:9.5f}   LOBSTER-S {y:9.5f}   ratio {y/x:.4f}")
