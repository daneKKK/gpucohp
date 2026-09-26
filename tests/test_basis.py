"""Open basis sets: rebuilt from the published tables, normalised, and their
analytic Fourier-Bessel transform agrees with numerical quadrature."""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from gpucohp.sto import read_lobster_basisfunctions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "gpucohp", "basis_data")


@pytest.mark.parametrize("tool,src,name", [
    ("make_koga_basis.py", "third_party/koga_thakkar", "koga.lobster"),
    ("make_bunge_basis.py", "third_party/bunge1993/RHF.TABLES.txt", "bunge.lobster"),
])
def test_shipped_basis_is_rebuilt_from_published_tables(tmp_path, tool, src, name):
    out = tmp_path / name
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), os.path.join(ROOT, src), str(out)],
                   check=True, capture_output=True)
    assert out.read_bytes() == open(os.path.join(DATA, name), "rb").read()


@pytest.mark.parametrize("name,nel", [("koga.lobster", 103), ("bunge.lobster", 53)])
def test_all_functions_normalised(name, nel):
    lib = read_lobster_basisfunctions(os.path.join(DATA, name))
    assert len(lib) == nel
    for el, fns in lib.items():
        for lab, f in fns.items():
            assert abs(f.selfoverlap() - 1.0) < 2e-4, (el, lab)   # Bunge prints 6 digits


@pytest.mark.parametrize("el,lab", [("Ti", "3d"), ("B", "2p"), ("Ti", "4s"), ("Hf", "5p")])
def test_fourier_bessel_transform(el, lab):
    f = read_lobster_basisfunctions(os.path.join(DATA, "koga.lobster"))[el][lab]
    for q in (0.0, 0.7, 2.5, 6.0):
        num = quad(lambda r: r * r * f.R(np.array([r]))[0] * spherical_jn(f.l, q * r), 0, 40, limit=400)[0]
        assert abs(f.Rt(np.array([q]))[0] - num) < 1e-7 * max(1.0, abs(num))
