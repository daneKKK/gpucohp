"""k-point weights: vasprun.xml (full precision) with fallback to equal weights."""
import re
import numpy as np


def read_kweights(vasprun="vasprun.xml", nk=None):
    try:
        txt = open(vasprun).read()
        m = re.search(r'<varray name="weights"\s*>(.*?)</varray>', txt, re.S)
        w = np.array([float(x) for x in re.findall(r"<v>\s*([-\d.Ee+]+)\s*</v>", m.group(1))])
        if nk is not None and len(w) != nk:
            raise ValueError(f"vasprun.xml has {len(w)} k-point weights, WAVECAR has {nk}")
        return w / w.sum()
    except (OSError, AttributeError):
        if nk is None:
            raise
        return np.full(nk, 1.0 / nk)
