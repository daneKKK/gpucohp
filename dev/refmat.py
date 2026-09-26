"""Parsers for LOBSTER's writeMatricesToFile dumps (overlap / transfer /
coefficient matrices) used as numerical references in tests."""
import re
import numpy as np


def read_matrices(path, maxk=None):
    """Return list of dicts: {kpoint(3 floats), names(list), mat(complex ndarray)}"""
    txt = open(path).read()
    blocks = re.split(r"\n(?=\w+ matrix for )", txt)
    out = []
    for b in blocks:
        m = re.match(r"(\w+) matrix for (?:spin (\d+) )?kpoint (\d+) at (\S+)\s+(\S+)\s+(\S+)", b)
        if not m:
            continue
        kpt = np.array([float(m.group(i)) for i in (4, 5, 6)])
        parts = re.split(r"\n\s*(Real parts|Imagw* parts)\s*\n", b)
        mats = {}
        for i in range(1, len(parts) - 1, 2):
            tag, data = parts[i], parts[i + 1]
            lines = [ln for ln in data.splitlines() if ln.strip()]
            hdr = lines[0].split()
            cols = hdr[1:]
            rows, names = [], []
            for ln in lines[1:]:
                p = ln.split()
                if not p:
                    continue
                # row label may contain spaces for "band 1"? columns are 'band', '1' pairs
                # row labels are basis function names (single token)
                try:
                    vals = [float(x) for x in p[1:]]
                except ValueError:
                    break
                names.append(p[0])
                rows.append(vals)
            mats[tag] = np.array(rows)
        ik = int(m.group(3))
        # LOBSTER writes real and imaginary parts as two separate blocks
        if out and out[-1]["ik"] == ik and out[-1]["kind"] == m.group(1):
            for tag, arr in mats.items():
                out[-1]["mat"] = out[-1]["mat"] + (1j * arr if tag.startswith("Imag") else arr)
            continue
        if maxk and len(out) >= maxk:
            break
        mat = 0
        for tag, arr in mats.items():
            mat = mat + (1j * arr if tag.startswith("Imag") else arr)
        out.append(dict(kind=m.group(1), kpoint=kpt, ik=ik, names=names, mat=np.asarray(mat, dtype=complex)))
    return out
