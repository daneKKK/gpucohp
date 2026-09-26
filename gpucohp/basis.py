"""Assemble the local (LCAO) basis for a structure.

LOBSTER ordering: atoms in POSCAR order; within an atom the radial functions
sorted by l, then n; within a radial function m = -l..l (VASP real harmonics).
"""
import numpy as np
from .sto import BasisFunction, RadialSTO


class LocalBasis:
    def __init__(self, structure, radials_per_element, requested):
        """radials_per_element: {el: {label: RadialSTO}}
        requested: {el: [labels]} e.g. {"Ti": ["3s","3p","3d","4s"]}"""
        self.structure = structure
        self.functions = []        # list of BasisFunction
        self.radials = {}          # (element,label) -> RadialSTO (unique radials)
        self.atom_slices = []
        for ia, el in enumerate(structure.symbol_of_atom):
            if el not in requested:
                raise ValueError(f"no basis functions requested for element {el}")
            labels = sorted(requested[el], key=lambda s: (int(_letter_l(s[-1])), int(s[:-1])))
            start = len(self.functions)
            for lab in labels:
                if lab not in radials_per_element.get(el, {}):
                    raise ValueError(f"basis set has no {lab} function for {el}. Polarisation functions such as 4p are "
                                 f"only in LOBSTER's pbeVaspFit2015: import them from your own LOBSTER run "
                                 f"with 'gpucohp-basis import <dir>', or leave {lab} out of basisFunctions")
                R = radials_per_element[el][lab]
                self.radials[(el, lab)] = R
                for m in range(2 * R.l + 1):
                    self.functions.append(BasisFunction(ia, el, R, m))
            self.atom_slices.append(slice(start, len(self.functions)))
        self.nbasis = len(self.functions)
        self.iatom = np.array([f.iatom for f in self.functions])
        self.l = np.array([f.radial.l for f in self.functions])
        self.m = np.array([f.m for f in self.functions])
        # index of radial (unique) per function
        keys = list(self.radials.keys())
        self.radial_keys = keys
        self.iradial = np.array([keys.index((f.element, f.radial.label)) for f in self.functions])

    def names(self):
        return [f"{f.element}{f.iatom + 1}_{f.name}" for f in self.functions]

    def describe(self):
        out = []
        for ia, el in enumerate(self.structure.symbol_of_atom):
            sl = self.atom_slices[ia]
            out.append(f"{el:>2s} " + " ".join(f.name for f in self.functions[sl]))
        return "\n".join(out)


def _letter_l(c):
    return {"s": 0, "p": 1, "d": 2, "f": 3}[c]
