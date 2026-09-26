"""lobsterin parsing, valence-shell choice, user basis library, output format."""
import os

import numpy as np
import pytest

from gpucohp import output
from gpucohp.cli import load_basis_library, valence_orbitals
from gpucohp.lobsterin import LobsterIn
from gpucohp.structure import ATOMIC_NUMBER
from gpucohp import userbasis


def _lobsterin(tmp_path, text):
    p = tmp_path / "lobsterin"
    p.write_text(text)
    return LobsterIn.read(str(p))


def test_lobsterin_defaults_and_unsupported(tmp_path):
    li = _lobsterin(tmp_path, "COHPstartEnergy -14\nCOHPendEnergy 6\nbasisSet Bunge\nincludeOrbitals sp\n"
                              "cohpbetween atom 1 and atom 2\nsaveProjectionToFile\nkSpaceCOHP\nwriteMatricesToFile\n")
    assert li.esteps == 401                              # LOBSTER: 0.05 eV spacing
    assert li.basisset == "bunge" and li.include_orbitals == "sp"
    assert li.between == [(1, 2, (0, 0, 0), False)]
    assert li.unsupported == ["kSpaceCOHP", "writeMatricesToFile"]   # harmless keywords stay silent


@pytest.mark.parametrize("el,zval,expect,libname", [
    ("C", 4, ["2s", "2p"], "bunge"), ("Ga", 3, ["4s", "4p"], "bunge"), ("Ga", 13, ["3d", "4s", "4p"], "bunge"),
    ("Ti", 12, ["3s", "3p", "3d", "4s"], "koga"), ("Hf", 10, ["5p", "5d", "6s"], "koga"),
    ("Nb", 13, ["4s", "4p", "4d", "5s"], "koga"), ("Fe", 8, ["3d", "4s"], "koga")])
def test_valence_orbitals_follow_potcar(el, zval, expect, libname, monkeypatch, tmp_path):
    monkeypatch.setenv("GPUCOHP_BASIS_DIR", str(tmp_path))
    lib = load_basis_library(libname, None, {}, log=lambda *a: None)
    assert valence_orbitals(list(lib[el]), ATOMIC_NUMBER[el], zval) == expect


def test_fallback_to_koga_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUCOHP_BASIS_DIR", str(tmp_path))
    msgs = []
    lib = load_basis_library("pbeVaspFit2015", None, {}, elements=["Ti", "B"], log=msgs.append)
    assert lib.fallback == ["Ti", "B"] and "Koga" in msgs[0]


DUMP = """This file lists the basisfunctions in the same order as all matrices employ them.
q and coeff are dimensionless, but alpha is given in 1/a_0

Ti 4p_y           at atom 1
q        alpha       coeff
4     1.200000    1.000000

Ti 4p_z           at atom 1
q        alpha       coeff
4     1.200000    1.000000

C 2s              at atom 2
q        alpha       coeff
2     1.600000    1.000000
"""
OUT = """setting up local basis functions...
Ti (pbevaspfit2015) 4p_y 4p_z 4p_x
 C (bunge) 2s
"""


def test_user_basis_import(tmp_path, monkeypatch):
    run, lib = tmp_path / "run", tmp_path / "lib"
    run.mkdir()
    (run / "basisFunctions.lobster").write_text(DUMP)
    (run / "lobsterout").write_text(OUT)
    monkeypatch.setenv("GPUCOHP_BASIS_DIR", str(lib))
    userbasis.import_run(str(run))
    userbasis.import_run(str(run))                       # second import adds nothing
    assert sorted(os.listdir(lib)) == ["pbevaspfit2015.lobster"]   # bunge is shipped, not imported
    got = load_basis_library("pbeVaspFit2015", None, {}, elements=["Ti"], log=lambda *a: None)
    assert list(got["Ti"]) == ["4p"] and got.fallback == []


def test_grosspop_layout_per_spin(tmp_path):
    class B:
        atom_slices = [slice(0, 2)]
        functions = [type("f", (), {"name": "4s"})(), type("f", (), {"name": "3d_xy"})()]

    class S:
        natoms, symbol_of_atom = 1, ["Fe"]

    p = tmp_path / "GROSSPOP.lobster"
    output.write_grosspop(str(p), S, B, [np.array([.59, .96]), np.array([.64, .43])],
                          [np.array([.55, .96]), np.array([.61, .44])])
    lines = p.read_text().splitlines()
    assert lines[2].split().count("Mulliken") == 2 and "For spin 2" in lines[3]
    assert lines[4].split()[-4:] == ["0.59", "0.64", "0.55", "0.61"]
