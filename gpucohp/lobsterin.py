"""Parser for the (case-insensitive) lobsterin control file.

Supported keywords (subset of LOBSTER's):
  COHPstartEnergy, COHPendEnergy, COHPSteps, gaussianSmearingWidth
  basisSet / useBasisSet <name>, basisFunctions <El> <orbitals...>
  includeOrbitals <spdf letters>     (valence shells from the POTCAR, restricted to these l)
  customSTOforAtom <El> <file>
  cohpBetween atom i atom j [cell a b c] [orbitalWise]
  cohpGenerator from <d1> to <d2> [type A type B] [orbitalWise]
  skipDOS, skipCOOP, skipCOHP, skipCOBI, skipPopulationAnalysis, skipGrossPopulation
  saveProjectionToFile / loadProjectionFromFile (accepted; gpucohp always projects)
Every other keyword is reported by gpucohp as not supported and ignored.
  augmentation onsite|full           (gpucohp extension; default onsite = LOBSTER behaviour)
  device cpu|cuda                    (gpucohp extension)
  precision single|double            (gpucohp extension)
"""
import re


# keywords whose absence changes no result: accepted without a warning
HARMLESS = {"saveprojectiontofile", "loadprojectionfromfile", "writebasisfunctions"}


class LobsterIn:
    def __init__(self):
        self.estart, self.eend, self.esteps = -10.0, 5.0, None   # None: 0.05 eV spacing, as LOBSTER
        self.sigma = 0.2
        self.basisset = "pbevaspfit2015"      # LOBSTER default; gpucohp falls back to koga
        self.basisfunctions = {}          # element -> list of labels
        self.custom_sto = {}              # element -> file
        self.include_orbitals = None      # e.g. "sp": l-channels kept when choosing valence shells
        self.between = []                 # (i, j, cell(3), orbitalwise)
        self.generators = []              # (dmin, dmax, typeA, typeB, orbitalwise)
        self.skip = set()
        self.flags = set()
        self.augmentation = "onsite"
        self.device = "auto"
        self.precision = "double"
        self.integration = "auto"         # "gaussian": gpucohp extension forceGaussianSmearing
        self.raw = []

    @classmethod
    def read(cls, path="lobsterin"):
        li = cls()
        for raw in open(path):
            line = re.split(r"[!#]|//", raw)[0].strip()
            if not line:
                continue
            li.raw.append(line)
            t = line.split()
            key = t[0].lower()
            args = t[1:]
            if key == "cohpstartenergy":
                li.estart = float(args[0])
            elif key == "cohpendenergy":
                li.eend = float(args[0])
            elif key == "cohpsteps":
                li.esteps = int(args[0])
            elif key == "gaussiansmearingwidth":
                li.sigma = float(args[0])
            elif key in ("forcegaussiansmearing", "usegaussiansmearing"):
                li.integration = "gaussian"
            elif key in ("basisset", "usebasisset"):
                li.basisset = args[0].lower()
            elif key == "basisfunctions":
                li.basisfunctions[args[0]] = [a.lower() for a in args[1:]]
            elif key == "includeorbitals":
                li.include_orbitals = "".join(args).lower()
            elif key == "customstoforatom":
                li.custom_sto[args[0]] = args[1]
            elif key == "cohpbetween":
                low = [a.lower() for a in args]
                i = int(low[low.index("atom") + 1])
                j = int(low[low.index("atom", low.index("atom") + 1) + 1])
                cell = (0, 0, 0)
                if "cell" in low:
                    c = low.index("cell")
                    cell = tuple(int(x) for x in low[c + 1:c + 4])
                li.between.append((i, j, cell, "orbitalwise" in low))
            elif key == "cohpgenerator":
                low = [a.lower() for a in args]
                d1 = float(low[low.index("from") + 1]); d2 = float(low[low.index("to") + 1])
                types = [args[k + 1] for k, a in enumerate(low) if a == "type"]
                ta, tb = (types + [None, None])[:2]
                li.generators.append((d1, d2, ta, tb, "orbitalwise" in low))
            elif key.startswith("skip"):
                li.skip.add(key[4:])
            elif key == "augmentation":
                li.augmentation = args[0].lower()
            elif key == "device":
                li.device = args[0].lower()
            elif key == "precision":
                li.precision = args[0].lower()
            else:
                li.flags.add(t[0])
        li.unsupported = sorted(k for k in li.flags if k.lower() not in HARMLESS)
        if li.esteps is None:
            li.esteps = int(round((li.eend - li.estart) / 0.05)) + 1
        return li
