"""LOBSTER 5.1.1 (res/<sys>/cpu) vs gpucohp (res/<sys>/gpu) for the benchmark systems."""
import re, sys, os

def ilist(path):
    """{(A, B, T, spin): value}. Handles spin as extra columns or as separate blocks."""
    rows, spin_block = {}, 1
    for line in open(path):
        if "for spin 2" in line and "for spin 1" not in line:
            spin_block = 2
        f = line.split()
        if len(f) < 8 or not f[0].isdigit():
            continue
        key = (f[1], f[2], int(f[4]), int(f[5]), int(f[6]))
        vals = [float(x) for x in f[7:]]
        if len(vals) == 2:
            rows[key + (1,)], rows[key + (2,)] = vals
        else:
            rows[key + (spin_block,)] = vals[0]
    return rows

def charges(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) >= 4 and f[0].isdigit():
            out.append((float(f[2]), float(f[3])))
    return out

def spill(path):
    return [float(x) for x in re.findall(r"abs\. charge spilling:\s+([\d.]+)%", open(path).read())]

def wall(path):
    t = open(path).read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    p = [float(x) for x in m.group(1).split(":")]
    s = p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) > 2 else 0)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1024 ** 2
    return s, rss

print(f"{'system':8s} {'rows':>5s} {'max|dICOHP|':>11s} {'mean':>8s} {'max|dICOOP|':>11s} {'worst ICOHP row':32s} "
      f"{'spill CPU':>12s} {'spill GPU':>12s} {'dq Mull/Loew':>12s} {'LOBSTER s':>9s} {'gpucohp s':>9s} {'RSS GB':>11s}")
for s in sys.argv[1:]:
    c, g = f"{s}/reference_lobster", f"{s}/reference_gpucohp"
    if not os.path.exists(g + "/ICOHPLIST.lobster"):
        continue
    A, B = ilist(c + "/ICOHPLIST.lobster"), ilist(g + "/ICOHPLIST.lobster")
    com = [k for k in A if k in B]
    d = [abs(A[k] - B[k]) for k in com]
    w = max(com, key=lambda k: abs(A[k] - B[k]))
    Ao, Bo = ilist(c + "/ICOOPLIST.lobster"), ilist(g + "/ICOOPLIST.lobster")
    do = max(abs(Ao[k] - Bo[k]) for k in Ao if k in Bo)
    qa, qb = charges(c + "/CHARGE.lobster"), charges(g + "/CHARGE.lobster")
    dm = max(abs(x[0] - y[0]) for x, y in zip(qa, qb)); dl = max(abs(x[1] - y[1]) for x, y in zip(qa, qb))
    tc, rc = wall(c + "/time.txt"); tg, rg = wall(g + "/time2.txt")
    wl = f"{w[0]}-{w[1]} s{w[5]} {A[w]:.3f}/{B[w]:.3f}"
    print(f"{s:8s} {len(com):5d}/{len(A)} {max(d):9.4f} {sum(d)/len(d):8.5f} {do:11.5f} {wl:32s} "
          f"{str(spill(c + '/lobsterout')):>12s} {str(spill(g + '/lobsterout')):>12s} {dm:5.2f}/{dl:4.2f}     "
          f"{tc:9.1f} {tg:9.1f} {rc:5.2f}/{rg:5.2f}")
