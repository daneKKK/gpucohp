"""Spherical Bessel functions j_L(x), L <= 6, vectorised (numpy or torch),
stable for small arguments (series) and explicit sin/cos forms otherwise."""
import math
import numpy as np


def spherical_jn(L, x):
    xp = np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            xp = torch
    except ImportError:
        pass
    x = xp.asarray(x, dtype=float) if xp is np else x
    small = xp.abs(x) < 0.5 * (L + 1) ** 0.5 + 0.1 if L > 0 else xp.abs(x) < 1e-3
    xs = xp.where(small, xp.ones_like(x), x)          # safe argument for the closed form
    s, c = xp.sin(xs), xp.cos(xs)
    inv = 1.0 / xs
    if L == 0:
        f = s * inv
    elif L == 1:
        f = (s * inv - c) * inv
    elif L == 2:
        f = ((3 * inv * inv - 1) * s - 3 * inv * c) * inv
    elif L == 3:
        f = ((15 * inv ** 3 - 6 * inv) * s - (15 * inv * inv - 1) * c) * inv
    elif L == 4:
        f = ((105 * inv ** 4 - 45 * inv * inv + 1) * s - (105 * inv ** 3 - 10 * inv) * c) * inv
    elif L == 5:
        f = ((945 * inv ** 5 - 420 * inv ** 3 + 15 * inv) * s - (945 * inv ** 4 - 105 * inv * inv + 1) * c) * inv
    elif L == 6:
        f = ((10395 * inv ** 6 - 4725 * inv ** 4 + 210 * inv * inv - 1) * s
             - (10395 * inv ** 5 - 1260 * inv ** 3 + 21 * inv) * c) * inv
    else:
        raise ValueError("L > 6 not implemented")
    # series for small x: j_L(x) = x^L/(2L+1)!! * sum_k (-x^2/2)^k / (k! (2L+3)(2L+5)...(2L+2k+1))
    x2 = x * x
    term = xp.ones_like(x)
    ser = xp.ones_like(x)
    for k in range(1, 12):
        term = term * (-0.5 * x2) / (k * (2 * L + 2 * k + 1))
        ser = ser + term
    dfact = 1.0
    for m in range(1, 2 * L + 2, 2):
        dfact *= m
    fs = (x ** L) / dfact * ser
    return xp.where(small, fs, f)
