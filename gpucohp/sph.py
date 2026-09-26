"""Real spherical harmonics in the VASP / LOBSTER ordering (m = -l..l):

l=0:  s
l=1:  p_y, p_z, p_x
l=2:  d_xy, d_yz, d_z2, d_xz, d_x2-y2
l=3:  f_y(3x2-y2), f_xyz, f_yz2, f_z3, f_xz2, f_z(x2-y2), f_x(x2-3y2)

All functions take a (...,3) array of Cartesian vectors and return (..., 2l+1).
They work on both numpy arrays and torch tensors (only elementary ops used).
Vectors of zero length give Y_00 = 1/sqrt(4pi) and 0 for l>0.
"""
import math
import numpy as np

ORB_NAMES = {
    0: ["s"],
    1: ["p_y", "p_z", "p_x"],
    2: ["d_xy", "d_yz", "d_z^2", "d_xz", "d_x^2-y^2"],
    3: ["f_y(3x^2-y^2)", "f_xyz", "f_yz^2", "f_z^3", "f_xz^2", "f_z(x^2-y^2)", "f_x(x^2-3y^2)"],
}
L_OF_LETTER = {"s": 0, "p": 1, "d": 2, "f": 3}


def _xp(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return torch
    except ImportError:
        pass
    return np


def real_ylm(l, v, eps=1e-30):
    """Real spherical harmonics Y_lm(v_hat) for a single l, shape (...,2l+1)."""
    xp = _xp(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r2 = x * x + y * y + z * z
    inv = 1.0 / xp.sqrt(xp.clip(r2, eps, None) if xp is np else r2.clamp(min=eps))
    x, y, z = x * inv, y * inv, z * inv
    if xp is np:
        stack = lambda lst: np.stack(lst, axis=-1)
        ones = np.ones_like(x)
    else:
        stack = lambda lst: xp.stack(lst, dim=-1)
        ones = xp.ones_like(x)
    pi = math.pi
    if l == 0:
        return stack([ones * math.sqrt(1.0 / (4 * pi))])
    if l == 1:
        c = math.sqrt(3.0 / (4 * pi))
        return stack([c * y, c * z, c * x])
    if l == 2:
        c1 = math.sqrt(15.0 / (4 * pi))
        c0 = math.sqrt(5.0 / (16 * pi))
        c2 = math.sqrt(15.0 / (16 * pi))
        return stack([c1 * x * y, c1 * y * z, c0 * (3 * z * z - 1.0), c1 * x * z, c2 * (x * x - y * y)])
    if l == 3:
        c3 = math.sqrt(35.0 / (32 * pi))
        c2 = math.sqrt(105.0 / (4 * pi))
        c1 = math.sqrt(21.0 / (32 * pi))
        c0 = math.sqrt(7.0 / (16 * pi))
        return stack([
            c3 * y * (3 * x * x - y * y),
            c2 * x * y * z,
            c1 * y * (5 * z * z - 1.0),
            c0 * z * (5 * z * z - 3.0),
            c1 * x * (5 * z * z - 1.0),
            math.sqrt(105.0 / (16 * pi)) * z * (x * x - y * y),
            c3 * x * (x * x - 3 * y * y),
        ])
    # general l > 3 (numpy only): real combinations of scipy complex harmonics.
    # Only used internally (Gaunt coefficients / Y_LM(d)), so the sign convention
    # merely has to be consistent between those two uses.
    if xp is not np:
        raise ValueError("l > 3 only supported for numpy input")
    from scipy import special as _sp
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    cols = []
    for m in range(-l, l + 1):
        if hasattr(_sp, "sph_harm_y"):
            Yc = _sp.sph_harm_y(l, abs(m), theta, phi)
        else:
            Yc = _sp.sph_harm(abs(m), l, phi, theta)
        if m == 0:
            cols.append(Yc.real)
        elif m > 0:
            cols.append(math.sqrt(2.0) * (-1) ** m * Yc.real)
        else:
            cols.append(math.sqrt(2.0) * (-1) ** m * Yc.imag)
    return np.stack(cols, axis=-1)


def angular_grid(ntheta=32, nphi=64):
    """Product Gauss-Legendre (theta) x uniform (phi) quadrature on the unit sphere.
    Returns unit vectors (n,3) and weights (n,) summing to 4*pi."""
    xg, wg = np.polynomial.legendre.leggauss(ntheta)
    phi = np.arange(nphi) * 2 * np.pi / nphi
    wphi = 2 * np.pi / nphi
    ct = xg[:, None]
    st = np.sqrt(1 - ct ** 2)
    x = (st * np.cos(phi)[None, :]).ravel()
    y = (st * np.sin(phi)[None, :]).ravel()
    z = (ct * np.ones_like(phi)[None, :]).ravel()
    w = (wg[:, None] * wphi * np.ones_like(phi)[None, :]).ravel()
    return np.stack([x, y, z], axis=1), w


def gaunt_real(lmax=3, ntheta=48, nphi=96):
    """Real Gaunt coefficients G[(l1,m1),(l2,m2),(L,M)] = int Y Y Y dOmega
    computed by quadrature.  Returned as dict l1,l2 -> array (2l1+1, 2l2+1, Lmax_tot, ...)
    Simpler: return function values as nested dict keyed by (l1,l2,L) -> array (2l1+1,2l2+1,2L+1)."""
    pts, w = angular_grid(ntheta, nphi)
    Y = {l: real_ylm(l, pts) for l in range(0, 2 * lmax + 1)}
    out = {}
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + L) % 2:
                    continue
                g = np.einsum("na,nb,nc,n->abc", Y[l1], Y[l2], Y[L], w)
                g[np.abs(g) < 1e-12] = 0.0
                out[(l1, l2, L)] = g
    return out
