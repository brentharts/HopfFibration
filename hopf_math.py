#!/usr/bin/env python3
"""hopf_math.py - the Hopf fibration S^1 -> S^3 -> S^2 as plain numpy, no Blender needed.

This is the exact core the other scripts (hopf_*.py) build on.  It does not import bpy.

CONVENTIONS (fixed here, used everywhere)
  A point of S^3 is (z1, z2) in C^2 with |z1|^2 + |z2|^2 = 1, equivalently the unit quaternion
  q = a + b i + c j + d k with z1 = a + b i and z2 = c + d i.
  Hopf map     h(z1, z2) = ( 2 Re(z1 conj z2), 2 Im(z1 conj z2), |z1|^2 - |z2|^2 )  in S^2.
  Fibre        e^{it} (z1, z2): the points that differ by a common phase.
  Quaternion   with q = a + b i + c j + d k and (z1, z2) = (a + d i, c + b i):  h(z1, z2) = q k conj(q), the unit vector k
               rotated by q.  (This pairing of components is the one for which the identity holds exactly with this
               file's axes; other conventions differ by a relabelling of the axes.  Checked in the tests.)
  Stereographic projection of w = (Re z1, Im z1, Re z2, Im z2) from the pole (0,0,0,1):
               P(w) = (w1, w2, w3) / (1 - w4).  The fibre over the base point (0,0,-1) is the circle through the
               pole, i.e. a straight line (the 'circle through infinity' of the README).
  Base points  are given as unit vectors (x, y, z), or as (elevation, azimuth) with
               x = sin(el) cos(az), y = sin(el) sin(az), z = -cos(el): el = 0 is the SOUTH pole (z = -1) here.
               NOTE fibration.py counts the other way (its flat circle is elevation 0, its line is elevation pi) and
               draws the mirror image; blender_frame() converts.

What is exact and what is not: everything here is exact up to floating point.  The linking number is a numerical Gauss
integral of polygons and is accurate to a few per cent, which is enough to tell +1/-1 from 0.

    python3 hopf_math.py --tests     checks the identities above, and audits fibration.py's makeFibre geometry
    python3 hopf_math.py             prints the audit table
"""
import math
import sys

import numpy as np

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------- S^3 -> S^2

def hopf(z1, z2):
    """Hopf map.  z1, z2 may be arrays (broadcast).  Returns an array of shape (..., 3)."""
    z1 = np.asarray(z1, dtype=complex)
    z2 = np.asarray(z2, dtype=complex)
    w = z1 * np.conj(z2)
    return np.stack([2 * w.real, 2 * w.imag, np.abs(z1) ** 2 - np.abs(z2) ** 2], axis=-1)


def base_point(elevation, azimuth):
    """The point of S^2 for fibration.py's (elevation, azimuth): elevation 0 = south pole (0,0,-1), pi = north pole."""
    return np.array([math.sin(elevation) * math.cos(azimuth), math.sin(elevation) * math.sin(azimuth),
                     -math.cos(elevation)])


def section(b):
    """A point (z1, z2) of S^3 above the base point b = (x, y, z) on S^2 (unit vector)."""
    x, y, z = (float(c) for c in b)
    if z < -1 + 1e-14:
        return 0.0 + 0j, 1.0 + 0j
    z1 = math.sqrt((1 + z) / 2)
    z2 = (x - 1j * y) / (2 * z1)
    return complex(z1), complex(z2)


def fibre_s3(b, n=400):
    """The fibre over b as n points of S^3, shape (n, 2) complex: e^{it}(z1, z2)."""
    z1, z2 = section(b)
    t = np.linspace(0.0, TWO_PI, n, endpoint=False)
    e = np.exp(1j * t)
    return np.stack([e * z1, e * z2], axis=1)


# ---------------------------------------------------------------------------------------------------- S^3 -> R^3

def to_r4(zz):
    zz = np.asarray(zz)
    return np.stack([zz[..., 0].real, zz[..., 0].imag, zz[..., 1].real, zz[..., 1].imag], axis=-1)


def stereographic(zz):
    """S^3 -> R^3 from the pole (0,0,0,1).  zz shape (..., 2) complex.  The pole itself maps to infinity."""
    w = to_r4(zz)
    d = 1.0 - w[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return w[..., :3] / d[..., None]


def fibre_r3(b, n=400):
    """The fibre over b in R^3 (polygon of n points).  Over the south pole this is a straight line: use fibre_line."""
    return stereographic(fibre_s3(b, n))


def fit_circle(P):
    """Least-squares circle through points P (n,3): returns centre, radius, unit normal, max residual."""
    P = np.asarray(P, dtype=float)
    c0 = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c0)
    nrm = vt[2]
    e1, e2 = vt[0], vt[1]
    xy = np.stack([(P - c0) @ e1, (P - c0) @ e2], axis=1)
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)[0]
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    centre = c0 + cx * e1 + cy * e2
    resid = np.abs(np.linalg.norm(P - centre, axis=1) - r).max() + np.abs((P - c0) @ nrm).max()
    return centre, r, nrm, resid


# ---------------------------------------------------------------------------------------------------- linking

def linking_number(A, B):
    """Gauss linking integral of two closed polygons A (m,3), B (n,3).  Midpoint rule; accurate to a few per cent."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    dA = np.roll(A, -1, axis=0) - A
    dB = np.roll(B, -1, axis=0) - B
    mA = A + dA / 2
    mB = B + dB / 2
    r = mA[:, None, :] - mB[None, :, :]
    d3 = np.linalg.norm(r, axis=2) ** 3
    cross = np.cross(dA[:, None, :], dB[None, :, :])
    return float((np.einsum("ijk,ijk->ij", r, cross) / d3).sum() / (4 * math.pi))


def min_distance(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return float(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min())


# ---------------------------------------------------------------------------------------------------- quaternions

def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def qconj(q):
    return np.array(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


def quaternion_of(z1, z2):
    """q = a + b i + c j + d k with z1 = a + d i, z2 = c + b i  (see the module docstring)."""
    return np.array([z1.real, z2.imag, z2.real, z1.imag])


# ---------------------------------------------------------------------------------------------------- audit of fibration.py

def _blender_circle(ele, baseRad, fibreRad, azimuth, n=360):
    """The circle makeFibre places: Bezier circle of radius fibreRad in the XY plane, centre (baseRad sin az, baseRad cos az, 0),
    rotation_euler.y = ele (already halved), rotation_euler.z = -az (Blender 'XYZ' Euler: R = Rz Ry Rx)."""
    t = np.linspace(0, TWO_PI, n, endpoint=False)
    circ = fibreRad * np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    cy, sy = math.cos(ele), math.sin(ele)
    cz, sz = math.cos(-azimuth), math.sin(-azimuth)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return circ @ (Rz @ Ry).T + np.array([baseRad * math.sin(azimuth), baseRad * math.cos(azimuth), 0.0])


def blender_fibre(elevation, azimuth, n=360, exact=False):
    """The circle fibration.py's makeFibre(elevation, azimuth) builds, reproduced without bpy.

    As shipped: ele = elevation/2, baseRad = 2*(1/(pi/2 - ele) - 2/pi), fibreRad = baseRad + 1/(baseRad + 1).
    With exact=True: baseRad = tan(ele), fibreRad = sec(ele); everything else unchanged.  That is the one-line fix that
    turns the drawing into the true Hopf fibration (see audit()).  The Bezier circle is itself a 3e-4 approximation of a
    circle, which is ignored here.  Returns (points (n,3), baseRad, fibreRad)."""
    ele = elevation / 2.0
    if exact:
        baseRad, fibreRad = math.tan(ele), 1.0 / math.cos(ele)
    else:
        baseRad = 2 * ((1 / (math.pi / 2 - ele)) - (2 / math.pi))
        fibreRad = baseRad + 1 / (baseRad + 1)
    return _blender_circle(ele, baseRad, fibreRad, azimuth, n), baseRad, fibreRad


def blender_frame(P):
    """Mirror (x -> -x): fibration.py draws the mirror image (the other-handed Hopf fibration) of this file's convention."""
    return np.asarray(P) * np.array([-1.0, 1.0, 1.0])


def exact_fibre_in_blender_frame(elevation, azimuth, n=400):
    """The true Hopf fibre that fibration.py's (elevation, azimuth) is aiming at: the fibre over base_point(pi - elevation,
    azimuth), mirrored."""
    return blender_frame(fibre_r3(base_point(math.pi - elevation, azimuth), n))


def circle_params(P):
    """(centre distance from the z axis, radius, tilt of the normal from the z axis)."""
    c, r, nrm, res = fit_circle(P)
    return math.hypot(c[0], c[1]), r, math.acos(min(1.0, abs(nrm[2])))


def audit(elevations=(0.6, 1.0, 1.5, 2.0, 2.5, 3.0), azimuth=0.0):
    """Compare fibration.py's circles with the exact Hopf fibres: radius, centre distance, tilt, and whether two
    neighbouring azimuths link.  Returns a list of dict rows.

    Exact fibres satisfy  d = tan(ele), r = sec(ele), tilt = ele  (ele = elevation/2, d = distance of the centre from the
    z axis); in particular r^2 = d^2 + 1.  makeFibre's baseRad/fibreRad are a smooth approximation to tan and sec."""
    rows = []
    for el in elevations:
        P, baseRad, fibreRad = blender_fibre(el, azimuth, 720)
        Pe, _, _ = blender_fibre(el, azimuth, 720, exact=True)
        P2, _, _ = blender_fibre(el, azimuth + 0.5, 720)
        P3, _, _ = blender_fibre(el, azimuth + 1.0, 720)
        rows.append(dict(el=el, shipped=circle_params(P), fixed=circle_params(Pe),
                         exact=circle_params(exact_fibre_in_blender_frame(el, azimuth)),
                         link=linking_number(P, P2), mind=min_distance(P, P3)))
    return rows


# ---------------------------------------------------------------------------------------------------- tests

def run_tests(verbose=True):
    rng = np.random.default_rng(7)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 1. hopf lands on S^2, constant on fibres, and section() inverts it
    maxerr = 0.0
    for _ in range(50):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        z1, z2 = section(v)
        maxerr = max(maxerr, abs(abs(z1) ** 2 + abs(z2) ** 2 - 1), np.abs(hopf(z1, z2) - v).max(),
                     np.abs(hopf(np.exp(1.234j) * z1, np.exp(1.234j) * z2) - v).max())
    check("hopf(section(b)) = b, |z|=1, phase-invariant", maxerr < 1e-12, "(max err %.1e)" % maxerr)

    # 2. quaternion form  h = q k conj(q)
    err = 0.0
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        z1, z2 = complex(q[0], q[3]), complex(q[2], q[1])
        r = qmul(qmul(q, np.array([0, 0, 0, 1.0])), qconj(q))[1:]
        err = max(err, np.abs(r - hopf(z1, z2)).max(), np.abs(quaternion_of(z1, z2) - q).max())
    check("hopf(z1,z2) = q k conj(q) for q = a+bi+cj+dk, (z1,z2) = (a+di, c+bi)", err < 1e-12, "(max err %.1e)" % err)

    # 3. a fibre is a great circle of S^3 and a round circle in R^3
    b = base_point(1.3, 0.8)
    F = fibre_s3(b)
    rank = np.linalg.matrix_rank(to_r4(F), tol=1e-9)
    check("fibre spans a 2-plane through the origin of R^4 (great circle)", rank == 2)
    c, r, nrm, res = fit_circle(stereographic(F))
    check("stereographic image of a fibre is a round circle", res < 1e-9, "(residual %.1e)" % res)

    # 4. exact geometry: centre distance d, radius r obey r^2 = d^2 + 1  (derived in the docstring of torus_data)
    worst = 0.0
    for el in (0.4, 1.0, 1.7, 2.5):
        for az in (0.0, 1.1, 4.0):
            d, r, t = circle_params(fibre_r3(base_point(el, az), 400))
            e = (math.pi - el) / 2
            worst = max(worst, abs(r * r - d * d - 1), abs(d - math.tan(e)), abs(r - 1 / math.cos(e)), abs(t - e))
    check("exact fibres: d = tan(ele), r = sec(ele), tilt = ele, ele = (pi - el)/2", worst < 1e-9, "(max err %.1e)" % worst)

    # 5. linking number of two fibres is +-1, and they never meet
    A = fibre_r3(base_point(1.0, 0.0), 300)
    B = fibre_r3(base_point(2.0, 2.0), 300)
    lk = linking_number(A, B)
    check("two fibres link once", abs(abs(lk) - 1) < 0.05, "(Gauss integral %.3f)" % lk)
    check("two fibres do not meet", min_distance(A, B) > 1e-3, "(min distance %.3f)" % min_distance(A, B))

    # 6. the fibre over the south pole is a line through the pole (unbounded as the sample reaches it)
    z1, z2 = section((0, 0, -1))
    F0 = to_r4(np.stack([np.exp(1j * np.linspace(0, TWO_PI, 8, endpoint=False)) * z1,
                         np.exp(1j * np.linspace(0, TWO_PI, 8, endpoint=False)) * z2], axis=1))
    check("fibre over the south pole passes through the projection pole", np.isclose(F0[:, 3].max(), 1.0))

    # 7. audit of fibration.py
    rows = audit()
    worst_fixed = max(max(abs(a - b) for a, b in zip(r_["fixed"], r_["exact"])) for r_ in rows)
    check("with baseRad = tan(ele), fibreRad = sec(ele) makeFibre's circles ARE the exact fibres (mirrored)",
          worst_fixed < 1e-6, "(max parameter diff %.1e)" % worst_fixed)
    worst_ship = max(abs(r_["shipped"][1] ** 2 - r_["shipped"][0] ** 2 - 1) for r_ in rows)
    check("as shipped they are NOT exact fibres (r^2 - d^2 - 1 is not 0)", worst_ship > 0.04,
          "(up to %.3f)" % worst_ship)
    if verbose:
        print("  audit of fibration.py makeFibre   [d = centre distance, r = radius, t = tilt of the plane]")
        for r_ in rows:
            sd, sr, st = r_["shipped"]
            xd, xr, xt = r_["exact"]
            print("       el=%.2f  shipped d=%.3f r=%.3f t=%.3f | exact d=%.3f r=%.3f t=%.3f | "
                  "link(az,az+.5)=%+.2f  min dist(az,az+1)=%.3f" % (r_["el"], sd, sr, st, xd, xr, xt, r_["link"], r_["mind"]))
    return ok


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    run_tests()
