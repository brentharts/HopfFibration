#!/usr/bin/env python3
"""hopf_clifford.py - Clifford parallels, the Clifford torus and the Villarceau circles of the Hopf fibration.

Clifford parallels (W. K. Clifford, 1873) are the great circles of S^3 that stay a CONSTANT distance apart.  Hopf (1931)
built his fibration from them: the fibres are a family of Clifford-parallel great circles.  There are two such
families, the RIGHT fibres  e^{it}(z1, z2)  (this repository's fibres) and the LEFT fibres  (e^{it} z1, e^{-it} z2),
the fibres of the mirror map hopf(z1, conj z2).  Through every point passes one of each; they coincide only through
the poles (z1 = 0 or z2 = 0).  (Wikipedia 'Clifford parallel': "right" and "left" parallels, quaternion description.)

What is checked (exact up to floating point):
  1. any two right fibres are Clifford parallel: the distance from a point of one to the other fibre is the same for
     every point, and equals HALF the angle between their base points on S^2.
  2. the right fibre and the left fibre through a point meet in exactly the two antipodal points +-p.
  3. the Clifford torus |z1| = |z2| (the preimage of the equator of S^2) is a flat torus; its stereographic image is the
     standard torus of major radius sqrt(2) and minor radius 1, and each fibre over the equator is a Villarceau circle of
     it: radius sqrt(2) = the major radius, tilted by arcsin(r/R) = 45 degrees.

    python3 hopf_clifford.py --tests
    python3 hopf_clifford.py --png figures/hopf_clifford.png
"""
import math
import sys

import numpy as np

import hopf_math as H


def right_fibre(b, n=720):
    """Right Clifford-parallel great circle over b (the Hopf fibre): e^{it}(z1, z2)."""
    return H.fibre_s3(b, n)


def left_fibre(b, n=720):
    """Left Clifford-parallel great circle over b: (e^{it} z1, e^{-it} conj(z2)), a fibre of hopf(z1, conj z2)."""
    z1, z2 = H.section(b)
    t = np.linspace(0.0, H.TWO_PI, n, endpoint=False)
    return np.stack([np.exp(1j * t) * z1, np.exp(-1j * t) * np.conj(z2)], axis=1)


def right_fibre_through(p, n=720):
    """The right fibre through the point p = (z1, z2) of S^3."""
    t = np.linspace(0.0, H.TWO_PI, n, endpoint=False)
    return np.exp(1j * t)[:, None] * np.asarray(p, complex)[None, :]


def left_fibre_through(p, n=720):
    t = np.linspace(0.0, H.TWO_PI, n, endpoint=False)
    p = np.asarray(p, complex)
    return np.stack([np.exp(1j * t) * p[0], np.exp(-1j * t) * p[1]], axis=1)


def s3_angle(P, Q):
    """Great-circle distance in S^3 between points (rows of P) and points (rows of Q); returns the matrix of angles."""
    A, B = H.to_r4(P), H.to_r4(Q)
    return np.arccos(np.clip(A @ B.T, -1, 1))


def distance_to_fibre(p, F):
    """Distance in S^3 from the point p to the fibre F (sampled)."""
    return float(s3_angle(np.asarray(p)[None, :], F).min())


def clifford_torus(n=60):
    """Stereographic image of the Clifford torus |z1| = |z2| = 1/sqrt(2): arrays X, Y, Z of shape (n, n)."""
    s = 1 / math.sqrt(2)
    a, b = np.meshgrid(np.linspace(0, H.TWO_PI, n), np.linspace(0, H.TWO_PI, n))
    zz = np.stack([s * np.exp(1j * a), s * np.exp(1j * b)], axis=-1)
    P = H.stereographic(zz)
    return P[..., 0], P[..., 1], P[..., 2]


def run_tests(verbose=True):
    rng = np.random.default_rng(11)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 1. Clifford parallelism of right fibres
    worst = 0.0
    for _ in range(6):
        b1, b2 = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
        F1, F2 = right_fibre(b1, 360), right_fibre(b2, 360)
        D = s3_angle(F1, F2)
        half = 0.5 * math.acos(float(np.clip(b1 @ b2, -1, 1)))
        worst = max(worst, np.ptp(D.min(axis=1)), abs(D.min() - half))
    check("right fibres are Clifford parallel: constant distance = half the base angle", worst < 2e-3,
          "(spread/err %.1e, limited by sampling)" % worst)

    # 2. left and right fibre through a point
    n_meet = []
    for _ in range(6):
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        p = (complex(v[0], v[1]), complex(v[2], v[3]))
        R, L = right_fibre_through(p, 7200), left_fibre_through(p, 7200)
        near = (s3_angle(R, L) < 2e-3).any(axis=1)          # which samples of R are close to L
        runs = int(np.sum(near & ~np.roll(near, 1)))         # number of separate stretches (circularly)
        n_meet.append(runs)
    check("right and left fibre through a point meet in 2 antipodal points", all(k == 2 for k in n_meet),
          "(separate contacts %s)" % n_meet)

    # coincide only through the poles
    p_pole = (1.0 + 0j, 0.0 + 0j)
    same = np.abs(s3_angle(right_fibre_through(p_pole, 90), left_fibre_through(p_pole, 90)).min(axis=0)).max() < 1e-6
    gen = (complex(0.6, 0.0), complex(0.0, 0.8))
    differ = s3_angle(right_fibre_through(gen, 90), left_fibre_through(gen, 90)).min(axis=0).max() > 0.1
    check("left and right fibres coincide through a pole, differ elsewhere", same and differ)

    # 3. Clifford torus and Villarceau circles
    X, Y, Z = clifford_torus(80)
    rho = np.hypot(X, Y)
    R_major, r_minor = math.sqrt(2), 1.0
    torus_err = np.abs((rho - R_major) ** 2 + Z ** 2 - r_minor ** 2).max()
    check("Clifford torus -> torus of major radius sqrt(2), minor radius 1", torus_err < 1e-9, "(err %.1e)" % torus_err)
    F = H.fibre_r3(H.base_point(math.pi / 2, 0.7), 720)
    c, r, nrm, res = H.fit_circle(F)
    on_torus = np.abs((np.hypot(F[:, 0], F[:, 1]) - R_major) ** 2 + F[:, 2] ** 2 - r_minor ** 2).max()
    tilt = math.acos(min(1.0, abs(nrm[2])))
    check("fibre over the equator lies on that torus", on_torus < 1e-9, "(err %.1e)" % on_torus)
    check("... and is a Villarceau circle: radius sqrt(2) = R, tilt arcsin(r/R) = 45 deg",
          abs(r - math.sqrt(2)) < 1e-9 and abs(tilt - math.asin(r_minor / R_major)) < 1e-9,
          "(radius %.6f, tilt %.1f deg)" % (r, math.degrees(tilt)))
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    X, Y, Z = clifford_torus(60)
    ax.plot_surface(X, Y, Z, alpha=0.15, color="tab:blue", linewidth=0)
    for az in np.linspace(0, H.TWO_PI, 12, endpoint=False):
        F = H.fibre_r3(H.base_point(math.pi / 2, az), 300)
        ax.plot(*F.T, color="tab:red", lw=1.4)
    for az in np.linspace(0, H.TWO_PI, 12, endpoint=False):
        F = H.stereographic(left_fibre(H.base_point(math.pi / 2, az), 300))
        ax.plot(*F.T, color="tab:green", lw=1.0, alpha=0.8)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_title("Clifford torus (R=sqrt2, r=1): right Villarceau circles (red), left (green)")
    ax.set_axis_off()
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
