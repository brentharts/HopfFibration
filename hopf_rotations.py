#!/usr/bin/env python3
"""hopf_rotations.py - the 12-fold rotation group of the Spectre's 4D lift acting on C^2, seen as Hopf fibres.

Background (Spectre repository, SPECTRE.md section 5 and HOPF.md section 5).  Every edge of the Spectre tiling is a unit
vector at a multiple of 30 degrees.  With w = e^{i pi/3} and z = e^{i pi/6} (z^2 = w) every vertex is an integer vector
(p, q, r, s) and sits at the plane point  P = a + z b,   a = p + q w,   b = r + s w   (a, b in the hexagonal lattice Z[w]),
so the vertex lattice is the ring Z[zeta_12] = Z^4 and (a, b) is a point of C^2.

What is computed (exact integer matrices, then checked on points):
  R60 = rotation by 60 degrees  = multiplication of P by w:   (a, b) -> (w a, w b)      order 6
  R30 = rotation by 30 degrees  = multiplication of P by z:   (a, b) -> (w b, a)         order 12, R30^2 = R60
So R60 is the SAME phase on both coordinates, which is exactly the Hopf circle action restricted to the sixth roots of unity:
rotating by 60 degrees moves a vector ALONG its Hopf fibre and never off it; the Hopf base point [a:b] is invariant.
R30 swaps the two planes and acts on the Hopf base S^2 as a rotation by pi (about the axis through the two fixed points
t = +-e^{-i pi/6} of t -> 1/(w t), t = b/a).

This is an ELEMENTARY observation: any complex 2-plane with a scalar phase action has it.  Nothing here says the Hopf
fibration explains the Spectre (see CONNECTIONS.md in the Spectre repository).  The internal space of the Spectre's
cut-and-project description is the contracting plane of the inflation, not the pair (a, b).

    python3 hopf_rotations.py --tests
    python3 hopf_rotations.py --png figures/hopf_rotations.png
"""
import itertools
import math
import sys

import numpy as np

import hopf_math as H

W = np.exp(1j * math.pi / 3)
Z = np.exp(1j * math.pi / 6)

# integer matrices on (p, q, r, s); columns are the images of the basis vectors (verified by _recompute in the tests)
R60 = np.array([[0, -1, 0, 0], [1, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 1]])
R30 = np.array([[0, 0, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]])


def ab(v):
    """(p, q, r, s) -> (a, b) in C^2."""
    p, q, r, s = v
    return p + q * W, r + s * W


def position(v):
    """Physical plane position P = a + z b."""
    a, b = ab(v)
    return a + Z * b


def base_point_of(v):
    """The Hopf base point (unit vector of S^2) of the vector v != 0: h(a, b)/|(a,b)|^2."""
    a, b = ab(v)
    n2 = abs(a) ** 2 + abs(b) ** 2
    return H.hopf(a, b) / n2


def _recompute(mult):
    """Integer matrix of 'multiply P by mult' on Z^4, found by brute force over small coordinates."""
    cols = []
    for e in np.eye(4, dtype=int):
        t = mult * position(e)
        cols.append(next(c for c in itertools.product(range(-2, 3), repeat=4) if abs(position(c) - t) < 1e-9))
    return np.array(cols).T


def fixed_axis_of_R30():
    """Unit vector n on S^2 fixed by the action of R30 on the Hopf base (the rotation-by-pi axis): the base point of
    (a, b) = (1, t) with t = e^{-i pi/6} (the fixed point of t -> 1/(w t))."""
    t = np.exp(-1j * math.pi / 6)
    a, b = 1.0 + 0j, t
    return H.hopf(a, b) / (abs(a) ** 2 + abs(b) ** 2)


def run_tests(verbose=True):
    rng = np.random.default_rng(1)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    I = np.eye(4, dtype=int)
    mp = np.linalg.matrix_power
    check("matrices are the integer matrices of multiplication by w and z",
          (_recompute(W) == R60).all() and (_recompute(Z) == R30).all())
    check("R60 has order 6, R30 has order 12, R30^2 = R60",
          (mp(R60, 6) == I).all() and not any((mp(R60, k) == I).all() for k in range(1, 6))
          and (mp(R30, 12) == I).all() and not any((mp(R30, k) == I).all() for k in range(1, 12))
          and (mp(R30, 2) == R60).all())
    vs = rng.integers(-6, 7, (40, 4))
    vs = vs[np.abs(vs).sum(1) > 0]
    check("R60 = (a,b) -> (w a, w b): the Hopf circle action",
          all(np.allclose(ab(R60 @ v), (W * ab(v)[0], W * ab(v)[1])) for v in vs))
    check("R30 = (a,b) -> (w b, a)", all(np.allclose(ab(R30 @ v), (W * ab(v)[1], ab(v)[0])) for v in vs))
    check("physical rotation: P(R60 v) = w P(v), P(R30 v) = z P(v)",
          all(abs(position(R60 @ v) - W * position(v)) < 1e-9 and abs(position(R30 @ v) - Z * position(v)) < 1e-9
              for v in vs))
    check("60 degrees fixes the Hopf base point (moves along the fibre)",
          all(np.allclose(base_point_of(R60 @ v), base_point_of(v)) for v in vs))
    # R30 acts on S^2 as an isometry of order 2 (a rotation by pi)
    bp = np.array([base_point_of(v) for v in vs])
    bp30 = np.array([base_point_of(R30 @ v) for v in vs])
    M = np.linalg.lstsq(bp, bp30, rcond=None)[0].T
    check("30 degrees acts on S^2 as a linear isometry", np.allclose(M @ bp.T, bp30.T) and np.allclose(M @ M.T, np.eye(3)))
    check("... of order 2 and determinant +1: a rotation by pi", np.allclose(M @ M, np.eye(3)) and abs(np.linalg.det(M) - 1) < 1e-9)
    n = fixed_axis_of_R30()
    check("... about the axis through the base points of (1, e^{-i pi/6}) and (1, -e^{-i pi/6})",
          np.allclose(M @ n, n) and np.allclose(M @ -n, -n) and np.allclose(
              H.hopf(1.0, -np.exp(-1j * math.pi / 6)) / 2, -n))
    # the 12-element orbit of a vector: how many distinct base points?
    v = np.array([3, 1, 2, -1])
    orbit = {tuple(np.round(base_point_of(np.linalg.matrix_power(R30, k) @ v), 9)) for k in range(12)}
    check("the 12 rotations of a vector have only 2 Hopf base points (60 deg moves along a fibre, 30 deg swaps)",
          len(orbit) == 2, "(%d)" % len(orbit))
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.array([3, 1, 2, -1])
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u_, v_ = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="0.9", alpha=0.3, linewidth=0)
    pts = [base_point_of(np.linalg.matrix_power(R30, k) @ v) for k in range(12)]
    for k, p in enumerate(pts):
        ax.scatter(*p, s=60, color=plt.cm.tab20(k % 2))
    n = fixed_axis_of_R30()
    ax.plot(*np.array([-1.3 * n, 1.3 * n]).T, color="k", lw=1, ls="--")
    ax.set_title("Hopf base S^2: the 12 rotations of one vector give 2 points;\ndashed = axis of the pi-rotation by R30")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    # fibre of the first point in R^3 with the six R60 images marked (equally spaced along the fibre)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    a, b = ab(v)
    nrm = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
    z1, z2 = a / nrm, b / nrm
    t = np.linspace(0, H.TWO_PI, 400)
    F = H.stereographic(np.stack([np.exp(1j * t) * z1, np.exp(1j * t) * z2], axis=1))
    ax2.plot(*F.T, color="tab:blue", lw=1.5)
    six = H.stereographic(np.array([[W ** k * z1, W ** k * z2] for k in range(6)]))
    ax2.scatter(*six.T, color="tab:red", s=60)
    ax2.set_title("its Hopf fibre in R^3 with the six 60-degree images (red)")
    ax2.set_axis_off()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
