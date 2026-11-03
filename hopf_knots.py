#!/usr/bin/env python3
"""hopf_knots.py - the maps f_{p,q} = Hopf(z1^p, z2^q): preimages are torus knots, two of them link p*q times.

Normalise (z1^p, z2^q) back onto S^3 and apply the Hopf map:

    f_{p,q}(z1, z2) = hopf( z1^p / n , z2^q / n ),     n = sqrt(|z1|^(2p) + |z2|^(2q)).

f_{1,1} is the Hopf map.  The preimage of a point b of S^2 is found by solving z1^p = lam e^{it} w1, z2^q = lam e^{it} w2,
where (w1, w2) is the section of the Hopf fibre over b: the moduli are fixed by |z1|^2 + |z2|^2 = 1 (one monotone equation
in lam) and the phases wind q times in z1 and p times in z2.  For coprime p, q the preimage is ONE closed curve, the
(q, p) torus knot, and the Gauss linking number of the preimages of two different points is p*q: the Hopf invariant of
f_{p,q}.  The same product rule (Q_H = m n for the rational map xi = Z1^m / Z0^n) is the starting point of the hopfion-crystal
construction of Hou, Xiang, Liu and Zang (arXiv:2504.03981); the maps are also checked symbolically in the fork
brentharts/openai_math_yang_mills (sympy_models/models/hopf_fibration.py, which this script tests numerically).

What is checked: the curve lies on S^3 and maps to the base point; z1 winds q times and z2 winds p times; the Gauss linking
integral of two preimages is p*q (to the accuracy of the midpoint rule, a few parts in 10^3).

    python3 hopf_knots.py --tests
    python3 hopf_knots.py --png figures/hopf_knots.png
"""
import math
import sys

import numpy as np

import hopf_math as H

PAIRS_FIGURE = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 4), (4, 3), (2, 5)]


def f_pq(z1, z2, p, q):
    """The map f_{p,q}: S^3 -> S^2 (a point of R^3 of unit length)."""
    a, b = np.asarray(z1, complex) ** p, np.asarray(z2, complex) ** q
    n = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
    return H.hopf(a / n, b / n)


def preimage(b, p, q, n=1400):
    """Preimage of the base point b under f_{p,q} (p, q coprime): a closed curve on S^3, shape (n, 2) complex."""
    if math.gcd(p, q) != 1:
        raise ValueError("p and q must be coprime for the preimage to be a single curve")
    w1, w2 = H.section(b)
    a1, a2, m1, m2 = np.angle(w1), np.angle(w2), abs(w1), abs(w2)

    def g(lam):
        return (lam * m1) ** (2.0 / p) + (lam * m2) ** (2.0 / q) - 1.0

    lo, hi = 0.0, 1.0
    while g(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    lam = 0.5 * (lo + hi)
    r1, r2 = (lam * m1) ** (1.0 / p), (lam * m2) ** (1.0 / q)
    u = np.linspace(0.0, H.TWO_PI, n, endpoint=False)        # t = p q u
    return np.stack([r1 * np.exp(1j * (q * u + a1 / p)), r2 * np.exp(1j * (p * u + a2 / q))], axis=1)


def winding(c):
    """Net number of turns of a closed complex-valued sequence."""
    return int(round(np.sum(np.diff(np.unwrap(np.append(np.angle(c), np.angle(c[0]))))) / H.TWO_PI))


def link_of_preimages(p, q, b1=None, b2=None, n=1400):
    b1 = H.base_point(1.1, 0.4) if b1 is None else b1
    b2 = H.base_point(2.0, 2.3) if b2 is None else b2
    return H.linking_number(H.stereographic(preimage(b1, p, q, n)), H.stereographic(preimage(b2, p, q, n)))


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    b1, b2 = H.base_point(1.1, 0.4), H.base_point(2.0, 2.3)
    for p, q in [(1, 1), (2, 1), (1, 3), (3, 2), (2, 5), (3, 4)]:
        C = preimage(b1, p, q)
        err = max(np.abs(f_pq(z1, z2, p, q) - b1).max() for z1, z2 in C[::50])
        on_s3 = np.abs(np.abs(C[:, 0]) ** 2 + np.abs(C[:, 1]) ** 2 - 1).max()
        check("f_(%d,%d): preimage lies on S^3 and maps to the base point" % (p, q), err < 1e-9 and on_s3 < 1e-12,
              "(err %.1e)" % err)
        check("   z1 winds q = %d times, z2 winds p = %d times: the (q, p) torus knot" % (q, p),
              (winding(C[:, 0]), winding(C[:, 1])) == (q, p))
        L = link_of_preimages(p, q, b1, b2)
        check("   Gauss linking number of two preimages = p q = %d" % (p * q), abs(abs(L) - p * q) < 0.05,
              "(%.3f)" % L)
    # f_{1,1} is the Hopf map: its preimage of b is the Hopf fibre over b
    C = preimage(b1, 1, 1, 400)
    check("f_(1,1) is the Hopf map: the preimage is the fibre over b",
          np.abs(H.hopf(C[:, 0], C[:, 1]) - b1).max() < 1e-12)
    try:
        preimage(b1, 2, 2)
        check("p = q = 2 is rejected (two components)", False)
    except ValueError:
        check("p, q not coprime is rejected (the preimage has gcd(p, q) components)", True)
    return ok


def _axes_equal(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b1, b2 = H.base_point(1.1, 0.4), H.base_point(2.0, 2.3)
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    for k, (p, q) in enumerate([(1, 1), (2, 1), (3, 2)]):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        ax.view_init(elev=24, azim=-58)
        A, B = H.stereographic(preimage(b1, p, q, 1500)), H.stereographic(preimage(b2, p, q, 1500))
        ax.plot(*A.T, color="#c0392b", lw=1.6)
        ax.plot(*B.T, color="#1f5fa8", lw=1.6)
        _axes_equal(ax, np.vstack([A, B]))
        ax.set_axis_off()
        L = H.linking_number(A, B)
        ax.set_title("f_(%d,%d): two preimages, %s\nlinking number %.3f  (p q = %d)"
                     % (p, q, "Hopf fibres" if (p, q) == (1, 1) else "(%d,%d) torus knots" % (q, p), L, p * q), fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    xs, ys = [], []
    for p, q in PAIRS_FIGURE:
        xs.append(p * q)
        ys.append(abs(link_of_preimages(p, q, b1, b2, 1200)))
    m = max(xs) + 1
    ax.plot([0, m], [0, m], color="0.6", lw=1, ls="--", label="linking number = p q")
    ax.scatter(xs, ys, color="#c0392b", zorder=3, label="Gauss integral of two preimages")
    labels = {}
    for (p, q), x, y in zip(PAIRS_FIGURE, xs, ys):          # one label per value of p q, listing every (p,q) with that product
        labels.setdefault(x, [y, []])[1].append("(%d,%d)" % (p, q))
    for x, (y, names) in labels.items():
        ax.annotate(" ".join(names), (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    ax.set_xlabel("p q")
    ax.set_ylabel("|linking number|")
    ax.set_title("The Hopf invariant of f_(p,q) = Hopf(z1^p, z2^q) is p q", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
