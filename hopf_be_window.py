#!/usr/bin/env python3
"""hopf_be_window.py - the one-link Bakry-Emery window of lattice SU(2), on the round sphere and on Berger spheres.

Setting (see YANG_MILLS_LATTICE_GAP.md, section 6, for the source).  In 4D lattice Yang-Mills with gauge group SU(2) = S^3
(unit quaternions), weigh a configuration by exp(-f), where for one link variable q = Q_e with all other links fixed at 1

    f(q) = -beta_W * sum_{6 plaquettes through e} Re q = -6 beta_W Re q ,

and the metric on the link is the round (bi-invariant) metric of the unit S^3, whose Ricci curvature is 2.  The Bakry-Emery
tensor of the measure e^{-f} dvol is Ric + Hess f.  The Hexagon preprint 2610.00341v5 states that at Q_e = -1 the Hessian is
-6 beta_W times the metric, so the tensor is 2 - 6 beta_W: negative once beta_W > 1/3, which is why no curvature argument for
the gauge-fixed SU(2) measure passes beta_W = 1/3.  At a general point q = (cos a, sin a n) the round-sphere tensor is
(2 + 6 beta_W cos a) g, because the Hessian of the linear function Re q on the unit S^3 is -Re q times the metric.

THE QUESTION ASKED HERE (not addressed by the preprint).  Replace the round metric on the link by a Berger metric g_s: the
Hopf fibre scaled by s.  Its Ricci curvature is 2 s^2 along the fibre and 4 - 2 s^2 across it (hopf_berger.py).  q = -1 is a
critical point of f, so there Hess f is a bilinear form that does not depend on the metric: -6 beta_W times the ROUND metric.
In g_s-orthonormal vectors the fibre direction is the round unit vector divided by s, so

    BE_fibre  = 2 s^2     - 6 beta_W / s^2 ,
    BE_across = 4 - 2 s^2 - 6 beta_W          (twice) ,

and both are positive exactly when  beta_W < min(s^4, 2 - s^2) / 3.  The first limit increases with s, the second decreases,
and they cross at s = 1, where both equal 1/3.  So among Berger metrics the round sphere maximises this window, and the
threshold 1/3 is not an artefact of choosing the round metric.

WHAT THIS IS NOT.  It is a computation at one point (Q_e = -1) of one link with the rest of the lattice held at the identity,
as in the preprint's remark; it is not a bound on the lattice theory.  A Berger metric on every link is not the bi-invariant
product structure the Dobrushin and Bakry-Emery arguments use, and nothing here says those arguments extend to it.

    python3 hopf_be_window.py --tests
    python3 hopf_be_window.py --png figures/hopf_be_window.png
"""
import math
import sys

import numpy as np

import hopf_berger as B
import hopf_math as H


def f_link(q, bW):
    """f(q) = -6 beta_W Re q on the unit S^3 in R^4 (q[0] = Re q)."""
    return -6.0 * bW * np.asarray(q, float)[..., 0]


def be_round(alpha, bW):
    """Bakry-Emery eigenvalue (all three equal) on the round S^3 at q = (cos alpha, sin alpha n)."""
    return 2.0 + 6.0 * bW * np.cos(alpha)


def be_berger(s, bW):
    """Bakry-Emery eigenvalues at q = -1 for the Berger metric g_s: (along the fibre, across, across)."""
    s = np.asarray(s, float)
    return np.stack([2 * s * s - 6.0 * bW / (s * s), 4 - 2 * s * s - 6.0 * bW + 0 * s, 4 - 2 * s * s - 6.0 * bW + 0 * s], axis=-1)


def window(s):
    """Largest beta_W for which the tensor at q = -1 is positive: min(s^4, 2 - s^2) / 3, floored at 0."""
    s = np.asarray(s, float)
    return np.maximum(np.minimum(s ** 4, 2.0 - s * s) / 3.0, 0.0)


def second_derivative_along_geodesic(q0, X, bW, h=1e-4):
    """d^2/dt^2 f(q0 cos t + X sin t) at t = 0 by central differences (X a unit tangent vector at q0)."""
    g = lambda t: f_link(q0 * math.cos(t) + X * math.sin(t), bW)
    return float((g(h) - 2 * g(0.0) + g(-h)) / h ** 2)


def run_tests(verbose=True):
    ok = True
    rng = np.random.default_rng(3)

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    q0 = np.array([-1.0, 0.0, 0.0, 0.0])
    # the Hessian at q = -1 is -6 beta_W times the round metric, in every direction
    worst = 0.0
    for _ in range(10):
        X = np.concatenate([[0.0], rng.normal(size=3)])
        X /= np.linalg.norm(X)
        bW = rng.uniform(0.05, 1.0)
        worst = max(worst, abs(second_derivative_along_geodesic(q0, X, bW) + 6 * bW))
    check("Hess f at q = -1 is -6 beta_W times the round metric (numeric second derivatives)", worst < 1e-5,
          "(max err %.1e)" % worst)
    # and (2 + 6 beta_W cos a) g at a general point
    worst = 0.0
    for _ in range(10):
        a = rng.uniform(0.1, 3.0)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        q = np.concatenate([[math.cos(a)], math.sin(a) * n])
        X = rng.normal(size=4)
        X -= (X @ q) * q
        X /= np.linalg.norm(X)
        bW = rng.uniform(0.05, 1.0)
        worst = max(worst, abs(second_derivative_along_geodesic(q, X, bW) - 6 * bW * math.cos(a)))
    check("Hess f at q = (cos a, sin a n) is +6 beta_W cos a times the metric, so BE = 2 + 6 beta_W cos a", worst < 1e-5,
          "(max err %.1e)" % worst)
    check("round S^3 (s = 1): BE = 2 - 6 beta_W at q = -1, zero exactly at beta_W = 1/3",
          np.allclose(be_berger(1.0, 1.0 / 3.0), 0.0) and np.allclose(be_round(math.pi, 0.2), 2 - 6 * 0.2))
    # the fibre direction at q = -1 is the kernel of the differential of the Hopf map, and has unit round length
    z = (-1.0 + 0j, 0.0 + 0j)
    eps = 1e-6
    r4 = lambda c: np.array([c[0].real, c[0].imag, c[1].real, c[1].imag])
    fib = lambda t: (np.exp(1j * t) * z[0], np.exp(1j * t) * z[1])
    Xf = (r4(fib(eps)) - r4(fib(-eps))) / (2 * eps)
    dh = (H.hopf(*fib(eps)) - H.hopf(*fib(-eps))) / (2 * eps)
    horiz = lambda t: H.hopf(-math.cos(t) + 0j, math.sin(t) + 0j)       # great circle from q = -1 along (0, 0, 1, 0)
    dhh = (horiz(eps) - horiz(-eps)) / (2 * eps)
    check("the fibre direction at q = -1 is a unit tangent vector, killed by the Hopf map; horizontal vectors are doubled",
          abs(np.linalg.norm(Xf) - 1) < 1e-8 and abs(Xf @ q0) < 1e-8 and np.abs(dh).max() < 1e-8
          and abs(np.linalg.norm(dhh) - 2.0) < 1e-5)
    # the Berger formulas against hopf_berger's Ricci
    ss = np.linspace(0.3, 1.8, 31)
    bW = 0.1
    ric = np.array([B.berger_ricci(s) for s in ss])                      # (transverse, transverse, fibre)
    be_num = np.stack([ric[:, 2] - 6 * bW / ss ** 2, ric[:, 0] - 6 * bW, ric[:, 1] - 6 * bW], axis=-1)
    check("BE = Ric (hopf_berger) + Hess: (2 s^2 - 6 b/s^2, 4 - 2 s^2 - 6 b, 4 - 2 s^2 - 6 b)",
          np.allclose(be_num, be_berger(ss, bW)))
    # the window
    grid = np.linspace(0.2, 1.4, 120001)
    w = window(grid)
    i = int(np.argmax(w))
    check("the window min(s^4, 2 - s^2)/3 is maximal at s = 1 and equals 1/3 there", abs(grid[i] - 1.0) < 1e-4
          and abs(w[i] - 1.0 / 3.0) < 1e-8, "(argmax s = %.5f, max %.8f)" % (grid[i], w[i]))
    check("the fibre limit s^4/3 increases and the across limit (2 - s^2)/3 decreases; they cross only at s = 1",
          np.all(np.diff(grid ** 4 / 3) > 0) and np.all(np.diff((2 - grid ** 2) / 3) < 0)
          and abs(1.0 ** 4 - (2 - 1.0)) < 1e-15)
    check("for beta_W slightly above 1/3 every Berger metric (s in [0.2, 1.4]) has a negative BE direction",
          all(be_berger(s, 1.0 / 3.0 + 1e-3).min() < 0 for s in np.linspace(0.2, 1.4, 241)))
    check("for beta_W slightly below 1/3 the round metric has a positive tensor, a squashed or stretched one does not",
          be_berger(1.0, 1.0 / 3.0 - 1e-3).min() > 0 and be_berger(0.8, 1.0 / 3.0 - 1e-3).min() < 0
          and be_berger(1.2, 1.0 / 3.0 - 1e-3).min() < 0)
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    s = np.linspace(0.2, 1.45, 400)
    ax[0].plot(s, s ** 4 / 3, color="#c0392b", label="fibre direction:  s^4 / 3")
    ax[0].plot(s, (2 - s ** 2) / 3, color="#1f5fa8", label="across the fibre:  (2 - s^2) / 3")
    ax[0].plot(s, window(s), color="k", lw=2.5, alpha=0.8, label="window  min(s^4, 2 - s^2) / 3")
    ax[0].fill_between(s, 0, window(s), color="0.85", zorder=0)
    ax[0].plot([1], [1 / 3], "o", color="k")
    ax[0].annotate("round S^3:\nboth directions\nfail at beta_W = 1/3", (1, 1 / 3), xytext=(1.17, 0.27), fontsize=8,
                   arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax[0].set_xlabel("fibre scale s  (1 = round S^3)")
    ax[0].set_ylabel("largest beta_W with Ric + Hess f > 0 at Q_e = -1")
    ax[0].set_ylim(0, 0.75)
    ax[0].set_title("Berger metrics on the link: the round sphere maximises the window", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    a = np.linspace(0, math.pi, 300)
    for bW, c in [(0.2, "#2a9d8f"), (1 / 3, "k"), (0.5, "#e67e22"), (0.8, "#c0392b")]:
        ax[1].plot(a, be_round(a, bW), color=c, label="beta_W = %.3g" % bW)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_xticks([0, math.pi / 2, math.pi])
    ax[1].set_xticklabels(["0  (Q_e = 1)", "pi/2", "pi  (Q_e = -1)"])
    ax[1].set_xlabel("angle a of Q_e = (cos a, sin a n) from the identity")
    ax[1].set_ylabel("Bakry-Emery tensor  (2 + 6 beta_W cos a) g")
    ax[1].set_title("Round S^3: the tensor first goes negative at Q_e = -1 for beta_W > 1/3", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("One link of lattice SU(2), other links at the identity (a one-point computation, not a bound on the lattice theory)",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
