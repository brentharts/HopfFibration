#!/usr/bin/env python3
"""hopf_berger.py - Berger spheres: the Hopf fibration with a stretched or squashed fibre, and the collapsing fibre.

A BERGER SPHERE is S^3 = SU(2) with the metric of the round unit sphere except that the Hopf fibres are scaled by a factor
s (fibre length 2 pi s; the base S^2 and the horizontal directions are unchanged).  s = 1 is the round sphere.  In general
relativity the spatial slices of the Bianchi IX (Mixmaster) universe are S^3 with three scale factors; when two are equal
the slice is a Berger sphere, and the unequal one is the Hopf fibre.  The 'pancake' cosmology of Dechant, Lasenby and
Hobson has a fibre that collapses (s -> 0) over a round base.

Exact results computed here (Milnor's formula for left-invariant metrics on SU(2), see below):
  * Ricci curvature of the Berger sphere along the fibre = 2 s^2, across it = 4 - 2 s^2; scalar curvature 8 - 2 s^2.
    So at s = 1 all equal 2 and 6 (the round unit S^3), and when the fibre collapses (s -> 0) the curvature STAYS FINITE
    (Ric -> 0 along the fibre, 4 across, scalar -> 8) while the volume 2 pi^2 s -> 0.
  * The Hopf fibres are geodesics for every s.  Their length is 2 pi s.
  * Round case: Ricci = 2 and scalar curvature 6.
Milnor (1976): for an orthonormal basis e1,e2,e3 of a unimodular 3D Lie algebra with [e2,e3]=l1 e1, [e3,e1]=l2 e2,
[e1,e2]=l3 e3, the Ricci form is diagonal with Ric(e_i) = 2 mu_j mu_k, mu_i = (l1+l2+l3)/2 - l_i.  For the unit S^3 all
l_i = 2.  Scaling the fibre vector e3 -> e3/s (fibre length 2 pi s) gives (l1, l2, l3) = (2/s, 2/s, 2 s).

Pancake fibre clock (from the Spectre repository, spectre_pancake.py): one level of the Spectre substitution rescales
lengths by lambda = sqrt(4 + sqrt(15)); if the substitution level is read as the logarithm of the pancake's proper time,
the levels per fibre turn are N(a0) = 2 pi a0 / ln(lambda).  At the default a0 = 1 this is 6.0900..., 1.5 % away from
6 = |Z/6|.  That is NUMEROLOGY (a0 is initial data, and N = 6 exactly would need a0 chosen to give it); this script
prints it so the number can be checked, and asserts that it misses 6.

    python3 hopf_berger.py --tests
    python3 hopf_berger.py --png figures/hopf_berger.png
"""
import math
import sys

import numpy as np

import hopf_math as H


def milnor_ricci(l1, l2, l3):
    """Principal Ricci curvatures of a unimodular 3D Lie group metric (Milnor 1976)."""
    h = (l1 + l2 + l3) / 2.0
    m1, m2, m3 = h - l1, h - l2, h - l3
    return 2 * m2 * m3, 2 * m3 * m1, 2 * m1 * m2


def berger_ricci(s):
    """(Ric transverse, Ric transverse, Ric along the fibre) for the Berger sphere with fibre scale s."""
    return milnor_ricci(2.0 / s, 2.0 / s, 2.0 * s)


def berger_scalar(s):
    return sum(berger_ricci(s))


def berger_volume(s):
    return 2 * math.pi ** 2 * s


def fibre_length(s):
    return 2 * math.pi * s


def berger_metric_fibre_speed(s, n=1000):
    """Length of a Hopf fibre in the Berger metric, from the round metric (|d/dt e^{it}(z1,z2)| = 1 per radian) scaled by s,
    computed numerically from the sampled fibre in R^4 so the number is independent of the formula above."""
    z1, z2 = H.section((0.3, 0.4, math.sqrt(1 - 0.25)))
    t = np.linspace(0, H.TWO_PI, n + 1)
    F = H.to_r4(np.stack([np.exp(1j * t) * z1, np.exp(1j * t) * z2], axis=1))
    return s * float(np.linalg.norm(np.diff(F, axis=0), axis=1).sum())


def levels_per_turn(a0=1.0):
    lam = math.sqrt(4 + math.sqrt(15))
    return 2 * math.pi * a0 / math.log(lam)


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    check("round S^3 (s = 1): Ricci = 2, scalar = 6", np.allclose(berger_ricci(1.0), 2.0) and abs(berger_scalar(1.0) - 6) < 1e-12)
    check("Berger sphere: Ric = (4 - 2 s^2, 4 - 2 s^2, 2 s^2), scalar 8 - 2 s^2",
          all(np.allclose(berger_ricci(s), (4 - 2 * s * s, 4 - 2 * s * s, 2 * s * s)) and abs(berger_scalar(s) - (8 - 2 * s * s)) < 1e-12
              for s in (0.1, 0.5, 1.0, 1.7, 3.0)))
    check("fibre collapse s -> 0: curvature stays finite (scalar -> 8) while the volume -> 0",
          abs(berger_scalar(1e-3) - 8) < 1e-5 and berger_volume(1e-3) < 0.02)
    check("Einstein only at s = 1 (fibre and transverse Ricci agree)",
          all(abs(berger_ricci(s)[0] - berger_ricci(s)[2]) > 1e-3 for s in (0.5, 0.9, 1.1, 2.0))
          and abs(berger_ricci(1.0)[0] - berger_ricci(1.0)[2]) < 1e-12)
    check("fibre length 2 pi s (numerical, from the sampled fibre)",
          all(abs(berger_metric_fibre_speed(s) - fibre_length(s)) < 1e-3 for s in (0.3, 1.0, 2.0)))
    N = levels_per_turn(1.0)
    check("NUMEROLOGY, asserted to miss: levels per turn at a0 = 1 is 6.0900.., not 6", abs(N - 6.0900) < 1e-3 and abs(N - 6) > 0.05,
          "(N = %.4f, %.2f %% off 6)" % (N, 100 * (N - 6) / 6))
    a0 = 3 * math.log(math.sqrt(4 + math.sqrt(15))) / math.pi
    check("... N = 6 exactly needs a0 = 3 ln(lambda)/pi, a value chosen to give the answer", abs(levels_per_turn(a0) - 6) < 1e-12,
          "(a0 = %.4f)" % a0)
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.linspace(0.02, 2.5, 200)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(s, [berger_ricci(x)[2] for x in s], label="Ricci along the fibre  2 s^2")
    ax[0].plot(s, [berger_ricci(x)[0] for x in s], label="Ricci across the fibre  4 - 2 s^2")
    ax[0].plot(s, [berger_scalar(x) for x in s], label="scalar  8 - 2 s^2")
    ax[0].plot(s, [berger_volume(x) for x in s], label="volume  2 pi^2 s", ls="--")
    ax[0].axvline(1, color="0.6", lw=0.8)
    ax[0].set_xlabel("fibre scale s  (1 = round S^3)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Berger sphere: collapsing the Hopf fibre")
    ax[1].axis("off")
    ax[1].text(0.0, 0.9, "Hopf fibration with fibre scaled by s\n\n"
               "s = 1: round S^3, Einstein (Ric = 2)\n"
               "s -> 0: fibre collapses, volume -> 0,\n"
               "        curvature stays finite (scalar -> 8)\n"
               "        (Cheeger-Gromov collapse onto S^2)\n\n"
               "Bianchi IX with two equal scale factors\n"
               "is a Berger sphere (Mixmaster).\n\n"
               "numerology (asserted in the tests):\n"
               "levels per turn N(a0=1) = %.4f, not 6" % levels_per_turn(1.0), va="top", family="monospace", fontsize=9)
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
