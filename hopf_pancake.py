#!/usr/bin/env python3
"""hopf_pancake.py - the collapsing Hopf fibre of the pancake cosmology: Berger slices, finite curvature, and light winding.

In a biaxial Bianchi IX universe the spatial slice is a Berger sphere: S^3 with the Hopf fibre scaled by R_1 and the base S^2
scaled by R_2 = R_3.  In the non-singular 'pancake' model of Dechant, Lasenby and Hobson (arXiv:0809.4335) the fibre scale
goes through zero linearly while the base stays finite.  This script uses the leading-order behaviour near the pancake,

    R_1(t) = a0 t ,     R_2(t) = R_3(t) = b0 ,

which is how the companion repository brentharts/Quantum_Gravity_Lab (qgl_pancake.py, qgl_horizon.py) states it.  It is the
leading term of a series, NOT a solution of the field equations: nothing here evolves the scalar field or checks the paper.
What it does show are the Hopf-fibration facts that do not depend on the dynamics:

  * the slice is the Berger sphere with s = R_1 / R_2, so its Ricci curvature is berger_ricci(s) / b0^2 (hopf_berger.py):
    as the fibre collapses the curvature stays finite (scalar -> 8 / b0^2) while the volume 2 pi^2 R_1 R_2^2 -> 0;
  * the fibre is a great circle of the round S^3 scaled by R_1, so its circumference is 2 pi R_1 (the Euler angle psi has
    period 4 pi and the metric is (R_1/2)^2 (d psi + cos(th) d phi)^2);
  * a null ray moving along the fibre sweeps psi' = 2 / R_1, so it winds (2 / (4 pi a0)) ln(t / t_in) times, without bound
    as t_in -> 0; a null ray across the base sweeps theta' = 2 / R_2 and first reaches the antipode (theta = pi) at
    t = pi b0 / 2.  The same fibre clock gives the 'levels per turn' 2 pi a0 / ln(lambda) of hopf_berger.levels_per_turn.

    python3 hopf_pancake.py --tests
    python3 hopf_pancake.py --png figures/hopf_pancake.png
"""
import math
import sys

import numpy as np

import hopf_berger as B

FIBRE_PERIOD = 4.0 * math.pi        # period of the Euler angle psi on the standard S^3 (checked in the tests)


def slice_ricci(t, a0=1.0, b0=3.0):
    """Ricci curvature of the slice R = (a0 t, b0, b0): (along the fibre, across, across), orthonormal frame.

    Uses the closed form of the Berger sphere, (2 s^2, 4 - 2 s^2, 4 - 2 s^2) / b0^2 with s = a0 t / b0, which the tests of
    hopf_berger.py assert equals Milnor's formula.  Milnor's formula itself (hopf_berger.berger_ricci) subtracts numbers of
    size 1/s and loses every digit once s is below about 1e-6, which is exactly the regime of the pancake."""
    s = a0 * np.asarray(t, float) / b0
    out = np.stack([2 * s * s, 4 - 2 * s * s, 4 - 2 * s * s], axis=-1) / b0 ** 2
    return out


def fibre_circumference(t, a0=1.0):
    return 2.0 * math.pi * a0 * np.asarray(t, float)


def fibre_angle(t, a0=1.0, t_in=1e-9):
    """Angle psi swept by a null ray along the fibre from t_in to t: int 2 dt / R_1 = (2 / a0) ln(t / t_in)."""
    return (2.0 / a0) * np.log(np.asarray(t, float) / t_in)


def fibre_turns(t, a0=1.0, t_in=1e-9):
    return fibre_angle(t, a0, t_in) / FIBRE_PERIOD


def base_angle(t, b0=3.0, t_in=1e-9):
    """Polar angle swept by a null ray across the base from t_in to t: int 2 dt / R_2 = 2 (t - t_in) / b0."""
    return 2.0 * (np.asarray(t, float) - t_in) / b0


def base_contact_time(b0=3.0, t_in=1e-9):
    """First time at which antipodal points of the base S^2 are in causal contact: base_angle = pi."""
    return math.pi * b0 / 2.0 + t_in


def efolds_per_fibre_turn(a0=1.0):
    return 2.0 * math.pi * a0


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    for a0, b0 in [(1.0, 3.0), (0.7, 1.0), (2.0, 18000.0)]:
        r = slice_ricci(1e-9, a0, b0)
        check("a0 = %g, b0 = %g: at t = 1e-9 the curvature is finite, scalar b0^2 -> 8, Ricci along the fibre -> 0" % (a0, b0),
              abs(r.sum() * b0 ** 2 - 8) < 1e-6 and abs(r[0]) * b0 ** 2 < 1e-6, "(%.8f)" % (r.sum() * b0 ** 2))
    t_round = 3.0 / 1.0                                   # s = 1: the slice is the round sphere of radius b0
    check("s = a0 t / b0 = 1 is the round S^3 of radius b0: Ricci = 2 / b0^2 in every direction",
          np.allclose(slice_ricci(t_round, 1.0, 3.0), 2.0 / 9.0))
    vol = lambda t, a0=1.0, b0=3.0: b0 ** 3 * B.berger_volume(a0 * t / b0)        # 2 pi^2 R_1 R_2 R_3
    check("the slice volume 2 pi^2 R_1 R_2^2 vanishes linearly with the fibre (ratio 1e-9 at t = 1e-9) while the scalar "
          "curvature does not", abs(vol(1e-9) / vol(1.0) - 1e-9) < 1e-15 and abs(slice_ricci(1e-9).sum() - 8 / 9.0) < 1e-6)
    check("closed form agrees with Milnor's formula (hopf_berger.berger_ricci) where the latter is accurate",
          all(np.allclose(slice_ricci(a * 3.0, 1.0, 3.0)[[1, 2, 0]] * 9.0, B.berger_ricci(a)) for a in (0.01, 0.3, 1.0, 1.9)))
    # the fibre circumference is the numerically measured length of a fibre of the round S^3 scaled by R_1
    for t in (0.1, 0.5, 2.0):
        s = 1.0 * t / 3.0
        check("t = %g: fibre circumference 2 pi a0 t = b0 x (measured length of the Hopf fibre scaled by s)" % t,
              abs(3.0 * B.berger_metric_fibre_speed(s) - fibre_circumference(t)) < 1e-3)
    # the analytic fibre angle equals the integral of 2 / R_1 (log-spaced trapezoid)
    t_in, t1, a0 = 1e-6, 1.0, 0.8
    tt = np.geomspace(t_in, t1, 400001)
    num = np.trapezoid(2.0 / (a0 * tt), tt)
    check("fibre angle (2/a0) ln(t/t_in) equals the integral of 2 / R_1", abs(num - fibre_angle(t1, a0, t_in)) < 1e-5,
          "(%.6f vs %.6f)" % (num, fibre_angle(t1, a0, t_in)))
    check("the winding is unbounded: turns grow like ln(1/t_in) / (2 pi a0)",
          abs((fibre_turns(1.0, 1.0, 1e-12) - fibre_turns(1.0, 1.0, 1e-6)) - math.log(1e6) / (2 * math.pi)) < 1e-9)
    check("base: antipodal points first meet at t = pi b0 / 2 (the base angle reaches pi)",
          abs(base_angle(base_contact_time(3.0)) - math.pi) < 1e-12 and abs(base_contact_time(3.0) - 4.7124) < 1e-3)
    # the 4 pi period (the identification behind the winding count) on the Euler-angle S^3
    th, ph = 1.1, 0.7

    def emb(ps):
        return np.array([math.cos(th / 2) * math.cos((ps + ph) / 2), math.cos(th / 2) * math.sin((ps + ph) / 2),
                         math.sin(th / 2) * math.cos((ps - ph) / 2), math.sin(th / 2) * math.sin((ps - ph) / 2)])
    check("the Euler angle psi has period 4 pi on S^3 (psi = 2 pi is the antipodal point)",
          np.allclose(emb(FIBRE_PERIOD), emb(0.0)) and np.allclose(emb(2 * math.pi), -emb(0.0)))
    lam = math.sqrt(4 + math.sqrt(15))
    check("e-folds per fibre turn 2 pi a0 = levels per turn x ln(lambda): the clock of hopf_berger.levels_per_turn",
          all(abs(B.levels_per_turn(a) * math.log(lam) - efolds_per_fibre_turn(a)) < 1e-12 for a in (0.5, 1.0, 2.3)))
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a0, b0, t_in = 1.0, 3.0, 1e-9
    t = np.geomspace(1e-4, b0 / a0, 600)                      # up to s = 1, where the slice is the round sphere
    R = slice_ricci(t, a0, b0)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ax[0].semilogx(t, R[:, 0] * b0 ** 2, label="Ricci along the fibre", color="#c0392b")
    ax[0].semilogx(t, R[:, 1] * b0 ** 2, label="Ricci across the fibre (each of 2 directions)", color="#1f5fa8")
    ax[0].semilogx(t, R.sum(axis=1) * b0 ** 2, label="scalar curvature", color="k")
    ax[0].axhline(8, color="0.6", ls="--", lw=0.8)
    ax[0].text(1.3e-4, 7.3, "limit 8 / b0^2", fontsize=8, color="0.4")
    ax[0].set_xlim(1e-4, b0 / a0 * 1.02)
    ax[0].set_xlabel("t   (right edge: s = a0 t / b0 = 1, the round S^3)")
    ax[0].set_ylabel("curvature  x  b0^2")
    ax[0].set_title("Slice curvature stays finite as the fibre collapses", fontsize=10)
    ax[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax[1].loglog(t, fibre_circumference(t, a0), color="#c0392b", label="fibre circumference  2 pi R_1")
    ax[1].loglog(t, np.full_like(t, math.pi * b0), color="#1f5fa8", label="base great circle  pi R_2")
    ax[1].set_xlabel("t")
    ax[1].set_title("Proper sizes: the fibre closes up, the base does not", fontsize=10)
    ax[1].legend(fontsize=8)
    tt = np.geomspace(1e-8, base_contact_time(b0, t_in), 600)
    ax[2].semilogx(tt, fibre_turns(tt, a0, t_in), color="#c0392b", label="turns round the fibre  (2/(4 pi a0)) ln(t/t_in)")
    ax[2].semilogx(tt, base_angle(tt, b0, t_in) / math.pi, color="#1f5fa8", label="base angle / pi  (2 t / b0) / pi")
    ax[2].axhline(1, color="0.6", ls="--", lw=0.8)
    ax[2].axvline(base_contact_time(b0), color="0.6", ls=":", lw=0.8)
    ax[2].text(base_contact_time(b0) * 0.8, 2.0, "t = pi b0 / 2:\nantipodes of the\nbase meet", fontsize=8, color="0.4", ha="right")
    ax[2].set_xlabel("t")
    ax[2].set_title("Light winds without bound round the collapsing fibre", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Leading-order pancake R_1 = a0 t, R_2 = R_3 = b0  (a0 = 1, b0 = 3, t_in = 1e-9); not a solution of the field equations",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
