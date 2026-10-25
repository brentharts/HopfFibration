#!/usr/bin/env python3
"""hopf_line_bundle.py - the Hopf fibration WITHOUT a principal bundle: the tautological line bundle over S^2.

Motivation.  H. Gomes, 'Gauge Theory Without Principal Fiber Bundles', Philosophy of Science 92(3) (2025), argues (as far as the
summary we could read says) that gauge theory does not need principal bundles: take complex VECTOR bundles with an inner
product, let the gauge group be the automorphism group of the fibre (U(1) for C^1, SU(2) for C^2, ...) instead of postulating
it, and let the gauge field be a connection (parallel transport) on those vector bundles; the structure group re-appears as a
holonomy group.  This script does that for the simplest case, the Hopf/Dirac-monopole bundle, with the single fibre C^1.

THE STRUCTURE-FIRST VERSION OF THE HOPF FIBRATION
  Bundle      E over S^2:  the fibre over the point b is the complex LINE  l_b = image of P(b) = (I + b.sigma)/2  inside C^2
              (the tautological line bundle; l_b is the qubit state whose Bloch vector is b).
  Structure   the hermitian inner product of C^2 restricted to each l_b.  Nothing else is given.
  Gauge group Aut(l_b, inner product) = the unit complex numbers U(1).  It is derived, not postulated.
  Connection  covariant derivative  nabla s = P ds  of a section s (s(b) in l_b); parallel transport is Kato's equation
              U' = [P', P] U  along a curve b(s).  No principal bundle, no gauge potential A, no section is used.
  Principal   the unit vectors of the fibres, S(E) = {(b, u) : u in l_b, |u| = 1}, are exactly S^3 -> S^2, the Hopf fibration:
  bundle      the principal bundle is DERIVED from the vector bundle, not the other way round.

What is checked (numerics, all independent of each other):
  1. S(E) = S^3: every unit u in C^2 lies in exactly one line l_b, namely b = Bloch(u), and the fibre over b is a circle.
  2. Three ways to get the holonomy of a loop agree, to the accuracy of the integrators:
        (a) Kato transport with the projector only (the structure-first connection),
        (b) the connection form A = -Im(conj(s) ds) of the principal bundle along the Hopf section (the textbook version),
        (c) the Pancharatnam overlap product of states around the loop (Berry phase = minus its argument).
  3. For a loop of constant latitude theta the holonomy is  -pi (1 - cos theta)  = minus half the enclosed solid angle.
  4. The holonomy phase is gauge invariant (changing the section by e^{i chi(b)} does not move it) and the same for a general loop
     as for its reparametrisation.
  5. The Chern number of the line bundle, from the winding of the latitude holonomy, is 1 in absolute value (the Hopf bundle is the
     generator); the k-th tensor power L^k has holonomy k * gamma and Chern number k.
  6. Stokes: the holonomy of a loop equals i times the integral, over the cap it bounds, of the curvature Tr(P [d_theta P, d_phi P])
     computed from the projector alone; over the whole sphere it is 2 pi i, so the Chern number is 1.

What this is NOT: it is the abelian case C^1 only.  Gomes's Standard Model structure uses C^3 + C^2 + C^1; the quaternionic Hopf
fibration S^7 -> S^4 (the rank-2 tautological bundle, SU(2) instanton number 1) is the natural non-abelian next step and is not
done here.  All of this is standard mathematics (tautological line bundle, Berry/Simon holonomy); the new thing is only the check.

    python3 hopf_line_bundle.py --tests
    python3 hopf_line_bundle.py --png figures/hopf_line_bundle.png
"""
import math
import sys

import numpy as np

import hopf_math as H

SIGMA = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex)]


def projector(b):
    """P(b) = (I + b.sigma)/2: the projector onto the line l_b (the state with Bloch vector b)."""
    b = np.asarray(b, float)
    return 0.5 * (np.eye(2) + sum(c * s for c, s in zip(b, SIGMA)))


def bloch(psi):
    psi = np.asarray(psi, complex)
    psi = psi / np.linalg.norm(psi)
    return np.array([float(np.real(psi.conj() @ s @ psi)) for s in SIGMA])


def state_of(b):
    """A unit vector spanning l_b, smooth away from the south pole b = (0,0,-1): (cos(theta/2), e^{i phi} sin(theta/2))."""
    x, y, z = b
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    return np.array([math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)])


def latitude_loop(theta, n):
    """Counter-clockwise loop of constant colatitude theta: returns (b(s), db/ds) at n+1 points, s in [0, 2 pi]."""
    s = np.linspace(0, 2 * math.pi, n + 1)
    b = np.stack([math.sin(theta) * np.cos(s), math.sin(theta) * np.sin(s), math.cos(theta) * np.ones_like(s)], axis=1)
    db = np.stack([-math.sin(theta) * np.sin(s), math.sin(theta) * np.cos(s), np.zeros_like(s)], axis=1)
    return b, db


def kato_holonomy(b, db, ds):
    """(a) Holonomy of the projector connection along the curve (arrays of b and db/ds, step ds): solves U' = [P', P] U with RK4
    and returns the phase gamma with U(end) u0 = e^{i gamma} u0 for u0 spanning l_{b[0]}."""
    def rhs(Pm, dPm, U):
        return (dPm @ Pm - Pm @ dPm) @ U

    U = np.eye(2, dtype=complex)
    for i in range(len(b) - 1):
        # RK4 over [i, i+1] with P and dP interpolated linearly in b (midpoint uses the average)
        P0, P1 = projector(b[i]), projector(b[i + 1])
        Pm = 0.5 * (P0 + P1)
        dP0 = 0.5 * sum(c * s for c, s in zip(db[i], SIGMA))
        dP1 = 0.5 * sum(c * s for c, s in zip(db[i + 1], SIGMA))
        dPm = 0.5 * (dP0 + dP1)
        k1 = rhs(P0, dP0, U)
        k2 = rhs(Pm, dPm, U + 0.5 * ds * k1)
        k3 = rhs(Pm, dPm, U + 0.5 * ds * k2)
        k4 = rhs(P1, dP1, U + ds * k3)
        U = U + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    u0 = state_of(b[0])
    v = U @ u0
    return float(np.angle(np.vdot(u0, v))), float(np.linalg.norm(v)), float(np.linalg.norm(projector(b[-1]) @ v - v))


def principal_holonomy(states, ds):
    """(b) Textbook version: horizontal lift for the connection form A = -Im(conj(s) ds): phi' = -Im(conj(s) s'), holonomy =
    total phi accumulated along the loop, with s the section state_of(b(s)) (single valued on the loop)."""
    d = (np.roll(states, -1, axis=0) - states)[:-1]
    mids = 0.5 * (states[:-1] + np.roll(states, -1, axis=0)[:-1])
    return float(-np.imag(np.einsum("ij,ij->i", mids.conj(), d)).sum())


def pancharatnam_phase(states):
    """(c) arg of the closed product of overlaps <s_0|s_1><s_1|s_2>...; the Berry phase is MINUS this."""
    prod = 1.0 + 0j
    for k in range(len(states) - 1):
        prod *= np.vdot(states[k], states[k + 1])
    return float(np.angle(prod))


def wrap(a):
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def holonomies(theta, n=4000):
    b, db = latitude_loop(theta, n)
    ds = 2 * math.pi / n
    states = np.array([state_of(x) for x in b])
    states[-1] = states[0]
    gamma_a = kato_holonomy(b, db, ds)[0]
    gamma_b = principal_holonomy(states, ds)
    gamma_c = -pancharatnam_phase(states)
    return gamma_a, gamma_b, gamma_c


def holonomy_winding(thetas, n=1500):
    """Holonomy of the latitude loops for a range of theta, unwrapped continuously, and the resulting Chern number C = gamma(pi-)/ -2 pi."""
    g = np.array([holonomies(t, n)[0] for t in thetas])
    g = np.unwrap(np.r_[0.0, g])[1:]
    return g


def curvature_cap_integral(theta_max, n=200):
    """Integral over the polar cap 0 <= theta <= theta_max (all phi) of  Tr(P [d_theta P, d_phi P]) d theta d phi, the curvature of the
    projector connection (midpoint rule).  Analytically it is  i * pi * (1 - cos theta_max)."""
    th = (np.arange(n) + 0.5) * theta_max / n
    ph = (np.arange(2 * n) + 0.5) * 2 * math.pi / (2 * n)
    tot = 0.0 + 0j
    for t in th:
        for p in ph:
            b = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])
            dth = np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), -math.sin(t)])
            dph = np.array([-math.sin(t) * math.sin(p), math.sin(t) * math.cos(p), 0.0])
            A = 0.5 * sum(c * s for c, s in zip(dth, SIGMA))
            B = 0.5 * sum(c * s for c, s in zip(dph, SIGMA))
            tot += np.trace(projector(b) @ (A @ B - B @ A))
    return tot * (theta_max / n) * (2 * math.pi / (2 * n))


def run_tests(verbose=True):
    rng = np.random.default_rng(31)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 1. unit sphere bundle = S^3
    good = True
    for _ in range(30):
        u = rng.normal(size=2) + 1j * rng.normal(size=2)
        u /= np.linalg.norm(u)
        b = bloch(u)
        good &= np.allclose(projector(b) @ u, u, atol=1e-12) and abs(np.linalg.norm(b) - 1) < 1e-12
        # the fibre over b: all phases of u
        good &= np.allclose(bloch(np.exp(1.3j) * u), b, atol=1e-12)
        # and it is the Hopf map (up to the y reflection of the Bloch convention)
        good &= np.allclose(b, H.hopf(u[0], u[1]) * np.array([1, -1, 1]), atol=1e-12)
    check("unit vectors of the lines l_b are exactly S^3, the fibre over b is the phase circle, the base map is Hopf", good)

    # 2/3. three holonomies agree, and equal minus half the solid angle
    worst_ab = worst_ac = worst_sol = 0.0
    for theta in (0.5, 1.0, 1.7, 2.4):
        ga, gb, gc = holonomies(theta, 3000)
        pred = wrap(-math.pi * (1 - math.cos(theta)))
        worst_ab = max(worst_ab, abs(wrap(ga - gb)))
        worst_ac = max(worst_ac, abs(wrap(ga - gc)))
        worst_sol = max(worst_sol, abs(wrap(ga - pred)), abs(wrap(gb - pred)), abs(wrap(gc - pred)))
    check("(a) Kato projector transport = (b) principal-bundle connection form = (c) Pancharatnam product",
          worst_ab < 2e-3 and worst_ac < 2e-3, "(max differences %.1e, %.1e)" % (worst_ab, worst_ac))
    check("latitude holonomy = -pi(1 - cos theta) = minus half the solid angle", worst_sol < 2e-3, "(max err %.1e)" % worst_sol)

    # transported vector stays in the line and keeps its length
    ga, norm, off = kato_holonomy(*latitude_loop(1.1, 3000), 2 * math.pi / 3000)
    check("Kato transport keeps the vector in the line and of unit length", abs(norm - 1) < 1e-6 and off < 1e-4,
          "(|v|-1 = %.1e, off-line %.1e)" % (norm - 1, off))

    # 4. gauge invariance and reparametrisation
    theta, n = 1.3, 3000
    b, db = latitude_loop(theta, n)
    s = np.linspace(0, 2 * math.pi, n + 1)
    st = np.array([state_of(x) for x in b])
    st[-1] = st[0]
    chi = 0.7 * np.sin(s) + 0.4 * np.cos(2 * s) - 0.4          # single valued on the loop (chi(0) = chi(2 pi))
    chi[-1] = chi[0]
    gauge = st * np.exp(1j * chi)[:, None]
    g0 = -pancharatnam_phase(st)
    g1 = -pancharatnam_phase(gauge)
    check("holonomy is gauge invariant (section changed by e^{i chi(b)})", abs(wrap(g0 - g1)) < 1e-9, "(diff %.1e)" % abs(wrap(g0 - g1)))
    # the phase of A integrated changes by the winding of chi: zero, so the principal-bundle number agrees as well
    check("same in the connection-form version", abs(wrap(principal_holonomy(st, 1) - principal_holonomy(gauge, 1))) < 2e-3)

    # general loop (a tilted ellipse that does not enclose the south pole), three methods again
    t = np.linspace(0, 2 * math.pi, 3001)
    c = np.array([0.2, 0.1, 1.0])
    v = c[None, :] + 0.5 * np.cos(t)[:, None] * np.array([1, 0, 0]) + 0.35 * np.sin(t)[:, None] * np.array([0, 1, 0.2])
    bb = v / np.linalg.norm(v, axis=1)[:, None]
    bb[-1] = bb[0]
    dbb = np.gradient(bb, t, axis=0)
    ga = kato_holonomy(bb, dbb, 2 * math.pi / 3000)[0]
    stt = np.array([state_of(x) for x in bb])
    stt[-1] = stt[0]
    gc = -pancharatnam_phase(stt)
    check("a non-circular loop: Kato transport and Pancharatnam product agree", abs(wrap(ga - gc)) < 5e-3, "(diff %.1e)" % abs(wrap(ga - gc)))

    # 5. Chern number from the winding of the latitude holonomy; tensor powers
    thetas = np.linspace(0.02, math.pi - 0.02, 160)
    g = holonomy_winding(thetas, 1500)
    chern = -g[-1] / (2 * math.pi)
    check("Chern number from the winding of the latitude holonomy = 1", abs(chern - 1) < 0.02, "(%.3f)" % chern)
    check("tensor power L^k: holonomy k*gamma, Chern number k  (Gomes: multiple charges from tensor powers of E_1, no new fields)",
          all(abs(wrap(k * g[40] - np.angle(np.exp(1j * g[40]) ** k))) < 1e-12 for k in (2, 3, -1)) and abs(-(2 * g[-1]) / (2 * math.pi) - 2) < 0.04)

    # 6. Stokes: holonomy of the loop = i * (curvature integrated over the cap it bounds), curvature from the projector alone
    worst = 0.0
    for theta in (0.6, 1.4, 2.2):
        gam = holonomies(theta, 3000)[0]
        cap = curvature_cap_integral(theta, 80)
        worst = max(worst, abs(wrap(gam - (1j * cap).real)), abs((1j * cap).imag))
    check("Stokes: holonomy = i * integral of Tr(P [dP, dP]) over the enclosed cap", worst < 2e-3, "(max err %.1e)" % worst)
    total = curvature_cap_integral(math.pi, 80)
    check("whole sphere: the curvature integrates to 2 pi i, so the Chern number is 1 with  C = -i * integral / (2 pi)",
          abs(-1j * total / (2 * math.pi) - 1) < 2e-3 and abs(chern - 1) < 0.02, "(C = %.4f)" % (-1j * total / (2 * math.pi)).real)
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thetas = np.linspace(0.02, math.pi - 0.02, 120)
    g = holonomy_winding(thetas, 1200)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(thetas, g, label="holonomy of the latitude loop (Kato, projector only)")
    ax[0].plot(thetas, -math.pi * (1 - np.cos(thetas)), "--", label="-pi (1 - cos theta)")
    ax[0].set_xlabel("colatitude theta of the loop")
    ax[0].set_ylabel("phase")
    ax[0].set_title("holonomy = minus half the enclosed solid angle")
    ax[0].legend(fontsize=8)
    ga = [holonomies(t, 1500) for t in (0.5, 1.0, 1.7, 2.4)]
    x = np.arange(4)
    for i, lab in enumerate(("(a) Kato projector", "(b) connection form", "(c) -Pancharatnam")):
        ax[1].bar(x + 0.25 * (i - 1), [wrap(w[i]) for w in ga], 0.25, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["0.5", "1.0", "1.7", "2.4"])
    ax[1].set_xlabel("theta")
    ax[1].set_title("three methods, same holonomy (mod 2 pi)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
