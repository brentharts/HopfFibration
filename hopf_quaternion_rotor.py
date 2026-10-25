#!/usr/bin/env python3
"""hopf_quaternion_rotor.py - rotors, the double cover SU(2) -> SO(3), spinors and the qubit on the Bloch sphere.

The Hopf map is "where does a rotation send the axis k":  h(q) = q k conj(q)  (hopf_math.py, module docstring).  So the
Hopf fibre over a point b of S^2 is the set of unit quaternions (rotors) q whose rotation carries k to b; two such rotors
differ by a rotation about k, i.e. by right multiplication with  e^{k t} = cos t + k sin t.  That is the circle.

What is checked (exact up to floating point):
  1. h(q) = R(q) k, with R(q) v = q v conj(q) the rotation matrix of the rotor q.
  2. the fibre is q e^{k t}: right multiplication by the circle  cos t + k sin t  is the phase e^{it} on (z1, z2).
  3. DOUBLE COVER: q and -q give the same rotation; a rotor with rotation angle 2 pi is -1, not +1 (spinor sign).
  4. QUBIT: for a state (alpha, beta) the Bloch vector (<sigma_x>, <sigma_y>, <sigma_z>) equals the Hopf map of (alpha, beta)
     with y reflected, a global phase is invisible, and under H = (Omega/2) n.sigma the Bloch vector rotates about n at
     angular rate Omega.  A full 2 pi rotation of the Bloch vector multiplies the state by -1.
  5. BERRY PHASE: carrying a state around a loop of constant latitude theta on the Bloch sphere (Pancharatnam product of
     overlaps) gives a phase equal to HALF the solid angle, pi (1 - cos theta), a gauge-invariant holonomy of the Hopf bundle.

    python3 hopf_quaternion_rotor.py --tests
    python3 hopf_quaternion_rotor.py --png figures/hopf_qubit.png
"""
import math
import sys

import numpy as np

import hopf_math as H

SIGMA = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex)]
K = np.array([0.0, 0.0, 0.0, 1.0])


def rotation_matrix(q):
    """3x3 rotation matrix of the unit quaternion q = (a, b, c, d): v -> q v conj(q)."""
    q = np.asarray(q, float)
    cols = [H.qmul(H.qmul(q, np.r_[0.0, e]), H.qconj(q))[1:] for e in np.eye(3)]
    return np.array(cols).T


def rotor(axis, angle):
    """The unit quaternion that rotates by `angle` about the unit vector `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def bloch(alpha, beta):
    psi = np.array([alpha, beta], complex)
    psi = psi / np.linalg.norm(psi)
    return np.array([float(np.real(psi.conj() @ s @ psi)) for s in SIGMA])


def evolve(psi, n, Omega_t):
    """exp(-i (Omega t / 2) n.sigma) psi : rotation of the Bloch vector by angle Omega t about n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    ns = sum(c * s for c, s in zip(n, SIGMA))
    U = math.cos(Omega_t / 2) * np.eye(2) - 1j * math.sin(Omega_t / 2) * ns
    return U @ np.asarray(psi, complex)


def rodrigues(v, n, angle):
    n = np.asarray(n, float) / np.linalg.norm(n)
    return v * math.cos(angle) + np.cross(n, v) * math.sin(angle) + n * (n @ v) * (1 - math.cos(angle))


def pancharatnam_phase(states):
    """arg of the closed product <psi_0|psi_1><psi_1|psi_2>...<psi_{N-1}|psi_0> (gauge invariant)."""
    prod = 1.0 + 0j
    n = len(states)
    for k in range(n):
        prod *= np.vdot(states[k], states[(k + 1) % n])
    return float(np.angle(prod))


def run_tests(verbose=True):
    rng = np.random.default_rng(21)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    qs = [v / np.linalg.norm(v) for v in rng.normal(size=(30, 4))]
    check("R(q) is a proper rotation", all(np.allclose(rotation_matrix(q) @ rotation_matrix(q).T, np.eye(3))
                                              and abs(np.linalg.det(rotation_matrix(q)) - 1) < 1e-12 for q in qs))
    check("h(q) = R(q) k  (the Hopf map is 'where the rotor sends the axis k')",
          all(np.allclose(rotation_matrix(q) @ np.array([0, 0, 1.0]),
                          H.hopf(complex(q[0], q[3]), complex(q[2], q[1]))) for q in qs))
    # fibre = right multiplication by exp(k t)
    good = True
    for q in qs[:10]:
        z1, z2 = complex(q[0], q[3]), complex(q[2], q[1])
        for t in rng.uniform(0, 2 * math.pi, 5):
            lhs = H.quaternion_of(np.exp(1j * t) * z1, np.exp(1j * t) * z2)
            r1 = H.qmul(q, np.r_[math.cos(t), 0, 0, math.sin(t)])
            good &= bool(np.allclose(lhs, r1))
    check("the fibre is q e^{k t}: the phase e^{it} on (z1,z2) = right multiplication by a rotation about k", good)
    check("q and -q are the same rotation (double cover)", all(np.allclose(rotation_matrix(q), rotation_matrix(-q)) for q in qs))
    r2pi = rotor([0, 0, 1.0], 2 * math.pi)
    check("a rotor of rotation angle 2 pi is -1 (spinor sign), 4 pi is +1",
          np.allclose(r2pi, [-1, 0, 0, 0], atol=1e-12) and np.allclose(rotor([0, 0, 1.0], 4 * math.pi), [1, 0, 0, 0], atol=1e-12))
    # qubit
    good = True
    for _ in range(20):
        a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
        nrm = math.hypot(abs(a), abs(b))
        a, b = a / nrm, b / nrm
        good &= np.allclose(bloch(a, b), H.hopf(a, b) * np.array([1, -1, 1]), atol=1e-12)
        good &= np.allclose(bloch(a, b), bloch(np.exp(0.77j) * a, np.exp(0.77j) * b), atol=1e-12)
    check("Bloch vector = Hopf map with y reflected; invisible to a global phase", good)
    good = True
    for _ in range(10):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        n = rng.normal(size=3)
        ang = rng.uniform(0, 2 * math.pi)
        r = bloch(*evolve(psi / np.linalg.norm(psi), n, ang))
        good &= np.allclose(r, rodrigues(bloch(*psi), n, ang), atol=1e-12)
    check("H = (Omega/2) n.sigma rotates the Bloch vector about n by Omega t", good)
    psi = np.array([0.6, 0.8j])
    full = evolve(psi, [1, 2, 3], 2 * math.pi)
    check("after one full turn of the Bloch vector the state is -psi", np.allclose(full, -psi) and np.allclose(bloch(*full), bloch(*psi)))
    # Berry / Pancharatnam
    worst = 0.0
    for theta in (0.4, 1.0, 1.9, 2.8):
        phis = np.linspace(0, 2 * math.pi, 4000, endpoint=False)
        states = [np.array([math.cos(theta / 2), np.exp(1j * ph) * math.sin(theta / 2)]) for ph in phis]
        gamma = pancharatnam_phase(states)
        solid = 2 * math.pi * (1 - math.cos(theta))
        worst = max(worst, abs(((gamma - solid / 2 + math.pi) % (2 * math.pi)) - math.pi))
    check("Pancharatnam/Berry phase around a latitude circle = half the solid angle", worst < 1e-3, "(max err %.1e)" % worst)
    return ok


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", alpha=0.3, linewidth=0)
    psi0 = np.array([math.cos(0.5), math.sin(0.5) * np.exp(0.3j)])
    n = np.array([0.3, 0.2, 1.0])
    ts = np.linspace(0, 2 * math.pi, 200)
    path_pts = np.array([bloch(*evolve(psi0, n, t)) for t in ts])
    ax.plot(*path_pts.T, color="tab:red", lw=2)
    ax.scatter(*bloch(*psi0), color="k")
    ax.plot(*np.array([-1.2 * n / np.linalg.norm(n), 1.2 * n / np.linalg.norm(n)]).T, color="k", ls="--", lw=1)
    ax.set_title("qubit under H = (Omega/2) n.sigma: the Bloch vector circles n;\none turn multiplies the state by -1")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    else:
        sys.exit(0 if run_tests() else 1)
