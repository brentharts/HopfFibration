#!/usr/bin/env python3
"""hopf_bkl.py - the Mixmaster walls as a hyperbolic reflection group, and the `ray` inversion family.

    python3 hopf_bkl.py --tests      checks (the reflection-law test needs the wada-coudene clone)
    python3 hopf_bkl.py --trend      how well the sharp-wall reflection law fits real trajectories (about 5 minutes)

Content (see HOPF_BKL.md):
  * the kinetic form of the diagonal Bianchi IX system, sum_{i<j} v_i v_j, is Lorentzian; the three curvature walls
    q_i^2 = exp(2 x_i) have Gram matrix with all off-diagonal cosines -1, i.e. they bound an *ideal* triangle
    (Coxeter (inf, inf, inf)); with the three symmetry walls x_i = x_j the domain is the (2, 3, inf) triangle of PGL(2, Z);
  * the ideal vertices are the Kasner states p = (1,0,0) and permutations, the three exits of `MIXMASTER.md`;
  * the family of reflection triangles (m, m, n) of brentharts/ray (paper section 'Inversion groups') contains the ideal
    triangle at m = n = infinity (kappa = 2, s = 1/2: the arbelos) and the (3,3,inf) triangle (kappa = 2, s = 1/4).
Needs numpy.
"""
import itertools
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------------------------------- Lorentz data
A = np.ones((3, 3)) - np.eye(3)                 # kinetic form T(v) = v^T A v / 2 = sum_{i<j} v_i v_j
Q = -np.linalg.inv(A)                          # form on covectors (walls); signature (-, +, +) once the sign is fixed
E = np.eye(3)


def gram(covs):
    G = np.array([[a @ Q @ b for b in covs] for a in covs])
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def interior_angles(covs):
    """Dihedral angles (radians) of the region {c . x <= 0 for all c}; cos(angle) = -gram."""
    G = gram(covs)
    return {(i, j): float(np.arccos(np.clip(-G[i, j], -1, 1))) for i in range(len(covs)) for j in range(i + 1, len(covs))}


def reflect_cov(u, c):
    """Reflection of covector u in the wall with covector c (preserves Q)."""
    return u - 2 * (u @ Q @ c) / (c @ Q @ c) * c


def reflection_matrix(c):
    return np.array([reflect_cov(E[k], c) for k in range(3)]).T


def order(M, nmax=60, tol=1e-9):
    P = np.eye(3)
    for k in range(1, nmax + 1):
        P = P @ M
        if np.abs(P - np.eye(3)).max() < tol:
            return k
    return np.inf


def bounce(i, v):
    """Reflection of a velocity off wall i of the sharp-wall billiard: v_i -> -v_i, v_j -> v_j + 2 v_i."""
    w = np.array(v, float)
    w[i] = -v[i]
    for j in range(3):
        if j != i:
            w[j] = v[j] + 2 * v[i]
    return w


# ---------------------------------------------------------------------------------------------------- ray's inversions
def ray_spheres(s, kap):
    """Parent unit circle and the two children, as in brentharts/ray examples/hopf_reflection.py."""
    sg = kap * s
    cg, off = np.sqrt(max(1 - sg ** 2, 0.0)), sg - s
    return [(np.zeros(3), 1.0), (np.array([cg, off, 0.0]), s), (np.array([cg, -off, 0.0]), s)]


def ray_cos(S, i, j):
    (c1, r1), (c2, r2) = S[i], S[j]
    return (r1 * r1 + r2 * r2 - np.sum((c1 - c2) ** 2)) / (2 * r1 * r2)


def ray_inv(x, S, i):
    c, r = S[i]
    v = x - c
    return c + r * r * v / np.sum(v * v)


def ray_word_err(S, word, order_, x=np.array([0.31, -0.17, 0.23])):
    y = x.copy()
    for _ in range(order_):
        for i in reversed(word):
            y = ray_inv(y, S, i)
    return float(np.linalg.norm(y - x))


# ---------------------------------------------------------------------------------------------------- the Wada repo
def _mixmaster():
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.environ.get("WADA_REPO"), os.path.join(here, "..", "wada-coudene")):
        if root and os.path.isdir(os.path.join(root, "lakes_of_wada")):
            sys.path.insert(0, root)
            try:
                from lakes_of_wada import mixmaster
                return mixmaster
            except Exception:
                pass
            finally:
                sys.path.remove(root)
    return None


# ---------------------------------------------------------------------------------------------------- tests
def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 1. the kinetic form is Lorentzian and the three curvature walls have all cosines -1 (an ideal triangle)
    ev = np.linalg.eigvalsh(Q)
    G = gram([E[0], E[1], E[2]])
    angs = interior_angles([E[0], E[1], E[2]])
    check("kinetic form has signature (1 timelike, 2 spacelike); the three curvature walls pairwise meet at angle 0 (ideal triangle)",
          (ev < 0).sum() == 1 and (ev > 0).sum() == 2 and np.abs(G - (2 * np.eye(3) - 1)).max() < 1e-12 and max(angs.values()) < 1e-6,
          "(Gram off-diagonal = %.3f)" % G[0, 1])

    # 2. adding the symmetry walls x_i = x_j gives the (2, 3, inf) triangle
    a, b, c = E[0] - E[1], E[1] - E[2], E[2]
    ang = interior_angles([a, b, c])
    want = {(0, 1): np.pi / 3, (0, 2): np.pi / 2, (1, 2): 0.0}
    check("walls x_1 = x_2, x_2 = x_3 and the curvature wall of axis 3 bound a triangle with angles pi/3, pi/2, 0 (Coxeter (2,3,inf))",
          all(abs(ang[k] - want[k]) < 1e-6 for k in want), "(%s)" % ", ".join("%.4f" % ang[k] for k in sorted(ang)))
    Ma, Mb, Mc = (reflection_matrix(x) for x in (a, b, c))
    oa, ob, oc = order(Ma @ Mb), order(Ma @ Mc), order(Mb @ Mc)
    check("reflections in these three walls: orders of ab, ac, bc are 3, 2 and infinity (the PGL(2,Z) Coxeter relations)",
          (oa, ob, oc) == (3, 2, np.inf), "(%s, %s, %s)" % (oa, ob, oc))

    # 3. cusps = Kasner states p = e_i = the exits
    cusp_ok = True
    for i in range(3):
        j, k = [t for t in range(3) if t != i]
        v = E[i]
        null = abs(v @ A @ v) < 1e-14                               # T(v) = 0: a Kasner state
        on_walls = abs(E[j] @ v) < 1e-14 and abs(E[k] @ v) < 1e-14   # lies on the two walls j and k
        cusp_ok &= null and on_walls
    check("the ideal vertex shared by walls j and k is the null vector e_i, the Kasner state p = (1,0,0): the exit with axis i", cusp_ok)

    # 4. the index 2 subgroup: doubling the triangle across the wall c gives (3,3,inf)
    Mc2 = Mc @ Mb @ Mc                                              # reflection in the image of wall b under c
    o1, o2, o3 = order(Ma @ Mb), order(Ma @ Mc2), order(Mb @ Mc2)
    check("reflections in a, b and c b c generate a (3,3,inf) triangle group (index 2 in the (2,3,inf) group)",
          (o1, o2, o3) == (3, 3, np.inf), "(%s, %s, %s)" % (o1, o2, o3))

    # 5. ray's family: m = n = inf is the arbelos with the same Gram matrix
    S = ray_spheres(0.5, 2.0)
    cpc, ccc = ray_cos(S, 0, 1), ray_cos(S, 1, 2)
    # the parent's normal points inward and the children's outward: parent/child enters as -cos, child/child as cos
    Gr = np.array([[1, -cpc, -cpc], [-cpc, 1, ccc], [-cpc, ccc, 1]])
    check("ray's inversion family at kappa = 2, s = 1/2 (parent unit circle, children of radius 1/2 tangent to it and to each other) has Gram matrix all -1 like the Mixmaster walls",
          abs(cpc - 1) < 1e-12 and abs(ccc + 1) < 1e-12 and np.abs(Gr - (2 * np.eye(3) - 1)).max() < 1e-12,
          "(cos parent/child %.3f, child/child %.3f)" % (cpc, ccc))

    # 6. ray's (3,3,inf): kappa = 2, s = 1/4: Coxeter relations (r0 r1)^3 = (r0 r2)^3 = 1, (r1 r2) parabolic
    S = ray_spheres(np.cos(np.pi / 3) / 2.0, 2.0)
    e01, e02 = ray_word_err(S, (0, 1), 3), ray_word_err(S, (0, 2), 3)
    e12 = min(ray_word_err(S, (1, 2), n) for n in range(1, 40))
    check("ray's inversions at kappa = 2, s = 1/4 satisfy (r0 r1)^3 = (r0 r2)^3 = 1 and r1 r2 has no finite order (3,3,inf)",
          e01 < 1e-12 and e02 < 1e-12 and e12 > 1e-3, "(errors %.0e, %.0e; best finite order for r1 r2: %.2f)" % (e01, e02, e12))

    # 7. the sharp wall reflection law in isolation, from the actual Mixmaster equations
    mm = _mixmaster()
    if mm is None:
        if verbose:
            print("  skip check 7: clone of brentharts/wada-coudene (lakes_of_wada.mixmaster) not found (set WADA_REPO)")
        return ok
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(6):
        v0 = np.array([-rng.uniform(0.3, 1.5), -rng.uniform(0.3, 1.5), 0.0])
        v0[2] = -v0[0] * v0[1] / (v0[0] + v0[1])                       # Kasner: T(v) = 0 and v_3 > 0
        x = np.array([-3.0, -3.0, -3.0 - 12.0 * v0[2] / 0.45])          # start far from the wall 3
        v = v0.copy()
        h = 1e-3
        for _ in range(int(60 / h)):
            k1x, k1v = v, mm.force(x)
            k2x, k2v = v + .5 * h * k1v, mm.force(x + .5 * h * k1x)
            k3x, k3v = v + .5 * h * k2v, mm.force(x + .5 * h * k2x)
            k4x, k4v = v + h * k3v, mm.force(x + h * k3x)
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            if v[2] < 0 and abs(mm.potential(x)) < 1e-7:
                break
        worst = max(worst, np.linalg.norm(v - bounce(2, v0)) / np.linalg.norm(v0))
    check("an isolated bounce off wall 3 of the Mixmaster equations changes a Kasner velocity as v_3 -> -v_3, v_j -> v_j + 2 v_3",
          worst < 1e-2, "(worst relative error %.1e)" % worst)
    return ok


# ---------------------------------------------------------------------------------------------------- trend
def trend(ntraj=100, seed=3):
    """For consecutive Kasner epochs of real trajectories: error of the sharp-wall prediction against how far the other two
    walls were from the bounce (largest of the other two x_j at the kappa peak)."""
    mm = _mixmaster()
    if mm is None:
        print("wada-coudene clone not found")
        return
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(ntraj):
        bp, bm = rng.uniform(-0.6, 0.6, 2)
        x, v, V = mm.turnaround_state(np.array([bp]), np.array([bm]))
        if V[0] >= 0:
            continue
        x, v = x[0], v[0]
        best, peak, epochs, state, T = None, None, [], "in", 0.0
        while T < 80:
            hh = 0.002 / (1 + np.abs(v).max())
            k1x, k1v = v, mm.force(x); k2x, k2v = v + .5 * hh * k1v, mm.force(x + .5 * hh * k1x)
            k3x, k3v = v + .5 * hh * k2v, mm.force(x + .5 * hh * k2x); k4x, k4v = v + hh * k3v, mm.force(x + hh * k3x)
            x = x + hh / 6 * (k1x + 2 * k2x + 2 * k3x + k4x); v = v + hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v); T += hh
            kap = abs(mm.potential(x)) / (v ** 2).sum()
            if not np.isfinite(kap) or np.abs(x).max() > 300:
                break
            if state == "in":                      # inside an epoch (kappa small): keep the vector at the minimum of kappa
                if kap < 1e-3 and (best is None or kap < best[0]):
                    best = (kap, v.copy())
                if best is not None and kap > max(300 * best[0], 1e-3):
                    state, peak = "wall", (kap, x.copy())
            else:                                  # in a wall bounce: keep x at the peak of kappa until kappa is small again
                if kap > peak[0]:
                    peak = (kap, x.copy())
                if kap < 1e-3:
                    epochs.append((best, peak[1])); best, state = (kap, v.copy()), "in"
        for (e0, e1) in zip(epochs[:-1], epochs[1:]):
            (ka, a), xp = e0
            (kb, b), _ = e1
            i = int(np.argmax(a))
            err = np.linalg.norm(bounce(i, a) - b) / np.linalg.norm(a)
            rows.append((max(xp[j] for j in range(3) if j != i), err))
    r = np.array(rows)
    print("%d consecutive epoch pairs" % len(r))
    for lo, hi in ((-1e9, -8), (-8, -4), (-4, -2), (-2, 5)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            print("largest other x_j at the bounce in [%g, %g): n = %4d  median error %.3f  90%% %.3f" %
                  (lo, hi, m.sum(), np.median(r[m, 1]), np.percentile(r[m, 1], 90)))


if __name__ == "__main__":
    if "--trend" in sys.argv:
        trend()
    else:
        sys.exit(0 if run_tests(verbose=True) else 1)
