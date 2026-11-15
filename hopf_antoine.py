#!/usr/bin/env python3
"""hopf_antoine.py - chains of round circles (the cores of Antoine's necklace) against Hopf fibres.  numpy + scipy, no Blender.

Antoine's necklace (and, built from it, the Alexander and Antoine horned spheres) is made of chains of linked solid tori:
adjacent tori link once, non-adjacent tori do not, and each torus is replaced by a smaller chain, again and again.  Each
clasp is a Hopf link.  Hopf FIBRES, on the other hand, link in EVERY pair.  This script makes both statements precise and
computes the Moebius invariants of the clasps (the machinery of hopf_oloid.py) to ask whether the clasps are Hopf pairs.

  1. STRAIGHT CHAIN  circles of radius 1 alternately in the planes y = 0 and z = 0 (normals e_y, e_z), centres d apart on
                     the x axis.  Every clasp is exactly the oloid family of hopf_oloid.py: invariants (|1 - d^2/2|, 0).
                     Adjacent circles link for 0 < d < 2; second neighbours (coplanar, 2d apart) are disjoint iff d > 1,
                     and at d = 1 (the OLOID) they are tangent.  A clasp is an exact Hopf pair iff d = sqrt(2).
  2. RING CHAIN      k round circles whose centres are the vertices of a regular k-gon, closed up like Antoine's first
                     stage.  With flat/vertical links (normals N and R) the clasp is NOT a Hopf pair: curvature spoils the
                     straight-chain value.  Twisting the planes about the tangent (normals cos a N + sin a R and
                     cos b R + sin b N) restores it: for each k and sigma there is a twist (a, b), found numerically, for
                     which EVERY clasp is an exact Hopf pair (two Moebius invariants equal), while the linking pattern
                     (neighbours only) is kept.  So no single Hopf fibration contains the chain (fibres of one fibration
                     all link), but every pair of neighbours lies in one (of one handedness or the other).
  3. GENERATIONS     'decorate' replaces a circle by a ring chain along it, scaled by lambda = 2 sin(pi/k) / sigma.  The
                     operator commutes with similarities, so the clasp invariants are the same at every generation: this is
                     the self-similarity of Antoine's construction seen in the Moebius invariants.  Generation 2 is built
                     explicitly and its linking matrix checked.
  4. CONTRAST        k Hopf fibres over a latitude circle of S^2: linking graph = complete graph; chain: cycle graph.

CONVENTIONS
  A circle is (centre, radius, unit normal), the same as hopf_oloid.py.  sigma = (centre spacing)/(link radius) of a straight
  chain, and in a ring chain the ratio chord/link radius, chord = 2 rho sin(pi/k); link planes alternate even/odd:
  even link: normal cos a N + sin a R_j, odd link: normal cos b R_j + sin b N, with N the normal of the ring circle and R_j
  the unit radial vector of the j-th centre.  a = b = 0 is the 'flat/vertical' chain of a physical chain.
  Hopf defect of a pair = theta_2 - theta_1 (the two principal angles of hopf_oloid.principal_angles): 0 iff Hopf pair.

What is exact and what is not:
  - clasp invariants, the d-family formula, the twist solution (two equations in two unknowns, residual < 1e-12) and the
    equality of the invariants at every generation are exact up to floating point.
  - linking numbers are Gauss integrals of polygons, accurate to a few per cent, used only to tell 0 from +-1.
  - 'No single Hopf fibration contains three consecutive circles' follows from lk(i-1, i+1) = 0 against the fact that two
    fibres of one fibration link.
  - NOT done here: thickening the circles to solid tori, the limit Cantor set, and the non-triviality of the fundamental group
    of its complement (Antoine's theorem; see R. Schwartz's notes on Antoine's necklace).  What is checked is that the
    generation-2 circles realise the nesting pattern: every child circle links only its two sibling neighbours.
  - observation, not proved: at sigma = sqrt 2 the Hopf clasp has Moebius invariant s1 = s2 = tan(pi/k) (checked for k =
    8, 12, 16, 24, 32 to 1e-8), i.e. base angle 2 arccos tan(pi/k), tending to the antipodal 180 deg of the straight chain.
  - the twist values themselves are one point on a 1-parameter family (the two equations leave one free parameter); the
    minimal-norm branch near (-pi/4, pi/4) is returned.  I have not checked whether the family has other components.

    python3 hopf_antoine.py --tests          runs the checks
    python3 hopf_antoine.py                  prints the summary
    python3 hopf_antoine.py --png out.png    draws the figure
"""
import math
import sys

import numpy as np

import hopf_math as H
import hopf_oloid as O

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------------------------------- circles and linking

def polygon(circ, m=48):
    """Polygon of m points on the circle (centre, radius, normal)."""
    c, r, n = circ[:3]
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    th = np.linspace(0.0, 2 * math.pi, m, endpoint=False)
    return np.asarray(c, float) + r * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b))


def linking_matrix(circs, m=48):
    """Gauss linking numbers of all pairs (float matrix).  Two circles can only link if |c_i - c_j| < r_i + r_j, so the
    other pairs are set to 0 without computing."""
    n = len(circs)
    cen = np.array([c[0] for c in circs], float)
    rad = np.array([c[1] for c in circs], float)
    P = [polygon(c, m) for c in circs]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(cen[i] - cen[j]) < rad[i] + rad[j]:
                M[i, j] = M[j, i] = H.linking_number(P[i], P[j])
    return M


def linked_pairs(M, tol=0.5):
    """Set of pairs (i, j), i < j, with |lk| > tol."""
    iu, ju = np.where(np.triu(np.abs(M) > tol, 1))
    return set(zip(iu.tolist(), ju.tolist()))


def cycle_pairs(k):
    return {(min(i, (i + 1) % k), max(i, (i + 1) % k)) for i in range(k)}


def path_pairs(n):
    return {(i, i + 1) for i in range(n - 1)}


def min_distance_circles(c1, c2, m=360):
    """Smallest distance between the two sampled polygons (an upper bound for the true distance of the circles)."""
    from scipy.spatial.distance import cdist
    return float(cdist(polygon(c1, m), polygon(c2, m)).min())


# ---------------------------------------------------------------------------------------------------- chains

def straight_chain(n, d):
    """n circles of radius 1, centres (j d, 0, 0), normals alternately e_z (xy plane) and e_y (xz plane)."""
    return [(np.array([j * d, 0.0, 0.0]), 1.0, np.array([0.0, 0.0, 1.0] if j % 2 == 0 else [0.0, 1.0, 0.0])) for j in range(n)]


def decorate(circ, k, sigma, a=0.0, b=0.0):
    """Replace the circle (c, r, N) by a ring chain of k links along it: centres on the circle, link radius
    r' = r 2 sin(pi/k) / sigma (so that chord / link radius = sigma), alternate links tilted by a and b (see CONVENTIONS)."""
    c, r, N = circ[:3]
    N = np.asarray(N, float) / np.linalg.norm(N)
    e1 = np.cross(N, [0.31, 0.52, 0.79])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(N, e1)
    rp = r * 2 * math.sin(math.pi / k) / sigma
    out = []
    for j in range(k):
        ph = 2 * math.pi * j / k
        R = math.cos(ph) * e1 + math.sin(ph) * e2
        n = (math.cos(a) * N + math.sin(a) * R) if j % 2 == 0 else (math.cos(b) * R + math.sin(b) * N)
        out.append((np.asarray(c, float) + r * R, rp, n / np.linalg.norm(n)))
    return out


UNIT_RING = (np.zeros(3), 1.0, np.array([0.0, 0.0, 1.0]))


def ring_chain(k, sigma, a=0.0, b=0.0):
    """Generation 1: a ring chain of k links (k even) along the unit circle in the xy plane."""
    return decorate(UNIT_RING, k, sigma, a, b)


def hopf_defect(c1, c2):
    """theta_2 - theta_1 (radians) of the pair, nan if an invariant is >= 1.  0 iff the pair is a Hopf pair."""
    s = O.moebius_invariants(c1, c2)
    if s[0] >= 1 - 1e-12:
        return float("nan")
    return math.acos(min(s[1], 1.0)) - math.acos(s[0])


def solve_twist(k, sigma=SQRT2):
    """The twist (a, b) for which the clasp of ring_chain(k, sigma, a, b) is an exact Hopf pair: Moebius invariants
    s1 = s2, i.e. G = M M^T with M the Lorentz Gram matrix of the two circle planes satisfies G01 = 0, G00 = G11.
    Returns (a, b) of smallest norm found from a few starting points, or None."""
    import warnings
    from scipy.optimize import fsolve

    def eqs(p):
        ch = ring_chain(k, sigma, p[0], p[1])
        W1 = O.circle_frame(*ch[0])
        W2 = O.circle_frame(*ch[1])
        M = W1 @ O.ETA @ W2.T
        G = M @ M.T
        return [G[0, 1], G[0, 0] - G[1, 1]]

    best = None
    for g in [(-math.pi / 4, math.pi / 4), (math.pi / 4, -math.pi / 4), (0, 0), (0.3, -0.3), (-0.3, 0.3), (0.8, -0.7), (-0.8, 0.7)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = fsolve(eqs, g, xtol=1e-14)
        if np.abs(eqs(sol)).max() < 1e-10 and (best is None or sol @ sol < best @ best):
            best = sol
    return None if best is None else (float(best[0]), float(best[1]))


def generations(levels, k, sigma, a, b):
    """list of generations; generation g is a list of circles, plus parent index arrays."""
    gens = [ring_chain(k, sigma, a, b)]
    parents = [np.zeros(k, int)]
    for _ in range(levels - 1):
        nxt, par = [], []
        for idx, circ in enumerate(gens[-1]):
            nxt += decorate(circ, k, sigma, a, b)
            par += [idx] * k
        gens.append(nxt)
        parents.append(np.array(par))
    return gens, parents


def fibre_circles(k, elevation=1.2, n=400):
    """k Hopf fibres over equally spaced points of a latitude circle of S^2: list of (circle data, polygon)."""
    out = []
    for j in range(k):
        P = H.fibre_r3(H.base_point(elevation, 2 * math.pi * j / k), n)
        c, r, nrm, _ = H.fit_circle(P)
        out.append(((c, r, nrm), P))
    return out


def best_untwisted_defect(k, sigmas=np.linspace(1.05, 1.9, 35)):
    """Smallest Hopf defect (radians) of the flat/vertical ring chain over a grid of sigma, with its sigma."""
    best = (float("inf"), None)
    for s in sigmas:
        ch = ring_chain(k, s)
        d = hopf_defect(ch[0], ch[1])
        if not math.isnan(d) and d < best[0]:
            best = (d, s)
    return best


# ---------------------------------------------------------------------------------------------------- tests

def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # --- 1. straight chain = the oloid family
    for d in (1.2, SQRT2, 1.8):
        ch = straight_chain(8, d)
        M = linking_matrix(ch)
        check("straight chain d = %.4f: only neighbours link (path graph)" % d, linked_pairs(M) == path_pairs(8))
    worst = 0.0
    for d in (1.1, 1.3, SQRT2, 1.7, 1.95):
        ch = straight_chain(6, d)
        for i in range(5):
            worst = max(worst, np.abs(O.moebius_invariants(ch[i], ch[i + 1]) - O.d_family_invariants(d)).max())
    check("every clasp of a straight chain has invariants (|1 - d^2/2|, 0)", worst < 1e-12, "(max err %.1e)" % worst)
    ch = straight_chain(4, 1.0)
    md = min_distance_circles(ch[0], ch[2])
    check("d = 1 (the oloid): second neighbours are tangent", md < 1e-9, "(distance %.1e)" % md)
    ch = straight_chain(4, 0.9)
    ch = straight_chain(4, 0.9)
    check("d < 1: second neighbours (coplanar, centres 2d < 2 apart) intersect", min_distance_circles(ch[0], ch[2], 3600) < 0.01)
    check("d = 1.2: second neighbours are disjoint", min_distance_circles(*[straight_chain(4, 1.2)[i] for i in (0, 2)]) > 0.1)
    M2 = linking_matrix(straight_chain(4, 2.05))
    check("d > 2: neighbours do not link", len(linked_pairs(M2)) == 0)
    a_, b_ = straight_chain(2, SQRT2)
    sep = O.separations(O.principal_angles(a_, b_))
    check("d = sqrt(2): the clasp is a Hopf pair over antipodal base points",
          abs(sep[0] - math.pi) < 1e-7 and sep[1] < 1e-7, "(separations %.3f, %.3f deg)" % tuple(np.degrees(sep)))

    # --- 2. ring chains
    for k in (8, 16):
        ch = ring_chain(k, 1.5)
        M = linking_matrix(ch)
        check("flat/vertical ring chain k = %d: linking graph is the %d-cycle" % (k, k), linked_pairs(M) == cycle_pairs(k))
    d8, d16, d32 = (best_untwisted_defect(k)[0] for k in (8, 16, 32))
    check("flat/vertical ring chain is NOT a Hopf chain: best defect over sigma is > 5 deg and falls like 1/k",
          math.degrees(d16) > 5 and d8 > d16 > d32 > 0,
          "(k = 8, 16, 32: %.2f, %.2f, %.2f deg)" % (math.degrees(d8), math.degrees(d16), math.degrees(d32)))
    for k in (8, 16, 24):
        tw = solve_twist(k)
        ch = ring_chain(k, SQRT2, *tw)
        M = linking_matrix(ch)
        invs = np.array([O.moebius_invariants(ch[i], ch[(i + 1) % k]) for i in range(k)])
        defect = max(abs(hopf_defect(ch[i], ch[(i + 1) % k])) for i in range(k))
        check("twisted ring chain k = %2d (a = %.4f, b = %.4f): linking graph is the cycle, every clasp is a Hopf pair" %
              (k, *tw), linked_pairs(M) == cycle_pairs(k) and defect < 1e-7 and np.ptp(invs, axis=0).max() < 1e-9,
              "(s = %.5f, base angle %.2f deg)" % (invs[0, 0], 2 * math.degrees(math.acos(invs[0, 0]))))

    worst = 0.0
    for k in (8, 12, 16, 24, 32):
        ch = ring_chain(k, SQRT2, *solve_twist(k))
        worst = max(worst, abs(O.moebius_invariants(ch[0], ch[1])[0] - math.tan(math.pi / k)))
    check("observation: at sigma = sqrt 2 the Hopf clasp has s = tan(pi/k), base angle 2 arccos tan(pi/k) -> 180 deg",
          worst < 1e-8, "(max err %.1e)" % worst)

    # --- one explicit clasp through the Moebius model
    k = 24
    tw = solve_twist(k)
    ch = ring_chain(k, SQRT2, *tw)
    model = O.great_circle_model(ch[0], ch[1])
    s_ = O.moebius_invariants(ch[0], ch[1])
    alpha = 2 * math.acos(s_[0])
    sep = O.plane_separations(model["P1"], model["P2"])
    check("explicit Moebius model of a clasp: great circles with separations (base angle, 0)",
          abs(sep[0] - alpha) < 1e-7 and sep[1] < 1e-7, "(%.3f, %.3f deg)" % tuple(np.degrees(sep)))
    ang = O.plane_principal_angles(model["P1"], model["P2"])
    check("... and they are Clifford parallel (equal principal angles)", abs(ang[0] - ang[1]) < 1e-7,
          "(%.3f, %.3f deg)" % tuple(np.degrees(ang)))

    # --- no single Hopf fibration contains three consecutive circles
    M = linking_matrix(ch)
    check("circles i-1 and i+1 do not link, although both are Hopf-paired with i",
          abs(M[0, 2]) < 0.5 and abs(M[0, 1]) > 0.5 and abs(M[1, 2]) > 0.5,
          "(lk = %.2f, %.2f, %.2f)" % (M[0, 1], M[1, 2], M[0, 2]))
    s02 = O.moebius_invariants(ch[0], ch[2])
    check("... and the pair (i-1, i+1) is not a Hopf pair (its invariants differ)", abs(s02[0] - s02[1]) > 1e-3,
          "(invariants %.3f, %.3f)" % tuple(s02))

    # --- 3. generations
    gens, parents = generations(2, k, SQRT2, *tw)
    g1, g2 = gens
    check("generation 2 has k^2 circles and radius ratio lambda = 2 sin(pi/k)/sigma",
          len(g2) == k * k and abs(g2[0][1] / g1[0][1] - 2 * math.sin(math.pi / k) / SQRT2) < 1e-12,
          "(lambda = %.5f)" % (g2[0][1] / g1[0][1]))
    worst = 0.0
    for i in range(0, len(g2), 7):
        j = i + 1 if (i % k) < k - 1 else i - (k - 1)
        worst = max(worst, np.abs(O.moebius_invariants(g2[i], g2[j]) - O.moebius_invariants(g1[0], g1[1])).max())
    check("every clasp at generation 2 has the SAME Moebius invariants as generation 1 (self-similarity)",
          worst < 1e-9, "(max diff %.1e)" % worst)
    M2 = linking_matrix(g2, m=72)
    expect = set()
    for p in range(k):
        expect |= {(p * k + i, p * k + j) for (i, j) in cycle_pairs(k)}
    got = linked_pairs(M2)
    check("generation 2: each child links exactly its two sibling neighbours (%d necklaces of %d), nothing else" % (k, k),
          got == expect, "(%d linked pairs, expected %d)" % (len(got), len(expect)))
    cen = np.array([c[0] for c in g2])
    rad = np.array([c[1] for c in g2])
    from scipy.spatial.distance import cdist
    D = cdist(cen, cen)
    near = np.triu(D < rad[:, None] + rad[None, :] + 0.2, 1) & (parents[1][:, None] != parents[1][None, :])
    md = min(min_distance_circles(g2[i], g2[j], 240) for i, j in zip(*np.where(near)))
    check("children of different parents are disjoint circles", md > 1e-3, "(smallest distance %.4f, link radius %.3f)" % (md, rad[0]))

    # --- 4. contrast with Hopf fibres
    fc = fibre_circles(6)
    Mf = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            Mf[i, j] = Mf[j, i] = H.linking_number(fc[i][1][::2], fc[j][1][::2])
    check("6 Hopf fibres: every pair links once (complete graph K6)", len(linked_pairs(Mf)) == 15 and
          np.abs(np.abs(Mf[np.triu_indices(6, 1)]) - 1).max() < 0.1, "(15 of 15 pairs, |lk| in [%.2f, %.2f])" %
          (np.abs(Mf[np.triu_indices(6, 1)]).min(), np.abs(Mf[np.triu_indices(6, 1)]).max()))
    worst = 0.0
    for i in range(6):
        for j in range(i + 1, 6):
            s = O.moebius_invariants(fc[i][0], fc[j][0])
            worst = max(worst, abs(s[0] - s[1]))
    check("... and every pair of fibres is a Hopf pair (equal invariants)", worst < 1e-9, "(max |s1 - s2| %.1e)" % worst)
    return ok


# ---------------------------------------------------------------------------------------------------- report and figure

def summary():
    print("straight chain, centre spacing d (radius 1): invariants (|1 - d^2/2|, 0); Hopf pair iff d = sqrt 2; oloid d = 1:")
    for d in (1.0, 1.2, SQRT2, 1.8):
        a_, b_ = straight_chain(2, d)
        t = O.principal_angles(a_, b_)
        print("   d = %.4f  principal angles (%.2f, %.2f) deg  Hopf defect %.2f deg" %
              (d, *np.degrees(t), math.degrees(t[1] - t[0])))
    print("ring chain, sigma = sqrt 2:  k   best untwisted defect (deg, at sigma)   twist (a, b)   clasp s   base angle")
    for k in (8, 16, 24, 32):
        dd, sg = best_untwisted_defect(k)
        tw = solve_twist(k)
        ch = ring_chain(k, SQRT2, *tw)
        s = O.moebius_invariants(ch[0], ch[1])
        print("                              %2d   %6.2f (sigma %.2f)                  (%+.4f, %+.4f)   %.5f   %.2f deg" %
              (k, math.degrees(dd), sg, *tw, s[0], 2 * math.degrees(math.acos(s[0]))))
    k = 24
    tw = solve_twist(k)
    gens, _ = generations(2, k, SQRT2, *tw)
    M1 = linking_matrix(gens[0])
    print("k = %d: generation 1 has %d linked pairs (cycle), generation 2 has %d circles of radius %.4f x generation 1" %
          (k, len(linked_pairs(M1)), len(gens[1]), gens[1][0][1] / gens[0][0][1]))
    s1 = O.moebius_invariants(gens[0][0], gens[0][1])
    s2 = O.moebius_invariants(gens[1][0], gens[1][1])
    print("   clasp invariants: generation 1 (%.6f, %.6f), generation 2 (%.6f, %.6f)" % (*s1, *s2))
    print("6 Hopf fibres on a latitude circle: all 15 pairs link; the chain links only neighbours")


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(13, 10))

    def draw(ax, circs, colors, lw=1.4, alpha=1.0):
        for c, col in zip(circs, colors):
            P = polygon(c, 90)
            P = np.vstack([P, P[:1]])
            ax.plot(*P.T, color=col, lw=lw, alpha=alpha)

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ch = straight_chain(7, SQRT2)
    draw(ax, ch, [cm.viridis(j / 6) for j in range(7)], lw=1.8)
    ax.set_box_aspect((3.2, 1, 1))
    ax.set_title("straight chain, d = sqrt 2: every clasp is an exact Hopf pair\n(over antipodal base points); oloid = d = 1 (tangent)")
    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    k = 24
    tw = solve_twist(k)
    gens, par = generations(2, k, SQRT2, *tw)
    draw(ax, gens[0], ["0.75"] * k, lw=1.0, alpha=0.8)
    for p, col in ((0, "tab:red"), (1, "tab:blue")):
        draw(ax, [c for c, q in zip(gens[1], par[1]) if q == p], [col] * k, lw=1.0)
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_title("Hopf ring chain (k = 24, twist %.3f, %.3f) and the children\nof two neighbouring links: same clasp invariants" % tw)
    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 3)
    for off, kk, title, edges in ((0.0, 8, "chain: cycle graph", "cycle"), (3.0, 8, "Hopf fibres: complete graph", "all")):
        th = 2 * math.pi * np.arange(kk) / kk
        xs, ys = off + np.cos(th), np.sin(th)
        pairs = cycle_pairs(kk) if edges == "cycle" else {(i, j) for i in range(kk) for j in range(i + 1, kk)}
        for i, j in pairs:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.5", lw=1)
        ax.scatter(xs, ys, s=40, color="tab:red" if edges == "all" else "tab:blue", zorder=3)
        ax.text(off, -1.35, title, ha="center")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("who links whom (8 circles)")

    ax = fig.add_subplot(2, 2, 4)
    ks = [8, 12, 16, 24, 32, 48]
    ax.plot(ks, [math.degrees(best_untwisted_defect(kk)[0]) for kk in ks], "o-", color="tab:orange", label="flat/vertical links (best sigma)")
    tw_def = []
    for kk in ks:
        chk = ring_chain(kk, SQRT2, *solve_twist(kk))
        tw_def.append(math.degrees(abs(hopf_defect(chk[0], chk[1]))))
    ax.plot(ks, tw_def, "s-", color="tab:green", label="twisted links (solved, computed)")
    ax.set_xlabel("links k in the ring")
    ax.set_ylabel("Hopf defect of a clasp (degrees)")
    ax.set_title("how far a clasp is from a Hopf pair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure(sys.argv[sys.argv.index("--png") + 1])
    elif "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    else:
        summary()
        run_tests()
