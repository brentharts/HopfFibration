#!/usr/bin/env python3
"""hopf_figures.py - four new matplotlib figures for the Hopf fibration (numpy + matplotlib only, no Blender).

Convention: exactly hopf_math.py's.  A point of S^3 is (z1, z2) in C^2, the Hopf map is
h = (2 Re z1 conj z2, 2 Im z1 conj z2, |z1|^2 - |z2|^2), a fibre is e^{it}(z1, z2), stereographic projection is from (0,0,0,1),
and base points are (elevation, azimuth) as in hopf_math.base_point (elevation 0 = SOUTH pole here).

Labels used in the docs and tests: Exact (identity checked to rounding), Counted (numerical, finite sampling).

  fig 1  linking atlas       Gauss linking number of every pair of fibres on a grid of base points (all the same sign, magnitude 1),
                             and the S^3 distance between fibres against half the base angle (Clifford parallels).
  fig 2  Poincare section    the half-plane {y = 0, x > 0} of R^3 is exactly {Im z1 = 0, Re z1 > 0}: every fibre crosses it ONCE, so the
                             half-plane is a copy of S^2 minus a point.  Constant-elevation tori meet it in round circles, centre
                             (1/sin(el/2), 0), radius cot(el/2): all in closed form.
  fig 3  audit of makeFibre  centre distance d and radius r of the circles fibration.py draws, against the exact tan and sec, and the defect
                             r^2 - d^2 - 1 (zero for a true Hopf fibre).
  fig 4  Hopf invariant      the map f_{p,q}(z1, z2) = hopf(z1^p, z2^q normalised) : S^3 -> S^2 has Hopf invariant p q.  The preimage of a
                             point is a (q,p) torus knot; two preimages link p q times.  Computed here by the Gauss integral.

    python3 hopf_figures.py --tests
    python3 hopf_figures.py                          writes figures/hopf_fig1..4 .png
    python3 hopf_figures.py --png-dir out --only 2,4
"""
import math
import os
import sys

import numpy as np

import hopf_math as H

TWO_PI = 2.0 * math.pi


# =========================================================================================================== fig 1: linking atlas

def fibre_grid(n_el=4, n_az=6, el_range=(0.9, 2.5)):
    """(elevation, azimuth) pairs away from the poles (the pole fibres are straight lines through infinity)."""
    els = np.linspace(el_range[0], el_range[1], n_el)
    azs = TWO_PI * np.arange(n_az) / n_az
    return [(float(e), float(a)) for e in els for a in azs]


def s3_fibre_distance(F1, F2):
    """Sampled distance on S^3 between two fibres (arrays (n,2) complex): arccos of the largest real inner product in R^4."""
    U = H.to_r4(F1)
    V = H.to_r4(F2)
    return float(math.acos(min(1.0, (U @ V.T).max())))


def linking_atlas(n_el=4, n_az=6, npts=240):
    """Returns dict with the base points, the matrix of Gauss linking numbers L (nan on the diagonal), the base angles, and the
    sampled S^3 distances between fibres."""
    pts = fibre_grid(n_el, n_az)
    bases = [H.base_point(e, a) for e, a in pts]
    fib3 = [H.fibre_s3(b, npts) for b in bases]
    fibr = [H.stereographic(F) for F in fib3]
    m = len(pts)
    L = np.full((m, m), np.nan)
    ang = np.full((m, m), np.nan)
    d3 = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1, m):
            L[i, j] = L[j, i] = H.linking_number(fibr[i], fibr[j])
            ang[i, j] = ang[j, i] = math.acos(float(np.clip(bases[i] @ bases[j], -1, 1)))
            d3[i, j] = d3[j, i] = s3_fibre_distance(fib3[i], fib3[j])
    return dict(pts=pts, L=L, ang=ang, d3=d3, n_el=n_el, n_az=n_az)


def fig_linking_atlas(path, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = linking_atlas(**kw)
    L, ang, d3 = A["L"], A["ang"], A["d3"]
    off = ~np.isnan(L)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1.05, 1]))
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("0.92")
    im = a1.imshow(L, cmap=cm, vmin=-1.5, vmax=1.5, origin="lower")
    plt.colorbar(im, ax=a1, fraction=0.046, label="Gauss linking number of fibre i and fibre j")
    n_az = A["n_az"]
    ticks = np.arange(0, len(A["pts"]), n_az)
    a1.set_xticks(ticks)
    a1.set_yticks(ticks)
    a1.set_xticklabels(["el=%.2f" % A["pts"][t][0] for t in ticks], rotation=45, fontsize=7)
    a1.set_yticklabels(["el=%.2f" % A["pts"][t][0] for t in ticks], fontsize=7)
    a1.set_title("every pair of fibres links exactly once\n(%d fibres, %d pairs; %d azimuths per elevation block)"
                 % (len(A["pts"]), off.sum() // 2, n_az), fontsize=9)
    iu = np.triu_indices_from(L, 1)
    lk = L[iu]
    a1.text(0.02, 0.02, "lk range  [%.3f, %.3f]" % (lk.min(), lk.max()), transform=a1.transAxes, fontsize=8,
            bbox=dict(fc="w", ec="0.6", alpha=0.9))
    x = ang[iu] / 2
    y = d3[iu]
    a2.scatter(x, y, s=10, c=np.abs(lk), cmap="viridis", vmin=0.9, vmax=1.1)
    xx = np.linspace(0, x.max(), 2)
    a2.plot(xx, xx, "k--", lw=1, label="distance = half the base angle")
    a2.set_xlabel("half the angle between the two base points on S$^2$ (rad)")
    a2.set_ylabel("distance between the two fibres in S$^3$ (rad)")
    a2.set_title("Clifford parallels: the fibres are equidistant\nmax |difference| = %.1e (sampled)" % np.abs(x - y).max(), fontsize=9)
    a2.legend(loc="upper left", fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)
    return A


# ====================================================================================================== fig 2: Poincare section

def section_halfplane(el, az):
    """Closed form: where the canonical section s(b) = (sin(el/2), cos(el/2) e^{-i az}) of the fibre over base_point(el, az) lands
    in the half-plane {y = 0, x > 0} of R^3, as (rho, z).  Exact."""
    s, c = np.sin(el / 2), np.cos(el / 2)
    den = 1.0 + c * np.sin(az)
    return s / den, c * np.cos(az) / den


def torus_meridian(el):
    """The torus swept by the fibres of elevation el meets the half-plane in the round circle with this centre and radius."""
    s, c = math.sin(el / 2), math.cos(el / 2)
    return (1.0 / s, 0.0), c / s


def hsv_colour(el, az):
    import matplotlib.colors as mc
    h = (np.asarray(az) % TWO_PI) / TWO_PI
    v = 0.30 + 0.70 * np.asarray(el) / math.pi
    return mc.hsv_to_rgb(np.stack([h, 0.85 * np.ones_like(h), v], axis=-1))


def fig_poincare_section(path, n_el=220, n_az=360):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    el_e = np.linspace(0.02, math.pi - 0.005, n_el + 1)           # cell edges
    az_e = np.linspace(0, TWO_PI, n_az + 1)
    el_c = 0.5 * (el_e[1:] + el_e[:-1])
    az_c = 0.5 * (az_e[1:] + az_e[:-1])
    EL, AZ = np.meshgrid(el_c, az_c, indexing="ij")
    rgb = hsv_colour(EL, AZ)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a1.imshow(rgb, origin="lower", extent=[0, TWO_PI, el_e[0], el_e[-1]], aspect="auto")
    a1.set_xlabel("azimuth of the base point")
    a1.set_ylabel("elevation of the base point (0 = south pole, $\\pi$ = north pole)")
    a1.set_title("S$^2$ (base space), coloured by position", fontsize=9)
    EE, AA = np.meshgrid(el_e, az_e, indexing="ij")               # (n_el+1, n_az+1) corners
    R_, Z_ = section_halfplane(EE, AA)
    V = np.stack([np.stack([R_[:-1, :-1], Z_[:-1, :-1]], -1), np.stack([R_[1:, :-1], Z_[1:, :-1]], -1),
                  np.stack([R_[1:, 1:], Z_[1:, 1:]], -1), np.stack([R_[:-1, 1:], Z_[:-1, 1:]], -1)], axis=2).reshape(-1, 4, 2)
    good = np.isfinite(V).all(axis=(1, 2)) & (np.abs(V).max(axis=(1, 2)) < 12)
    a2.add_collection(PolyCollection(V[good], facecolors=rgb.reshape(-1, 3)[good], edgecolors="face", linewidths=0.3,
                                     rasterized=True))
    th = np.linspace(0, TWO_PI, 400)
    for e in (0.7, 1.1, 1.6, 2.1, 2.6):
        (cx, cz), r = torus_meridian(e)
        a2.plot(cx + r * np.cos(th), cz + r * np.sin(th), color="k", lw=0.9)
        a2.annotate("el=%.1f" % e, (cx + r * 0.72, cz + r * 0.72), fontsize=7)
        a1.axhline(e, color="k", lw=0.7)
    for a in np.arange(0, TWO_PI, TWO_PI / 8):
        e = np.linspace(0.05, math.pi - 0.02, 300)
        r_, z_ = section_halfplane(e, a)
        keep = np.hypot(r_, z_) < 12
        a2.plot(r_[keep], z_[keep], color="w", lw=0.6, alpha=0.8)
        a1.axvline(a, color="w", lw=0.6, alpha=0.8)
    a2.axvline(0, color="r", lw=2)
    a2.text(0.12, -2.75, "red line: the fibre over the south pole (the z axis)", color="r", fontsize=8,
            bbox=dict(fc="w", ec="r", alpha=0.9))
    a2.set_xlim(-0.1, 6)
    a2.set_ylim(-3, 3)
    a2.set_aspect("equal")
    a2.set_xlabel("$\\rho = \\sqrt{x^2 + y^2}$")
    a2.set_ylabel("z")
    a2.set_title("half-plane {y = 0, x > 0} of R$^3$: each fibre crosses it once\n"
                 "black: tori of constant elevation = round circles; white: constant azimuth", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


# ===================================================================================================== fig 3: audit of makeFibre

def audit_table(els):
    rows = []
    for el in els:
        Ps, _, _ = H.blender_fibre(el, 0.0, 720)
        Pe, _, _ = H.blender_fibre(el, 0.0, 720, exact=True)
        ds, rs, ts = H.circle_params(Ps)
        de, re_, te = H.circle_params(Pe)
        rows.append(dict(el=el, shipped=(ds, rs, ts), fixed=(de, re_, te)))
    return rows


def fig_audit(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    els = np.linspace(0.1, 3.0, 30)
    rows = audit_table(els)
    ds = np.array([r["shipped"][0] for r in rows])
    rs = np.array([r["shipped"][1] for r in rows])
    ts = np.array([r["shipped"][2] for r in rows])
    fine = np.linspace(0.05, 3.05, 300)
    fig = plt.figure(figsize=(12, 7.5))
    a1 = fig.add_subplot(2, 3, 1)
    a1.semilogy(fine, np.tan(fine / 2), "k-", label="exact  tan(el/2)")
    a1.semilogy(els, ds, "o", ms=4, color="C3", label="makeFibre baseRad (measured)")
    a1.set_title("distance of the circle's centre from the z axis", fontsize=9)
    a1.set_xlabel("elevation passed to makeFibre")
    a1.legend(fontsize=7)
    a2 = fig.add_subplot(2, 3, 2)
    a2.semilogy(fine, 1 / np.cos(fine / 2), "k-", label="exact  sec(el/2)")
    a2.semilogy(els, rs, "o", ms=4, color="C3", label="makeFibre fibreRad (measured)")
    a2.set_title("radius of the circle", fontsize=9)
    a2.set_xlabel("elevation passed to makeFibre")
    a2.legend(fontsize=7)
    a3 = fig.add_subplot(2, 3, 3)
    a3.plot(els, rs ** 2 - ds ** 2 - 1, "o-", ms=4, color="C3", label="shipped")
    a3.plot(els, [r["fixed"][1] ** 2 - r["fixed"][0] ** 2 - 1 for r in rows], "s-", ms=4, color="C2", label="baseRad = tan, fibreRad = sec")
    a3.axhline(0, color="k", lw=0.6)
    a3.set_title("defect  r$^2$ - d$^2$ - 1   (0 for a Hopf fibre)", fontsize=9)
    a3.set_xlabel("elevation passed to makeFibre")
    a3.legend(fontsize=7)
    a4 = fig.add_subplot(2, 3, 4)
    a4.plot(els, ts, "o", ms=4, color="C3", label="tilt of the shipped plane")
    a4.plot(fine, fine / 2, "k-", label="exact tilt = el/2")
    a4.set_title("tilt of the circle's plane (rad)", fontsize=9)
    a4.set_xlabel("elevation passed to makeFibre")
    a4.legend(fontsize=7)
    a5 = fig.add_subplot(2, 3, 5)
    a5.plot(els, np.abs(rs / (1 / np.cos(els / 2)) - 1) * 100, "o-", ms=4, color="C3", label="radius")
    a5.plot(els, np.abs(ds / np.tan(els / 2) - 1) * 100, "s-", ms=4, color="C0", label="centre distance")
    a5.set_yscale("log")
    a5.set_title("relative error of the shipped circle (%)", fontsize=9)
    a5.set_xlabel("elevation passed to makeFibre")
    a5.legend(fontsize=7)
    a6 = fig.add_subplot(2, 3, 6)
    for el, col in ((0.8, "C0"), (1.8, "C1"), (2.6, "C4")):
        e = el / 2                                           # the circle's plane is spanned by y and (cos e, 0, -sin e): draw it face on
        for P, ls, lab in ((H.blender_fibre(el, 0.0, 400)[0], "--", None),
                           (H.blender_fibre(el, 0.0, 400, exact=True)[0], "-", "el = %.1f" % el)):
            a6.plot(P[:, 1], P[:, 0] * math.cos(e) - P[:, 2] * math.sin(e), ls, color=col, lw=1.2, label=lab)
    a6.set_aspect("equal")
    a6.set_title("circles seen face-on in their own plane (azimuth 0)\ndashed: shipped, solid: exact", fontsize=9)
    a6.set_xlabel("y")
    a6.set_ylabel("in-plane axis (cos(el/2) x - sin(el/2) z)")
    a6.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


# ===================================================================================== fig 4: Hopf invariant of f_{p,q}

def compose_map(p, q, z1, z2):
    """f_{p,q}(z1, z2) = hopf(z1^p, z2^q) / normalisation: S^3 -> S^2.  Its Hopf invariant is p q."""
    a, b = z1 ** p, z2 ** q
    n = math.sqrt(abs(a) ** 2 + abs(b) ** 2) if np.isscalar(a) else np.sqrt(abs(a) ** 2 + abs(b) ** 2)
    return H.hopf(a / n, b / n)


def preimage_curve(p, q, el, az, n=1500):
    """The preimage of the base point base_point(el, az) under f_{p,q}, as n points of S^3 (shape (n, 2) complex).

    z1 = r1 e^{i q phi}, z2 = r2 e^{i (p phi - az/q)}, with r1^p / r2^q = tan(el/2) and r1^2 + r2^2 = 1.
    For coprime p, q this is ONE closed curve, the (q, p) torus knot on the torus |z1| = r1.  Exact."""
    t = math.tan(el / 2)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid ** p / (1 - mid * mid) ** (q / 2.0) < t:
            lo = mid
        else:
            hi = mid
    r1 = 0.5 * (lo + hi)
    r2 = math.sqrt(1 - r1 * r1)
    phi = np.linspace(0, TWO_PI, n, endpoint=False)
    return np.stack([r1 * np.exp(1j * q * phi), r2 * np.exp(1j * (p * phi - az / q))], axis=1)


DEFAULT_PAIR = ((1.4, 0.3), (2.0, 2.5))


def hopf_invariant_by_linking(p, q, pair=DEFAULT_PAIR, n=1500):
    """Gauss linking number of the preimages of two base points under f_{p,q}.  Counted (a numerical integral); equals +-p q."""
    A = H.stereographic(preimage_curve(p, q, *pair[0], n=n))
    B = H.stereographic(preimage_curve(p, q, *pair[1], n=n))
    return H.linking_number(A, B), A, B


def fig_hopf_invariant(path, cases=((1, 1), (2, 1), (3, 1), (2, 3))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11.5, 8.2))
    for i, (p, q) in enumerate(cases):
        lk, A, B = hopf_invariant_by_linking(p, q)
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.plot(*A.T, color="C0", lw=1.6, label="preimage of b$_1$")
        ax.plot(*B.T, color="C3", lw=1.6, label="preimage of b$_2$")
        ax.set_title("$f(z_1,z_2) = $ Hopf$(z_1^{%d}, z_2^{%d})$:  linking = %+.3f,  p q = %d" % (p, q, lk, p * q), fontsize=9)
        ax.set_box_aspect((1, 1, 0.8))
        ax.legend(fontsize=7, loc="upper left")
        pts = np.vstack([A, B])
        r = np.abs(pts).max() * 1.05
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r * 0.8, r * 0.8)
    fig.suptitle("Pontryagin: the Hopf invariant is the linking number of two preimage curves "
                 "(preimages are (q,p) torus knots, stereographic projection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


# ================================================================================================================ tests

def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # fig 1 (reduced size to keep the test fast)
    A = linking_atlas(n_el=3, n_az=4, npts=200)
    iu = np.triu_indices_from(A["L"], 1)
    lk = A["L"][iu]
    check("fig1: every pair of fibres has linking number of the same sign and magnitude 1 (to 5 %)",
          np.all(np.abs(np.abs(lk) - 1) < 0.05) and len(set(np.sign(lk))) == 1, "(range [%.3f, %.3f])" % (lk.min(), lk.max()))
    dev = np.abs(A["d3"][iu] - A["ang"][iu] / 2).max()
    check("fig1: distance between fibres in S^3 = half the base angle (Clifford parallels)", dev < 2e-3, "(max dev %.1e)" % dev)

    # fig 2: closed forms against the numerical stereographic image of the section, one crossing per fibre, round circles
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40):
        el, az = rng.uniform(0.2, 2.9), rng.uniform(0, TWO_PI)
        z1, z2 = H.section(H.base_point(el, az))
        P = H.stereographic(np.array([[z1, z2]]))[0]
        r_, z_ = section_halfplane(el, az)
        worst = max(worst, abs(P[1]), abs(math.hypot(P[0], P[1]) - r_), abs(P[2] - z_), 0.0 if P[0] > 0 else 1.0)
    check("fig2: section(b) lands in the half-plane {y=0, x>0} and the closed form (rho, z) is exact", worst < 1e-12,
          "(max err %.1e)" % worst)
    crossings = []
    for _ in range(25):
        el, az = rng.uniform(0.3, 2.9), rng.uniform(0, TWO_PI)
        F = H.fibre_s3(H.base_point(el, az), 4000)
        w = H.to_r4(F)
        y = w[:, 1]
        sc = np.where(y >= 0, 1, -1)            # the sample at t = 0 has Im z1 = 0 exactly: count it once, as positive
        flips = np.where(sc != np.roll(sc, -1))[0]
        crossings.append(sum(1 for i in flips if w[i, 0] + w[(i + 1) % len(w), 0] > 0))
    check("fig2: every fibre crosses the half-plane exactly once", set(crossings) == {1}, "(counts %s)" % sorted(set(crossings)))
    worst = 0.0
    for el in (0.5, 1.2, 2.0, 2.9):
        (cx, cz), r = torus_meridian(el)
        az = np.linspace(0, TWO_PI, 200, endpoint=False)
        rr, zz = section_halfplane(el, az)
        worst = max(worst, np.abs(np.hypot(rr - cx, zz - cz) - r).max())
    check("fig2: a constant-elevation torus meets the half-plane in the circle centre (1/sin(el/2), 0), radius cot(el/2)",
          worst < 1e-12, "(max err %.1e)" % worst)

    # fig 3: audit grid
    rows = audit_table(np.linspace(0.3, 2.9, 8))
    exact_ok = all(abs(r["fixed"][0] - math.tan(r["el"] / 2)) < 1e-9 and abs(r["fixed"][1] - 1 / math.cos(r["el"] / 2)) < 1e-9
                   and abs(r["fixed"][2] - r["el"] / 2) < 1e-9 for r in rows)
    check("fig3: with baseRad = tan(el/2), fibreRad = sec(el/2) the circle has d = tan, r = sec, tilt = el/2", exact_ok)
    defect = [abs(r["shipped"][1] ** 2 - r["shipped"][0] ** 2 - 1) for r in rows]
    check("fig3: the shipped circles are not exact fibres (defect r^2 - d^2 - 1 up to %.2f)" % max(defect), max(defect) > 0.04)
    check("fig3: shipped tilt is exact (el/2) even though the radii are approximate",
          all(abs(r["shipped"][2] - r["el"] / 2) < 1e-9 for r in rows))

    # fig 4
    worst = 0.0
    b_cases = [(1.4, 0.3), (2.0, 2.5), (0.9, 5.0)]
    for p, q in ((1, 1), (2, 1), (3, 1), (2, 3), (3, 2)):
        for el, az in b_cases:
            C = preimage_curve(p, q, el, az, n=200)
            img = compose_map(p, q, C[:, 0], C[:, 1])
            worst = max(worst, np.abs(img - H.base_point(el, az)).max())
    check("fig4: the curve preimage_curve(p, q, b) is mapped to the base point b by f_{p,q}", worst < 1e-12, "(max err %.1e)" % worst)
    lks = {}
    for p, q in ((1, 1), (2, 1), (3, 1), (2, 3)):
        lks[(p, q)] = hopf_invariant_by_linking(p, q, n=1200)[0]
    check("fig4: linking number of two preimages = +-p q  for (p,q) = (1,1), (2,1), (3,1), (2,3)",
          all(abs(abs(v) - p * q) < 0.06 * p * q for (p, q), v in lks.items()),
          "(%s)" % ", ".join("%d%d: %+.3f" % (p, q, v) for (p, q), v in lks.items()))
    check("fig4: all the linking numbers carry the same sign (one orientation convention)",
          len({np.sign(v) for v in lks.values()}) == 1)
    return ok


FIGS = {1: ("hopf_fig1_linking_atlas.png", fig_linking_atlas),
        2: ("hopf_fig2_poincare_section.png", fig_poincare_section),
        3: ("hopf_fig3_makefibre_audit.png", fig_audit),
        4: ("hopf_fig4_hopf_invariant.png", fig_hopf_invariant)}


def main():
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    outdir = "figures"
    if "--png-dir" in sys.argv:
        outdir = sys.argv[sys.argv.index("--png-dir") + 1]
    only = set(range(1, 5))
    if "--only" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
    os.makedirs(outdir, exist_ok=True)
    for k in sorted(only):
        name, fn = FIGS[k]
        fn(os.path.join(outdir, name))


if __name__ == "__main__":
    main()
