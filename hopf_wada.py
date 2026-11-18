#!/usr/bin/env python3
"""hopf_wada.py - Wada basins on S^2 lifted through the Hopf map to S^3.

    python3 hopf_wada.py              print the summary
    python3 hopf_wada.py --tests      run the checks
    python3 hopf_wada.py --png FILE   draw the figure

The idea.  Newton's method for z^3 - 1 on the Riemann sphere CP^1 = S^2 has three basins with a common boundary (the
Julia set): the standard Wada example.  Give a point (z1, z2) of S^3 the label of w = z1/z2.  That label is constant
on Hopf fibres, so it is the pull-back of the S^2 label along the Hopf map h.  Claims, in the labels of this repo:

  A (Exact, a short proof)  h is a continuous open surjection, so for every set U of S^2
                            closure(h^-1 U) = h^-1 closure(U) and interior(h^-1 U) = h^-1 interior(U); hence
                            boundary(h^-1 U) = h^-1 boundary(U).  The three lifted basins are three disjoint open
                            sets of S^3 with the common boundary h^-1(J) = J x S^1 (locally), so Wada lifts.
  B (Exact, quantitative)   h is a Riemannian submersion S^3 -> S^2(1/2): for x in S^3 and a set A of S^2,
                            dist_S3(x, h^-1 A) = dist_S2(h(x), A) / 2   (unit-sphere angle on the right).  So the
                            radius needed to see all three lakes from a boundary point is the same number on S^2 and S^3.
  C (Counted)               the voxel test of the Wada repo (any dimension) on a 3D grid of the stereographic image
                            agrees with A: the profile of the S^3 picture, and the boundary voxels lie over the Julia set.

Nothing here says the lifted sets are 'lakes' in Wada's original (accessibility) sense; the property tested is the
modern one, every boundary point of one domain is a boundary point of all.  The lemma A is ours, not checked against a source.
Convention: that of hopf_math.py; the stereographic coordinate of the base point b = (x, y, z) is w = (x + i y)/(1 - z) = z1/z2.
"""
import math
import os
import sys

import numpy as np

import hopf_math as hm

ROOTS = np.exp(2j * np.pi * np.arange(3) / 3)


# ---------------------------------------------------------------------------------------------------- labels

def newton_label(w, n_iter=80, tol=1e-3):
    """Basin of z^3 - 1 under Newton's method, for complex w (any shape): 1, 2, 3 = root index + 1, 0 = undecided
    (orbit not within tol of a root after n_iter steps; includes w = infinity)."""
    z = np.array(w, dtype=complex)
    with np.errstate(all="ignore"):
        for _ in range(n_iter):
            z = z - (z ** 3 - 1) / (3 * z ** 2)
        d = np.abs(z[..., None] - ROOTS)
    lab = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=-1) + 1
    best = np.take_along_axis(np.where(np.isfinite(d), d, np.inf), (lab - 1)[..., None], axis=-1)[..., 0]
    return np.where(best < tol, lab, 0)


def w_of_z(z1, z2):
    with np.errstate(all="ignore"):
        return np.asarray(z1, dtype=complex) / np.asarray(z2, dtype=complex)


def label_s3(z1, z2):
    """The lifted label: the Newton basin of w = z1/z2 (constant on Hopf fibres)."""
    return newton_label(w_of_z(z1, z2))


def label_s2(b):
    """The Newton basin of the base point b = (x, y, z) (array (..., 3)): w = (x + i y)/(1 - z)."""
    b = np.asarray(b, dtype=float)
    with np.errstate(all="ignore"):
        w = (b[..., 0] + 1j * b[..., 1]) / (1 - b[..., 2])
    return newton_label(w)


def point_of_s3(p):
    """Inverse stereographic projection R^3 -> S^3 (pole (0,0,0,1)); returns (z1, z2)."""
    p = np.asarray(p, dtype=float)
    r2 = np.sum(p ** 2, axis=-1)
    x4 = 2 * p / (1 + r2)[..., None]
    w4 = (r2 - 1) / (r2 + 1)
    return x4[..., 0] + 1j * x4[..., 1], x4[..., 2] + 1j * w4


def random_s3(n, rng):
    v = rng.normal(size=(n, 4))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v[:, 0] + 1j * v[:, 1], v[:, 2] + 1j * v[:, 3]


def random_s2(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


# ---------------------------------------------------------------------------------------------------- the Wada repo

def _wadatest():
    """Import lakes_of_wada.wadatest from a clone of brentharts/wada-coudene (env WADA_REPO, else ../wada-coudene)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.environ.get("WADA_REPO"), os.path.join(here, "..", "wada-coudene")):
        if root and os.path.isdir(os.path.join(root, "lakes_of_wada")):
            sys.path.insert(0, root)
            try:
                from lakes_of_wada import wadatest
                return wadatest
            except Exception:       # missing scipy etc.
                pass
            finally:
                sys.path.remove(root)
    return None


# ---------------------------------------------------------------------------------------------------- the 3D picture

def voxel_labels(n, half=3.0):
    """Labels of the n^3 voxel centres of [-half, half]^3 in the stereographic R^3 picture of S^3; returns
    (label array, voxel size)."""
    ax = (np.arange(n) + 0.5) / n * 2 * half - half
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    z1, z2 = point_of_s3(np.stack([X, Y, Z], axis=-1))
    return label_s3(z1, z2), 2 * half / n


def sphere_grid_labels(nlat=1000, nlon=2000):
    """Labels on an equirectangular grid of S^2 (cell centres); returns (labels, base points)."""
    el = (np.arange(nlat) + 0.5) / nlat * math.pi          # elevation 0 = south pole, as hopf_math.base_point
    az = (np.arange(nlon) + 0.5) / nlon * 2 * math.pi
    E, A = np.meshgrid(el, az, indexing="ij")
    B = np.stack([np.sin(E) * np.cos(A), np.sin(E) * np.sin(A), -np.cos(E)], axis=-1)
    return label_s2(B), B


# ---------------------------------------------------------------------------------------------------- tests

def run_tests(verbose=True):
    rng = np.random.default_rng(11)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 0. the label is the Newton basin: three-fold symmetric, and the roots are in their own basins
    check("roots label themselves 1, 2, 3", list(newton_label(ROOTS)) == [1, 2, 3])
    w = rng.normal(size=500) + 1j * rng.normal(size=500)
    check("label(w e^{2 pi i/3}) = label(w) permuted cyclically (symmetry of z^3 - 1)",
          np.all(newton_label(w * ROOTS[1]) == (newton_label(w) % 3) + 1))

    # 1. S^3 label is the pull-back of the S^2 label along the Hopf map, and constant on fibres
    z1, z2 = random_s3(4000, rng)
    h = hm.hopf(z1, z2)
    same = np.mean(label_s3(z1, z2) == label_s2(h))
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=z1.shape))
    fib = np.mean(label_s3(z1, z2) == label_s3(ph * z1, ph * z2))
    check("label_s3 = label_s2 o hopf  (4000 random points)", same > 0.999, "(agree %.4f)" % same)
    check("label_s3 is constant on Hopf fibres", fib > 0.9995, "(agree %.4f)" % fib)

    # 2. claim B: Riemannian submersion.  distance from x to the fibre over b' is theta/2
    from scipy.optimize import minimize_scalar
    worst = 0.0
    for _ in range(40):
        x = random_s3(1, rng)
        x = np.array([x[0][0], x[1][0]])
        bp = random_s2(1, rng)[0]
        s = np.array(hm.section(bp))
        # S^3 distance between x and e^{it} s: arccos Re <x, e^{it} s>, minimised over t numerically
        f = lambda t: math.acos(max(-1.0, min(1.0, (np.vdot(x, np.exp(1j * t) * s)).real)))
        ts = np.linspace(0, 2 * np.pi, 721)
        t0 = ts[np.argmin([f(t) for t in ts])]
        m = minimize_scalar(f, bounds=(t0 - 0.01, t0 + 0.01), method="bounded", options={"xatol": 1e-12})
        theta = math.acos(max(-1.0, min(1.0, float(np.dot(hm.hopf(x[0], x[1]), bp)))))
        worst = max(worst, abs(m.fun - theta / 2))
    check("dist_S3(x, fibre over b') = (angle between h(x) and b')/2", worst < 1e-6, "(max err %.1e)" % worst)

    # 2b. the same for a whole lifted basin: S^3 nearest-sample distance vs S^2 nearest-sample distance / 2
    N3 = 600000
    Z1, Z2 = random_s3(N3, rng)
    L3 = label_s3(Z1, Z2)
    P4 = np.stack([Z1.real, Z1.imag, Z2.real, Z2.imag], axis=1)
    B2 = random_s2(300000, rng)
    L2 = label_s2(B2)
    worst_gap, below = 0.0, 0.0
    for _ in range(6):
        x = random_s3(1, rng)
        x4 = np.array([x[0][0].real, x[0][0].imag, x[1][0].real, x[1][0].imag])
        hx = hm.hopf(x[0][0], x[1][0])
        for k in (1, 2, 3):
            d3 = math.acos(min(1.0, float(np.max(P4[L3 == k] @ x4))))
            d2 = math.acos(min(1.0, float(np.max(B2[L2 == k] @ hx)))) / 2
            worst_gap = max(worst_gap, d3 - d2)
            below = min(below, d3 - d2)
    check("nearest-sample distance to a lifted basin matches dist_S2/2 (S^3 sample is an upper bound)",
          below > -1e-9 and worst_gap < 0.05, "(S^3 minus S^2/2: min %.1e, max %.3f; sampling error)" % (below, worst_gap))

    # 3. Wada profile in R^3 (stereographic voxels) with the Wada repo's own dimension-free tests
    wt = _wadatest()
    if wt is None:
        if verbose:
            print("  skip 3D checks: clone of brentharts/wada-coudene not found (set WADA_REPO)")
        return ok
    lab, vox = voxel_labels(96)
    prof = wt.wada_profile(lab, radii=(1, 2, 4, 8))
    check("3D picture has exactly three basins", prof["K"] == 3 and prof["boundary_pixels"] > 5000,
          "(K = %d, %d boundary voxels)" % (prof["K"], prof["boundary_pixels"]))
    check("3D Wada profile reaches 1 by radius 8 voxels", prof["profile"][8] > 0.99,
          "(%s)" % {r: round(v, 4) for r, v in prof["profile"].items()})
    # needed radius, in physical units, shrinks as the resolution is refined
    need = {}
    for n in (48, 96):
        lb, v = voxel_labels(n)
        need[n] = wt.wada_radius(lb, percentiles=(50, 90))["radius"]
    r50 = [need[n][50] * (6.0 / n) for n in (48, 96)]
    r90 = [need[n][90] * (6.0 / n) for n in (48, 96)]
    check("needed radius (in box units) shrinks with resolution, median and 90th percentile",
          r50[1] < 0.75 * r50[0] and r90[1] < 0.75 * r90[0],
          "(48 -> 96: median %.3f -> %.3f, p90 %.3f -> %.3f)" % (r50[0], r50[1], r90[0], r90[1]))

    # 4. claim A at voxel scale: every boundary voxel of the S^3 picture lies over a point near the Julia set of S^2.
    # A voxel of size s at p spans S^3 length <= s*2/(1+|p|^2)*sqrt(3) <= 2 sqrt(3) s; its image on the unit S^2 is within
    # twice that angle (h is a Riemannian submersion onto S^2(1/2)).
    bm = wt.boundary_mask(lab)
    idx = np.argwhere(bm)
    sel = idx[rng.choice(len(idx), size=min(400, len(idx)), replace=False)]
    ax = (np.arange(96) + 0.5) / 96 * 6.0 - 3.0
    pts = np.stack([ax[sel[:, 0]], ax[sel[:, 1]], ax[sel[:, 2]]], axis=1)
    z1, z2 = point_of_s3(pts)
    base = hm.hopf(z1, z2)
    rho = 2 * 2 * math.sqrt(3) * vox
    hit = 0
    for b in base:
        e1 = np.cross(b, [0.3, 0.5, 0.8]); e1 /= np.linalg.norm(e1); e2 = np.cross(b, e1)
        r_, a_ = np.sqrt(rng.uniform(0, 1, 300)) * rho, rng.uniform(0, 2 * np.pi, 300)
        q = (np.cos(r_)[:, None] * b + np.sin(r_)[:, None] * (np.cos(a_)[:, None] * e1 + np.sin(a_)[:, None] * e2))
        labs = set(label_s2(q).tolist()) - {0}
        hit += len(labs) >= 2
    check("boundary voxels of S^3 lie over S^2 boundary points (a disc of angle %.2f around h(x) sees >= 2 basins)" % rho,
          hit / len(base) > 0.97, "(%d of %d)" % (hit, len(base)))
    return ok


# ---------------------------------------------------------------------------------------------------- picture

def fibre_points(b, n=360, clip=3.0):
    P = hm.fibre_r3(b, n)
    P = np.where(np.linalg.norm(P, axis=1)[:, None] > clip * 1.6, np.nan, P)
    return P


def make_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(5)
    cols = ["#f2f2f2", "#d1495b", "#2e86ab", "#edae49"]
    cmap = ListedColormap(cols)
    fig = plt.figure(figsize=(16, 5.4))

    ax = fig.add_subplot(1, 3, 1)
    n = 900
    x = np.linspace(-2, 2, n)
    W = x[None, :] + 1j * x[:, None]
    ax.imshow(newton_label(W), origin="lower", extent=(-2, 2, -2, 2), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("Newton basins of z^3 - 1 on S^2\n(the base of the fibration, w = z1/z2)")
    ax.set_xlabel("Re w"); ax.set_ylabel("Im w")

    ax = fig.add_subplot(1, 3, 2)
    nr, nz = 900, 1100
    rho = np.linspace(0, 4, nr)
    zz = np.linspace(-3, 3, nz)
    R, Z = np.meshgrid(rho, zz)
    z1, z2 = point_of_s3(np.stack([R, np.zeros_like(R), Z], axis=-1))
    ax.imshow(label_s3(z1, z2), origin="lower", extent=(0, 4, -3, 3), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("the same labels on the half-plane y = 0, x > 0 of R^3\n(a Poincare section: every fibre crosses it once)")
    ax.set_xlabel("rho = sqrt(x^2 + y^2)"); ax.set_ylabel("z"); ax.set_aspect("equal")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    lab, B = sphere_grid_labels(700, 1400)
    sub = np.zeros_like(lab, dtype=bool)
    m = np.zeros_like(lab, dtype=bool)
    m[1:-1, 1:-1] = (lab[1:-1, 1:-1] != lab[:-2, 1:-1]) | (lab[1:-1, 1:-1] != lab[2:, 1:-1]) | \
                    (lab[1:-1, 1:-1] != lab[1:-1, :-2]) | (lab[1:-1, 1:-1] != lab[1:-1, 2:])
    bidx = np.argwhere(m & (np.abs(B[..., 2]) < 0.85))
    for i, j in bidx[rng.choice(len(bidx), size=48, replace=False)]:
        P = fibre_points(B[i, j])
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="k", lw=0.7, alpha=0.85)
    for k in (1, 2, 3):
        idx = np.argwhere((lab == k) & (np.abs(B[..., 2]) < 0.7))
        for i, j in idx[rng.choice(len(idx), size=7, replace=False)]:
            P = fibre_points(B[i, j])
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cols[k], lw=1.6)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)
    ax.set_title("Hopf fibres over the three basins (colour) and over\nJulia-set points (black), stereographic R^3")
    fig.suptitle("Wada basins of z^3 - 1 on S^2 and their Hopf lift to S^3: three open sets, one common boundary J x S^1", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summary():
    wt = _wadatest()
    print(__doc__.split("Nothing here")[0])
    if wt is None:
        print("(Wada repo not found: set WADA_REPO to a clone of brentharts/wada-coudene for the 3D numbers)")
        return
    for n in (48, 96, 128):
        lab, vox = voxel_labels(n)
        p = wt.wada_profile(lab, radii=(1, 2, 4, 8))
        r = wt.wada_radius(lab, percentiles=(50, 90, 99))["radius"]
        print("3D n=%3d  boundary voxels %7d  profile %s" % (n, p["boundary_pixels"],
              {k: round(v, 3) for k, v in p["profile"].items()}))
        print("         needed radius in box units (box = 6): %s" % {q: round(v * 6.0 / n, 3) for q, v in r.items()})


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests(verbose=True) else 1)
    if "--png" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--png") + 1])
        sys.exit(0)
    summary()
