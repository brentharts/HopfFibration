#!/usr/bin/env python3
"""hopf_bianchi.py - the three exits of the Mixmaster universe as the three left-invariant Hopf fibrations of S^3.

    python3 hopf_bianchi.py              print the summary
    python3 hopf_bianchi.py --tests      run the checks
    python3 hopf_bianchi.py --png FILE   draw the figure

Setting.  The spatial slice of a vacuum Bianchi IX universe is SU(2) = S^3 (unit quaternions) with a left-invariant metric
h_ab in the basis e_a of left-invariant vector fields, [e_a, e_b] = eps_abc e_c (the fields q -> q u/2, u = i, j, k).  The
diagonal metrics h = diag(a^2, b^2, c^2) are the Mixmaster universe; its exit basins (which axis collapses into a pancake) are
computed in lakes_of_wada.mixmaster of brentharts/wada-coudene (MIXMASTER.md there).  This script is the Hopf side.

  1  (Exact)   The integral curves of the left-invariant field e_u are the fibres q exp(t u) of the Hopf fibration
               h_u(q) = q u conj(q) in S^2, u = i, j, k: three Hopf fibrations, all of one handedness, h_k being hopf_math.hopf.
               When axis u collapses (a pancake, fibre length 2 pi a_u -> 0) the fibre that shrinks is the h_u fibre.
  2  (Exact)   Conjugation by q0 = (1 + i + j + k)/2 maps h_{u'} to h_u with u' = q0^-1 u q0: the cyclic permutation of the axes is
               an isometry of the round S^3 that permutes the three fibrations; a transposition is conjugation by (i + j)/sqrt 2
               (i <-> j, k -> -k), which maps the fibration h_k to itself with the orientation of the fibres reversed.  So the S_3
               symmetry of the Mixmaster picture is realised by isometries of S^3 permuting the Hopf fibrations.
  3  (Counted) SO(3) covariance of the full (non-diagonal) Bianchi IX equations: if h(tau) solves them, so does R h R^T.  With R = Ad_q,
               the rotated solution collapses along the direction R e_l, and R e_l = q u_l conj(q) = h_{u_l}(q): the exit direction in
               the frame is the *Hopf map of the orientation q, taken in the fibration of the exit axis l*.  Checked below against an
               independent integrator of the full 3 x 3 ADM equations (not the diagonal ones).

What is not claimed: the full 3 x 3 integrator is used only with data of the form R diag R^T (metric and momentum diagonalised by the
same frame), not for generic momenta; the exit rule is the one of lakes_of_wada.mixmaster (ours, not Cornish and Levin's).
"""
import math
import os
import sys

import numpy as np

import hopf_math as hm

EPS = np.zeros((3, 3, 3))
for _a, _b, _c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[_a, _b, _c] = 1.0
    EPS[_a, _c, _b] = -1.0
UNITS = {"i": np.array([0, 1.0, 0, 0]), "j": np.array([0, 0, 1.0, 0]), "k": np.array([0, 0, 0, 1.0])}
AXES = [UNITS["i"], UNITS["j"], UNITS["k"]]          # u_1, u_2, u_3 = i, j, k = the axes a, b, c


# ---------------------------------------------------------------------------------------------------- quaternions

def rand_quaternion(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def conj_action(q, v):
    """q v conj(q) for a pure quaternion v given as a 3-vector; returns a 3-vector."""
    return hm.qmul(hm.qmul(q, np.r_[0.0, v]), hm.qconj(q))[1:]


def rotation_matrix(q):
    """The matrix R with R v = q v conj(q) (rotation of the imaginary quaternions)."""
    return np.stack([conj_action(q, e) for e in np.eye(3)], axis=1)


def hopf_u(q, u):
    """h_u(q) = q u conj(q), u a unit quaternion (4-vector); a point of S^2 (imaginary part)."""
    return hm.qmul(hm.qmul(q, u), hm.qconj(q))[1:]


def zz_of_quaternion(q):
    """(z1, z2) = (a + d i, c + b i) for q = a + b i + c j + d k: the convention of hopf_math, with hopf(z1, z2) = q k conj(q)."""
    return complex(q[0], q[3]), complex(q[2], q[1])


def fibre_of(q, u, n=360):
    """The fibre of h_u through q, q exp(t u), as n quaternions."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.array([hm.qmul(q, np.cos(s) * np.array([1.0, 0, 0, 0]) + np.sin(s) * u) for s in t])


def fibre_r3_of(q, u, n=360):
    zz = np.array([zz_of_quaternion(p) for p in fibre_of(q, u, n)])
    return hm.stereographic(zz)


# ---------------------------------------------------------------------------------------------------- full Bianchi IX (3 x 3)

def connection_batch(h):
    """Levi-Civita connection of the left-invariant metric h (shape (m, 3, 3)) on the Lie algebra [e_a, e_b] = eps_abc e_c:
    nabla_a e_b = Gam[d, a, b] e_d, from the Koszul formula; algebraic, no assumption of diagonality.  Also returns c^d_ab."""
    cu = np.einsum("abd->dab", EPS)                               # c^d_ab = eps_abd
    C = np.einsum("mxe,eyz->mxyz", h, cu)                         # C[x,y,z] = <[e_y, e_z], e_x>
    # Koszul: 2 Gam[c,a,b] = C[c,a,b] - C[a,b,c] + C[b,c,a]
    G = np.empty_like(C)
    for c in range(3):
        for a in range(3):
            for b in range(3):
                G[:, c, a, b] = 0.5 * (C[:, c, a, b] - C[:, a, b, c] + C[:, b, c, a])
    return np.einsum("mdc,mcab->mdab", np.linalg.inv(h), G), cu


def ricci_batch(h):
    """3-Ricci tensor (lower indices) of the left-invariant metric h (shape (m, 3, 3))."""
    Gam, cu = connection_batch(h)
    R = (np.einsum("mdbc,mfad->mfabc", Gam, Gam) - np.einsum("mdac,mfbd->mfabc", Gam, Gam)
         - np.einsum("dab,mfdc->mfabc", cu, Gam))
    return np.einsum("maabc->mbc", R)


def momentum_constraint(h, K):
    """Momentum constraint D_j (K^j_i - delta^j_i K) of the homogeneous model, as a 3-vector (shape (m, 3)): with unimodular
    structure constants only Gam^d_{ji} K^j_d survives.  Linear in K; zero on vacuum data."""
    Gam, _ = connection_batch(h)
    Kmix = np.linalg.inv(h) @ K                                   # K^j_d
    return np.einsum("mdji,mjd->mi", Gam, Kmix)


def adm_rhs(h, hp):
    """Second derivative of the metric in the BKL time dt = N dtau, N = sqrt(det h) (vacuum ADM, K_ij = -h'_ij / (2N))."""
    hi = np.linalg.inv(h)
    N = np.sqrt(np.linalg.det(h))
    Np = 0.5 * N * np.einsum("mab,mba->m", hi, hp)
    K = -hp / (2 * N[:, None, None])
    Kt = np.einsum("mab,mba->m", hi, K)
    Kp = N[:, None, None] * (ricci_batch(h) + Kt[:, None, None] * K - 2 * K @ hi @ K)
    return -2 * Np[:, None, None] * K - 2 * N[:, None, None] * Kp


def adm_constraint(h, hp, with_scale=False):
    """Hamiltonian constraint R + K^2 - K_ij K^ij (zero on solutions); with ``with_scale`` also the sum of the absolute sizes of
    the three terms, to make it relative."""
    hi = np.linalg.inv(h)
    N = np.sqrt(np.linalg.det(h))
    K = -hp / (2 * N[:, None, None])
    Kud = hi @ K
    R = np.einsum("mab,mba->m", hi, ricci_batch(h))
    K2, KK = np.trace(Kud, axis1=1, axis2=2) ** 2, np.einsum("mab,mba->m", Kud, Kud)
    c = R + K2 - KK
    return (c, np.abs(R) + K2 + np.abs(KK)) if with_scale else c


def eigen_rates(h, hp):
    """Eigenvalues q_i of h, unit eigenvectors, and x_i' = d ln q_i / dtau = e_i^T h' e_i / q_i (first-order perturbation theory)."""
    q, E = np.linalg.eigh(h)
    xp = np.einsum("mai,mab,mbi->mi", E, hp, E) / q
    return q, E, xp


def full_exit(h0, hp0, h_step=0.01, nmax=30000, kappa=1e-3, p_exit=0.95):
    """Integrate the full 3 x 3 system (RK4, per-trajectory step h_step/(1+max|x'|)) until the Kasner exit rule of
    lakes_of_wada.mixmaster holds; return (unit direction of the collapsing eigenvector, exit found, relative constraint drift)."""
    h, hp = h0.copy(), hp0.copy()
    m = len(h)
    n_out = np.full((m, 3), np.nan)
    found = np.zeros(m, bool)
    drift = 0.0
    alive = np.arange(m)

    def f(hh, pp):
        return pp, adm_rhs(hh, pp)

    for it in range(nmax):
        if alive.size == 0:
            break
        ha, pa = h[alive], hp[alive]
        _, _, xp = eigen_rates(ha, pa)
        dt = (h_step / (1.0 + np.abs(xp).max(axis=1)))[:, None, None]
        k1h, k1p = f(ha, pa)
        k2h, k2p = f(ha + 0.5 * dt * k1h, pa + 0.5 * dt * k1p)
        k3h, k3p = f(ha + 0.5 * dt * k2h, pa + 0.5 * dt * k2p)
        k4h, k4p = f(ha + dt * k3h, pa + dt * k3p)
        ha = ha + dt / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
        pa = pa + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        h[alive], hp[alive] = ha, pa
        q, E, xp = eigen_rates(ha, pa)
        S = xp.sum(axis=1)
        V = np.sum(q ** 2, axis=1) - 2 * (q[:, 0] * q[:, 1] + q[:, 1] * q[:, 2] + q[:, 0] * q[:, 2])
        p = xp / np.where(S == 0, 1.0, S)[:, None]
        ex = (np.abs(V) < kappa * np.maximum(np.sum(xp ** 2, axis=1), 1e-300)) & (p.max(axis=1) > p_exit) & (S < 0)
        if ex.any():
            idx = np.argmax(p[ex], axis=1)
            n_out[alive[ex]] = E[ex][np.arange(ex.sum()), :, idx]
            found[alive[ex]] = True
        if it % 1000 == 0:
            c_, sc_ = adm_constraint(ha, pa, with_scale=True)
            drift = max(drift, float(np.max(np.abs(c_) / sc_)))
        alive = alive[~ex]
    return n_out, found, drift


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
    rng = np.random.default_rng(23)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # 1. h_k is hopf_math.hopf; the fibre of h_u through q is q exp(t u) and h_u is constant on it
    err1 = err2 = 0.0
    for _ in range(40):
        q = rand_quaternion(rng)
        z1, z2 = zz_of_quaternion(q)
        err1 = max(err1, np.abs(hopf_u(q, UNITS["k"]) - hm.hopf(z1, z2)).max())
        for u in AXES:
            F = fibre_of(q, u, 17)
            err2 = max(err2, max(np.abs(hopf_u(p, u) - hopf_u(q, u)).max() for p in F))
    check("h_k(q) = q k conj(q) is hopf_math.hopf; each h_u is constant on its fibres q exp(t u)", max(err1, err2) < 1e-12,
          "(%.1e, %.1e)" % (err1, err2))

    # 2. the fibre of h_u through q is the integral curve of the left-invariant field q -> q u: tangent = q u
    err = 0.0
    for _ in range(20):
        q = rand_quaternion(rng)
        for u in AXES:
            t = 0.37
            d = (hm.qmul(q, np.cos(t + 1e-6) * np.eye(4)[0] + np.sin(t + 1e-6) * u)
                 - hm.qmul(q, np.cos(t - 1e-6) * np.eye(4)[0] + np.sin(t - 1e-6) * u)) / 2e-6
            pt = hm.qmul(q, np.cos(t) * np.eye(4)[0] + np.sin(t) * u)
            err = max(err, np.abs(d - hm.qmul(pt, u)).max())
    check("d/dt q exp(t u) = (q exp(t u)) u, the left-invariant vector field of u", err < 1e-8, "(%.1e)" % err)

    # 3. symmetry: conjugation by q0 permutes the three fibrations; a transposition does too (with a sign)
    q0 = np.array([1, 1, 1, 1]) / 2.0
    q0i = hm.qconj(q0)
    err = 0.0
    for _ in range(20):
        q = rand_quaternion(rng)
        Cq = hm.qmul(hm.qmul(q0, q), q0i)
        for u in AXES:
            up = hm.qmul(hm.qmul(q0i, u), q0)           # u' = q0^-1 u q0
            err = max(err, np.abs(hopf_u(Cq, u) - conj_action(q0, hopf_u(q, up)[0:3])).max())
    perm = [tuple(int(t) for t in np.round(conj_action(q0i, e))) for e in np.eye(3)]
    check("conjugation by q0 = (1+i+j+k)/2 maps the h_u' fibration to the h_u fibration, u' = q0^-1 u q0 (cyclic permutation)",
          err < 1e-12 and sorted(map(lambda t: tuple(np.abs(t)), perm)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
          "(err %.1e; q0^-1 {i,j,k} q0 = %s)" % (err, perm))
    s = (UNITS["i"] + UNITS["j"]) / math.sqrt(2)
    img = [np.round(conj_action(s, e), 12) for e in np.eye(3)]
    check("conjugation by (i + j)/sqrt 2 sends i -> j, j -> i, k -> -k (a transposition with a fibre-orientation sign)",
          np.allclose(img, [[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

    # 4. linking of great circles: |lk| = 1 always; fibres of one fibration all link +1; across the three left-invariant
    #    fibrations the sign is not constant (they are not Clifford parallel to each other)
    same, cross = [], []
    for _ in range(12):
        q, qq = rand_quaternion(rng), rand_quaternion(rng)
        for a in range(3):
            for b in range(3):
                A, B = fibre_r3_of(q, AXES[a], 300), fibre_r3_of(qq, AXES[b], 300)
                if np.isfinite(A).all() and np.isfinite(B).all() and hm.min_distance(A, B) > 0.02:
                    (same if a == b else cross).append(hm.linking_number(A, B))
    same, cross = np.array(same), np.array(cross)
    check("fibres of one left-invariant Hopf fibration link +1 pairwise; fibres of two different ones link +-1 with both signs occurring",
          np.all(np.abs(same - 1) < 0.05) and np.all(np.abs(np.abs(cross) - 1) < 0.05) and cross.min() < -0.9 and cross.max() > 0.9,
          "(%d same-family pairs, %d cross-family pairs; cross values in [%.3f, %.3f])" % (len(same), len(cross), cross.min(), cross.max()))

    # 5. the full 3 x 3 ADM integrator reproduces the diagonal Mixmaster equations, and its constraint is conserved
    x = rng.normal(size=3) * 0.4
    v = rng.normal(size=3)
    qd = np.exp(x)
    hpp = adm_rhs(np.diag(qd)[None], np.diag(qd * v)[None])[0]
    force = np.array([(qd[(i + 1) % 3] - qd[(i + 2) % 3]) ** 2 - qd[i] ** 2 for i in range(3)])
    check("3 x 3 ADM equations with [e_a, e_b] = eps_abc e_c reduce to x_i'' = (q_j - q_k)^2 - q_i^2 on diagonal data",
          np.abs(np.diag(hpp) / qd - v ** 2 - force).max() < 1e-12)
    mm = _mixmaster()
    if mm is None:
        if verbose:
            print("  skip checks 6 and 7: clone of brentharts/wada-coudene (lakes_of_wada.mixmaster) not found (set WADA_REPO)")
        return ok

    # 6. SO(3) covariance and the Hopf map: collapse direction of the rotated solution = R e_l = h_{u_l}(q)
    n_samp = 36
    bp, bm = rng.uniform(-0.7, 0.7, n_samp), rng.uniform(-0.7, 0.7, n_samp)
    x0, v0, V0 = mm.turnaround_state(bp, bm)
    good = V0 < 0
    bp, bm, x0, v0 = bp[good], bm[good], x0[good], v0[good]
    lab = mm.exit_basins(bp, bm)
    keep = lab > 0
    bp, bm, x0, v0, lab = bp[keep][:24], bm[keep][:24], x0[keep][:24], v0[keep][:24], lab[keep][:24]
    qs = [rand_quaternion(rng) for _ in range(len(lab))]
    Rs = np.array([rotation_matrix(q) for q in qs])
    q0d = np.exp(x0)
    h0 = np.einsum("mab,mb,mcb->mac", Rs, q0d, Rs)                          # R diag(q) R^T
    hp0 = np.einsum("mab,mb,mcb->mac", Rs, q0d * v0, Rs)                    # R diag(q v) R^T
    check("initial data satisfy the 3 x 3 Hamiltonian constraint", np.abs(adm_constraint(h0, hp0)).max() < 1e-9,
          "(%.1e)" % np.abs(adm_constraint(h0, hp0)).max())
    n_dir, found, drift = full_exit(h0, hp0)
    match_own, match_label, dev = 0, 0, 0.0
    for m in range(len(lab)):
        if not found[m]:
            continue
        target = hopf_u(qs[m], AXES[lab[m] - 1])
        c = abs(float(np.dot(n_dir[m], target)))
        dev = max(dev, 1 - c)
        match_own += c > 1 - 1e-6
        # which of the three exit directions h_{u_i}(q) does the full integrator hit?
        hit = [abs(float(np.dot(n_dir[m], hopf_u(qs[m], u)))) > 1 - 1e-6 for u in AXES]
        match_label += hit[lab[m] - 1]
    nf = int(found.sum())
    check("full integrator: exit found for the rotated data", nf >= len(lab) - 2, "(%d of %d)" % (nf, len(lab)))
    check("collapse direction of the rotated solution = h_u(q) in the fibration of the diagonal exit axis (up to sign), "
          "i.e. the diagonal and the full integrators agree on the exit axis",
          match_own >= nf - 1 and nf > 15, "(%d of %d match to 1e-6; worst 1 - |cos| = %.1e; constraint drift %.1e)" %
          (match_own, nf, dev, drift))

    # 10-11. non-aligned momenta: the momentum constraint is linear in K; its kernel is {K : [h^-1 K, ...]=0} = K commuting with h
    def kernel(h):
        basis = []
        for a in range(3):
            for b in range(a, 3):
                E = np.zeros((3, 3)); E[a, b] = E[b, a] = 1.0
                basis.append(E)
        M = np.array([momentum_constraint(h[None], E[None])[0] for E in basis]).T      # 3 x 6
        u, sv, vt = np.linalg.svd(M)
        null = vt[np.sum(sv > 1e-9):]
        return [sum(c * E for c, E in zip(v, basis)) for v in null]
    dims, comm = [], 0.0
    for hh in (np.diag([0.7, 1.1, 1.6]), np.diag([0.7, 1.4, 1.4]), np.eye(3) * 1.2):
        ker = kernel(hh)
        dims.append(len(ker))
        comm = max([comm] + [np.abs(hh @ k - k @ hh).max() for k in ker])
    check("momentum-constraint kernel has dimension 3 (generic), 4 (biaxial), 6 (round) and consists of K commuting with h",
          dims == [3, 4, 6] and comm < 1e-12, "(dims %s, worst |[h,K]| %.1e)" % (dims, comm))
    # the frame-rotation charge J = sqrt(det h) eps_ijk [K, h^-1]_jk is conserved by the flow even off the constraint surface
    hq = np.diag([0.7, 1.1, 1.6])[None].copy()
    Kq = (np.diag([-0.4, -0.3, -0.5]) + np.array([[0, .15, -.1], [.15, 0, .2], [-.1, .2, 0]]))[None]
    hpq = -2 * np.sqrt(np.linalg.det(hq))[:, None, None] * Kq

    def charge(h, hp):
        Kc = -hp / (2 * np.sqrt(np.linalg.det(h))[:, None, None]); hi = np.linalg.inv(h)
        Cm = Kc @ hi - hi @ Kc
        A = np.stack([Cm[:, 1, 2] - Cm[:, 2, 1], Cm[:, 2, 0] - Cm[:, 0, 2], Cm[:, 0, 1] - Cm[:, 1, 0]], axis=1)
        return np.sqrt(np.linalg.det(h))[:, None] * A
    J0 = charge(hq, hpq); dt = 1e-3
    for _ in range(1500):
        f = lambda a, b: (b, adm_rhs(a, b))
        k1 = f(hq, hpq); k2 = f(hq + .5 * dt * k1[0], hpq + .5 * dt * k1[1])
        k3 = f(hq + .5 * dt * k2[0], hpq + .5 * dt * k2[1]); k4 = f(hq + dt * k3[0], hpq + dt * k3[1])
        hq = hq + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); hpq = hpq + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    dJ = np.abs(charge(hq, hpq) - J0).max()
    check("frame-rotation charge J is conserved along the flow for non-aligned momenta", dJ < 1e-9 and np.abs(J0).max() > 0.1,
          "(|J| = %.3f, drift %.1e)" % (np.abs(J0).max(), dJ))
    return ok


# ---------------------------------------------------------------------------------------------------- picture

def make_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#d1495b", "#2e86ab", "#edae49"]
    rng = np.random.default_rng(2)
    fig = plt.figure(figsize=(16, 5.4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    ax.plot_surface(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), color="#eeeeee", alpha=0.35, linewidth=0)
    for i, e in enumerate(np.eye(3)):
        ax.plot([-e[0], e[0]], [-e[1], e[1]], [-e[2], e[2]], color=cols[i], lw=2)
        ax.scatter(*e, color=cols[i], s=60); ax.scatter(*(-e), color=cols[i], s=60)
        ax.text(*(1.15 * e), "+e%d" % (i + 1)); ax.text(*(-1.2 * e), "-e%d" % (i + 1))
    ax.set_title("the base S^2: the exit axes are the vertices\nof an octahedron (R e_l = h_{u_l}(q))")
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    for i, u_ in enumerate(AXES):
        for _ in range(7):
            P = fibre_r3_of(rand_quaternion(rng), u_, 300)
            P = np.where(np.linalg.norm(P, axis=1)[:, None] > 4.5, np.nan, P)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cols[i], lw=1.1, alpha=0.9)
        P = fibre_r3_of(np.array([1.0, 0, 0, 0]), u_, 300)
        P = np.where(np.linalg.norm(P, axis=1)[:, None] > 4.5, np.nan, P)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cols[i], lw=3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)
    ax.set_title("fibres of the three left-invariant Hopf fibrations h_i, h_j, h_k\n(thick: through the identity), stereographic R^3")

    ax = fig.add_subplot(1, 3, 3)
    mm = _mixmaster()
    if mm is not None:
        n = 160
        pic = mm.mixmaster_picture(n=n, half=1.2)
        from matplotlib.colors import ListedColormap
        ax.imshow(pic.label.T, origin="lower", extent=(-1.2, 1.2, -1.2, 1.2),
                  cmap=ListedColormap(["#999999", "#ffffff"] + cols), vmin=-1, vmax=3, interpolation="nearest")
        ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_title("Mixmaster exit basins (wada-coudene): colour l = the\nfibration h_{u_l} whose fibres collapse")
    ax.set_xlabel("beta+"); ax.set_ylabel("beta-")
    fig.suptitle("Three exits = three left-invariant Hopf fibrations of S^3 = SU(2)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests(verbose=True) else 1)
    if "--png" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--png") + 1])
        sys.exit(0)
    print(__doc__)
