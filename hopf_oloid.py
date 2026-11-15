#!/usr/bin/env python3
"""hopf_oloid.py - the oloid against the Hopf fibration: plain numpy (+ scipy for the convex hull), no Blender needed.

The oloid is the convex hull of two linked circles of radius 1 in perpendicular planes, each passing through the other's
centre (so the centres are 1 apart).  This script asks three questions and answers them numerically, with tests.

  1. SHAPE      build the oloid, check its surface area is 4 pi (that of the unit sphere), print its volume, find the
                family of straight generators that rule it, and check they all have the same length sqrt(3).
  2. LINK       the skeleton of the oloid is a Hopf link (linking number +-1).  Is it a pair of Hopf FIBRES, up to a Moebius
                map?  Stereographic projection preserves round circles and Moebius maps are exactly the conformal maps
                of S^3, so this is the right notion.  Answer: NO, but it is close, and the family d = (centre distance)
                contains one exact Hopf pair, the 'two-circle roller' d = sqrt(2).
  3. ROLL       roll the oloid on a plane without slipping, track its orientation as a unit quaternion in S^3, and project
                with the Hopf map.  The orientation returns after one cycle but the quaternion does NOT: it returns as -q.

CONVENTIONS
  Oloid         circle A: centre (0,0,0), radius 1, in the plane z = 0, normal e_z.
                circle B: centre (d,0,0), radius 1, in the plane y = 0, normal e_y.   d = 1 is the oloid.
                Centre of mass (1/2, 0, 0) by symmetry (the hull's centroid is checked against it).
  Generators    A(t) = (cos t, sin t, 0) joined to B(s) = (d + cos s, 0, sin s).  For d = 1 the segment A(t)B(s) lies in a
                common tangent plane of both circles exactly when  cos s = cos t / (1 - cos t)  (coplanarity of A'(t),
                B'(s) and B(s) - A(t)); this needs cos t <= 1/2, i.e. each circle contributes the arc of length 4 pi / 3
                (two thirds of it) to the surface.  One closed loop of generators has four patches and is the period.
  Moebius       the Lorentz model: a point x of R^3 is the null vector (x, (|x|^2-1)/2, (|x|^2+1)/2) of R^{4,1} with the
                form x.x' + a a' - b b'; a sphere with centre p and radius R is the unit spacelike vector
                (p/R, (P-1)/(2R), (P+1)/(2R)), P = |p|^2 - R^2; a plane {x.n = h} is (n, h, h).  A circle is the
                intersection of two orthogonal spheres, i.e. a 2-plane W in R^{4,1}.  Two circles have TWO Moebius
                invariants: the singular values s1 >= s2 of the Lorentz Gram matrix of orthonormal bases of W1, W2.
                s_i = cos(theta_i), the principal angles of the 2-planes when both circles are great circles of S^3.
  Hopf pair     Two great circles of S^3 are fibres of ONE Hopf fibration iff they are Clifford parallel (isoclinic),
                theta1 = theta2 = theta, and then the base points of the two fibres are 2 theta apart on S^2 (this is
                the 'Clifford distance = half the base angle' of hopf_clifford.py).  A 2-plane in R^4 is a point of
                S^2_+ x S^2_- (self-dual and anti-self-dual parts of its bivector); two planes with principal angles
                theta1, theta2 sit theta1 + theta2 apart on one factor and |theta1 - theta2| on the other.  So the pair
                is a Hopf pair exactly when one of the two separations is 0, and the other one is the base angle.
  Rolling       the oloid is developable, so it rolls on a plane about the straight generator in contact.  With body-frame
                generator direction e(u) and outward unit normal n(u) of the tangent plane along it, the world normal is
                fixed (-e_z), which forces the body angular velocity to be  omega = -((e x n).n') e  (rotation about the
                generator), and the world position follows from 'the contact line does not slip'.  Orientation is the
                unit quaternion q (v -> q v conj(q), hopf_quaternion_rotor.rotation_matrix), updated by RIGHT
                multiplication q -> q (cos(d/2), sin(d/2) e) (body-frame rotation), so the Hopf map h(q) = q k conj(q)
                of hopf_math.py is the world direction of the BODY z axis (the normal of circle A); the fibre is the spin
                about that axis.  The same with q -> q r gives the world direction of any other body axis.

What is exact and what is not:
  - the Moebius invariants, the (theta1 + theta2, |theta1 - theta2|) separations, the great-circle model and the d-family
    formula s1 = |1 - d^2/2|, s2 = 0 are exact up to floating point.  Moebius-equivalence of two circle pairs with the same
    (s1, s2) is the standard classification; here it is backed by an explicit map (great_circle_model) that sends both
    circles to great circles, which the tests check on sample points.
  - area and volume are those of the hull of 2n sample points (error about 1e-6 for n = 1500).
  - the rolling is a discretisation with N = 6000 steps per period (errors about 1e-6); the generator family is exact.
  - the numbers sqrt(2)/2 and 3/4 for the range of the centre-of-mass height, and 8 pi/(3 sqrt 3) for the displacement per
    period, are read off the simulation (they agree to 6 digits); they are not derived here.

    python3 hopf_oloid.py --tests            runs the checks
    python3 hopf_oloid.py                    prints the summary
    python3 hopf_oloid.py --png out.png      draws the figure
"""
import math
import sys

import numpy as np

import hopf_math as H

ETA = np.diag([1.0, 1.0, 1.0, 1.0, -1.0])
OLOID_CENTRE = np.array([0.5, 0.0, 0.0])
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------- the shape

def circles(d=1.0, n=720):
    """The two circle polygons A (z = 0 plane, centre 0) and B (y = 0 plane, centre (d,0,0)), each (n, 3)."""
    th = np.linspace(0.0, TWO_PI, n, endpoint=False)
    A = np.stack([np.cos(th), np.sin(th), 0 * th], axis=1)
    B = np.stack([d + np.cos(th), 0 * th, np.sin(th)], axis=1)
    return A, B


def circle_data(d=1.0):
    """(centre, radius, normal) of the two circles."""
    return ((np.zeros(3), 1.0, np.array([0.0, 0.0, 1.0])), (np.array([d, 0.0, 0.0]), 1.0, np.array([0.0, 1.0, 0.0])))


def hull_stats(d=1.0, n=1500):
    """Area, volume and centroid of the convex hull of the two circles (2n sample points)."""
    from scipy.spatial import ConvexHull
    A, B = circles(d, n)
    pts = np.vstack([A, B])
    hull = ConvexHull(pts)
    ref = pts.mean(axis=0)
    vol = 0.0
    cm = np.zeros(3)
    for sm in hull.simplices:
        p = pts[sm]
        v = abs(np.linalg.det(p - ref)) / 6.0
        vol += v
        cm += v * (p.sum(axis=0) + ref) / 4.0
    return {"area": float(hull.area), "volume": float(hull.volume), "centroid": cm / vol, "points": pts}


def generators(N=6000):
    """One period of the generator loop of the oloid (d = 1), N samples, in the body frame.

    Parameter eta in [-pi/2, 3 pi/2): t = pi + (2 pi/3) sin(eta) runs over the arc [pi/3, 5 pi/3] of circle A and back,
    s = sign(cos eta) arccos(cos t / (1 - cos t)); this is smooth through the turning points (s is linear in eta there).
    Returns dict with PA, PB (the ends), e (unit direction), L (length), n (outward unit normal of the tangent plane)."""
    eta = np.linspace(-math.pi / 2, 3 * math.pi / 2, N, endpoint=False)
    t = math.pi + (2 * math.pi / 3) * np.sin(eta)
    c = np.cos(t)
    s = np.sign(np.cos(eta)) * np.arccos(np.clip(c / (1 - c), -1.0, 1.0))
    PA = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1)
    PB = np.stack([1 + np.cos(s), 0 * s, np.sin(s)], axis=1)
    Ad = np.stack([-np.sin(t), np.cos(t), 0 * t], axis=1)
    Bd = np.stack([-np.sin(s), 0 * s, np.cos(s)], axis=1)
    D = PB - PA
    L = np.linalg.norm(D, axis=1)
    e = D / L[:, None]
    # the tangent plane contains e and A'(t) and also e and B'(s); at the turning points one of the two degenerates
    n1 = np.cross(e, Ad)
    n2 = np.cross(e, Bd)
    pick = (np.linalg.norm(n1, axis=1) >= np.linalg.norm(n2, axis=1))[:, None]
    n = np.where(pick, n1, n2)
    n /= np.linalg.norm(n, axis=1)[:, None]
    flip = np.sign(np.einsum("ij,ij->i", n, 0.5 * (PA + PB) - OLOID_CENTRE))
    n *= flip[:, None]
    return {"eta": eta, "t": t, "s": s, "PA": PA, "PB": PB, "e": e, "L": L, "n": n}


# ---------------------------------------------------------------------------------------------------- Moebius invariants

def circle_frame(centre, radius, normal):
    """The 2-plane W of R^{4,1} of the spheres through the circle: rows (plane through the circle, sphere with the circle
    as an equator).  Both unit spacelike and Lorentz-orthogonal."""
    centre = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    plane = np.r_[n, n @ centre, n @ centre]
    P = centre @ centre - radius ** 2
    sphere = np.r_[centre / radius, (P - 1) / (2 * radius), (P + 1) / (2 * radius)]
    return np.array([plane, sphere])


def moebius_invariants(circ1, circ2):
    """Singular values (s1 >= s2) of the Lorentz Gram matrix of the two circle planes.  circ = (centre, radius, normal)."""
    W1 = circle_frame(*circ1[:3])
    W2 = circle_frame(*circ2[:3])
    M = W1 @ ETA @ W2.T
    return np.linalg.svd(M, compute_uv=False)


def principal_angles(circ1, circ2):
    """theta_i = arccos(s_i), ascending.  Only meaningful when both s_i < 1 (the pair is then Moebius equivalent to a pair
    of great circles of S^3 with these principal angles); otherwise returns nan for the offending angle."""
    s = moebius_invariants(circ1, circ2)
    return np.array([math.acos(x) if x < 1 - 1e-12 else float("nan") for x in s])


def separations(theta):
    """(theta1 + theta2, |theta1 - theta2|): the angular separations of the two planes on the two S^2 factors, largest
    first.  The pair is a Hopf pair iff the second is 0; the first is then the base angle."""
    th1, th2 = float(theta[0]), float(theta[1])
    return np.array(sorted([th1 + th2, abs(th1 - th2)], reverse=True))


def great_circle_model(circ1, circ2):
    """An explicit Moebius map of R^3 u {oo} onto S^3 sending both circles to great circles.

    Needs a unit timelike vector t Lorentz-orthogonal to W1 and W2 (exists iff W1 + W2 is positive definite, i.e. both
    Moebius invariants are < 1).  Returns dict with 'map' (points (m,3) -> (m,4) on S^3) and the 2-planes P1, P2
    (2x4 orthonormal rows) of the two great circles in R^4."""
    W = np.vstack([circle_frame(*circ1[:3]), circle_frame(*circ2[:3])])
    if np.linalg.eigvalsh(W @ ETA @ W.T).min() < 1e-9:
        raise ValueError("the circles are not Moebius equivalent to a pair of great circles (an invariant is >= 1)")
    t = np.linalg.svd(W @ ETA)[2][-1]
    nt = t @ ETA @ t
    if nt >= 0:
        raise ValueError("no timelike vector orthogonal to both circle planes")
    t = t / math.sqrt(-nt)
    if t[4] < 0:
        t = -t
    V = np.eye(5) + np.outer(ETA @ t, t)                 # projection of e_i onto t-perp:  v + <v,t> t
    lam, U = np.linalg.eigh(V @ ETA @ V.T)
    keep = lam > 1e-9
    F = (U[:, keep] / np.sqrt(lam[keep])).T @ V           # 4 Lorentz-orthonormal rows spanning t-perp

    def to_s3(pts):
        x = np.asarray(pts, float).reshape(-1, 3)
        r2 = (x ** 2).sum(axis=1)
        X = np.c_[x, (r2 - 1) / 2, (r2 + 1) / 2]
        X = X / (-(X @ ETA @ t))[:, None]                 # rescale so that <X,t> = -1
        Y = X - t                                          # = X + <X,t> t, now in t-perp with |Y| = 1
        return Y @ ETA @ F.T

    Wc = W @ ETA @ F.T                                     # the planes W1, W2 in the Euclidean R^4 of t-perp
    planes = []
    for k in (0, 2):
        planes.append(np.linalg.svd(Wc[k:k + 2])[2][2:])   # orthogonal complement of W_k: the great circle's plane
    return {"map": to_s3, "P1": planes[0], "P2": planes[1], "t": t}


def plane_principal_angles(P1, P2):
    s = np.linalg.svd(np.asarray(P1) @ np.asarray(P2).T, compute_uv=False)
    return np.arccos(np.clip(np.sort(s)[::-1], -1, 1))


def plane_to_s2(P):
    """The (self-dual, anti-self-dual) unit vectors of the bivector of the 2-plane with orthonormal rows P (2x4)."""
    u, v = P
    p = lambda i, j: u[i] * v[j] - u[j] * v[i]
    sd = np.array([p(0, 1) + p(2, 3), p(0, 2) - p(1, 3), p(0, 3) + p(1, 2)])
    ad = np.array([p(0, 1) - p(2, 3), p(0, 2) + p(1, 3), p(0, 3) - p(1, 2)])
    return sd, ad


def plane_separations(P1, P2):
    """Angles between the self-dual parts and between the anti-self-dual parts of two planes, largest first.
    A plane has no orientation; reversing one basis sends both angles alpha -> pi - alpha, so the second plane is oriented
    with  cos(alpha_+) + cos(alpha_-) = 2 cos(theta1) cos(theta2) >= 0, which is the convention of the principal angles
    theta_i in [0, pi/2]."""
    a1, b1 = plane_to_s2(P1)
    a2, b2 = plane_to_s2(P2)
    if a1 @ a2 + b1 @ b2 < 0:
        a2, b2 = -a2, -b2
    return np.array(sorted([math.acos(np.clip(a1 @ a2, -1, 1)), math.acos(np.clip(b1 @ b2, -1, 1))], reverse=True))


def d_family_invariants(d):
    """Closed form for the pair (A, B) with centre distance d: s1 = |1 - d^2/2|, s2 = 0."""
    return np.array([abs(1 - d * d / 2.0), 0.0])


# ---------------------------------------------------------------------------------------------------- quaternions and rolling

def quat_matrices(q):
    """Rotation matrices (v -> q v conj(q)) of unit quaternions q (m, 4) = (a, b, c, d), vectorised."""
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (c * c + d * d)
    R[:, 0, 1] = 2 * (b * c - a * d)
    R[:, 0, 2] = 2 * (b * d + a * c)
    R[:, 1, 0] = 2 * (b * c + a * d)
    R[:, 1, 1] = 1 - 2 * (b * b + d * d)
    R[:, 1, 2] = 2 * (c * d - a * b)
    R[:, 2, 0] = 2 * (b * d - a * c)
    R[:, 2, 1] = 2 * (c * d + a * b)
    R[:, 2, 2] = 1 - 2 * (b * b + c * c)
    return R


def quat_of_matrix(Rm):
    """Unit quaternion (a, b, c, d) with quat_matrices = Rm (one of the two signs)."""
    tr = np.trace(Rm)
    if tr > 0:
        S = math.sqrt(tr + 1) * 2
        q = np.array([S / 4, (Rm[2, 1] - Rm[1, 2]) / S, (Rm[0, 2] - Rm[2, 0]) / S, (Rm[1, 0] - Rm[0, 1]) / S])
    else:
        i = int(np.argmax(np.diag(Rm)))
        j, k = (i + 1) % 3, (i + 2) % 3
        S = math.sqrt(Rm[i, i] - Rm[j, j] - Rm[k, k] + 1) * 2
        v = np.zeros(3)
        v[i] = S / 4
        v[j] = (Rm[j, i] + Rm[i, j]) / S
        v[k] = (Rm[k, i] + Rm[i, k]) / S
        q = np.r_[(Rm[k, j] - Rm[j, k]) / S, v]
    return q / np.linalg.norm(q)


def rotor(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def align_to_down(n):
    """A rotation matrix taking the unit vector n to (0, 0, -1)."""
    target = np.array([0.0, 0.0, -1.0])
    ax = np.cross(n, target)
    s = np.linalg.norm(ax)
    c = float(n @ target)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def roll(N=6000, align=True):
    """Roll the oloid on the plane z = 0 for one period of the generator loop (N steps, the last one closing the loop).

    Returns dict: gens (see generators), q (N+1, 4) the continuous quaternion lift, R (N+1,3,3), x (N+1,3) the world position
    of the body-frame origin, dphi (N,) the rotation angle of each step, e_mid (N,3) its body-frame axis, com (N+1,3) the
    world centre of mass.  With align=True the world is rotated about z so that the displacement per period points along +x."""
    g = generators(N)
    e, n, PA = g["e"], g["n"], g["PA"]
    k2 = (np.arange(N) + 1) % N
    em = e + e[k2]
    em /= np.linalg.norm(em, axis=1)[:, None]
    nm = n + n[k2]
    nm /= np.linalg.norm(nm, axis=1)[:, None]
    dphi = -np.einsum("ij,ij->i", np.cross(em, nm), n[k2] - n)
    pam = 0.5 * (PA + PA[k2])
    q = np.empty((N + 1, 4))
    qmid = np.empty((N, 4))
    q[0] = quat_of_matrix(align_to_down(n[0]))
    for k in range(N):
        qmid[k] = H.qmul(q[k], rotor(em[k], dphi[k] / 2))
        q[k + 1] = H.qmul(q[k], rotor(em[k], dphi[k]))
    Rmid = quat_matrices(qmid)
    dx = -np.einsum("kij,kj->ki", Rmid, np.cross(dphi[:, None] * em, pam))
    x = np.vstack([np.zeros((1, 3)), np.cumsum(dx, axis=0)])
    R = quat_matrices(q)
    x -= R[0] @ PA[0]                                      # put the first contact point A(pi/3) at the origin
    if align:
        com0 = x + np.einsum("kij,j->ki", R, OLOID_CENTRE)
        disp = com0[-1] - com0[0]
        phi = math.atan2(disp[1], disp[0])
        Rz = np.array([[math.cos(-phi), -math.sin(-phi), 0], [math.sin(-phi), math.cos(-phi), 0], [0, 0, 1]])
        qz = rotor([0, 0, 1], -phi)
        R = np.einsum("ij,kjl->kil", Rz, R)
        x = x @ Rz.T
        q = np.array([H.qmul(qz, qq) for qq in q])
    com = x + np.einsum("kij,j->ki", R, OLOID_CENTRE)
    return {"gens": g, "q": q, "R": R, "x": x, "dphi": dphi, "e_mid": em, "com": com}


def hopf_shadow(q, axis=(0.0, 0.0, 1.0)):
    """Hopf image of the lift: the world direction R a of the body axis a.  For a = e_z this is exactly H.hopf(z1, z2)
    with (z1, z2) = (a + d i, c + b i) of q = a + b i + c j + d k; for another axis use the lift q r with r e_z conj(r) = a."""
    a = np.asarray(axis, float)
    return quat_matrices(q) @ a


def solid_angle(P):
    """Signed solid angle (mod 4 pi) enclosed by the closed polygon P (m,3) on S^2: sum of the signed triangle areas
    (p0, P_i, P_{i+1}) from a fixed generic p0 not on the curve."""
    P = np.asarray(P, float)
    P = P / np.linalg.norm(P, axis=1)[:, None]
    for p0 in (np.array([0.3, 0.5, 0.81]), np.array([-0.62, 0.11, -0.77]), np.array([0.05, -0.9, 0.43])):
        p0 = p0 / np.linalg.norm(p0)
        if (P @ p0).max() < 0.9999:
            break
    a, b = P[:-1], P[1:]
    num = a @ np.cross(a, b).T if False else np.einsum("j,ij->i", p0, np.cross(a, b))
    den = 1 + a @ p0 + np.einsum("ij,ij->i", a, b) + b @ p0
    return float(np.sum(2 * np.arctan2(num, den)))


def body_increments(q):
    """Rotation vectors (body frame) of the steps q_k -> q_{k+1} = q_k (cos(d/2), sin(d/2) w)  -> d w, shape (m-1, 3)."""
    out = np.empty((len(q) - 1, 3))
    for k in range(len(q) - 1):
        r = H.qmul(H.qconj(q[k]), q[k + 1])
        v = r[1:]
        nv = np.linalg.norm(v)
        out[k] = 0 * v if nv < 1e-300 else v / nv * 2 * math.atan2(nv, r[0])
    return out


def fibre_phase(q, axis):
    """For a closed loop of unit quaternions whose body-axis shadow R a closes: returns (psi_total, psi_dyn, Omega), all
    reduced mod 4 pi, where q_end = q_0 (cos(psi/2), sin(psi/2) a) is the net rotation about the body axis a, psi_dyn is
    its dynamic part (the integral of the body angular velocity along a) and Omega the solid angle of the shadow.
    The identity  psi_total = psi_dyn + Omega  (mod 4 pi)  is the Berry / Pancharatnam phase = half the solid angle in
    the repo's convention (hopf_quaternion_rotor.py); it is tested on a synthetic loop where it fixes the sign."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    r = H.qmul(H.qconj(q[0]), q[-1])
    if np.linalg.norm(np.cross(r[1:], a)) > 1e-6:
        raise ValueError("the shadow of this axis does not close")
    psi = (2 * math.atan2(r[1:] @ a, r[0])) % (2 * TWO_PI)
    dyn = float((body_increments(q) @ a).sum()) % (2 * TWO_PI)
    Om = solid_angle(quat_matrices(q) @ a) % (2 * TWO_PI)
    return psi, dyn, Om


def wrap4pi(x):
    """x reduced to (-2 pi, 2 pi]."""
    return (x + TWO_PI) % (2 * TWO_PI) - TWO_PI


def com_extrema(com):
    """Number of local extrema of the periodic height series com[:-1, 2] (flat noise below 1e-9 ignored), and their sample indices."""
    z = com[:, 2]
    dz = np.diff(z)
    big = np.abs(dz) > 1e-9
    idx = np.where(big)[0]
    sg = np.sign(dz[idx])
    chg = np.where(sg != np.roll(sg, 1))[0]
    return len(chg), idx[chg]


# ---------------------------------------------------------------------------------------------------- tests

def run_tests(verbose=True):
    rng = np.random.default_rng(11)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    # --- 1. shape
    hs = hull_stats(1.0, 1500)
    check("area of the convex hull = 4 pi (the unit sphere's)", abs(hs["area"] / (4 * math.pi) - 1) < 1e-4,
          "(area/4pi = %.7f)" % (hs["area"] / (4 * math.pi)))
    check("centroid of the hull = (1/2, 0, 0)", np.abs(hs["centroid"] - OLOID_CENTRE).max() < 1e-6,
          "(%s)" % np.array2string(hs["centroid"], precision=7))
    hs2 = hull_stats(1.0, 3000)
    check("volume is stable under refinement (about 3.0524)", abs(hs["volume"] - hs2["volume"]) < 1e-5,
          "(V = %.6f, %.6f)" % (hs["volume"], hs2["volume"]))
    g = generators(6000)
    check("generators are all of length sqrt(3)", np.abs(g["L"] - math.sqrt(3)).max() < 1e-9, "(%.12f)" % g["L"].mean())
    pts = hs["points"]
    viol = max((pts @ g["n"][k] - g["PA"][k] @ g["n"][k]).max() for k in range(0, 6000, 60))
    check("every generator lies in a supporting plane of the hull (so it is on the boundary)", viol < 1e-9,
          "(max violation %.1e)" % viol)
    check("A(t) and B(s) of the generators are on the circles",
          np.abs(np.linalg.norm(g["PB"][:, [0, 2]] - [1, 0], axis=1) - 1).max() < 1e-12)

    # --- 2. the link
    A, B = circles(1.0, 400)
    lk = H.linking_number(A, B)
    check("the two circles of the oloid link once (Hopf link)", abs(abs(lk) - 1) < 0.05, "(Gauss integral %.3f)" % lk)
    check("they are disjoint", H.min_distance(A, B) > 0.1, "(min distance %.3f)" % H.min_distance(A, B))
    A2, B2 = circles(2.5, 400)
    check("moved apart (d = 2.5) they do not link", abs(H.linking_number(A2, B2)) < 0.05)

    # the Lorentz machinery on genuine Hopf fibres
    worst = 0.0
    for (e1, a1, e2, a2) in [(1.0, 0.0, 2.0, 2.0), (0.7, 0.3, 2.5, 4.0), (0.4, 1.0, 0.9, 1.4)]:
        b1, b2 = H.base_point(e1, a1), H.base_point(e2, a2)
        c1, r1, n1, _ = H.fit_circle(H.fibre_r3(b1, 400))
        c2, r2, n2, _ = H.fit_circle(H.fibre_r3(b2, 400))
        s = moebius_invariants((c1, r1, n1), (c2, r2, n2))
        al = math.acos(np.clip(b1 @ b2, -1, 1))
        worst = max(worst, abs(s[0] - math.cos(al / 2)), abs(s[1] - math.cos(al / 2)))
    check("two Hopf fibres: s1 = s2 = cos(base angle / 2)  (Clifford distance = half the base angle)", worst < 1e-9,
          "(max err %.1e)" % worst)

    # the identity behind 'separations': principal angles (t1, t2) -> separations t1 + t2 and |t1 - t2| on the two S^2
    worst = 0.0
    for _ in range(30):
        P = []
        for _k in range(2):
            Mx = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
            P.append(Mx)
        th = np.sort(plane_principal_angles(P[0], P[1]))
        want = separations(th)
        got = plane_separations(P[0], P[1])
        worst = max(worst, np.abs(want - got).max())
    check("2-planes of R^4: separations on the two S^2 factors are t1 + t2 and |t1 - t2|", worst < 1e-9,
          "(max err %.1e)" % worst)

    # explicit Moebius map to great circles
    c1, c2 = circle_data(1.0)
    model = great_circle_model(c1, c2)
    A, B = circles(1.0, 500)
    Y1, Y2 = model["map"](A), model["map"](B)
    e1 = max(abs(np.linalg.norm(Y1, axis=1) - 1).max(), np.abs(Y1 - Y1 @ model["P1"].T @ model["P1"]).max())
    e2 = max(abs(np.linalg.norm(Y2, axis=1) - 1).max(), np.abs(Y2 - Y2 @ model["P2"].T @ model["P2"]).max())
    check("explicit Moebius map sends circle A and circle B to great circles of S^3", max(e1, e2) < 1e-9,
          "(max err %.1e)" % max(e1, e2))
    th = principal_angles(c1, c2)
    th_direct = np.sort(plane_principal_angles(model["P1"], model["P2"]))
    check("principal angles of those great circles (direct SVD) match the Lorentz invariants",
          np.abs(th - th_direct).max() < 1e-9, "(%.4f, %.4f rad)" % tuple(th_direct))
    check("oloid: principal angles are (60 deg, 90 deg)", abs(th[0] - math.pi / 3) < 1e-12 and abs(th[1] - math.pi / 2) < 1e-12)
    sep = separations(th)
    sep_direct = plane_separations(model["P1"], model["P2"])
    check("oloid: separations on the two S^2 are 150 deg and 30 deg (so NOT a Hopf pair)",
          np.abs(sep - sep_direct).max() < 1e-9 and abs(math.degrees(sep[0]) - 150) < 1e-9 and abs(math.degrees(sep[1]) - 30) < 1e-9,
          "(%.3f, %.3f deg)" % tuple(np.degrees(sep_direct)))

    # the d-family
    worst = 0.0
    for d in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.9):
        a, b = circle_data(d)
        worst = max(worst, np.abs(moebius_invariants(a, b) - d_family_invariants(d)).max())
    check("family of centre distance d: s1 = |1 - d^2/2|, s2 = 0", worst < 1e-12, "(max err %.1e)" % worst)
    a, b = circle_data(math.sqrt(2))
    sr = separations(principal_angles(a, b))
    check("d = sqrt(2) (the two-circle roller) IS a Hopf pair, over antipodal base points",
          abs(sr[0] - math.pi) < 1e-7 and sr[1] < 1e-7, "(separations %.4f, %.4f deg)" % tuple(np.degrees(sr)))
    mr = great_circle_model(a, b)
    Pa, Pb = mr["P1"], mr["P2"]
    check("   ... and the model circles are Clifford parallel (equal principal angles, both 90 deg)",
          np.abs(plane_principal_angles(Pa, Pb) - math.pi / 2).max() < 1e-7)
    others = [abs(separations(principal_angles(*circle_data(d)))[1]) for d in (0.5, 0.8, 1.0, 1.2, 1.6, 1.9)]
    check("no other d in (0, 2) gives a Hopf pair (second separation > 0)", min(others) > 0.05,
          "(smallest %.3f rad)" % min(others))

    # --- 3. rolling
    r = roll(6000)
    g, q, R, x, com = r["gens"], r["q"], r["R"], r["x"], r["com"]
    N = len(g["eta"])
    errN = np.abs(np.einsum("kij,kj->ki", R[:-1], g["n"]) - [0, 0, -1]).max()
    check("contact: the world normal along the generator stays (0,0,-1)", errN < 1e-5, "(max err %.1e)" % errN)
    zc = (x[:-1] + np.einsum("kij,kj->ki", R[:-1], g["PA"]))[:, 2]
    zb = (x[:-1] + np.einsum("kij,kj->ki", R[:-1], g["PB"]))[:, 2]
    check("both ends of the contact segment stay on the ground z = 0", max(abs(zc).max(), abs(zb).max()) < 1e-5,
          "(max %.1e)" % max(abs(zc).max(), abs(zb).max()))
    low = min((x[k] + pts @ R[k].T)[:, 2].min() for k in range(0, N, 120))
    check("the oloid never goes below the ground", low > -1e-5, "(lowest point %.1e)" % low)
    # no slipping: the contact points are fixed in the world over the step (|x'| from the integration is consistent)
    v = (x[1:] + np.einsum("kij,kj->ki", R[1:], g["PA"][np.r_[np.arange(1, N), 0]])) - \
        (x[:-1] + np.einsum("kij,kj->ki", R[:-1], g["PA"]))
    # step-to-step the world position of the *moving* contact point advances; the slip is the velocity of a body point
    # momentarily at rest.  Test it directly: the body point PA[k] has (nearly) the same world position before and after
    # the rotation of step k.
    Rm = quat_matrices(np.array([H.qmul(q[k], rotor(r["e_mid"][k], r["dphi"][k] / 2)) for k in range(0, N, 40)]))
    pa_mid = 0.5 * (g["PA"][::40] + g["PA"][(np.arange(0, N, 40) + 1) % N])
    dx = x[1:][::40] - x[:-1][::40]
    slip = np.abs(dx + np.einsum("kij,kj->ki", Rm, np.cross(r["dphi"][::40, None] * r["e_mid"][::40], pa_mid))).max()
    check("no slipping: the contact point has zero velocity", slip < 1e-12, "(%.1e)" % slip)
    check("orientation matrices are those of the quaternion lift (Q.rotation_matrix)",
          np.abs(quat_matrices(q[:3]) - np.array([__import__("hopf_quaternion_rotor").rotation_matrix(qq) for qq in q[:3]])).max() < 1e-12)
    Rend = R[-1] @ R[0].T
    check("after one period the ORIENTATION returns (R_end = R_0)", np.abs(Rend - np.eye(3)).max() < 1e-5,
          "(max err %.1e)" % np.abs(Rend - np.eye(3)).max())
    dq = float(q[-1] @ q[0])
    check("... but the quaternion lift returns as -q (the cycle is the nontrivial loop of SO(3))", abs(dq + 1) < 1e-5,
          "(q_end . q_0 = %.6f)" % dq)
    disp = com[-1] - com[0]
    check("the oloid has moved: displacement per period along +x", disp[0] > 4 and abs(disp[1]) < 1e-6 and abs(disp[2]) < 1e-5,
          "(%.5f vs 8pi/(3 sqrt3) = %.5f)" % (disp[0], 8 * math.pi / (3 * math.sqrt(3))))
    z = com[:, 2]
    check("height of the centre of mass lies in [sqrt(2)/2, 3/4]",
          abs(z.min() - math.sqrt(2) / 2) < 1e-5 and abs(z.max() - 0.75) < 1e-5, "([%.6f, %.6f])" % (z.min(), z.max()))
    ne, idx = com_extrema(com)
    check("8 extrema of the height per period (2 maxima and 2 minima per half period)", ne == 8, "(found %d)" % ne)
    half = np.abs(z[:N // 2] - z[N // 2:N]).max()
    check("height is periodic with HALF the period", half < 1e-6, "(max diff %.1e)" % half)
    lat = com[:, 1] - com[0, 1]
    check("the path meanders sideways about its mean line", lat.max() - lat.min() > 0.1,
          "(lateral excursion %.3f)" % (lat.max() - lat.min()))
    check("lift is continuous (quaternion steps are small)", np.linalg.norm(np.diff(q, axis=0), axis=1).max() < 0.02,
          "(max step %.3f)" % np.linalg.norm(np.diff(q, axis=0), axis=1).max())

    # --- Hopf map of the lift
    z1z2 = [(complex(qq[0], qq[3]), complex(qq[2], qq[1])) for qq in q[::300]]
    err = max(np.abs(H.hopf(a, b) - R[k * 300][:, 2]).max() for k, (a, b) in enumerate(z1z2))
    check("Hopf map of the lift = world direction of the body z axis (normal of circle A)", err < 1e-12, "(max err %.1e)" % err)
    sh = hopf_shadow(q)
    check("the S^2 shadow of the cycle is a closed curve", np.linalg.norm(sh[-1] - sh[0]) < 1e-5)

    # synthetic loop fixes the sign of  psi_total = psi_dyn + Omega
    worst = 0.0
    for a_, gam in [(0.9, 0.0), (0.9, 0.37), (1.7, -1.3), (2.4, 2.0)]:
        m = 4000
        qs = np.array([H.qmul(H.qmul(rotor([0, 0, 1], tt), rotor([0, 1, 0], a_)), rotor([0, 0, 1], gam * tt))
                       for tt in np.linspace(0, TWO_PI, m + 1)])
        psi, dyn, Om = fibre_phase(qs, [0, 0, 1])
        worst = max(worst, abs(wrap4pi(psi - dyn - Om)))
    check("synthetic latitude loop with extra spin: psi_total = psi_dyn + Omega (mod 4 pi)", worst < 1e-3, "(max err %.1e)" % worst)

    # the oloid loop on the three body axes
    out = []
    for name, ax in (("x (line of centres)", [1, 0, 0]), ("y (normal of B)", [0, 1, 0]), ("z (normal of A)", [0, 0, 1])):
        psi, dyn, Om = fibre_phase(q, ax)
        out.append((name, psi, dyn, Om))
        check("oloid cycle, body axis %-19s psi_total = psi_dyn + Omega (mod 4 pi)" % name,
              abs(wrap4pi(psi - dyn - Om)) < 1e-3 and abs(psi - TWO_PI) < 1e-3,
              "(psi = %.4f, dyn = %.4f, Omega = %.4f)" % (psi, dyn, Om))
    return ok


# ---------------------------------------------------------------------------------------------------- report and figure

def summary():
    hs = hull_stats(1.0, 1500)
    print("oloid: area/4pi = %.6f, volume = %.6f, centroid = %s" % (hs["area"] / (4 * math.pi), hs["volume"],
                                                                      np.array2string(hs["centroid"], precision=6)))
    c1, c2 = circle_data(1.0)
    th = principal_angles(c1, c2)
    sep = separations(th)
    print("oloid circle pair: Moebius invariants (s1, s2) = (%.4f, %.4f), principal angles (%.2f, %.2f) deg" %
          (*moebius_invariants(c1, c2), *np.degrees(th)))
    print("  separations on the two S^2 factors: %.2f deg and %.2f deg  (a Hopf pair would have 0 deg for one)" %
          tuple(np.degrees(sep)))
    print("  family d = centre distance (radius 1):   d   s1=|1-d^2/2|   theta1    separations")
    for d in (0.5, 1.0, math.sqrt(2), 1.7):
        a, b = circle_data(d)
        t = principal_angles(a, b)
        print("                                         %.4f  %.4f      %6.2f   %6.2f %6.2f deg" %
              (d, moebius_invariants(a, b)[0], math.degrees(t[0]), *np.degrees(separations(t))))
    r = roll(6000)
    q, com = r["q"], r["com"]
    ne, _ = com_extrema(com)
    print("rolling: displacement per period %.5f, centre of mass height in [%.6f, %.6f], %d extrema per period" %
          (np.linalg.norm(com[-1] - com[0]), com[:, 2].min(), com[:, 2].max(), ne))
    print("  q_end . q_0 = %.6f  (R_end = R_0 but the lift ends at -q_0)" % (q[-1] @ q[0]))
    for name, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        psi, dyn, Om = fibre_phase(q, ax)
        print("  body axis %s: net fibre rotation psi = %.4f = dynamic %.4f + solid angle %.4f (mod 4 pi)" % (name, psi, dyn, Om))


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    r = roll(6000)
    q, R, x, com, g = r["q"], r["R"], r["x"], r["com"], r["gens"]
    N = len(g["eta"])
    fig = plt.figure(figsize=(13, 10))

    # (a) top view of three periods
    ax = fig.add_subplot(2, 2, 1)
    disp = com[-1] - com[0]
    for m in range(3):
        sh = m * disp
        ax.plot(com[:, 0] + sh[0], com[:, 1] + sh[1], color="tab:red", lw=1.5)
        for k in range(0, N, 150):
            pa = x[k] + R[k] @ g["PA"][k] + sh
            pb = x[k] + R[k] @ g["PB"][k] + sh
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cm.viridis(k / N), lw=1)
    ax.set_aspect("equal")
    ax.set_title("rolling on the plane, three periods (top view)\nred: centre of mass; segments: contact line (length sqrt 3)")

    # (b) height of the centre of mass
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(np.linspace(0, 1, N + 1), com[:, 2], color="tab:red")
    ax.axhline(math.sqrt(2) / 2, color="0.6", lw=0.8, ls="--")
    ax.axhline(0.75, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("fraction of the period")
    ax.set_ylabel("height of the centre of mass")
    ax.set_title("8 extrema per period, range [sqrt(2)/2, 3/4]; period of the shape = half")

    # (c) S^2 shadows
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", alpha=0.25, linewidth=0)
    for axis, col, lab in (([0, 0, 1], "tab:blue", "normal of circle A (Hopf map of q)"), ([0, 1, 0], "tab:orange", "normal of circle B")):
        s = quat_matrices(q) @ np.array(axis, float)
        ax.plot(*s.T, color=col, lw=2, label=lab)
        ax.scatter(*s[0], color=col, s=25)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("S^2 shadow of one period: world direction of a body axis")

    # (d) the S^3 lift, two periods, stereographic
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    q2 = np.vstack([q, -q[1:]])
    zz = np.stack([q2[:, 0] + 1j * q2[:, 3], q2[:, 2] + 1j * q2[:, 1]], axis=1)
    P = H.stereographic(zz)
    P[np.linalg.norm(P, axis=1) > 6] = np.nan
    h = len(q)
    ax.plot(*P[:h].T, color="tab:blue", lw=2, label="period 1: q")
    ax.plot(*P[h - 1:].T, color="tab:red", lw=2, label="period 2: -q  (the loop closes only now)")
    ax.scatter(*P[0], color="k", s=30)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("the lift to S^3 (stereographic): one period ends at the antipode -q")
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
