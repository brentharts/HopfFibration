#!/usr/bin/env python3
"""hopf_quaternionic.py - the quaternionic Hopf fibration S^3 -> S^7 -> S^4 (the second of the four Hopf fibrations).

Points of S^7 are pairs (q1, q2) of quaternions with |q1|^2 + |q2|^2 = 1.  The map to S^4 in H x R = R^5 is

    h(q1, q2) = ( 2 q1 conj(q2) ,  |q1|^2 - |q2|^2 ),

the same formula as the complex Hopf map with complex numbers replaced by quaternions (hopf_math.hopf).  The fibre over a point is the
S^3 of unit quaternions u acting by RIGHT multiplication, (q1 u, q2 u): since q1 u conj(q2 u) = q1 u conj(u) conj(q2) = q1 conj(q2).

What is checked (exact up to floating point):
  1. h lands on the unit 4-sphere; it is surjective in the sense that a section (1, m)/sqrt(1+|m|^2) hits the point stereographic in m.
  2. the fibre is invariant: h(q1 u, q2 u) = h(q1, q2) for every unit quaternion u, and the fibre is a full S^3 (the orbit of u has dimension 3).
  3. NON-COMMUTATIVITY matters: LEFT multiplication (u q1, u q2) does NOT preserve h in general; right multiplication does.
  4. RESTRICTION: with q1, q2 complex (in span{1, i}) h reduces to the complex Hopf map (hopf_math.hopf) and the image lies in the S^2 inside S^4;
     the complex fibres are the circles inside the S^3 fibres.
  5. COMPOSITION: the S^3 fibre through a point of the S^3 in S^7 (q2 = 0 subset) is the point itself (h = north pole), and (q1, q2) with q1 q2 on a
     complex line gives a copy of the S^1 -> S^3 -> S^2 fibration inside S^3 -> S^7 -> S^4.
  6. TOPOLOGY (numerical): the tautological quaternionic line bundle over S^4 = HP^1 has |second Chern number| = 1; the instanton number
     (1/8 pi^2) * integral of tr F^F, with F = P dP dP for the real rank-4 projector P of the line, is computed on a grid in a stereographic chart
     and equals 2 * 1 in magnitude (the real 4x4 trace of an sp(1) curvature is twice the complex 2x2 one).  This one is Counted, not Exact
     (4th-order differences, grid error about 1-2 per cent).

    python3 hopf_quaternionic.py --tests
    python3 hopf_quaternionic.py --instanton        (the integral, slow-ish)
"""
import math
import sys

import numpy as np

import hopf_math as H


def qm(p, q):
    """Quaternion product of arrays with the component axis FIRST: shapes (4, ...)."""
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def qc(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0]).reshape((4,) + (1,) * (np.ndim(q) - 1))


def hopf_s7(q1, q2):
    """Quaternionic Hopf map; q1, q2 shape (4, ...). Returns shape (5, ...): (2 q1 conj(q2) [4 comps], |q1|^2 - |q2|^2)."""
    w = 2 * qm(q1, qc(q2))
    return np.concatenate([w, (np.sum(q1 ** 2, 0) - np.sum(q2 ** 2, 0))[None]], axis=0)


def random_s7(rng, n=1):
    v = rng.normal(size=(8, n))
    v /= np.linalg.norm(v, axis=0)
    return v[:4], v[4:]


def random_unit_quaternion(rng, n=1):
    u = rng.normal(size=(4, n))
    return u / np.linalg.norm(u, axis=0)


def section(m):
    """A point of S^7 over the point of S^4 whose stereographic coordinate is the quaternion m: (1, m)/sqrt(1+|m|^2)."""
    m = np.asarray(m, float)
    one = np.zeros_like(m)
    one[0] = 1.0
    s = 1 / np.sqrt(1 + np.sum(m ** 2, 0))
    return one * s, m * s


def line_projector(m):
    """Real rank-4 orthogonal projector (8x8) onto the quaternionic line {(u, m u)} in R^8, m of shape (4, ...). Returns shape (..., 8, 8)."""
    q1 = np.zeros_like(m)
    q1[0] = 1.0
    cols = []
    for k in range(4):
        u = np.zeros_like(m)
        u[k] = 1.0
        cols.append(np.concatenate([qm(q1, u), qm(m, u)], axis=0))     # right multiplication by 1, i, j, k
    E = np.stack(cols, axis=-1)                                         # (8, ..., 4)
    E = np.moveaxis(E, 0, -2)                                           # (..., 8, 4)
    return np.einsum("...ik,...jk->...ij", E, E) / (1 + np.sum(m ** 2, 0))[..., None, None]


def instanton_number(L=6.0, n=41):
    """(1/8 pi^2) int tr(F^F) over R^4 (m coordinates) for the line bundle's Grassmann connection F_{mn} = P [d_m P, d_n P]; grid n^4."""
    xs = np.linspace(-L, L, n)
    h = xs[1] - xs[0]
    grid3 = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"))            # (3, n, n, n)

    def P_at(i):
        m = np.concatenate([np.full((1, n, n, n), xs[i]), grid3], axis=0)
        return line_projector(m)

    cache = {}

    def P(i):
        if i not in cache:
            cache[i] = P_at(i)
            for k in [k for k in cache if k < i - 3]:
                del cache[k]
        return cache[i]

    def d4(f, a):
        """4th-order central difference along array axis a (periodic wrap; the 2 border layers are excluded below)."""
        return (-np.roll(f, -2, a) + 8 * np.roll(f, -1, a) - 8 * np.roll(f, 1, a) + np.roll(f, 2, a)) / (12 * h)

    total = 0.0
    for i in range(2, n - 2):
        dP = [(-P(i + 2) + 8 * P(i + 1) - 8 * P(i - 1) + P(i - 2)) / (12 * h)] + [d4(P(i), a) for a in (0, 1, 2)]
        Pi = P(i)

        def F(a, b):
            return Pi @ (dP[a] @ dP[b] - dP[b] @ dP[a])

        dens = 2 * (np.trace(F(0, 1) @ F(2, 3), axis1=-2, axis2=-1) - np.trace(F(0, 2) @ F(1, 3), axis1=-2, axis2=-1)
                    + np.trace(F(0, 3) @ F(1, 2), axis1=-2, axis2=-1))
        total += dens[2:-2, 2:-2, 2:-2].sum() * h ** 4
    return total / (8 * math.pi ** 2)


def run_tests(verbose=True, instanton=True):
    rng = np.random.default_rng(21)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    q1, q2 = random_s7(rng, 500)
    b = hopf_s7(q1, q2)
    check("h maps S^7 onto the unit 4-sphere", np.abs(np.sum(b ** 2, 0) - 1).max() < 1e-12)
    m = rng.normal(size=(4, 200))
    s1, s2 = section(m)
    b = hopf_s7(s1, s2)
    # stereographic point of S^4: (2m, |m|^2 - 1)/(|m|^2+1) up to the orientation/convention of the formula: here h = (2 conj-type, ...)
    pred = np.concatenate([2 * qm(s1, qc(s2)), (np.sum(s1 ** 2, 0) - np.sum(s2 ** 2, 0))[None]])
    check("the section (1, m)/sqrt(1+|m|^2) maps to a unit vector with last coordinate (1-|m|^2)/(1+|m|^2)",
          np.abs(b - pred).max() < 1e-12 and np.abs(b[4] - (1 - np.sum(m ** 2, 0)) / (1 + np.sum(m ** 2, 0))).max() < 1e-12)
    u = random_unit_quaternion(rng, 500)
    r1, r2 = qm(q1, u), qm(q2, u)
    check("right multiplication by a unit quaternion preserves h (the fibre action)", np.abs(hopf_s7(r1, r2) - hopf_s7(q1, q2)).max() < 1e-12)
    l1, l2 = qm(u, q1), qm(u, q2)
    check("left multiplication does NOT preserve h (non-commutativity)", np.abs(hopf_s7(l1, l2) - hopf_s7(q1, q2)).max() > 0.1)
    # fibre is a full S^3: the 4 right-translates by 1,i,j,k are linearly independent
    p = np.concatenate([q1[:, :1], q2[:, :1]], axis=0)
    cols = []
    for k in range(4):
        e = np.zeros((4, 1)); e[k] = 1
        cols.append(np.concatenate([qm(q1[:, :1], e), qm(q2[:, :1], e)], axis=0)[:, 0])
    check("the fibre through a point spans R^4 (it is a 3-sphere, not a circle)", np.linalg.matrix_rank(np.array(cols), tol=1e-9) == 4)
    # restriction to complex numbers: q = a + b i  ->  components (a, b, 0, 0)
    z1 = rng.normal(size=100) + 1j * rng.normal(size=100)
    z2 = rng.normal(size=100) + 1j * rng.normal(size=100)
    nrm = np.sqrt(abs(z1) ** 2 + abs(z2) ** 2)
    z1, z2 = z1 / nrm, z2 / nrm
    c1 = np.stack([z1.real, z1.imag, 0 * z1.real, 0 * z1.real]); c2 = np.stack([z2.real, z2.imag, 0 * z1.real, 0 * z1.real])
    bq = hopf_s7(c1, c2)
    hc = H.hopf(z1, z2)
    check("restricted to complex q1, q2, h reduces to the complex Hopf map (hopf_math.hopf)",
          np.abs(bq[0] - hc[:, 0]).max() < 1e-12 and np.abs(bq[1] - hc[:, 1]).max() < 1e-12 and np.abs(bq[4] - hc[:, 2]).max() < 1e-12
          and np.abs(bq[2:4]).max() < 1e-12)
    # q2 = 0 : north pole of S^4, the whole S^3 maps to it
    w1 = random_unit_quaternion(rng, 50)
    nb = hopf_s7(w1, np.zeros_like(w1))
    check("q2 = 0 (an S^3 inside S^7) is one fibre: it maps to the single point (0,0,0,0,1)", np.abs(nb - np.array([[0], [0], [0], [0], [1]])).max() < 1e-12)
    # projector sanity (used by the topology computation)
    Pm = line_projector(rng.normal(size=(4, 20)))
    check("line projector is an orthogonal projector of rank 4", np.abs(Pm @ Pm - Pm).max() < 1e-12 and np.abs(np.trace(Pm, axis1=-2, axis2=-1) - 4).max() < 1e-12)
    if instanton:
        I = instanton_number()
        check("instanton number: |(1/8 pi^2) int tr F^F| = 2 (= 2 x |c2| = 2 x 1 for the real 4x4 trace), 4th-order differences, grid error about 1-2 per cent", abs(abs(I) - 2) < 0.1, "(computed %.3f)" % I)
    return ok


if __name__ == "__main__":
    if "--instanton" in sys.argv:
        print("instanton number %.4f" % instanton_number())
    else:
        sys.exit(0 if run_tests() else 1)
