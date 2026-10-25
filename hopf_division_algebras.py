#!/usr/bin/env python3
"""hopf_division_algebras.py - the four Hopf fibrations from the four normed division algebras R, C, H, O.

                S^0 -> S^1  -> S^1         (R, dim 1)
                S^1 -> S^3  -> S^2         (C, dim 2)
                S^3 -> S^7  -> S^4         (H, dim 4)
                S^7 -> S^15 -> S^8         (O, dim 8)

One formula covers all four:  F(x, y) = ( |x|^2 - |y|^2 , 2 x conj(y) )  on pairs (x, y) of the algebra with |x|^2 + |y|^2 = 1.
The fibre over the base point with 'slope' m is {(x, m x)}: for any x this gives F = (|x|^2 (1 - |m|^2), 2 |x|^2 conj(m)), which
depends on m and |x|^2 only.  (That needs  x (conj(x) y) = |x|^2 y, true in all four algebras: they are ALTERNATIVE.)

Algebras are built by the Cayley-Dickson doubling, (a, b)(c, d) = (ac - conj(d) b, da + b conj(c)).

What is checked:
  1. |x y| = |x||y| for dims 1, 2, 4, 8 (division algebras) and it FAILS at dim 16 (the sedenions have zero divisors, so the
     construction stops: this is the algebraic shadow of Hurwitz's theorem; Adams proved no other sphere fibrations exist).
  2. commutativity fails at dim 4, associativity fails at dim 8, the alternative law holds at dim 8.
  3. F lands on the sphere S^m (m = dim of the algebra), and its fibres have dimension 0, 1, 3, 7 (numerical Jacobian rank).
  4. the fibre over m is {(x, m x)} in all four algebras.
  5. For C and H the fibre is also {(x u, y u)} (right multiplication by a unit u).  For O it is NOT: (x u, y u) leaves the
     fibre, because octonions are not associative.  This is why S^15 -> S^8 is a fibre bundle but not a principal bundle.

    python3 hopf_division_algebras.py --tests
"""
import sys

import numpy as np


def conj(x):
    x = np.array(x, float)
    out = -x
    out[0] = x[0]
    return out


def mul(x, y):
    """Cayley-Dickson product of two real vectors of the same power-of-2 length."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([mul(a, c) - mul(conj(d), b), mul(d, a) + mul(b, conj(c))])


def norm2(x):
    return float(np.dot(x, x))


def hopf_F(x, y):
    """(|x|^2 - |y|^2, 2 x conj(y)) as a vector of length dim + 1."""
    return np.concatenate([[norm2(x) - norm2(y)], 2 * mul(x, conj(y))])


def _unit(rng, n):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def fibre_dimension(dim, rng, eps=1e-6):
    """Dimension of the fibre of F restricted to S^{2 dim - 1}, from the rank of the numerical Jacobian."""
    n = 2 * dim
    p = _unit(rng, n)

    def f(v):
        return hopf_F(v[:dim], v[dim:])
    J = np.array([(f(p + eps * e) - f(p - eps * e)) / (2 * eps) for e in np.eye(n)]).T      # (dim+1) x n
    # tangent space of the sphere at p
    Q, _ = np.linalg.qr(np.c_[p, rng.normal(size=(n, n - 1))])
    T = Q[:, 1:]
    s = np.linalg.svd(J @ T, compute_uv=False)
    rank = int((s > 1e-6).sum())
    return (n - 1) - rank


def run_tests(verbose=True):
    rng = np.random.default_rng(5)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    for dim in (1, 2, 4, 8):
        err = max(abs(np.linalg.norm(mul(x, y)) - np.linalg.norm(x) * np.linalg.norm(y))
                  for x, y in ((rng.normal(size=dim), rng.normal(size=dim)) for _ in range(50)))
        check("dim %-2d: |xy| = |x||y| (normed division algebra)" % dim, err < 1e-12, "(max err %.1e)" % err)
    err16 = max(abs(np.linalg.norm(mul(x, y)) - np.linalg.norm(x) * np.linalg.norm(y))
                for x, y in ((rng.normal(size=16), rng.normal(size=16)) for _ in range(50)))
    check("dim 16 (sedenions): |xy| = |x||y| FAILS, the doubling stops", err16 > 1e-3, "(max err %.2f)" % err16)

    def violation(dim, f):
        return max(f(*(rng.normal(size=dim) for _ in range(3))) for _ in range(20))
    comm = lambda a, b, c: np.linalg.norm(mul(a, b) - mul(b, a))
    assoc = lambda a, b, c: np.linalg.norm(mul(mul(a, b), c) - mul(a, mul(b, c)))
    alt = lambda a, b, c: np.linalg.norm(mul(mul(a, a), b) - mul(a, mul(a, b)))
    check("C is commutative, H is not", violation(2, comm) < 1e-12 and violation(4, comm) > 1e-3)
    check("H is associative, O is not", violation(4, assoc) < 1e-12 and violation(8, assoc) > 1e-3)
    check("O is alternative: (a a) b = a (a b)", violation(8, alt) < 1e-12)

    for dim in (1, 2, 4, 8):
        x, y = rng.normal(size=dim), rng.normal(size=dim)
        s = np.sqrt(norm2(x) + norm2(y))
        x, y = x / s, y / s
        check("dim %-2d: F lands on S^%d" % (dim, dim), abs(np.linalg.norm(hopf_F(x, y)) - 1) < 1e-12)
    fd = [fibre_dimension(d, rng) for d in (1, 2, 4, 8)]
    check("fibre dimensions 0, 1, 3, 7  (S^0, S^1, S^3, S^7)", fd == [0, 1, 3, 7], "(%s)" % fd)

    for dim in (1, 2, 4, 8):
        m = rng.normal(size=dim)
        vals = []
        for _ in range(5):
            x = rng.normal(size=dim)
            x = x / np.linalg.norm(x) / np.sqrt(1 + norm2(m))          # |x|^2 (1 + |m|^2) = 1
            vals.append(hopf_F(x, mul(m, x)))
        check("dim %-2d: fibre over m is {(x, m x)}: F is the same for every x" % dim,
              max(np.abs(v - vals[0]).max() for v in vals) < 1e-12 and abs(np.linalg.norm(vals[0]) - 1) < 1e-12)

    def right_mult_defect(dim):
        worst = 0.0
        for _ in range(20):
            x, y = _unit(rng, dim), _unit(rng, dim)
            u = _unit(rng, dim)
            n = np.sqrt(norm2(x) + norm2(y))
            x, y = x / n, y / n
            worst = max(worst, np.abs(hopf_F(mul(x, u), mul(y, u)) - hopf_F(x, y)).max())
        return worst
    check("C and H: (x u, y u) stays in the fibre", right_mult_defect(2) < 1e-12 and right_mult_defect(4) < 1e-12)
    d8 = right_mult_defect(8)
    check("O: (x u, y u) LEAVES the fibre (non-associative: not a principal bundle)", d8 > 1e-3, "(defect %.2f)" % d8)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
