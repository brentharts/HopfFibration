# Wada basins lifted through the Hopf map

Script: `hopf_wada.py` (numpy + scipy, 10 checks in `--tests`, about 6 s). `python3 hopf_wada.py` prints the 3D numbers, `--png FILE` draws the figure.
The 3D checks import `lakes_of_wada.wadatest` from a clone of [brentharts/wada-coudene](https://github.com/brentharts/wada-coudene)
(`WADA_REPO`, default `../wada-coudene`); without it they are skipped and the rest still runs.
Labels as in the rest of this repository: **Exact**, **Counted**, **Quoted**, **Idea**.

![Newton basins of z^3 - 1 on S^2, on a Poincare half-plane in R^3, and as Hopf fibres](figures/hopf_wada.png?raw=true)

*Left: Newton basins of z^3 - 1 on the base S^2 (w = z1/z2). Middle: the same labels on the half-plane y = 0, x > 0 of R^3, a Poincare section
that every fibre crosses once (`HOPF_FIGURES.md`, figure 2). The fine ripples along the axis rho = 0 are unresolved Julia-set detail: the axis is the fibre over w = 0, which lies in J.
Right: Hopf fibres over points of the three basins (colour) and over Julia-set points (black).*

## 1. The statement

Newton's method for z^3 - 1 on the Riemann sphere CP^1 = S^2 has three basins with a common boundary, the Julia set J: the standard Wada
example (**Quoted**, e.g. the [Yale Newton basins pages](https://gauss.math.yale.edu/public_html/People/frame/Fractals/MandelSet/ComplexNewton/NewtonBasins/Basins3.html)).
Give a point (z1, z2) of S^3 the label of w = z1/z2. This is constant on Hopf fibres, so it is the pull-back of the S^2 label along the Hopf map h
(`hopf_math` convention: the base point (x, y, z) has w = (x + iy)/(1 - z)).

**Claim A (Exact; our own short proof, not checked against a source).** h is a continuous, open, surjective map. For such a map, for every U in S^2,
closure(h^-1 U) = h^-1 closure(U), and interior(h^-1 U) = h^-1 interior(U) (the inclusion "interior of the preimage is inside the preimage of the interior" is the
openness: if V is open inside h^-1 U then h(V) is open inside U; the other is continuity). Hence

    boundary( h^-1 U ) = h^-1 ( boundary U ).

So the three lifted basins are disjoint open sets of S^3 whose boundaries all equal h^-1(J), locally J x S^1: every boundary point of one is a boundary point
of the other two. **Wada lifts through the Hopf map**, and the same argument works for any fibre bundle projection.

**Claim B (Exact, quantitative).** h is a Riemannian submersion of the unit S^3 onto S^2 of radius 1/2, so for x in S^3 and A in S^2

    dist_S3( x, h^-1 A ) = dist_S2( h(x), A ) / 2     (S^2 distance as an angle on the unit sphere).

This is the "Clifford distance equals half the base angle" of `hopf_clifford.py`, applied to a whole set. So the radius needed to see all three lakes from a
boundary point of the lifted picture is exactly half the radius needed on S^2: the lift neither creates nor destroys intermingling at any scale.

What is *not* claimed: that the lifted sets are lakes of Wada in the original sense (accessible by canals); that the result says anything about Hopf-fibre structure
of the Julia set beyond the product J x S^1 (the S^1 is the fibre; nothing here is twisted by the bundle, since the label ignores the fibre coordinate).

## 2. What `hopf_wada.py --tests` checks

| # | check | label | result |
|---|---|---|---|
| 1 | roots label themselves; z^3 - 1 three-fold symmetry of the labelling | Exact | passes |
| 2 | label_s3 = label_s2 o h on 4000 random points of S^3; constant on fibres | Counted | agreement 1.0000 |
| 3 | dist_S3(x, fibre over b') = (angle h(x), b')/2, minimised over the fibre phase numerically, 40 random pairs | Exact | max error 3e-14 |
| 4 | claim B for lifted basins: nearest-sample distance on S^3 (600k points) against nearest-sample distance on S^2 (300k points) over two | Counted | S^3 value is never below S^2/2, and at most 0.036 above (sampling error) |
| 5 | 3D voxel picture (stereographic R^3, box [-3, 3]^3, 96^3) has exactly three basins | Counted | K = 3, 230 024 boundary voxels |
| 6 | the Wada repo's own `wada_profile` on the 3D array: fraction of boundary voxels that see all three basins within r voxels | Counted | 0.34, 0.89, 1.00, 1.00 for r = 1, 2, 4, 8 |
| 7 | `wada_radius` (radius needed to see every basin), in box units, as resolution is refined 48 to 96 to 128 | Counted | median 0.177, 0.088, 0.066; 90th percentile 0.28, 0.14, 0.105: shrinks like 1/resolution (checked 48 to 96 in the tests) |
| 8 | claim A at voxel scale: for 400 random boundary voxels, a disc of angle 0.43 around h(x) on S^2 sees at least two basins | Counted | 400 of 400 |

How to read 6 and 7: for an ordinary partition the profile stays far below 1 and the needed radius stalls; here it reaches 1 and shrinks with the pixel size, as it does
for the 2D Newton picture in the Wada repo (`figures/newton.png`, `radius_scaling.png`). This is evidence at the pixel scale, as in that repository, not a proof; the proof is claim A.

## 3. Caveats

- **The voxel chart is stereographic.** Voxels near infinity cover little of S^3 and voxels near the origin cover most of it; the Wada profile is in voxels, not in S^3 distance.
  Claim B is the metric statement and is tested separately (checks 3, 4).
- **230 024 boundary voxels is 26 % of the box.** Newton's boundary is dense at 96^3: that is a statement about resolution, not a sign of a problem.
- **The pictures are of one example.** Newton for z^3 - 1 is the Wada example with an exact Julia set; nothing here uses the Hopf fibre twisting, because the label is constant on fibres.
  The lift is topologically a product over any contractible set of S^2.
- **The right panel is not a visualisation of the lakes**, only of fibres over their points; the solid regions would need a volume rendering of the voxel labels.

## 4. What this opens (Ideas, not done)

1. **A non-trivial lift.** Take a dynamical system on S^3 that is *not* constant on fibres but covers a map of S^2 (equivariant maps (z1, z2) goes to (P, Q) with P, Q homogeneous of degree d, as in `hopf_knots.py`),
   and ask whether the fibre direction mixes the basins. The label of this note is the case where it does not.
2. **Hopf lift of a Cantor Julia set.** If J is a Cantor set (c outside the Mandelbrot set), h^-1(J) = J x S^1 and the complement has pi_1 equal to F_infinity x Z
   (item 3 of `HOPF_OLOID_AND_NECKLACES.md` section 4); that would give a tame Cantor-type set with Wada-like complement structure to compare with Antoine's necklace.
3. **Physical Wada sets on S^2.** Any S^2-valued exit problem (the celestial sphere of the Majumdar-Papapetrou binary shadow of `PHYSICS.md`) lifts the same way, with the
   fibre as the photon's polarization phase (item 4 of our earlier list; the factor between the fibre phase and the polarization angle is unchecked).
