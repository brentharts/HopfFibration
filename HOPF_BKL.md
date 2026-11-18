# The Mixmaster walls as a reflection group, and the inversion family of `ray`

Script: `hopf_bkl.py` (`--tests`; `--trend` for section 3; `--eras` for section 5). Numpy only; the isolated-bounce check and `--trend` need the `brentharts/wada-coudene` clone. Labels: **Exact**, **Counted**, **Known**
(a standard result in the literature that this note uses but did not look up or re-derive), **Idea**.

Source of the connection: `brentharts/ray` (`fractalkit`, paper section 'Inversion groups', `examples/hopf_reflection.py`) studies the ring and its two children as three inversions and finds a one-parameter family of reflection
triangles (m, m, n), with the Hopf clasp at n = 2. The Mixmaster picture of `wada-coudene/MIXMASTER.md` has three walls. This note puts the two side by side.

## 1. The walls of the diagonal Bianchi IX system

Our equations are x_i'' = (q_j - q_k)^2 - q_i^2, q = e^x, with constraint sum_{i<j} x_i' x_j' = V. The kinetic form T(v) = sum_{i<j} v_i v_j has the matrix A = J - I (eigenvalues 2, -1, -1); the form on covectors is minus its inverse,
with entries 1/2 on the diagonal and -1/2 off it, signature (one timelike, two spacelike). So the velocity space with the constraint T = 0 (Kasner) is a Lorentzian light cone, and the billiard lives in the hyperbolic plane.

1. **Exact (`run_tests` 1).** The three curvature walls q_i^2 = e^{2 x_i} (covectors e_i) have normalised Gram matrix with all off-diagonal entries -1: they pairwise meet at angle 0, so they bound an **ideal triangle**, the
   Coxeter group (inf, inf, inf).
2. **Exact (tests 2, 3).** Adding the symmetry walls x_1 = x_2 and x_2 = x_3 gives a triangle with angles pi/3, pi/2 and 0 (walls x_1 = x_2, x_2 = x_3 and curvature wall 3); the reflections have orders 3, 2 and infinity:
   the Coxeter group (2, 3, inf) = PGL(2, Z) (**Known**: Misner, Chitre; Damour, Henneaux and Nicolai for the Kac-Moody reading; we reproduce the triangle from our own equations, which is the new part).
3. **Exact (test 4).** Reflections in a (x_1 = x_2), b (x_2 = x_3) and c b c (c the curvature wall) generate a (3, 3, inf) triangle group, index 2 in the (2, 3, inf) group.
4. **Exact (test 3).** The ideal vertex where curvature walls j and k meet is the null vector e_i: the Kasner state p = (1, 0, 0), in which axis i collapses. **The three cusps of the billiard triangle are the three Mixmaster exits**,
   and the circle at infinity of the hyperbolic plane is the Kasner circle sum p = sum p^2 = 1.

## 2. The family of `ray`

For a parent circle of radius 1 and two children of radius s with centres (c_g, +-off), `ray` finds cos(parent/child) = kappa s and cos(child/child) = 1 - 2(kappa - 1)^2, and the reflection triangle (m, m, n) at
kappa_n = 1 + cos(pi/2n), s_m = cos(pi/m)/kappa_n. Two members of the family are the walls above:

- **Exact (test 5).** m = n = infinity: kappa = 2, s = 1/2. The parent and the two children are three mutually tangent circles (an arbelos), with cosines +1 and -1, and the Lorentz Gram matrix is again all -1, identical to the
  Mixmaster curvature walls. The inversion group of this configuration is the (inf, inf, inf) group of section 1.
- **Exact (test 6).** m = 3, n = infinity: kappa = 2, s = 1/4. The inversions satisfy (r0 r1)^3 = (r0 r2)^3 = 1 to 2e-15 and r1 r2 has no finite order: the (3, 3, inf) subgroup of section 1, point 3.

The Hopf clasp (n = 2, kappa = 1 + 1/sqrt 2) is not in this part of the family; it is at the other end (a right angle between the children) and gives no cusp. What the two have in common is the parametrisation by one tangency/crossing
angle. **That is the whole of the structural match**; the horned-sphere trees and their linking numbers are three-dimensional objects and the Mixmaster billiard is not.

## 3. How far the sharp-wall picture is the dynamics

**Exact in isolation (the isolated-bounce check).** Integrating the Mixmaster equations from a Kasner state far from the other walls, an isolated bounce off wall 3 changes the velocity as v_3 -> -v_3, v_j -> v_j + 2 v_3 to 3e-3 relative
(6 states). This is the reflection in the wall of the Lorentz form.

**Counted, a limit of the picture.** On real trajectories (100 starts on the maximum-expansion slice, 950 consecutive pairs of Kasner epochs) the law fits poorly: median relative error 0.25. It improves only when the other walls
are far from the bounce:

| largest other x_j at the bounce | pairs | median error | 90th percentile |
|---|---|---|---|
| below -4 (to -8) | 7 | 0.064 | 0.11 |
| -4 to -2 | 65 | 0.23 | 0.80 |
| above -2 | 878 | 0.25 | 0.78 |

So for 92 % of the bounces the walls overlap and the billiard is a poor description; the exit rule of `MIXMASTER.md` lives in that regime. The bins below -4 hold only 7 pairs, so the trend is suggestive and not a measurement of a rate.
Epoch pairs are as detected by the minimum of |V| / sum v^2 (below 1e-3), see `trend()`.

## 4. What this says for the Wada question

- **Says.** The three Mixmaster exits are the three cusps of a hyperbolic reflection triangle, and the same triangle is a member of the `ray` family of three inversions. The Lorentzian picture gives the exit labels a geometric home:
  the circle at infinity.
- **Does not say.** That the exit basins are Wada because of this. `ray`'s own paper notes that these groups have round limit circles (Fuchsian), no wild limit set. A sharp-wall billiard in a finite-area triangle recurs and has
  no exits except on a measure zero set, so the three-way exit competition that gives the Wada property comes from the soft walls, which section 3 shows matter.
- **Idea, not done.** (a) The Kasner circle is RP^1 inside CP^1 = S^2; its preimage under the Hopf map is a torus in S^3, so the exit directions of `HOPF_BIANCHI.md` (points of RP^2 over a rotated Mixmaster point) sit over
  it in a way that has not been worked out. (b) The BKL epoch map acts on the circle by the generators above (Farey/Gauss map). The reduced-word coding of the exit basins is now done in `wada-coudene` (`MIXMASTER.md` section 2e): the exit axis is the last letter in 87 % of decided states, the first bounces
  carry almost no information about the exit, and the cylinders of the first four bounces nearly all meet all three basins. The link to the group (the unfolded geodesic's endpoint on the circle) is not computed. (c) `ray` builds three-dimensional trees by recursion on clasped pairs; whether a recursion of *Mixmaster epochs* has the same self-similarity of its link graph is untested.

## 5. The two other forks: `spectre-monotile-py` and `4D_Wilson_Loop_RG`

Read for this section: both READMEs, `GEOMETRY_BRIDGE.md`, `STATUS_MAP.md`, `TILE_E8_BRIDGE.md`, `MATH_GAPS.md` and the Spectre-versus-BKL section of the Spectre README. I did not read the 244 commits, the Lean material or any of the `drafts/` proofs.

**The Spectre fork already has the Mixmaster link at the level of closed orbits.** Its `spectre_cosmology.py` (7 tests, they pass here) shows that the BKL orbit of u* = 3 + sqrt 15 closes in eras of 6 and 1 with multiplier (4 + sqrt 15)^2 = lambda^4, where lambda^2 = 4 + sqrt 15 is the Spectre's inflation factor, and notes that this is true of every periodic orbit and is not special to the Spectre (`MATH_GAPS.md` O8). What this note adds is the same fact in the wall group of section 1:

- **Exact (`run_tests`, closed-orbit check).** The (2,3,inf) reflection group of the Mixmaster walls contains an element with eigenvalue (4 + sqrt 15)^2 = 61.98 (trace 63 in the 3-dimensional representation), and the shortest reduced word found has length 12 (`123213232323`, 48 such words of that length). It contains a run of alternating walls, as an era of length 6 would. So the Spectre's inflation factor is a stretching factor of a closed orbit of *our* billiard group; as in the Spectre fork's own note, that holds for every hyperbolic element and so says nothing special about the Spectre.
- **Exact (`run_tests`, Hurwitz check).** The 24 Hurwitz units, which are the units of the Hurwitz-pair model of E8 in the Spectre fork (`e6_fibres.py`), lie on 6 fibres of 4 in each of the three left-invariant Hopf fibrations of `HOPF_BIANCHI.md`, and conjugation by (1 + i + j + k)/2 (the three-fold symmetry of the exits) carries family i to j to k. So the symmetry of the Mixmaster exits acts on the 24-cell that sits under the Spectre fork's E8 model. That is all that is shown; no relation of the exits to E8 or to the tiling is claimed.
- **Counted, negative (`python3 hopf_bkl.py --eras`).** Era lengths in the wall words of `MIXMASTER.md` 2e, taken as maximal runs of two alternating walls, have a distribution close to the Gauss-Kuzmin law of the BKL map if every run counts (total variation distance 0.044 over r = 2 to 11, with an offset of one letter chosen after seeing the data). The agreement does not survive: dropping the first and last run of each word, which are cut by the start of the integration and by the exit, leaves 1956 interior runs, 86 % of them of length 2, at distance 0.38. The long runs are the truncated ones. With walls that overlap (section 3) the interior of a soft-wall trajectory does not show BKL eras.
- **Counted, negative (a window test that cannot be run).** The Spectre's 4D lift has a window with a fractal boundary in a contracting internal plane (`lift4d.py`). I asked whether the six complete vertex types of the tiling (the angle sets (120,240), (90,270), (120,120,120), (90,90,180), (90,90,90,90), (180,180); 201 128 vertices of a level-5 patch) partition that plane into regions, which would allow the Wada profile of this repository to be applied. They do not: on a 96 x 96 raster of the internal plane only 32 % of the occupied pixels hold a single vertex type (median majority share 0.60). In this plane the vertex type is not a function of the internal position, so there is no labelled partition to test. This is a statement about this plane and this choice of labels; the Spectre fork notes that its plane is fitted, not derived from a model-set theorem.

**The Wilson-loop fork has no link to this work that I could find.** It is a programme on Balaban's renormalisation group for four-dimensional Yang-Mills (renormalised Wilson loops, perimeter and corner counterterms), whose README states that the documents are LLM generated and mostly unverified; I make no statement on its mathematics. Its `GEOMETRY_BRIDGE.md` connects only to the Spectre fork (E8, Hopf fibres of E8, hexagonal corner types), and says that none of it supports or refutes a step of the programme. The only common ground with this repository is the topic of `PHYSICS.md` section 4 (the mass gap), where my notes call for a mapping of Wada basins in homogeneous Yang-Mills, which is not related to loop renormalisation. An exit-basin computation for homogeneous SU(2) Yang-Mills (the x^2 y^2 potential) is the nearest thing I can name; I did not do it, and its valleys do not give clean exits, so it would need a definition first.

