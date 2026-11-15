# The oloid and Antoine's necklace against the Hopf fibration

Scripts: `hopf_oloid.py` (numpy + scipy, 38 checks in `--tests`, about 1.5 s) and `hopf_antoine.py` (numpy + scipy, 26 checks, about 3 s).
`python3 hopf_oloid.py` and `python3 hopf_antoine.py` print summaries; `--png FILE` draws the figures below; `python3 hopf_tests.py` runs everything.
Last run: 15 Nov 2026, all tests pass.

Conventions are those of `hopf_math.py` (a point of S³ is (z₁, z₂) ∈ ℂ² with |z₁|²+|z₂|² = 1, the Hopf map is h = (2 Re z₁z̄₂, 2 Im z₁z̄₂, |z₁|²−|z₂|²),
a fibre is e^{it}(z₁, z₂), stereographic projection is from (0,0,0,1)). A *circle* below is always a round circle in ℝ³ ∪ {∞}, given as
(centre, radius, unit normal).

Labels, extending those of [`HOPF_FIGURES.md`](HOPF_FIGURES.md):

- **Exact**: an identity checked to rounding.
- **Counted**: numerical with finite sampling (linking numbers are midpoint-rule Gauss integrals, used only to tell 0 from ±1; area and volume are those of a convex hull of sample points).
- **Observed**: found numerically to many digits, not proved.
- **Quoted**: a result from a source in the [References](#references), not re-derived here.
- **Idea**: a possible connection that nothing here tests. These are all in [section 4](#4-possible-new-connections).

**No source we have found connects the oloid or the horned sphere to the Hopf fibration.** Everything in sections 2 and 3 is a computation in this repository;
the interpretation is ours.

## 1. What was found, in one table

| # | statement | label | where |
|---|---|---|---|
| 1 | the two circles of the oloid form a Hopf link (linking number ±1) | Counted | `hopf_oloid.py` |
| 2 | as a pair of circles up to Möbius maps the oloid has principal angles (60°, 90°), i.e. separations 150° and 30° on the two S² factors of the plane pair: **not** a Hopf pair | Exact | `hopf_oloid.py` |
| 3 | for two circles of radius 1 in perpendicular planes, centres d apart, the Möbius invariants are (\|1 − d²/2\|, 0); the pair is an exact Hopf pair iff d = √2 (the two-circle roller), over antipodal base points | Exact | `hopf_oloid.py` |
| 4 | one rolling period of the oloid returns the orientation R but its lift to S³ ends at −q: the cycle is the non-trivial loop of SO(3) | Exact up to the discretisation (1e-6) | `hopf_oloid.py` |
| 5 | on every body axis the net fibre rotation of that loop is 2π and equals (dynamic part) + (solid angle of the S² shadow) mod 4π | Counted | `hopf_oloid.py` |
| 6 | Hopf fibres link in every pair (complete graph); a necklace of round circles links only neighbours (cycle graph) | Counted | `hopf_antoine.py` |
| 7 | a straight chain with centre spacing d has the oloid-family clasp at every link; second neighbours touch exactly at d = 1 (the oloid) | Exact | `hopf_antoine.py` |
| 8 | a closed ring of k flat/vertical links is **not** a Hopf chain (defect 31°, 16°, 8° at k = 8, 16, 32) | Exact | `hopf_antoine.py` |
| 9 | after twisting the link planes about the tangent, every clasp of the ring is an exact Hopf pair while only neighbours link | Exact (twist solved to 1e-12) | `hopf_antoine.py` |
| 10 | at σ = √2 the invariant of that clasp is s = tan(π/k), base angle 2 arccos tan(π/k) | Observed (agreement 5e-16 for k = 8, 12, 16, 24, 32) | `hopf_antoine.py` |
| 11 | no single Hopf fibration contains three consecutive circles of a chain | Counted | `hopf_antoine.py` |
| 12 | replacing every circle by a scaled ring chain along it keeps the clasp invariants unchanged (self-similarity); generation 2 (576 circles) has exactly the nesting linking pattern | Exact (2e-13) / Counted | `hopf_antoine.py` |

What is **not** done: nothing here builds the oloid's or the necklace's solid tori, the limit Cantor set, the horned sphere itself, or the fundamental
group of any complement. See [section 3.5](#35-generations-and-what-is-not-shown).

## 2. The oloid

### 2.1 What the sources say

**Quoted** from [Oloid (Wikipedia)](https://en.wikipedia.org/wiki/Oloid): the oloid is the convex hull of two linked, congruent circles lying in perpendicular planes,
each centre lying on the other circle (so the centres are one radius apart). Each circle contributes two thirds of its length to the boundary, two arcs of 4π/3.
The surface area equals that of a sphere of the same radius. The surface is developable and, as it rolls on a plane, it touches along a line segment of constant length;
the centre of mass follows a meandering path with two maxima and two minima of height per rolling cycle. The related two-circle roller (1966) has centre distance √2
times the radius, and then its centre of gravity stays at constant height.

### 2.2 The shape

Circle A: centre 0, radius 1, in the plane z = 0. Circle B: centre (d, 0, 0), radius 1, in the plane y = 0. The oloid is d = 1.

- **Counted.** Hull of 2 × 1500 sample points: area / 4π = 0.999998, volume 3.0524 (3.052409 and 3.052416 at 1500 and 3000 points), centroid (1/2, 0, 0).
- **Exact.** The straight segment A(t)B(s) lies in a common tangent plane of both circles iff cos s = cos t / (1 − cos t) (coplanarity of A′(t), B′(s), B(s) − A(t)).
  This needs cos t ≤ ½, i.e. t ∈ [π/3, 5π/3], an arc of 4π/3 as quoted above. Every such segment has length √3 (checked to 1e-9), and each lies in a supporting plane of the hull.
- **Counted.** The Gauss linking number of A and B is −1.000: a Hopf link.

### 2.3 The circle pair as a Möbius object

Stereographic projection preserves round circles, and Möbius maps are exactly the conformal maps of S³. So the right question about a pair of circles is whether
it is a pair of Hopf fibres *up to a Möbius map*.

In the Lorentz model (ℝ^{4,1}, a sphere is a unit spacelike vector, a circle is the 2-plane W of the spheres through it) a pair of circles has two Möbius invariants,
the singular values s₁ ≥ s₂ of the Lorentz Gram matrix of orthonormal bases of W₁, W₂. Write sᵢ = cos θᵢ. When both sᵢ < 1 an explicit Möbius map
(`hopf_oloid.great_circle_model`) sends both circles to great circles of S³ whose planes have principal angles θ₁, θ₂ (**Exact**, checked on sample points to 1e-9).

Two great circles are fibres of one Hopf fibration iff they are Clifford parallel, θ₁ = θ₂ = θ, and their base points are then 2θ apart on S² (the "Clifford distance
equals half the base angle" of `hopf_clifford.py`). A 2-plane in ℝ⁴ is a point of S² × S² (the self-dual and anti-self-dual parts of its bivector), and two planes with principal
angles θ₁, θ₂ are θ₁ + θ₂ apart on one factor and |θ₁ − θ₂| on the other (**Exact**, checked on random planes). So **a pair of circles is a Hopf pair exactly when one of the two
separations is 0, and the other is then the base angle.**

Result for the family of centre distance d (**Exact**, formula checked on seven values of d to 1e-12):

| d | s₁ = \|1 − d²/2\| | θ₁ | θ₂ | separations | |
|---|---|---|---|---|---|
| 0.5 | 0.875 | 28.96° | 90° | 118.96°, 61.04° | |
| **1 (oloid)** | **0.5** | **60°** | **90°** | **150°, 30°** | not a Hopf pair |
| **√2 (two-circle roller)** | **0** | **90°** | **90°** | **180°, 0°** | **Hopf pair, antipodal base points** |
| 1.7 | 0.445 | 63.58° | 90° | 153.58°, 26.42° | |

The oloid's circles are therefore 30° from being Hopf fibres, measured in the second S² factor. Since s₁ = 0 only at d = √2, **d = √2 is the only Hopf pair in the whole family**; the other values in the table (and 1.2, 1.6, 1.9, all tested) have
a second separation of at least 0.28 rad. At d = √2 the pair is Möbius-equivalent to the unit circle and the z axis.

### 2.4 Rolling, lifted to S³

![the oloid rolling: top view, height, S² shadows, S³ lift](figures/hopf_oloid.png?raw=true)

*Top left: three periods of rolling on the plane, contact segments coloured by time. Top right: height of the centre of mass. Bottom left: the S² shadow of one period,
the world direction of the normal of circle A (blue, the Hopf map of q) and of circle B (orange). Bottom right: the lift to S³ (stereographic), period 1 in blue and period 2
(which is −q) in red; the loop closes only after two periods.*

The oloid is developable, so it rolls about the straight generator in contact. With the body-frame generator direction e(u) and outward unit normal n(u), the requirement that
the world normal stay fixed gives the body angular velocity ω = −((e × n)·n′) e, and "the contact line does not slip" gives the position
(**Exact** kinematics, N = 6000 steps per period, residuals about 1e-6; the checks are that the world normal stays (0,0,−1), both ends of the contact segment stay on z = 0, the body never
goes below the ground, and the contact point has zero velocity to 1e-12).
The orientation is the unit quaternion q (v ↦ q v q̄, as in `hopf_quaternion_rotor.py`), updated by right multiplication, so the Hopf map h(q) = q k q̄ of `hopf_math.py` is the world
direction of the body z axis, the normal of circle A. This is checked against `hopf_math.hopf` to 1e-14.

- **Orientation returns, the quaternion does not.** After one period (the closed loop of four generator patches) R_end = R₀, but q_end · q₀ = −1.000000.
  One rolling period is the non-trivial loop of SO(3), the same double cover as the SU(2) → SO(3) of `hopf_quaternion_rotor.py`. The lift of the S² shadow closes after two periods.
- **Fibre phase.** For the loop on a body axis a, write q_end = q₀ (cos ψ/2, sin ψ/2 a). Then ψ = 2π on every axis, and ψ = ψ_dyn + Ω (mod 4π), where ψ_dyn is the integral of the
  body angular velocity along a and Ω the solid angle of the S² shadow of a. This is the Berry phase = half the solid angle of `hopf_quaternion_rotor.py`
  (and the holonomy of [`HOPF_WITHOUT_PRINCIPAL_BUNDLES.md`](HOPF_WITHOUT_PRINCIPAL_BUNDLES.md)); a synthetic latitude loop with extra spin fixes the sign convention.

  | axis | ψ_dyn | Ω | ψ |
  |---|---|---|---|
  | x (line of the centres) | 7.1910 | 11.6586 | 6π ≡ 2π |
  | y (normal of B) | ≡ 0 (4π) | 2π | 2π |
  | z (normal of A) | 0 | 2π | 2π |

  For circle A's normal the whole phase is geometric: ψ_dyn = 0.
- **Heights and displacement. Observed.** The centre-of-mass height lies in [0.707106, 0.750000] = [√2/2, 3/4], has 8 extrema per period (2 maxima and 2 minima per *half*
  period, which is the "cycle" of the quoted statement), and is periodic with half the period. The displacement per period is 4.83680, which agrees with 8π/(3√3) to the digits
  printed; the lateral excursion of the meander is 0.555. These closed forms are read off the simulation, not derived.

## 3. Antoine's necklace

### 3.1 What the sources say

**Quoted** from [Alexander horned sphere (Wikipedia)](https://en.wikipedia.org/wiki/Alexander_horned_sphere): the surface is homeomorphic to S² but wildly embedded, locally flat except at a Cantor set
of horn tips; its closed interior is a 3-cell; the unbounded component of the complement is not simply connected. It is built by repeatedly attaching interlocked punctured tori. Bing showed that two
solid horned spheres glued along their boundaries give S³.
The same page describes the horned sphere as a way of putting a surface around the wildness of Antoine's necklace; its Cantor set of tips is of Antoine type.
[Antoine's necklace (Wikipedia)](https://en.wikipedia.org/wiki/Antoine%27s_necklace) is a Cantor set in ℝ³ with non-simply-connected complement, built from nested chains of linked solid tori; Alexander used it
(1924) to build *Antoine's* horned sphere, "similar to but not the same as" Alexander's. **So the necklace is the cleaner object for what follows;** the horned sphere is related, not identical.

**Quoted** from [R. Schwartz, *Notes on Antoine's Necklace*](https://www.math.brown.edu/~res/MathNotes/antoine.pdf): in the self-similar construction the tori are disjoint, adjacent tori are linked and non-adjacent tori are unlinked,
each torus of one generation is replaced by a K-necklace of linked tori, and K must be large (the notes say K can be taken to be 20 with a modified construction and doubt that K < 20 is possible).
The notes work with the link of core circles, which is the object used here.

**Quoted** from [*Real Analysis, Quantitative Topology, and Geometric Complexity*](https://arxiv.org/abs/math/0010071): Antoine's necklaces are homeomorphic to the middle-thirds Cantor set but not ambiently so,
and in dimension 2 this does not happen (every Cantor set in the plane is tame).

### 3.2 A straight chain is the oloid family

Circles of radius 1 alternately in the planes y = 0 and z = 0, centres d apart on the x axis. Every adjacent pair is exactly the pair of section 2.3, so every clasp has invariants (\|1 − d²/2\|, 0) (**Exact**, 1e-14).

- Neighbours link for 0 < d < 2 and only neighbours link for 1 < d < 2 (path graph, **Counted** at d = 1.2, √2, 1.8).
- Second neighbours are coplanar circles 2d apart: they intersect for d < 1, are **tangent exactly at d = 1** (distance 2e-16), and are disjoint for d > 1. The oloid is the borderline of the chain family.
- At d = √2 every clasp of the chain is an exact Hopf pair over antipodal base points.

### 3.3 Hopf fibres link in every pair; a necklace only in neighbours

![chains of round circles against Hopf fibres](figures/hopf_antoine.png?raw=true)

*Top left: straight chain at d = √2. Top right: a closed Hopf ring chain (k = 24, grey) with the children of two neighbouring links (red, blue). Bottom left: who links whom, for 8 circles.
Bottom right: Hopf defect of a clasp (θ₂ − θ₁) against the number k of links, for flat/vertical links (orange, best σ) and for links twisted so that the defect vanishes (green, computed).*

- **Counted.** Six Hopf fibres over a latitude circle of S²: all 15 pairs link (|lk| = 1.00), and all 15 pairs are Hopf pairs (|s₁ − s₂| = 1e-15). This is the linking atlas (figure 1) of [`HOPF_FIGURES.md`](HOPF_FIGURES.md)
  seen as a graph: the complete graph. Linking numbers of fibres are the Hopf invariant of [Hou et al., arXiv:2504.03981](https://arxiv.org/abs/2504.03981v1) (Q_H = linking number of preimages).
- A ring of k round circles whose centres are the vertices of a regular k-gon, alternate links in the ring plane and perpendicular to it, links only its two neighbours: the cycle graph
  (**Counted**, k = 8, 16, and the twisted chains below).

So the necklace is the anti-Hopf configuration: the clasps are Hopf links but the whole is not contained in any Hopf fibration, because fibres of one fibration all link.

### 3.4 Ring chains: curvature spoils the Hopf clasp, a twist restores it

- **Exact.** For the flat/vertical ring the clasp has θ₂ = 90° and θ₁ < 90°, never a Hopf pair. The smallest defect over a grid of σ = (chord)/(link radius) is 31.40° (k = 8), 15.89° (16), 10.59° (24), 7.97° (32): it falls roughly like 1/k but is never 0.
- **Exact.** Tilt the link planes about the tangent: even links get normal cos a N + sin a R, odd links cos b R + sin b N (N the ring normal, R the radial direction). Two equations (the two Möbius invariants equal) in the two unknowns (a, b)
  have a solution for every k tried, residual below 1e-12, with the cycle linking pattern intact:

  | k | twist (a, b) at σ = √2 | s = s₁ = s₂ | base angle 2 arccos s |
  |---|---|---|---|
  | 8 | (+0.8716, −0.6992) | 0.41421 | 131.06° |
  | 16 | (+0.8052, −0.7656) | 0.19891 | 157.05° |
  | 24 | (+0.7941, −0.7767) | 0.13165 | 164.87° |
  | 32 | (+0.7902, −0.7805) | 0.09849 | 168.70° |

  The twist is close to ±45° and appears to tend to π/4 as k grows (observed; the straight chain is invariant under a common rotation of all links about its axis, so any common twist is equivalent in the limit).
- **Observed.** s = tan(π/k) to 5e-16 for k = 8, 12, 16, 24, 32: tan(π/8) = 0.41421, tan(π/16) = 0.19891, tan(π/24) = 0.13165. As k → ∞ the base angle tends to the antipodal 180° of the straight chain at d = √2. Not proved.
- **Exact (one clasp, k = 24).** The explicit Möbius model sends circles 0 and 1 to great circles with separations (164.870°, 0.000°) and equal principal angles 82.435°: Clifford parallel, so each adjacent pair lies in one Hopf fibration.
- **Counted.** Circles i−1 and i+1 do not link (lk = −1.00, +1.00, 0.00 for the pairs (0,1), (1,2), (0,2)), and the pair (0,2) has Möbius invariants (3.0, 1.0), outside the range of a great-circle pair. No single Hopf fibration contains three consecutive circles.
- The twist is one point of a one-parameter family (two equations, three unknowns σ, a, b); the minimal-norm branch near (−π/4, π/4) is returned. Whether the family has other components was not checked.

### 3.5 Generations, and what is not shown

Generation 2 (`hopf_antoine.decorate`): each circle is replaced by a ring chain of k links along it, with link radius r′ = r · 2 sin(π/k)/σ and the same twist.
The operator commutes with similarities, so the clasp invariants cannot change (**Exact**: 2.1e-13 at k = 24, 576 circles, scale factor λ = 0.18459). **Counted:** 576 linked pairs, exactly the 24 necklaces of 24 sibling neighbours,
no other pair linked, and children of different parents stay 0.042 apart (link radius 0.034). At k = 16 the pattern is still correct but the nearest circles of different parents are only 0.005 apart (link radius 0.076; polygon-sampled upper bound), which is the same tightness that makes the self-similar necklaces of Schwartz's notes need many links.

This is the geometry of the *cores*. It does not construct the solid tori, take the limit, show the limit is a Cantor set, or compute π₁ of the complement; the non-triviality of that group is
Antoine's theorem, quoted above, not a result of this repository. It does not construct either horned sphere.

## 4. Possible new connections

All of these are **Idea**s unless marked. Strength follows the README: *direct* is the same mathematical object, *indirect* is a shared space.

1. **The √2 coincidence (direct, unexplained).** At d = √2 the two-circle roller has constant centre-of-mass height (**Quoted**, section 2.1) and is exactly a Hopf pair (**Exact**, section 2.3). No mechanism links the two statements; a first test is whether
   other properties of the roller (for example the lift of its rolling loop to S³, which has not been computed) also differ from the oloid's.
2. **Prove s = tan(π/k).** The Lorentz Gram matrix of a clasp is a 2 × 2 matrix of explicit trigonometric expressions in k, σ, a, b. A closed form for the twist family would turn the table of section 3.4 into a theorem and might identify the one-parameter family.
3. **The Hopf lift of a Cantor set (direct, a derivation of ours, unchecked against a source).** For a Cantor set C ⊂ S², the set h⁻¹(C) = C × S¹ has complement h⁻¹(S² ∖ C) ≅ (S² ∖ C) × S¹ (an S¹-bundle over a non-compact surface is trivial), so
   π₁ = F∞ × ℤ: infinitely generated free times the fibre circle. C itself is tame (every Cantor set in the plane is, section 3.1), so the complement is not simply connected although nothing is wild. Compare this group with π₁ of Antoine's complement
   (which has been computed, see the thesis by Raspat cited under [References](#references)); in particular whether the ℤ is central there.
4. **Kleinian groups in Möb(S³) (indirect).** Wild Cantor sets, wild knots and a wild 2-sphere occur as limit sets of Kleinian groups in Möb(S³) (**Quoted**: Apanasov and Tetenov, Hinojosa, Boege–Hinojosa–Verjovsky, Souto–Stover below). Möb(S³) is the group whose Lorentz model
   `hopf_oloid.py` already uses, and Hinojosa's groups are generated by reflections in a necklace of pearls. Question: is there a pearl necklace whose consecutive pearls bound the Hopf clasps of section 3.4, and what is its limit set?
   None of the sources states that either horned sphere is such a limit set, so we do not claim it.
5. **Chirality of the clasps (direct, not computed).** For a pair of great circles that are Clifford parallel the handedness (left or right fibration) is defined once the circles are oriented. By the mirror symmetry of the ring chain the two clasps of one link
   should have opposite handedness, so the necklace would alternate between left and right Hopf pairs. Checking this needs oriented circles and a common orientation of ℝ³; the script reports only unsigned invariants.
6. **Heegaard splittings (indirect).** The Hopf fibration gives S³ as two solid tori glued along the Clifford torus (genus 1); Bing's result (**Quoted**) gives S³ as two solid horned spheres glued along a wild sphere (genus 0, with crumpled cubes).
   We have no map between the two.
7. **Other developable rollers (direct, not run).** Run `hopf_oloid.roll`'s construction on the sphericon and on the two-circle roller and ask whether one period lifts to −q as for the oloid, and what the S² shadow of the circle normals looks like.

## References

- **Oloid.** [en.wikipedia.org/wiki/Oloid](https://en.wikipedia.org/wiki/Oloid). Construction, area, developability, constant contact segment, the two-circle roller (centre distance √2, constant height), the sphericon.
  Used for the quoted facts in section 2.1 and as the starting shape of `hopf_oloid.py`. Relation to the Hopf fibration: none stated; ours.
- **Alexander horned sphere.** [en.wikipedia.org/wiki/Alexander_horned_sphere](https://en.wikipedia.org/wiki/Alexander_horned_sphere). Wild embedding, Cantor set of tips, exterior not simply connected, relation to Antoine's necklace, Bing's S³ = two solid horned spheres.
- **Antoine's necklace.** [en.wikipedia.org/wiki/Antoine's_necklace](https://en.wikipedia.org/wiki/Antoine%27s_necklace). Nested chains of linked tori; Antoine's horned sphere is not the same as Alexander's.
- R. Schwartz, [*Notes on Antoine's Necklace*](https://www.math.brown.edu/~res/MathNotes/antoine.pdf) (12 Nov 2022). Adjacent tori linked, non-adjacent unlinked, self-similar construction, the K ≥ 20 remark, the link of core circles.
- [*Real Analysis, Quantitative Topology, and Geometric Complexity*](https://arxiv.org/abs/math/0010071). Antoine's necklaces are not ambiently equivalent to the standard Cantor set; in dimension 2 this does not happen (author not recorded here).
- T. J. Raspat, *Complementary Groups of Antoine's Necklaces*, M.S. thesis, University of Pittsburgh, 2010 ([d-scholarship.pitt.edu/9048](https://d-scholarship.pitt.edu/9048/)). A presentation of the fundamental group of the complement; surveyed from the abstract only.
- *Kleinian groups in higher dimensions*, [arXiv:math/0701370](https://arxiv.org/abs/math/0701370) (author not recorded here), Theorem 9.14, attributed there to B. Apanasov and A. Tetenov: a convex-cocompact 2-quasifuchsian group in Möb(S³) whose limit set is a wild 2-sphere.
- G. Hinojosa, *Wild Knots as limit sets of Kleinian Groups*, [arXiv:math/0509124](https://arxiv.org/abs/math/0509124). Schottky-type groups generated by reflections in a necklace of pearls whose limit set is a wild knot.
- M. Boege, G. Hinojosa, A. Verjovsky, *Wild knots in higher dimensions as limit sets of Kleinian groups*, [arXiv:0907.1429](https://arxiv.org/abs/0907.1429).
- J. Souto, M. Stover, *A Cantor set with hyperbolic complement*, [arXiv:1205.4668](https://arxiv.org/abs/1205.4668). A Cantor set in S³ with complete hyperbolic complement, built by mimicking Antoine's necklace with θ-graphs.
- W.-T. Hou, Z. Xiang, Y. Liu, J. Zang, *Construction of Hopfion Crystals*, [arXiv:2504.03981v1](https://arxiv.org/abs/2504.03981v1). The Hopf invariant as the linking number of preimages, which is why the linking graphs of section 3.3 are the right way to compare chains with fibres.
  Relation to this document: used as the definition of the invariant, nothing more.
- In this repository: [`README.md`](README.md) (references and scripts), [`HOPF_FIGURES.md`](HOPF_FIGURES.md) (linking atlas, Clifford distance = half the base angle), [`HOPF_WITHOUT_PRINCIPAL_BUNDLES.md`](HOPF_WITHOUT_PRINCIPAL_BUNDLES.md) (holonomy and Berry phase),
  and the scripts `hopf_math.py`, `hopf_clifford.py`, `hopf_quaternion_rotor.py`, `hopf_oloid.py`, `hopf_antoine.py`.
