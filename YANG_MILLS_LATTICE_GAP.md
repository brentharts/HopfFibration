# A unique state and an explicit gap for 4D lattice Yang-Mills

Notes on one preprint, written for this repo because its SU(2) case lives on the three-sphere S3, the total space of the Hopf
fibration.

> **Source.** D. Bhattacharjee, U. Bhattacharya, J. M. Lockwood, T. Romanov, *Explicit lattice Yang-Mills mass gap for every
> compact simple group reducing the continuum gap to one cube*, Hexagon preprint
> [2610.00341v5](https://hexagonmath.org/2610.00341v5), 31 Oct 2026. Not on arXiv, not peer reviewed. The site now lists a v6
> under the title *Yang-Mills Mass Gap on the Lattice and One Cube from the Continuum*; these notes follow v5.
>
> **What these notes are.** A summary of the *statements* (introduction and theorems, pp. 1-8 of the 150-page PDF) plus a
> text search of the rest. The proofs are not checked here. The preprint's results are the authors' claims; several rest on a
> computer-assisted certificate. The authors say explicitly that this is **not** a solution of the Clay Millennium problem.

## 1. The claim in one paragraph

For four-dimensional lattice Yang-Mills theory with Wilson action, the authors prove, for every compact simple gauge group and
for explicit ranges of the coupling, that

- there is a **unique infinite-volume state** (boxes with free boundary conditions converge to the same limit along every
  exhausting sequence),
- that state is translation and hypercubic invariant and reflection positive,
- truncated correlations of Lipschitz observables **decay exponentially** at an explicit rate, and
- the **transfer matrix has an explicit spectral gap** m > 0.

For SU(2) the ranges reach beta_W = 0.345, and for SU(3) beta_W = 0.76. A second, separate part reduces the continuum question
to a Dobrushin-Shlosman condition on **one cube of fixed physical size**; that condition is stated, not proved.

## 2. Setting and normalisation

Wilson's lattice gauge theory puts a group element Q_e in G, a subgroup of U(N), on every link e of Z^4. A configuration is
weighed by

    exp( N * beta * sum_p Re Tr Q_p )

where Q_p is the ordered product of the four link variables around plaquette p. The paper's coupling is the conventional inverse
coupling

    beta_W = N^2 * beta.

For SU(2), identified with the unit quaternions, (1/2) Tr Q_p = Re Q_p and beta_W = 4 beta.

**Small beta_W is strong coupling.** Everything proved here sits at strong coupling. The continuum limit is the weak-coupling end
(beta_W -> infinity), and the authors place the difficulty exactly where strong coupling ends and Balaban's weak-coupling analysis
begins.

## 3. The two ideas that make explicit constants possible

**Gauge invariance makes links redundant.** In a box, the variables on a forest of links can be set to the identity without
changing any gauge-invariant expectation. The paper uses a **line field**: a field of straight lines with one fixed link per
site, chosen so that *every plaquette contains exactly one fixed link* (Lemma 2.1: this holds if and only if, at every site x,
the permutation pi_x fixes the direction sigma(x) and permutes the other three directions cyclically).

**Counting partners.** A link of Z^4 lies in 6 plaquettes and shares each with 3 other links, so 18 partners. After fixing the line
field, each free link has exactly **12 free partners**, two in each of its plaquettes (Lemma 2.2). Both methods improve:

| method | what it bounds | role of the count |
|---|---|---|
| Dobrushin | influence of one link's variable on the conditional law of another, measured by the Kantorovich modulus | 18 partners gives beta_W < 2/9 without gauge fixing; 12 partners gives beta_W < 1/3 with it |
| Bakry-Emery (curvature) | Hessian of the action against the Ricci curvature of the group | Hessian bound drops from 16 to 29/2 in the gauge |

For SU(2) the single-link conditional law is a **von Mises-Fisher law on the three-sphere**, and its sharp Kantorovich modulus is
1/4 (proved in the paper's Appendix I).

## 4. Results

### SU(2)

| Theorem | range | conclusion |
|---|---|---|
| 1.1 | 0 <= beta_W < 1/3 | unique state, reflection positivity, decay at every rate below -log(3 beta_W), gap m >= -log(3 beta_W) > 0 |
| 1.1 (negative side) | -1/3 < beta_W < 0 | everything except reflection positivity and the gap |
| 1.2 (ceiling) | beta_W >= 1/3 | the single-link Dobrushin argument in this gauge fails; the coefficients are attained, so no argument of that type reaches 1/3 |
| 1.3 (plaquette blocks) | 0 <= beta_W <= 0.345 | decay below -(1/2) log alpha(beta_W), gap m >= -(1/2) log alpha(beta_W); alpha(1/3) <= 0.9561, alpha(0.345) <= 0.9973; at beta_W = 1/3 the gap is m >= 0.0224 |

Plaquette blocks (the three free links of one plaquette treated as a block) are what carry the result past the ceiling in 1.2.

### Every compact simple group (curvature route)

**Theorem 1.4.** In the gauge, |Hess S_0(v, v)| <= (29/2) |v|^2 for every N, against the sharp constant 16 without gauge fixing.
Consequences: for SU(N) with |beta| < 1/29, and for SO(N), N >= 3, with |beta| < (N-2)/(58 N), the conclusions of Theorem 1.1 hold.
For SU(3) this is beta_W < 9/29 = 0.310.

**Theorem 1.5.** Let the Lie algebra of G be semisimple, so kappa_G > 0, the least Ricci curvature of G for the bi-invariant metric
<X, Y> = -Tr(rho(X) rho(Y)). If N |beta| < (2/29) kappa_G, the conclusions of Theorem 1.1 hold, with decay at every rate below

    m_G(beta) = arccosh( 1 + (1/12) * ( kappa_G / (N |beta|) - 29/2 ) ) > 0

and a transfer-matrix gap m >= m_G(beta). For a simple G, kappa_G = h^v / (2 l_rho), with h^v the dual Coxeter number and l_rho the
Dynkin index of the representation. As beta -> 0 the gap grows like log( kappa_G / (6 N |beta|) ).

Table 1 of the paper, for the conventional coupling beta_W = N^2 beta:

| group | rho | h^v | l_rho | kappa_G | beta_W < |
|---|---|---|---|---|---|
| SU(N), N >= 2 | C^N | N | 1 | N/2 | N^2/29 |
| SO(N), N >= 5 | R^N | N-2 | 2 | (N-2)/4 | N(N-2)/58 |
| SO(3) | R^3 | 2 | 4 | 1/4 | 3/58 |
| Sp(n), n >= 2 | C^2n | n+1 | 1 | (n+1)/2 | 2n(n+1)/29 |
| G2 | 7 | 4 | 2 | 1 | 14/29 |
| F4 | 26 | 9 | 6 | 3/4 | 39/29 |
| E6 | 27 | 12 | 6 | 1 | 54/29 |
| E7 | 56 | 18 | 12 | 3/4 | 84/29 |
| E8 | 248 | 30 | 60 | 1/4 | 124/29 |

The constant 29/2 depends only on the group through U(N), so the exceptional groups are covered "with an explicit gap" (the
authors' phrase). With a computer-assisted constant h_F <= 14.420001 (their Theorem L.5), every range in Table 1 grows by a factor
29/28.840002, about 1.0055.

### SU(3)

| Theorem | range | conclusion |
|---|---|---|
| 1.6 | 0 <= beta_W <= 3/4 | decay below -log(1.33319 beta_W), gap m >= -log(1.33319 beta_W) > 0. Proved by bounding the Kantorovich modulus of the one-link measure uniformly over staple sums, using a computer-assisted evaluation on 2341 boxes with rigorous error bounds |
| 1.7 (plaquette blocks) | 0 <= beta_W <= 19/25 = 0.76 | alpha_3 < 1 with alpha_3(0.72) <= 0.9182, alpha_3(3/4) <= 0.9653, alpha_3(0.76) <= 0.99950; gap m >= -(1/2) log alpha_3; at beta_W = 3/4, m >= 0.0177 (against 1.1e-4 from Theorem 1.6); at beta_W = 0.76, m >= 2.5e-4 |

### Summary of ranges (Table 2 of the paper)

| method | SU(2), beta_W | SU(3), beta_W |
|---|---|---|
| curvature, Shen-Zhu-Zhu (ungauged) | < 1/12 | < 3/16 |
| curvature, sharp (ungauged) | < 1/8 | < 9/32 |
| Dobrushin, sharp quarter (ungauged) | < 2/9 | |
| curvature, forest gauge | < 4/29 | < 9/29 |
| Dobrushin, forest gauge | < 1/3 | <= 3/4 |
| plaquette blocks, forest gauge | <= 0.345 | <= 0.76 |

The first three rows hold for the ungauged measure; the last three are this paper's. No Bakry-Emery argument for the ungauged
measure can pass |beta| = 1/32.

## 5. Where it stops: the continuum

All seven theorems are statements **at fixed lattice spacing**. They do not touch the continuum limit, the crossover to Balaban's
regime, or a gap in physical units, which together are the Clay problem. An obstruction proved in the authors' earlier work says
that a lattice gap uniform in lattice units forces trivial continuum two-point functions, so a physical-units gap needs a
different mechanism.

Section 10 supplies one conditionally. Three implications are proved:

1. **(I1)** A Dobrushin-Shlosman condition on one cube of fixed *physical* size, uniform along the continuum tuning, gives a mass
   gap in physical units for every continuum limit, and exponential decay of renormalised correlations (including observables
   smoothed by the spatial gradient flow).
2. **(I2)** The flowed, renormalised observables have reflection-positive continuum limits along subsequences, and such a limit is
   nontrivial exactly when one ratio of lattice two-point functions stays away from zero.
3. **(I3)** Tree bounds on truncated correlations imply that ratio bound, and carry the conditional gap to polynomials in the
   smeared fields.

**The inputs to (I1)-(I3) are not proved**: the cube condition along the tuning, the slice ratio in four dimensions, and the tree
bounds. So the nontrivial existence of the continuum limit and the gap in physical units remain open. The paper measures these
inputs in Section 11: the cube condition holds at strong coupling and fails at every fixed cube size at weak coupling, and for
SU(N) it needs cubes of side at least of order (beta_W / log beta_W)^(1/4) lattice spacings. Section 12 poses the cube condition
for block averages and examines a cut-off version of Balaban's averaging operation; several of its properties are proved and the
general case (a block map that is coercive for all fields and composes across all scales) is stated as open.

## 6. Where S3 appears (the connection to this repo)

The preprint never uses the word "Hopf" (a text search of all 150 pages finds no occurrence), so everything below is the
geometry it uses, read against the objects in this repo. It is a reading, not a claim by the authors.

- **SU(2) is S3.** The SU(2) proofs identify the group with the unit quaternions, hence with the unit sphere S3 in R^4: the
  total space of the Hopf fibration S1 -> S3 -> S2. This is the same quaternion convention as `hopf_quaternion_rotor.py`.
- **The single-link law lives on S3.** Conditioned on the other links, a link variable has a von Mises-Fisher law on S3 (a density
  of the form exp(<b, Q>) with respect to Haar measure).
- **A concrete curvature computation.** Along a geodesic through a link variable Q_e, with one active link equal to -1 and all
  others 1, the six plaquettes through e have variable -1, and the Hessian of -(beta_W/2) * sum_p Re Tr Q_p is -6 beta_W times
  the metric of the unit S3. The unit S3 has Ricci curvature 2, so the Bakry-Emery tensor is 2 - 6 beta_W, which is negative once
  beta_W > 1/3. This is why no curvature argument for the gauge-fixed SU(2) measure passes beta_W = 1/3.
- **Which metric.** The paper uses the bi-invariant metric, the round S3. In the vocabulary of `hopf_berger.py` (fibre scaled by
  s) this is the s = 1 sphere; the Berger deformation changes the Ricci curvature, and nothing in the preprint uses it.

Open question for this repo, not from the paper: the bi-invariant metric is one point in the Berger family. Whether a squashed
metric on SU(2) gives anything useful for these estimates is not addressed by the preprint and is not claimed here.

## 7. Reading notes and caveats

- **Status.** An unrefereed preprint from a site that is not arXiv, with a newer version (v6) and a changed title. Treat the
  specific constants (0.345, 0.76, 2341 boxes, 14.420001) as the authors' claims until independently checked.
- **Computer assistance.** Theorem 1.6 and the sharpened constant in Table 1 rest on certified computations. The site links a
  [Lean 4 formalization](https://github.com/creelie/4Y-M/blob/main/YangMills/StrongCoupling/GaugeFix.lean) (a gauge-fixing file under
  `StrongCoupling`) and a
  [forest-certificate repository](https://github.com/romanov360/yang-mills-4d-forest-certificate). Neither has been examined here.
- **Strong coupling only.** The explicit gaps are small numbers at the edge of their ranges (for example 2.5e-4 at SU(3),
  beta_W = 0.76). They say nothing directly about physical-units gaps.
- **Scope of this summary.** Statements taken from pages 1-8; the appendices (transfer-matrix reconstruction, spectral measures,
  the sharp quarter for the SU(2) single-link law, the certified computation) were not read.

## 8. Cross-reference: the fork's `YM_CONNECTIONS.md`

[brentharts/openai_math_yang_mills](https://github.com/brentharts/openai_math_yang_mills) is a fork of OpenAI's math manuscript
release that carries its own fork-level notes (they are not part of OpenAI's release). Its
[`YM_CONNECTIONS.md`](https://github.com/brentharts/openai_math_yang_mills/blob/main/YM_CONNECTIONS.md) ranks every connection to
Yang-Mills it has checked, and its row 1 is this same line of work (Bhattacharjee-Bhattacharya-Lockwood, with Romanov's forest
certificates). It rederives the forest, the partner counts and the window arithmetic, and quotes the certificates without rerunning
them. How its numbers line up with this note:

| item | the fork's `YM_CONNECTIONS.md` | this note and the preprint |
|---|---|---|
| 4D Hessian constant | h <= 14.420001, period-2 forest | Theorem L.5 gives h_F <= 14.420001; the earlier constant is 29/2 = 14.5 (Theorem 1.4) |
| spectral-gap window | moves from 1/29 to 1/28.84, "about 0.55%" | Theorem 1.4: \|beta\| < 1/29 for SU(N). Window 1/(2h) = 1/28.840002, and 29/28.840002 = 1.0055 (the preprint's footnote 9) |
| how much more is possible | "at most about 3%" with this method | 14.420001 / 13.974 = 1.032, using the fork's floating-point lower bound 13.974 (not checked here) |
| continuum | a gap bounded below in lattice units uniformly in the spacing cannot be the continuum step (the authors' no-go theorem) | the obstruction cited in section 5 above |
| 3D analogue | period-3 forest, 7 free partners per link; SU(2) Dobrushin uniqueness for beta_W < 4/7 with m >= -log(7 beta_W / 4); Hessian h = 9.599991, range \|beta\| < 1/(2h); no valid Hessian constant is below 9.3479 | not in this preprint; quoted from the fork and Romanov's 3D forest-gauge repository, not checked here |

One pattern that ties the 3D and 4D statements together (my observation, not a claim of either source): for SU(2) the sharp
Kantorovich modulus is 1/4, so p free partners give a Dobrushin sum of p * beta_W / 4. With p = 18 (no gauge fixing) that is
beta_W < 2/9; with p = 12 (the 4D forest) it is m >= -log(3 beta_W) and beta_W < 1/3, which is Theorem 1.1; with p = 7 (the 3D
forest) it is m >= -log(7 beta_W / 4) and beta_W < 4/7, which is the fork's 3D statement.

Two caveats. The fork cites its Bhattacharjee-Bhattacharya-Lockwood reference by Zenodo DOI 10.5281/zenodo.23729004, which is not one
of the DOIs on the Hexagon page (23444338 for all versions, 23841532 for the Lean release), so it may point at an earlier version.
And the fork is as explicit as this note that none of this reaches the continuum limit: it holds only at strong coupling, which is
the opposite end from beta -> infinity.
