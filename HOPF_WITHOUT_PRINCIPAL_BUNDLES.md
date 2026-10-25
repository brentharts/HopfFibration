# The Hopf fibration without a principal bundle

This page explains one idea and one script, [`hopf_line_bundle.py`](hopf_line_bundle.py). It is written for a reader who has met
the Hopf fibration (see the README and [`hopf_math.py`](hopf_math.py)) but not gauge theory or fibre bundles.

**Honesty note.** The idea is from Henrique Gomes, *Gauge Theory Without Principal Fiber Bundles*, Philosophy of Science 92(3),
511–527 (2025). We could only read a machine summary of the paper, not its full text, so what is said about the paper below is the
summary's account and should be checked against the paper. Everything about the Hopf fibration itself is standard mathematics, and
is checked numerically by the script.

Status labels: **Standard** (textbook, quoted), **Checked** (verified numerically here), **Not done**.

---

## 1. Two ways to build the same thing

The Hopf fibration S¹ → S³ → S² is usually presented as a **principal bundle**: a space S³ with a *group* U(1) acting on it by
phases, and S² the space of orbits. A gauge field is then a *connection* on that bundle, a rule saying which direction in S³ counts
as "horizontal". Physicists meet this as the Dirac magnetic monopole: S³ is the bundle, the connection is the monopole's vector
potential, U(1) is the electromagnetic gauge group.

There is a second description, using a **vector bundle** (**Standard**). Over each point b of S² put a complex *line* ℓ_b ⊂ ℂ²:
the line of the qubit state whose Bloch vector is b, with projector

    P(b) = (I + b·σ) / 2 .

This is the **tautological line bundle**. Its only structure is the complex inner product inherited from ℂ². Two things come out
of that structure automatically:

* The **gauge group is derived**: the symmetries of a one-dimensional complex inner-product space are the unit complex numbers,
  U(1). It was never postulated.
* The **connection is derived**: a section s(b) ∈ ℓ_b has covariant derivative ∇s = P ds (project the ordinary derivative back
  into the line). Parallel transport along a curve is Kato's equation U′ = [P′, P] U. No potential A and no principal bundle appear.

And the principal bundle comes back for free: the unit vectors of the lines, {(b, u) : u ∈ ℓ_b, |u| = 1}, are exactly S³ → S²,
the Hopf fibration. So here **the principal bundle is built from the vector bundle**, not the other way round.

## 2. What Gomes's argument is (as summarised)

In general relativity one structure, the tangent bundle with its metric, tells every tensor and spinor field how to be parallel
transported. Textbook gauge theory has no equally clear counterpart: one *postulates* a gauge group (SU(3)×SU(2)×U(1)) and a
principal bundle, and why all the matter fields should fit that group is left unexplained. Gomes calls this the *coordination problem*.

His proposal, "structure-first", as the summary gives it:

* take internal **vector bundles** with fibres ℂ³, ℂ², ℂ¹, each with a complex inner product (the ℂ³ and ℂ² also with an orientation);
* the gauge groups SU(3), SU(2), U(1) are then the **automorphism groups** of those fibres, not postulated;
* matter fields are sections of **tensor bundles** built from them, as tensors are built from the tangent bundle; gauge bosons are
  **connections** on them; the structure group reappears as a **holonomy** group;
* principal bundles are called auxiliary (they "coordinate" the parallel transports), possibly still useful but not necessary;
* the worked example is one Standard Model generation without the Higgs; the Higgs, Yukawa couplings, three generations,
  supersymmetry and exceptional groups are left open.

The Hopf case above is the simplest instance (a single ℂ¹ fibre). It does not reproduce his Standard Model construction.

## 3. What `hopf_line_bundle.py` checks (all **Checked**)

| # | statement | result |
|---|---|---|
| 1 | the unit vectors of the lines ℓ_b are exactly S³; the fibre over b is the circle of phases; the base map is the Hopf map | exact to rounding |
| 2 | **three independent ways** to compute the holonomy of a loop on S² agree: (a) Kato transport with the projector only; (b) the principal-bundle connection form A = −Im(s̄ ds) along the Hopf section; (c) minus the argument of the Pancharatnam overlap product around the loop | agree to 4·10⁻⁶ (a vs b) and 5·10⁻¹³ (a vs c) |
| 3 | for a loop of constant colatitude θ the holonomy is −π(1 − cos θ), i.e. **minus half the enclosed solid angle** | error 4·10⁻⁶ |
| 4 | the holonomy is **gauge invariant** (changing the local section by e^{iχ(b)} does not change it), also for a non-circular loop | 10⁻¹⁵ |
| 5 | **Stokes**: the holonomy of a loop equals *i* times the integral of the curvature Tr(P[∂P, ∂P]) over the cap it bounds, with the curvature computed from the projector alone | 1.6·10⁻⁴ |
| 6 | over the whole sphere the curvature integrates to 2πi, so the **Chern number is 1** (also read off from how the latitude holonomy winds) | 1.0001 and 1.000 |
| 7 | the tensor power L^k has holonomy kγ and Chern number k: *multiple U(1) charges from tensor powers of one fibre, with no new fields* (the point Gomes makes for multiple U(1) charges) | exact (and k = 2 winding) |

Run `python3 hopf_line_bundle.py --tests`, or `python3 hopf_line_bundle.py --png figures/hopf_line_bundle.png`.

![holonomy of latitude loops](figures/hopf_line_bundle.png)

How to read it: carry a vector in ℓ_b around a loop on S², always projecting back into the line (that is the whole connection). It
comes back multiplied by a phase e^{iγ}. For the loop at colatitude θ, γ = −π(1 − cos θ). That phase *is* the holonomy group
element: the structure group U(1) shows up as the set of such phases (any phase can be reached by choosing the loop), which is how
Gomes says the structure group reappears. The same number is the Berry phase of a spin-½ following a slowly rotating field, which
is why `hopf_quaternion_rotor.py` already found "half the solid angle".

## 4. What is and is not shown

* **Shown**: for the Hopf/monopole bundle the gauge group, the connection, the holonomy, the curvature and the Chern number can
  all be obtained from the vector bundle with its inner product, with no principal bundle and no postulated group; the principal
  bundle is recovered as the unit sphere bundle. This is textbook (tautological bundle, Berry 1984, Simon 1983); the script checks it
  numerically and with three independent methods.
* **Not shown**: anything about the Standard Model, ℂ³ and ℂ² fibres, the Higgs, three generations, or the paper's philosophical
  claims (ontological commitment, the coach metaphor and so on). Nothing here tests or supports them.
* **Abelian case only.** The next natural step is the rank-2 tautological bundle over S⁴ (the quaternionic Hopf fibration
  S⁷ → S⁴; fibre ℂ², automorphism group SU(2), instanton number 1). **Not done.**
* **No link to the tilings.** There is no connection here with the Spectre or Chair44 beyond the Hopf structure discussed in the
  [Spectre repository](https://github.com/brentharts/spectre) (`HOPF.md`); the paper does not mention tilings.

## 5. References

1. H. Gomes, *Gauge Theory Without Principal Fiber Bundles*, Philosophy of Science **92**(3), 511–527 (2025; online 28 Oct 2024),
   <https://www.cambridge.org/core/services/aop-cambridge-core/content/view/248F68985B0DD811F5AF03F477937347/S0031824824000497a.pdf/gauge_theory_without_principal_fiber_bundles.pdf>
   *(summary read, not the full text)*
2. [Hopf fibration](https://en.wikipedia.org/wiki/Hopf_fibration), Wikipedia: principal-bundle description, Bloch sphere, Dirac monopole.
3. M. V. Berry, *Quantal phase factors accompanying adiabatic changes*, Proc. R. Soc. Lond. A **392** (1984) 45–57; B. Simon,
   *Holonomy, the quantum adiabatic theorem, and Berry's phase*, Phys. Rev. Lett. **51** (1983) 2167; T. Kato, *On the adiabatic theorem of
   quantum mechanics*, J. Phys. Soc. Japan **5** (1950) 435; S. Pancharatnam, Proc. Indian Acad. Sci. A **44** (1956) 247. *(cited from memory)*
4. [`hopf_quaternion_rotor.py`](hopf_quaternion_rotor.py), [`hopf_math.py`](hopf_math.py) in this repository.
