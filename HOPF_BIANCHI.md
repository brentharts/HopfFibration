# The three Mixmaster exits are the three left-invariant Hopf fibrations

Script: `hopf_bianchi.py` (numpy; 9 checks in `--tests`, about 7 s). `--png FILE` draws the figure. The last two checks need a clone of
[brentharts/wada-coudene](https://github.com/brentharts/wada-coudene) (`WADA_REPO`, default `../wada-coudene`; module `lakes_of_wada.mixmaster`, notes in its `MIXMASTER.md`);
without it they are skipped. Labels: **Exact**, **Counted**, **Idea**, as elsewhere in this repository.

![the octahedron of exit axes, the three left-invariant Hopf fibrations, and the Mixmaster exit basins](figures/hopf_bianchi.png?raw=true)

*Left: the exit axes on the base S^2 are the vertices of an octahedron. Middle: fibres of the three left-invariant Hopf fibrations in the stereographic picture (thick: through
the identity; the one through the stereographic pole is a straight line). Right: the Mixmaster exit basins, colour l = the fibration whose fibres collapse.*

## 1. The dictionary

The spatial slice of a vacuum Bianchi IX universe is SU(2) = S^3, the unit quaternions, with a left-invariant metric. Take the left-invariant fields e_a = (1/2) (q goes to q u_a), u_a = i, j, k,
with [e_a, e_b] = eps_abc e_c. A diagonal metric diag(a^2, b^2, c^2) is the Mixmaster universe, and in the wada-coudene exit rule the exit through axis l is the pancake limit in which a_l goes to 0
(a Kasner epoch with p_l close to 1).

| Mixmaster | Hopf side | label |
|---|---|---|
| axis l = a, b, c | the left-invariant unit quaternion u_l = i, j, k | definition |
| integral curves of e_l | fibres q exp(t u_l) of the Hopf fibration h_l(q) = q u_l conj(q) in S^2; h_k is `hopf_math.hopf` | Exact (2e-16) |
| collapse of axis l (fibre length 2 pi a_l goes to 0) | the fibres of h_l shrink: the Berger pancake of `hopf_pancake.py` with fibration h_l | Exact (the length of the fibre is 2 pi a_l, as in the README) |
| the S_3 symmetry of the exit picture (cyclic: rotation by 120 degrees; transposition: reflection) | isometries of the round S^3: conjugation by q0 = (1+i+j+k)/2 maps the h_{u'} fibres to the h_u fibres, u' = q0^-1 u q0; conjugation by (i+j)/sqrt 2 sends i, j, k to j, i, -k | Exact (4e-16) |
| the final collapse direction in the frame, for the solution rotated by R = Ad_q | R e_l = q u_l conj(q) = h_{u_l}(q) | Counted: 23 of 23, see section 2 |

So the exit label ℓ(beta) in the anisotropy plane says **which of the three Hopf fibrations of S^3 is the one that collapses**, and the orientation q of the frame, through the Hopf map of that
fibration, says in which direction of S^2 it collapses. The six directions ±R e_l are the vertices of an octahedron on the base S^2.

## 2. The check that is not a restatement

The wada-coudene module integrates only the diagonal equations x_i'' = (q_j - q_k)^2 - q_i^2. `hopf_bianchi.py` has an independent integrator of the full 3 x 3 ADM equations for a
left-invariant metric h_ab (Ricci tensor from the Koszul formula for any h_ab, no diagonality used; lapse N = sqrt(det h), the BKL time). Checks:

1. With [e_a, e_b] = eps_abc e_c the full equations reduce, on diagonal data, to the diagonal equations to 1e-12 (**Exact**; this fixes the normalisation).
2. Take 23 random points of the maximum-expansion slice of the Mixmaster picture and 23 random orientations q in S^3, and start the full integrator from R diag(q_i) R^T and R diag(q_i v_i) R^T with
   R = Ad_q. The initial data satisfy the 3 x 3 Hamiltonian constraint to 2e-15 and the relative violation stays below 1.4e-8 along the way. **Counted**: the full integrator exits in all
   23 cases, and in all 23 the direction of the collapsing eigenvector of h equals h_{u_l}(q) up to sign, to 4e-16, where l is the exit axis returned by the diagonal integrator. So the two integrators
   agree on the exit axis, and the exit direction is the Hopf map of the orientation in the fibration of that axis.

Why the agreement is so exact: the Bianchi IX equations are covariant under h goes to R h R^T for R in SO(3) (the automorphisms of the Lie algebra), so the rotated solution is the rotated
exit with the rotated direction. The checks test the bookkeeping, the independence of the two integrators and the exit rule, not new dynamics. **What is not tested:** momenta not diagonalised by
the same frame as the metric. Those are treated in section 3b: in vacuum they do not exist.

## 3. What else came out

- **Two different left-invariant fibrations are not Clifford parallel.** Fibres of one fibration link +1 pairwise (36 pairs); fibres of two different ones are also great circles and link +-1, with both
  signs occurring (66 pairs). The three families are three Hopf fibrations of the same handedness but they are not mutually 'parallel'.
- **The Wada lift of `HOPF_WADA.md` does not apply.** That lemma needs basins that live on the base of a fixed fibration. Here beta is a parameter of the metric, not a point of S^3: for a
  fixed beta the outcome map is the single Hopf map h_{ℓ(beta)}, and the Wada boundary of the beta-plane is the set where it is undecided *which of the three fibrations* is the one.
  That is a statement of this note's reading, not a theorem.
- **The Berger line is the symmetric case.** beta_- = 0 is b = c, the biaxial metric: the Hopf fibration of the axis that differs (a here) is a symmetry of the metric, and the exit is its own
  axis (all 25 points tested in `MIXMASTER.md`, with a solid band around it).

## 3b. Non-aligned momenta

**Exact (checked numerically, `run_tests` 10-11).** The momentum constraint of the left-invariant 3 x 3 system is linear in K, and its kernel is exactly the set of K that commute with h:
dimension 3 for generic h, 4 for biaxial h, 6 for round h. A K that commutes with a generic h is diagonal in the eigenframe of h. So vacuum Bianchi IX data are always 'aligned' up to a rotation:
(h, K) = R (diagonal Mixmaster data) R^T with R in SO(3), and the phase space is Mixmaster x SO(3) (= Mixmaster x S^3/+-1). In the degenerate biaxial and round cases extra K are allowed, but they are
still rotations of aligned data (in the round case every K is). Consequence: the exit direction is F(m, q) = +-h_{u_l(m)}(q) in RP^2, where l(m) is the exit axis of the Mixmaster point m,
so the exit basins are B_l x SO(3) and the Wada property of the Mixmaster picture lifts by the product/open-projection argument of `HOPF_WADA.md`.

**Counted.** The frame-rotation charge J = sqrt(det h) eps_ijk [K, h^-1]_jk is conserved to 7e-14 over t = 1.5 for non-aligned data (it is the Noether charge of h -> R h R^T), and the vacuum
momentum constraint is J = 0 (at the start the constraint vector is proportional to J). So non-aligned momenta are not a new region of the vacuum phase space; they are excluded.

**Counted, off-shell (exploration only, not a test).** Forcing a small J != 0 while rescaling K to keep the Hamiltonian constraint at 4e-16 (40 aligned points with a known exit, so 40 trajectories per setting):

| off-diagonal size s | exit found | nearest axis = aligned exit | median distance of exit direction from an axis |
|---|---|---|---|
| 1e-4 | 40/40 | 40 | 0.01 deg |
| 1e-3 | 40/40 | 40 | 0.06 deg |
| 1e-2 | 39/40 | 36 | 0.64 deg |
| 1e-1 | 37/40 | 24 | 8.5 deg |

Such data are not vacuum solutions (they would need a source such as a rotating fluid), so this is not physics of the vacuum model. It shows that the exits are stable for small |J| and then start
to switch label, which is where a source-carrying model would be interesting. Set-up: 40 points, one random off-diagonal direction each; the numbers depend on that sample.

## 4. Ideas, not done

1. **A bundle over the beta-plane.** The data (beta, q) form a trivial S^3-bundle over the beta-plane whose exit map is h_{ℓ(beta)}: a 'jumping Hopf fibration'. Whether the jumps carry a topological
   invariant (they happen across the fractal boundary) is not examined.
2. **Berger-sphere pancake and the Hopf fibre as an exit channel in the Bianchi IX billiard.** The leading-order pancake of `hopf_pancake.py` (R_1 = a0 t, R_2 = b0) is the exit of this picture;
   a comparison of its subleading terms with the actual exit of a trajectory has not been done.
