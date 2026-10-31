# Four new Hopf-fibration figures, in depth

Script: `hopf_figures.py` (numpy + matplotlib only, no Blender). `python3 hopf_figures.py --tests` runs the checks quoted below (about 1.5 s);
`python3 hopf_figures.py` writes the four PNGs into `figures/` (about 5 s); `--png-dir DIR` and `--only 2,4` select where and which.

Conventions are exactly those of `hopf_math.py`: a point of S³ is (z₁, z₂) ∈ ℂ² with |z₁|²+|z₂|² = 1, the Hopf map is
h = (2 Re z₁z̄₂, 2 Im z₁z̄₂, |z₁|²−|z₂|²), a fibre is e^{it}(z₁, z₂), stereographic projection is from (0,0,0,1), and a base point is given as
(elevation, azimuth) with elevation 0 = the **south** pole (`hopf_math.base_point`). That is *not* `fibration.py`'s counting; `hopf_math.blender_frame`
converts.

Labels, as in the rest of the repository: **Exact** (identity checked to rounding), **Counted** (numerical, finite sampling), **Quoted** (a standard
result, not re-derived here).

| figure | file | what it adds |
|---|---|---|
| 1 | `figures/hopf_fig1_linking_atlas.png` | all pairwise linking numbers at once; the fibres are equidistant parallels |
| 2 | `figures/hopf_fig2_poincare_section.png` | a half-plane that every fibre crosses once: S² drawn inside ℝ³, in closed form |
| 3 | `figures/hopf_fig3_makefibre_audit.png` | the audit of `makeFibre` as continuous curves instead of a six-row table |
| 4 | `figures/hopf_fig4_hopf_invariant.png` | Hopf invariant = linking number of preimages, for maps of invariant 1, 2, 3, 6 |

---

## Figure 1 — the linking atlas

![linking atlas](figures/hopf_fig1_linking_atlas.png?raw=true)

**What is drawn.** Twenty-four fibres: four elevations (0.90, 1.43, 1.97, 2.50) times six equally spaced azimuths, kept away from the poles because the
two polar fibres are straight lines through infinity. Left: the 24 × 24 matrix of Gauss linking numbers of fibre *i* with fibre *j* (grey diagonal =
undefined). Right: for the same 276 pairs, the distance between the two fibres *in S³* against half the angle between their base points on S².

**How to read the left panel.** There is nothing to read, and that is the statement: every one of the 276 off-diagonal cells is the same colour. The
Gauss integrals lie in [0.9999, 1.0002] (n = 240 points per fibre), all of one sign. In the convention of this repository the sign is +1; the other
Hopf fibration (the mirror image, which is what `fibration.py` draws) has −1 everywhere. The README's sentence "every fibre links with every other
fibre" is thereby checked for 276 pairs instead of being illustrated with two. **Counted** (a midpoint-rule Gauss integral), accurate to about 2·10⁻⁴ here,
and unambiguous because the values are integers.

**How to read the right panel.** Two fibres over base points at angle θ on S² are at distance θ/2 from each other in S³, and every point of one is at that
same distance from the other: they are Clifford parallels. The dots sit on the line y = x with a largest deviation of 2.2·10⁻⁴, which is the resolution of
the sampled fibres (240 points, step 0.026 rad), not a discrepancy. The identity behind it is |⟨z, w⟩|² = (1 + b·b′)/2 for the hermitian inner product of
points over b and b′, hence distance = arccos|⟨z, w⟩| = θ/2 (`hopf_clifford.py` covers the Clifford parallels). Here it is **Counted** from the samples.

**Why both panels together.** Linking number 1 says the fibres cannot be pulled apart; the equidistance says they are nonetheless never close to each
other in S³ unless their base points are. In ℝ³ the stereographic image distorts distances badly (the minimal ℝ³ distance of two fibres is not a function
of θ), which is why the right panel is measured in S³ and not in the picture Blender shows.

---

## Figure 2 — a half-plane that sees every fibre once

![poincare section](figures/hopf_fig2_poincare_section.png?raw=true)

**The observation.** Take the half-plane H = {y = 0, x > 0} of ℝ³. Under the stereographic projection, y = 0 means Im z₁ = 0 and x > 0 means Re z₁ > 0, so
H is exactly the image of the set **{z₁ real and positive}**. Along a fibre e^{it}(z₁, z₂) the phase of z₁ advances uniformly, so the fibre passes through
"z₁ real and positive" exactly once per turn (unless z₁ = 0, which is the single fibre over the south pole: the z axis, the boundary of H).
So H is a *Poincaré section of the whole fibration*, and the map "base point ↦ the point where its fibre crosses H" is a bijection from S² minus the south
pole (a disc) onto H.

**Closed form.** With s = sin(el/2), c = cos(el/2), the canonical section of the fibre over (el, az) is (z₁, z₂) = (s, c·e^{−i·az}), and it lands at

    ρ = s / (1 + c·sin az),        z = c·cos az / (1 + c·sin az)        (ρ = √(x²+y²), the half-plane coordinates).

Checked against the numerically projected section at 40 random base points: largest error 1.4·10⁻¹⁴. **Exact.** The test also samples 25 random fibres at 4000
points each and counts crossings of H: one every time. **Counted.**

**The tori.** The fibres at one elevation sweep out a torus (constant |z₁| = sin(el/2)). Eliminating az from the formula above gives

    ρ² + z² + 1 = 2ρ / sin(el/2)     ⇔     (ρ − 1/s)² + z² = (c/s)²,

so the torus meets H in a **round circle with centre (1/sin(el/2), 0) and radius cot(el/2)**, checked to 2·10⁻¹⁴. These are the black circles. They form a
pencil with limit point (1, 0): as el → π the circle shrinks to the point (1, 0), the fibre over the north pole (the unit circle of the xy-plane); as el → 0 it
flattens into the line ρ = 0, the fibre over the south pole. The nested circles of Figure 1 in the README are the same picture seen as a rotation of this
half-plane about the z axis. **Exact.**

**The colours.** Left panel: S² as a rectangle (azimuth across, elevation up), hue = azimuth, brightness = elevation. Right panel: the same colour at the
image point in H. Following the colours around the point (1, 0) once, you meet every azimuth once — this is the north pole of S², where all azimuths
meet. The white curves (constant azimuth) are the half-plane cross-sections of the Hopf bands.

**Why it matters.** H shows the fibre bundle as a *picture of S² minus a point*, living in ℝ³, with no projection error beyond the stereographic map itself. It
also shows geometrically why the bundle is not trivial: the section above exists on everything except the south pole, and near the south pole its phase
e^{−i·az} winds once as the base point circles the pole. That winding is the obstruction (Chern number ±1), which `hopf_line_bundle.py` and
`HOPF_WITHOUT_PRINCIPAL_BUNDLES.md` compute from the projector alone.

---

## Figure 3 — the `makeFibre` audit as curves

![makeFibre audit](figures/hopf_fig3_makefibre_audit.png?raw=true)

The README and `hopf_math.py --tests` report an audit of `fibration.py` as a table. This figure samples 30 elevations from 0.1 to 3.0, reproduces the circle
`makeFibre` builds at each (without Blender, `hopf_math.blender_fibre`), fits a circle, and compares with the exact fibre: centre distance d = tan(el/2), radius
r = sec(el/2), tilt = el/2, and the defect r² − d² − 1, which is zero for a true Hopf fibre.

| elevation | shipped d | exact d | shipped r | exact r | defect r²−d²−1 | rel. error of r |
|---|---|---|---|---|---|---|
| 0.1 | 0.0419 | 0.0500 | 1.0017 | 1.0013 | 0.0016 | 0.04 % |
| 0.6 | 0.3006 | 0.3093 | 1.0695 | 1.0468 | 0.0534 | 2.2 % |
| 0.7 | 0.3650 | 0.3650 | 1.0976 | 1.0645 | 0.0715 | 3.1 % |
| 1.0 | 0.5945 | 0.5463 | 1.2217 | 1.1395 | 0.1390 | 7.2 % |
| 1.5 | 1.1634 | 0.9316 | 1.6256 | 1.3667 | 0.2892 | 19 % |
| 2.0 | 2.2306 | 1.5574 | 2.5402 | 1.8508 | 0.4767 | 37 % |
| 2.5 | 4.9612 | 3.0096 | 5.1290 | 3.1714 | 0.6926 | 62 % |
| 3.0 | 26.977 | 14.101 | 27.013 | 14.137 | 0.9298 | 91 % |

**What the six panels show.**
- *Tilt (bottom left)* is exact: the shipped circles lie in the right planes, to 10⁻⁹. Only the sizes are approximate.
- *Centre distance (top left)* is **below** the exact tan(el/2) for el < 0.70 and **above** it beyond: the shipped curve crosses the exact one at el ≈ 0.700,
  where the error is 0.0 %. The centre-distance error panel (bottom middle) therefore dips to a spike there; it is not a sign that the formula is good at
  0.7 specifically.
- *Radius (top middle)* is above the exact sec(el/2) at every sampled elevation and the two separate faster and faster toward the pole: near el = 3 the shipped circle is
  almost twice the exact one.
- *Defect (top right)* grows monotonically from 0.0016 to 0.93. The two-line replacement (`baseRad = tan(ele)`, `fibreRad = 1/cos(ele)`) is the green line at
  zero, to 10⁻⁹. **Exact.**
- *Face-on circles (bottom right)* draw each circle in its own plane (the tilt is exact, so shipped and exact share the plane): dashed = shipped, solid =
  exact, for el = 0.8, 1.8, 2.6. The shipped circles are not just a little too large; their centres are displaced as well, so they are not the Hopf circles.

**What this does and does not say.** The README already states that `makeFibre` is "close to but not exactly" Hopf fibres and that the picture is the
mirror image of this repository's convention once corrected. The figure adds how the error behaves: small and benign for el below 1, large near the
pole. It does not say the Blender renders are wrong as pictures — nearby shipped fibres still link and still do not intersect, which is what the render shows —
only that they are a deformation of the fibration and not the fibration. `fibration.py` is untouched.

---

## Figure 4 — the Hopf invariant as a linking number

![hopf invariant](figures/hopf_fig4_hopf_invariant.png?raw=true)

**The construction.** For integers p, q ≥ 1 define

    f_{p,q}(z₁, z₂) = Hopf( z₁ᵖ, z₂^q ) / normalisation  :  S³ → S².

Classically (Hopf 1931) the Hopf invariant of a map S³ → S² is the linking number of the preimages of two regular values. The Hopf map itself has
invariant 1, and composing with the map (z₁, z₂) ↦ (z₁ᵖ, z₂^q)/norm, which has degree p·q, multiplies the invariant by p·q, so **H(f_{p,q}) = p·q**
(**Quoted**, standard).

**The preimage curves, in closed form.** The preimage of the base point b = (el, az) is the set where (z₁ᵖ, z₂^q) is a complex multiple of the canonical
lift (sin(el/2), cos(el/2)·e^{−i·az}). Solving gives, for coprime p and q, **one closed curve**

    z₁ = r₁·e^{i·q·φ},      z₂ = r₂·e^{i(p·φ − az/q)},      φ ∈ [0, 2π),
    r₁ᵖ / r₂^q = tan(el/2),    r₁² + r₂² = 1,

the (q, p) torus knot on the torus |z₁| = r₁. The test checks that f_{p,q} maps these points back to b for (p,q) ∈ {(1,1), (2,1), (3,1), (2,3), (3,2)} at three
base points each: largest error 6·10⁻¹⁵. **Exact.**

**The numbers.** The Gauss linking integral of the preimages of two base points, (1.4, 0.3) and (2.0, 2.5), computed on 1500-point polygons after stereographic
projection:

| (p, q) | linking number | p·q |
|---|---|---|
| (1, 1) | +1.0000 | 1 |
| (2, 1) | +2.0000 | 2 |
| (3, 1) | +3.0000 | 3 |
| (2, 3) | +6.0001 | 6 |
| (3, 2) | +6.0001 | 6 |

**Counted** (a numerical integral), and all of one sign, which is one orientation convention applied consistently. In the figure the blue and red
curves are the two preimages; for (1,1) they are the Hopf fibres themselves, two linked round circles, and for (2,3) each is a trefoil-type (3,2) torus knot
winding around the other six times in total.

**Why this figure is in the repository.** It connects the picture of fibres to the quantity that decides the Hopf insulator in the companion repository
[quantum-lattice](https://github.com/brentharts/quantum-lattice) (`HOPF_INSULATOR.md`, `HOPF_INSULATOR_FIGURES.md`): there a map from the Brillouin-zone torus to S² has
Hopf invariant 0, −1 or +2 depending on a parameter h, and the same linking-number definition is evaluated on that map's preimages. The case p·q = 2 here
is the model for the +2 phase, in the sense that both are maps of Hopf invariant 2; this figure does not claim the preimages have the same shape (the
insulator's preimage of each of the two points tested at h = 0.5 is a pair of loops, not one knot).

---

## How the checks map to the tests

| claim | test in `--tests` | tolerance | label |
|---|---|---|---|
| all 276 pairs link ±1 with one sign | fig1 (reduced 12-fibre atlas) | 5 % | Counted |
| fibres are Clifford parallels | fig1 | 2·10⁻³ | Counted |
| closed form of the section in H | fig2 | 10⁻¹² | Exact |
| one crossing per fibre | fig2 | 25 fibres, exactly 1 | Counted |
| torus meets H in the circle (1/s, 0), radius c/s | fig2 | 10⁻¹² | Exact |
| exact version of `makeFibre` is a true fibre | fig3 | 10⁻⁹ | Exact |
| shipped circles are not fibres (defect > 0.04); shipped tilt is exact | fig3 | 0.04; 10⁻⁹ | Counted; Exact |
| preimage curve maps to b | fig4 | 10⁻¹² | Exact |
| linking = p·q, one sign | fig4 | 6 % | Counted |

The atlas test uses 3 elevations × 4 azimuths to keep it fast; the figure itself uses 4 × 6.

## What is not claimed

- Nothing here depends on, or says anything about, the Spectre-tiling motivation of `hopf_rotations.py` and `hopf_berger.py`.
- Figure 2's colour scheme is its own (hue = azimuth, brightness = elevation), chosen for legibility; it is not `fibration.py`'s colouring, which
  `hopf_s2_view.py` reproduces.
- The Blender helpers remain untested for lack of Blender; none of these scripts imports `bpy`.
