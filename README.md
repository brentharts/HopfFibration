# HopfFibration

Download Blender 2.9 (or later) for free here: 
https://www.blender.org/download/

## Description
### Overview of the Hopf Fibration
The Hopf Fibration is the fibre bundle created by projecting the four-dimenstional sphere **S3** onto the three-dimentional sphere **S2**, every point on **S2** has a corresponding fibre in three-dimensional space **R3** which is a perfect two-dimensional sphere (or circle) **S1**. This structure is written as **S1 -> S3 -> S2**, which can be read as; "The fibre-space **S1** is embedded in the full-space **S3** which is projected to the base-space **S2**". 

This fibre bundle is space-filling in **R3**, every fibre links with every other fibre and no two fibres ever intersect. Lines of constant elevation (&epsilon;) on **S2** are seen as tori in the fibre-space, lines of constant azimuth (&alpha;) appear has Hopf bands. There are two edge cases at the poles of **S2**; at the south pole (&alpha; = 0) the fibres converge to a circle lying flat in the X-Y plane, at the north pole (&alpha; = &pi;) the fibre is a circle through infinity (here visualised as the straight line lying on the Z-axis).

### This project
The file *fibration.py* is a visualisation tool which uses the Python Blender API. 
This includes the function `makeFibre(elevation,azimuth)` which creates the fibre corresponding to the point on the sphere **S2** at the coordinate (&epsilon;,&alpha;). 
Fibres are visualised as tori in order to be visible, the cross-sectional diameter can be adjusted with the optional argument `sectionRad`. Also included is a function to convert (&epsilon;,&alpha;) into RGB colours for visualisation.

## Example renders

![Hopf fibration: 6 nested tori](https://github.com/For-The-Wolf/HopfFibration/blob/main/Example%20Renders/hopf_fibration_1.JPG?raw=true)
**Figure 1** Fibres at 100 equally spaced azimuths and 8 equally spaced elevations. This results in 6 nested tori and the two edge cases.

![Hopf fibration: section](https://github.com/For-The-Wolf/HopfFibration/blob/main/Example%20Renders/hopf_fibration_3.JPG?raw=true)
**Figure 2** As in Figure 1 but fewer fibres are rendered showing a cross-section.

![Hopf fibration: 25 fibres at e = pi/2](https://github.com/For-The-Wolf/HopfFibration/blob/main/Example%20Renders/hopf_elevation_0_5pi.JPG?raw=true)

**Figure 3** 25 fibres all at &epsilon; = &pi;/2. Two fibres are shown in pink to highlight how the fibres link but do not intersect. Edge cases also shown in blue and red.

![Hopf fibration: colour trial](https://github.com/For-The-Wolf/HopfFibration/blob/main/Example%20Renders/hopf_fibration_4.JPG?raw=true)
**Figure 4** An earlier colouring scheme, I have still not finalised this, there is a lot to play around with.

## Directions

Simply make a new Blender file or download the provided example(*fibration.blend*), create a Text Editor pane, open the *fibration.py* file and use the keyboard shortcut `Alt`+`P` to run the script. 
This will run an example visualisation which generates fibres constructing six tori and the two edge cases at &epsilon; = {0,pi/2} (a flat cricle and a circle through infinity respectively).

![Hopf fibration visualisation in blender](https://github.com/For-The-Wolf/HopfFibration/blob/main/Example%20Renders/hopf_fibration_SS.JPG?raw=true)

## To do
- [ ] Improve the automatic colouring of fibres
- [x] Add a visualisation of **S2** showing points corresponding to rendered fibres of the same colour (`hopf_s2_view.py`, matplotlib; the Blender helper `add_s2_to_blender` is untested)
- [~] Add functionality for easy animation creation (`hopf_anim.py` writes GIFs without Blender; the Blender keyframe helper is untested) 

## Python scripts that do not need Blender

The scripts below use only numpy (and matplotlib/Pillow for pictures). Each has `--tests`; `python3 hopf_tests.py` runs them all.
Convention used throughout: a point of S3 is (z1, z2) in C2 with |z1|^2+|z2|^2 = 1, the Hopf map is
h = (2 Re z1 conj z2, 2 Im z1 conj z2, |z1|^2-|z2|^2), a fibre is e^{it}(z1, z2), and the stereographic projection is from (0,0,0,1).

| script | what it does |
|---|---|
| `hopf_math.py` | the exact core: map, fibres, stereographic projection, Gauss linking number, quaternion form, and an **audit of `makeFibre`** (below) |
| `hopf_clifford.py` | Clifford parallels (constant distance = half the base angle), left and right families, the Clifford torus and Villarceau circles |
| `hopf_s2_view.py` | the base S2 with one dot per fibre, in the colour `makeFibre` gives that fibre |
| `hopf_rotations.py` | the 12-fold rotation group of the Spectre tiling's 4D lift acting on C2: 60 degrees = the Hopf circle action, 30 degrees = a rotation by pi of the base |
| `hopf_berger.py` | Berger spheres (fibre scaled by s), exact Ricci curvature, the collapsing fibre (Mixmaster / "pancake"), and a labelled numerology check |
| `hopf_quaternion_rotor.py` | rotors, the SU(2)->SO(3) double cover, the qubit and Bloch sphere, Berry phase = half the solid angle |
| `hopf_line_bundle.py` | the Hopf fibration **without a principal bundle**: the tautological line bundle, connection and holonomy from the projector alone, three independent holonomy methods, Chern number 1 (see [`HOPF_WITHOUT_PRINCIPAL_BUNDLES.md`](HOPF_WITHOUT_PRINCIPAL_BUNDLES.md)) |
| `hopf_division_algebras.py` | the four Hopf fibrations S0->S1, S1->S3, S3->S7, S7->S15 from R, C, H, O by one formula; why the octonionic one is not a principal bundle |
| `hopf_anim.py` | GIF animations (a fibre sweeping a torus; a qubit rotating), and Blender keyframe helper |

### Audit of `makeFibre`

`hopf_math.py --tests` reproduces the circles that `fibration.py` builds, without Blender, and compares them with the exact
Hopf fibres. The exact fibre over elevation `ele` (halved, as in `makeFibre`) is a circle whose centre is at distance
`tan(ele)` from the z axis, with radius `sec(ele)`, in a plane tilted by `ele`. `fibration.py` has the correct tilt, the right
linking and the right overall picture, but uses a smooth approximation to `tan` and `sec`:

    baseRad  = 2*((1/(pi/2 - ele)) - (2/pi))            # approximates tan(ele)
    fibreRad = baseRad + 1/(baseRad + 1)                # approximates sec(ele)

so the circles are *close to* but not exactly Hopf fibres (for example r^2 - d^2 - 1 is 0.05 at elevation 0.6 and 0.69 at
2.5 instead of 0). Replacing the two lines by `baseRad = tan(ele)` and `fibreRad = 1/cos(ele)` makes every circle an exact
fibre, as the test checks (the picture is then the mirror image of the usual convention, which is also a Hopf fibration).
`fibration.py` itself is unchanged; the patch is for you to take or leave. Near the poles the circles get large (radius 14 at
elevation 3.0), so the cross-section `sectionRad` and the clipping distance may want adjusting.

### Notes on what the scripts do and do not claim

`hopf_rotations.py` and `hopf_berger.py` come from a study of the Spectre aperiodic monotile
([brentharts/spectre](https://github.com/brentharts/spectre), see `HOPF.md` and `CONNECTIONS.md` there). The 60-degree statement is
exact but elementary (any complex plane with a scalar phase action has it); the "6.0900 levels per turn is almost 6" is numerology,
asserted in the tests to miss 6. Nothing here claims the Hopf fibration explains the Spectre. The Blender helpers (`add_s2_to_blender`,
`keyframes_for_blender`) could not be run in the build environment (no Blender) and are untested.
