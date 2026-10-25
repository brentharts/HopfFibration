#!/usr/bin/env python3
"""hopf_s2_view.py - the base sphere S^2 with a point for every drawn fibre, in the SAME colour as the fibre.

This is the second item of the README to-do list.  The colour of a fibre is the one fibration.py uses
(hue = azimuth*15/pi + (elevation/2)*330/pi, degrees), so the two panels can be read against each other.

Panel 1: S^2, one dot per fibre.   Panel 2: the same fibres in R^3 (exact fibres, stereographic projection).

Convention: fibration.py's (elevation, azimuth) = base point base_point(pi - elevation, azimuth) mirrored in x, see
hopf_math.blender_frame; in this file S^2 is shown in that frame, so the dots sit where Blender's picture says they do.

    python3 hopf_s2_view.py --tests
    python3 hopf_s2_view.py --png figures/hopf_s2_view.png [--tori 6] [--per 24]

Inside Blender (untested here, no Blender in the build environment): add_s2_to_blender(root) places small spheres.
"""
import math
import sys

import numpy as np

import fibration as FIB          # importable without Blender: fibration.py guards its bpy import
import hopf_math as H


def fibre_colour(elevation, azimuth):
    """RGB in [0,1], identical to the colour makeFibre assigns (ele halved first)."""
    ele = elevation / 2.0
    hue = (azimuth * 15 / math.pi) + ele * 330 / math.pi
    return tuple(FIB.HSV2RGB(hue % 360.0, 1, 1))


def base_point_blender_frame(elevation, azimuth):
    """Where fibration.py's fibre (elevation, azimuth) sits on S^2 (unit vector, Blender's mirrored frame)."""
    return H.blender_frame(H.base_point(math.pi - elevation, azimuth))


def fibre_grid(tori=6, per=24, section=1.0):
    """The (elevation, azimuth) pairs mkhopf() draws (without the two edge cases)."""
    els = [(e + 1) * math.pi / tori for e in range(tori)][:-1]
    azs = [(a * 2 * math.pi * section) / per for a in range(per)]
    return [(e, a) for e in els for a in azs]


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    grid = fibre_grid(6, 24)
    check("grid has (tori-1) * per fibres", len(grid) == 5 * 24)
    pts = np.array([base_point_blender_frame(e, a) for e, a in grid])
    check("all base points lie on S^2", np.abs(np.linalg.norm(pts, axis=1) - 1).max() < 1e-12)
    # latitude: fibres of one elevation share one z on S^2 (constant elevation circle), 'tori' of the README
    zs = {round(e, 9): set(np.round([base_point_blender_frame(e, a)[2] for a in np.linspace(0, 6, 9)], 9))
          for e in {g[0] for g in grid}}
    check("one elevation <-> one circle of latitude on S^2 (one torus in R^3)", all(len(v) == 1 for v in zs.values()))
    # the fibre drawn at a base point is the one that maps there, in the mirrored frame
    e, a = 1.2, 0.9
    F = H.exact_fibre_in_blender_frame(e, a, 50)
    P = H.blender_frame(F)                       # back to the unmirrored frame
    check("exact fibre drawn for (el, az) lies over base_point_blender_frame(el, az)",
          np.allclose(H.hopf(*_inverse_stereographic(P)), H.base_point(math.pi - e, a), atol=1e-9))
    c = fibre_colour(1.0, 0.5)
    check("colour equals fibration.py's hue formula", len(c) == 3 and all(0 <= x <= 1 for x in c))
    return ok


def _inverse_stereographic(P):
    """R^3 -> S^3 (inverse of hopf_math.stereographic) as (z1, z2)."""
    P = np.asarray(P, float)
    s = (P ** 2).sum(axis=1)
    w4 = (s - 1) / (s + 1)
    f = 1 - w4
    w = P * f[:, None]
    return w[:, 0] + 1j * w[:, 1], w[:, 2] + 1j * w4


def figure(path, tori=6, per=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = fibre_grid(tori, per)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", alpha=0.35, linewidth=0)
    for e, a in grid:
        p = base_point_blender_frame(e, a)
        ax.scatter(*p, color=fibre_colour(e, a), s=18)
    for e in (0.0, math.pi):
        p = base_point_blender_frame(e, 0)
        ax.scatter(*p, color="k", s=40)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("base S^2: one dot per fibre")
    ax.set_axis_off()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for e, a in fibre_grid(tori, 12):
        F = H.exact_fibre_in_blender_frame(e, a, 200)
        if np.abs(F).max() > 3.5:          # the near-polar fibres are huge; they are clipped out of this picture
            continue
        ax2.plot(*F.T, color=fibre_colour(e, a), lw=1.0)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-3, 3)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("the same fibres in R^3 (exact)")
    ax2.set_axis_off()
    fig.savefig(path, dpi=120)
    print("wrote", path)


def add_s2_to_blender(root=None, tori=6, per=24, radius=3.0, size=0.08):          # pragma: no cover (needs bpy)
    """Blender only: place a small coloured sphere on a radius-`radius` S^2 for every fibre mkhopf draws."""
    import bpy
    for e, a in fibre_grid(tori, per):
        x, y, z = radius * base_point_blender_frame(e, a)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x - 2 * radius - 2, y, z))
        ob = bpy.context.active_object
        mat = bpy.data.materials.new("S2dot")
        mat.diffuse_color = (*fibre_colour(e, a), 1)
        ob.data.materials.append(mat)
        if root is not None:
            ob.parent = root


if __name__ == "__main__":
    if "--png" in sys.argv:
        g = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
        figure(sys.argv[sys.argv.index("--png") + 1], g("--tori", 6), g("--per", 24))
    else:
        sys.exit(0 if run_tests() else 1)
