#!/usr/bin/env python3
"""hopf_anim.py - animations of the Hopf fibration, with and without Blender (README to-do: 'easy animation creation').

Two animations, both from the exact fibres of hopf_math.py:
  sweep    the base point runs once round a circle of latitude of S^2 and its fibre sweeps out the torus over that circle
           (left: S^2 with the moving point; right: the fibre in R^3 and the trail of earlier fibres).
  rotate   a qubit evolves under H = (Omega/2) n.sigma (hopf_quaternion_rotor.evolve): the Hopf image of the state (its Bloch
           vector with y reflected) circles the axis, and the fibre over it is the set of phases of the same physical state.

    python3 hopf_anim.py --tests
    python3 hopf_anim.py --gif figures/hopf_sweep.gif [--mode sweep|rotate] [--frames 48] [--el 1.2]

The GIF writer needs matplotlib and Pillow.  For Blender, keyframes_for_blender() (below, untested here because the build
environment has no Blender) keys the base point of a single fibre object; use it from fibration.py's HOPF root.
"""
import math
import os
import sys
import tempfile

import numpy as np

import hopf_math as H
import hopf_quaternion_rotor as Q


def sweep_points(el, n):
    """n base points round the circle of latitude of elevation el."""
    return [H.base_point(el, 2 * math.pi * k / n) for k in range(n)]


def rotate_points(n_frames, axis=(0.3, 0.2, 1.0), psi0=None):
    psi0 = np.array([math.cos(0.5), math.sin(0.5) * np.exp(0.3j)]) if psi0 is None else psi0
    out = []
    for k in range(n_frames):
        psi = Q.evolve(psi0, axis, 2 * math.pi * k / n_frames)
        a, b = psi
        out.append(H.hopf(a, b))
    return out


def render_frames(mode="sweep", frames=24, el=1.2, size=(8, 4), dpi=70):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    pts = sweep_points(el, frames) if mode == "sweep" else rotate_points(frames)
    trail = [H.fibre_r3(p, 160) for p in pts]
    imgs = []
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    for k, p in enumerate(pts):
        fig = plt.figure(figsize=size, dpi=dpi)
        a1 = fig.add_subplot(1, 2, 1, projection="3d")
        a1.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", alpha=0.3, linewidth=0)
        a1.plot(*np.array(pts[:k + 1]).T, color="tab:red", lw=1)
        a1.scatter(*p, color="tab:red", s=40)
        a1.set_axis_off()
        a1.set_box_aspect((1, 1, 1))
        a2 = fig.add_subplot(1, 2, 2, projection="3d")
        for j in range(k):
            a2.plot(*trail[j].T, color="0.8", lw=0.5)
        a2.plot(*trail[k].T, color="tab:red", lw=1.8)
        lim = 3.0
        a2.set_xlim(-lim, lim)
        a2.set_ylim(-lim, lim)
        a2.set_zlim(-lim, lim)
        a2.set_axis_off()
        a2.set_box_aspect((1, 1, 1))
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    return imgs


def write_gif(path, mode="sweep", frames=24, el=1.2):
    imgs = render_frames(mode, frames, el)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    return path


def keyframes_for_blender(fibre_object, el=1.2, frames=48, fps=24):          # pragma: no cover (needs bpy)
    """Blender only: key rotation_euler.z of one fibre object so its base point sweeps a latitude circle, matching
    makeFibre's convention (rotation_euler.z = -azimuth)."""
    import bpy
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, frames
    for k in range(frames):
        fibre_object.rotation_euler.z = -2 * math.pi * k / frames
        fibre_object.keyframe_insert("rotation_euler", index=2, frame=1 + k)


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    pts = sweep_points(1.2, 12)
    zs = {round(p[2], 12) for p in pts}
    check("sweep: base points on one circle of latitude of S^2", len(zs) == 1 and all(abs(np.linalg.norm(p) - 1) < 1e-12 for p in pts))
    # the swept fibres fill the torus |z1| = const: same |z1| on every fibre
    mods = {round(abs(H.section(p)[0]), 9) for p in pts}
    check("the fibres of a latitude circle lie on one torus |z1| = const", len(mods) == 1)
    rp = rotate_points(16)
    n_ref = np.array([0.3, -0.2, 1.0]) / np.linalg.norm([0.3, -0.2, 1.0])      # axis n with y reflected (see rotate_points)
    check("rotate: the Hopf image of the state circles the axis n (y reflected): constant component along it",
          len(rp) == 16 and np.ptp([float(p @ n_ref) for p in rp]) < 1e-9)
    try:
        import matplotlib  # noqa: F401
        from PIL import Image  # noqa: F401
        with tempfile.TemporaryDirectory() as d:
            path = write_gif(os.path.join(d, "t.gif"), "sweep", 4)
            size = os.path.getsize(path)
            check("writes a GIF with 4 frames", size > 1000 and Image.open(path).n_frames == 4, "(%d bytes)" % size)
    except ImportError:
        check("GIF writer (matplotlib + Pillow) available - skipped", True)
    return ok


if __name__ == "__main__":
    if "--gif" in sys.argv:
        arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        print("wrote", write_gif(arg("--gif", "figures/hopf_sweep.gif"), arg("--mode", "sweep"),
                                 int(arg("--frames", 48)), float(arg("--el", 1.2))))
    else:
        sys.exit(0 if run_tests() else 1)
