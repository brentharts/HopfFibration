#!/usr/bin/env python3
"""hopf_tests.py - run the self-tests of every hopf_*.py script (none needs Blender).

    python3 hopf_tests.py
Exit status 0 only if all pass.
"""
import importlib
import sys
import time

MODULES = ["hopf_math", "hopf_clifford", "hopf_s2_view", "hopf_rotations", "hopf_berger",
           "hopf_quaternion_rotor", "hopf_division_algebras", "hopf_anim"]

if __name__ == "__main__":
    failed = []
    for name in MODULES:
        t0 = time.time()
        print("== %s" % name)
        ok = importlib.import_module(name).run_tests(verbose=False)
        print("   %s  (%.1f s)" % ("ok" if ok else "FAIL", time.time() - t0))
        if not ok:
            failed.append(name)
    print("all passed" if not failed else "FAILED: " + ", ".join(failed))
    sys.exit(1 if failed else 0)
