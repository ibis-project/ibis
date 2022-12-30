from __future__ import annotations


def define_env(env):
    import ibis

    env.variables["ibis"] = ibis
    env.variables["sorted"] = sorted
