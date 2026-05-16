"""Deprecated. SteadyDancer inference moved to subprocess via
ec2.models.steadydancer.SteadyDancerModel.animate(), which invokes the
upstream `generate_dancer.py` CLI.

This file is kept as a compatibility shim only — importing it raises so
callers fail loudly rather than silently using a stale path.
"""
from __future__ import annotations


def steadydancer_infer(*args: object, **kwargs: object) -> None:
    raise RuntimeError(
        "steadydancer_infer() was removed; SteadyDancer is now invoked as "
        "`generate_dancer.py` via subprocess. See ec2.models.steadydancer.SteadyDancerModel."
    )
