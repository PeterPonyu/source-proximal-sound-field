"""Corrected common-domain sparse Bayesian learning for TB-M2-SBLFIX (registered 2026-11-05).

Identical to v2.sbl_common (version-1 updates, monopoles on the common search domain at lambda/4, coarsened to at most
6,000 nodes, plus plane waves) except for the guard: dictionary monopoles within 5 cm of any microphone *or any target*
are removed. On MeshRIR's planar 0.05 m grid the unguarded dictionary placed atoms on or within millimetres of target
points, which made predictions singular (TB-M2 record).
"""
from __future__ import annotations
import contextlib
import numpy as np
from scipy.spatial.distance import cdist
import estimators as E
import v2


@contextlib.contextmanager
def guarded_dictionary(stats=None):
    orig = E.sbl_dictionary

    def guarded(prob, spacing_wl=0.25, n_dirs=None, max_nodes=v2.SBL_NODES):
        lo, hi = prob.search_box()
        nodes = E.grid_nodes(lo, hi, spacing_wl * prob.lam, max_nodes)
        keep_mic = cdist(prob.x, nodes).min(axis=0) > E.GUARD
        keep_tgt = cdist(prob.q, nodes).min(axis=0) > E.GUARD
        if stats is not None:
            stats.append(dict(nodes=int(len(nodes)), removed_by_mics=int((~keep_mic).sum()),
                              removed_by_targets_only=int((keep_mic & ~keep_tgt).sum())))
        nodes = nodes[keep_mic & keep_tgt]
        R = 0.5 * float(np.linalg.norm(prob.box[1] - prob.box[0]))
        if n_dirs is None:
            N = int(np.clip(np.ceil(prob.k * R) + 1, 3, 24))
            n_dirs = 2 * (N + 1) ** 2
        from fields import fibonacci_sphere
        return nodes, fibonacci_sphere(n_dirs)
    E.sbl_dictionary = guarded
    try:
        yield
    finally:
        E.sbl_dictionary = orig


def sbl_guarded(p, rule, stats=None):
    with guarded_dictionary(stats):
        return E.sbl_fit(p, rule)
