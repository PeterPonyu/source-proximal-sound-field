"""TB-PROX-S2: mo26-prox with the JASA 2026 paper's own stage-2 stop (PROTOCOLS/TB-PROX-S2.md).

The registered `prox2.stage_refine` (TB-PROX) stops stage 2 as soon as the penalised objective L + zeta * sum |w| rises.
Algorithm 2 of the paper computes L[i] with its eq. 26 (the negative log marginal likelihood, no penalty) and stops when
L[i+1] - L[i] > 0 (its eq. 68). `stage_refine_nlml` is the registered function with that one change: the quantity that
is recorded and compared is the unpenalised NLML. The weight step (proximal, the same threshold eta * zeta), the radius
and centre steps, the projection, the iteration cap and every setting are the registered ones.

`mo26_prox_s2` runs the registered `prox2.mo26_prox` unchanged, with `prox2.stage_refine` swapped for
`stage_refine_nlml` for the duration of the call (the frozen module is imported, never edited), so stage 1, the
dictionary, the scaling and the prediction are the registered code by construction.
"""
from __future__ import annotations
import contextlib
import prox2 as PX


def stage_refine_nlml(x, ks, cs, Bases, ys, cen, rad, W, lo, hi, cfg, record=None):
    """JASA 2026 Algorithm 2, stage 2, with the paper's stop: on the active set, joint steps of the weights (proximal),
    radii and centres (plain gradient); stop as soon as the NLML (eq. 26) rises (eq. 68), keeping the previous iterate;
    at most refine_max steps."""
    Lt, gW, gc, gr = PX.refine_loss_grads(x, ks, cs, Bases, ys, cen, rad, W)
    cur = Lt
    if record is not None:
        record.append(cur)
    for _ in range(cfg['refine_max']):
        Wn = PX.group_prox(W - cfg['eta'] * gW, cfg['eta'] * cfg['zeta'])
        cn, rn = PX._project(cen - cfg['eta_chi'] * gc, rad - cfg['eta_rho'] * gr, cen, rad, x, lo, hi)
        Ln, gWn, gcn, grn = PX.refine_loss_grads(x, ks, cs, Bases, ys, cn, rn, Wn)
        if record is not None:
            record.append(Ln)
        if Ln > cur:
            break
        W, cen, rad, gW, gc, gr, cur = Wn, cn, rn, gWn, gcn, grn, Ln
    return cen, rad, W


@contextlib.contextmanager
def paper_stop():
    """Within the block, prox2.mo26_prox calls stage_refine_nlml in place of the registered stage_refine."""
    old = PX.stage_refine
    PX.stage_refine = stage_refine_nlml
    try:
        yield
    finally:
        PX.stage_refine = old


def mo26_prox_s2(prep, f, record=None):
    """The registered prox2.mo26_prox for bin f (prep built with prox2.MO26), with the paper's stage-2 stop."""
    with paper_stop():
        return PX.mo26_prox(prep, f, record)
