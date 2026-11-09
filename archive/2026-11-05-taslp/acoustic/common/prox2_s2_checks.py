"""Known-answer checks of the TB-PROX-S2 change (common/prox2_s2.py). Run before registration and before any registered
output:  python3 -B prox2_s2_checks.py OUT.json   (exit 0 and "passed": true when every check passes).

Problem: a free-field monopole 0.5 m outside a 24-microphone box (one bin at 500 Hz, 40 dB SNR) with the registered
MO26 settings and dictionary (common/prox2.py, imported unchanged). Every call of prox2.refine_loss_grads is logged
(NLML, penalty and the iterate it was evaluated at), for the registered stop and for the paper's stop.

1. swap: inside prox2_s2.paper_stop() prox2.stage_refine is stage_refine_nlml; after the block, even after an
   exception, it is the registered function again.
2. pipeline: prox2_s2.mo26_prox_s2 and the registered prox2.mo26_prox on the same bin give bit-identical stage-1
   records and active sets; their stage-2 evaluations are bit-identical up to the first stop; each run stopped by its
   own rule (registered: the penalised objective rose; paper's: the NLML rose; or the cap); and when both stop at the
   same step, the predictions are bit-identical.
3. stop rules with rough rates (centre and radius rates 1e-2, at which stage 2 diverges in development), from the same
   stage-1 active set: the same two properties, and both runs stop before the cap, so each rule is exercised.
4. a discriminating case (weight step 1e-2, centre and radius rates 1e-4, from the same active set; found by a scan of
   step sizes on this problem): the two rules stop at different steps, the evaluations are bit-identical up to the
   earlier stop, and each run stopped by its own rule.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prox2 as PX  # noqa: E402
import prox2_s2 as S2  # noqa: E402
import prox2_checks as PC  # noqa: E402
import v2  # noqa: E402

C0 = 343.0


def band_one_bin(seed=11, M=24, fc=500.0, snr_db=40.0):
    rng = np.random.default_rng(seed)
    nodes = PC.region_nodes()
    src = np.array([nodes[:, 0].max() + 0.5, 0.6, 0.5])
    mic = np.sort(rng.choice(len(nodes), M, replace=False)); tgt = np.setdiff1d(np.arange(len(nodes)), mic)
    k = 2 * np.pi * fc / C0
    p = lambda pts: np.exp(1j * k * np.linalg.norm(pts - src, axis=1)) / np.linalg.norm(pts - src, axis=1)
    y = p(nodes[mic]); s = np.sqrt(np.mean(np.abs(y) ** 2))
    y = y + (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2) * s * 10 ** (-snr_db / 20)
    return v2.Band(nodes[mic], y[:, None], nodes[tgt], np.array([fc]), C0, (nodes.min(0), nodes.max(0)))


class Log:
    """Wraps prox2.refine_loss_grads and logs (NLML, penalty, iterate) of every call."""

    def __init__(self):
        self.calls, self.orig = [], PX.refine_loss_grads

    def __enter__(self):
        def wrapped(x, ks, cs, Bases, ys, cen, rad, W):
            out = self.orig(x, ks, cs, Bases, ys, cen, rad, W)
            self.calls.append(dict(L=out[0], pen=float(np.sum(np.linalg.norm(W, axis=0))), W=W.copy(), cen=cen.copy(), rad=rad.copy()))
            return out
        PX.refine_loss_grads = wrapped
        return self

    def __exit__(self, *a):
        PX.refine_loss_grads = self.orig


def same(a, b):
    return a['L'] == b['L'] and a['pen'] == b['pen'] and np.array_equal(a['W'], b['W']) and np.array_equal(a['cen'], b['cen']) \
        and np.array_equal(a['rad'], b['rad'])


def stop_ok(calls, value, cap):
    """The run evaluated the start and one trial per step; it stopped at the first trial whose value exceeded the last
    accepted one, or after cap accepted steps. Returns (rule holds, stopped before the cap)."""
    acc = value(calls[0]); n = len(calls) - 1
    for i in range(1, len(calls)):
        v = value(calls[i])
        if v > acc:
            return i == n, True
        acc = v
    return n == cap, False


def prefix(a, b):
    n = min(len(a), len(b))
    return all(same(x, y) for x, y in zip(a[:n], b[:n])), n


def check_swap():
    orig = PX.stage_refine
    with S2.paper_stop():
        inside = PX.stage_refine is S2.stage_refine_nlml
    after = PX.stage_refine is orig
    try:
        with S2.paper_stop():
            raise RuntimeError
    except RuntimeError:
        pass
    after_exc = PX.stage_refine is orig
    return dict(name='swap', inside=inside, restored=after, restored_after_exception=after_exc, passed=bool(inside and after and after_exc))


def check_pipeline(prep):
    zeta = prep.cfg['zeta']; cap = prep.cfg['refine_max']
    r_reg, r_s2 = [], []
    with Log() as lg_reg:
        a = PX.mo26_prox(prep, 0, r_reg)
    with Log() as lg_s2:
        b = S2.mo26_prox_s2(prep, 0, r_s2)
    n1 = a['info']['iterations'][0]
    stage1 = (n1 == b['info']['iterations'][0] and a['info']['after_stage1'] == b['info']['after_stage1'] and r_reg[:n1] == r_s2[:n1])
    pre, n = prefix(lg_reg.calls, lg_s2.calls)
    reg_ok, reg_early = stop_ok(lg_reg.calls, lambda c: c['L'] + zeta * c['pen'], cap)
    s2_ok, s2_early = stop_ok(lg_s2.calls, lambda c: c['L'], cap)
    records = r_reg[n1:] == [c['L'] + zeta * c['pen'] for c in lg_reg.calls] and r_s2[n1:] == [c['L'] for c in lg_s2.calls]
    same_stop = len(lg_reg.calls) == len(lg_s2.calls)
    preds = (not same_stop) or np.array_equal(a['pred'], b['pred'])
    return dict(name='pipeline', stage1_identical=bool(stage1), active_after_stage1=a['info']['after_stage1'],
                stage2_evaluations=[len(lg_reg.calls), len(lg_s2.calls)], common_prefix_identical=bool(pre), common_prefix=n,
                registered_stop_rule=bool(reg_ok), registered_stopped_early=bool(reg_early), paper_stop_rule=bool(s2_ok),
                paper_stopped_early=bool(s2_early), records_are_the_stopped_quantities=bool(records), same_stop=bool(same_stop),
                predictions_identical_when_same_stop=bool(preds),
                passed=bool(stage1 and pre and reg_ok and s2_ok and records and preds and a['info']['after_stage1'] > 0))


def check_rates(prep, name, eta, rate, require):
    """From the stage-1 active set of bin 0, both stage-2 functions with the weight step eta and the centre and radius
    rates `rate`; require 'both_early' (both stop before the cap) or 'differ' (they stop at different steps)."""
    cfg = dict(prep.cfg, eta=eta, eta_chi=rate, eta_rho=rate)
    zeta, cap = cfg['zeta'], cfg['refine_max']
    c0 = prep.cfg
    W = PX.stage_weights([prep.Bases[0]], [prep.ys[0]], [prep.Ks[0]], c0['zeta'], c0['eta'], c0['iters'], c0['eps1'])
    keep = np.flatnonzero(W[0] > 0)
    args = lambda: (prep.x, [prep.ks[0]], [prep.cs[0]], [prep.Bases[0]], [prep.ys[0]], prep.cen[keep].copy(), prep.rad[keep].copy(),
                    W[:, keep].copy(), prep.lo, prep.hi, cfg)
    with Log() as lg_reg:
        PX.stage_refine(*args())
    with Log() as lg_s2:
        S2.stage_refine_nlml(*args())
    pre, n = prefix(lg_reg.calls, lg_s2.calls)
    reg_ok, reg_early = stop_ok(lg_reg.calls, lambda c: c['L'] + zeta * c['pen'], cap)
    s2_ok, s2_early = stop_ok(lg_s2.calls, lambda c: c['L'], cap)
    differ = len(lg_reg.calls) != len(lg_s2.calls)
    req = (reg_early and s2_early) if require == 'both_early' else differ
    return dict(name=name, eta=eta, rate=rate, active=int(len(keep)), stage2_evaluations=[len(lg_reg.calls), len(lg_s2.calls)],
                common_prefix_identical=bool(pre), common_prefix=n, registered_stop_rule=bool(reg_ok), registered_stopped_early=bool(reg_early),
                paper_stop_rule=bool(s2_ok), paper_stopped_early=bool(s2_early), stops_differ=bool(differ), requirement=require,
                passed=bool(pre and reg_ok and s2_ok and req and len(keep) > 0))


def main(out):
    import warnings
    warnings.simplefilter('ignore')
    prep = PX.Prepared(band_one_bin(), PX.MO26)
    checks = [check_swap(), check_pipeline(prep), check_rates(prep, 'rough_rates', prep.cfg['eta'], 1e-2, 'both_early'),
              check_rates(prep, 'discriminating', 1e-2, 1e-4, 'differ')]
    res = dict(checks=checks, dictionary=int(len(prep.cen)), passed=all(c['passed'] for c in checks))
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for c in checks:
        print(('PASS ' if c['passed'] else 'FAIL ') + c['name'], {k: v for k, v in c.items() if k not in ('name', 'passed')})
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
