import sys, time, json, numpy as np
sys.path.insert(0, 'common'); sys.path.insert(0, 'V2-dev')
import estimators as E, prox as PX, run_dev as RD
mode, tau = sys.argv[1], float(sys.argv[2])
PX.Q_MAX, PX.ETA_C, PX.ETA_R, PX.TAU = 2048, 0.1, 0.01, tau
if mode == 'fixed':
    PX.STEP = 'fixed'
elif mode == 'reset':      # backtracking restarted from ETA at every iteration (probe only)
    src = open('common/prox.py').read()
    PX.STEP = 'backtracking'
    orig = PX._stage_weights
    def _sw(Hr, ys, Ks, iters=PX.ITERS, record=None):
        F, Q = len(ys), Ks[0].shape[0]
        W = np.full((F, Q), 1.0 / Q); flat = [K.reshape(Q, -1) for K in Ks]
        Lt, g = PX._weights_loss(Hr, ys, Ks, flat, W); ts = []
        for _ in range(iters):
            t = PX.ETA
            while True:
                Wn = PX.group_prox(W - t * g, t); Ln, gn = PX._weights_loss(Hr, ys, Ks, flat, Wn); D = Wn - W
                if Ln <= Lt + float(np.sum(g * D)) + float(np.sum(D * D)) / (2 * t) or t < PX.T_MIN: break
                t *= 0.5
            ts.append(t); W, Lt, g = Wn, Ln, gn
        if record is not None: record.extend(ts)
        return W
    PX._stage_weights = _sw
else:
    PX.STEP = 'backtracking'
band, T, *_ = RD.sim_band('base', 400920)
p = band.probs[0]
lo, hi = p.search_box()
cen0, rad0 = PX.dictionary(p.x, lo, hi, p.lam)
Ks = [PX.gram_stack(PX.factors(p.x, cen0, rad0, p.k))[0]]
y = p.y / np.sqrt(np.mean(np.abs(p.y) ** 2)); Hr = [p.H + PX.RIDGE * np.eye(p.M)]
a = time.process_time(); rec = []
W = PX._stage_weights(Hr, [y], Ks, record=rec)
s1 = time.process_time() - a
keep = np.flatnonzero(np.any(W > 0, axis=0))
a = time.process_time()
r = PX.mo26_prox(p)
tot = time.process_time() - a
print(json.dumps(dict(mode=mode, tau=tau, Q=len(rad0), stage1_s=round(s1, 1), survivors_stage1=int(len(keep)), total_s=round(tot, 1),
                      final_regions=r['info']['regions'], nmse_db=round(10*np.log10(E.nmse(r['pred'], T[:, 0])), 2),
                      rec_tail=[round(float(v), 5) for v in rec[-3:]] if rec else None)))
