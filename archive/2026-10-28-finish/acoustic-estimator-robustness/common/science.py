"""Prespecified CPU acoustic models. The modal forward model never calls the Green estimator.

The room is a rigid Neumann rectangular box with phenomenological uniform bulk damping.
Its finite modal projection is not an exact impedance-wall model and has no anechoic limit.
"""
from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import beta, qmc

RECORD = Path(__file__).resolve().parents[1]
ROOT = RECORD.parents[2]
OLD = ROOT / 'archive/2026-10-27-science/acoustic-estimator-robustness/frozen/revision_controls.py'
spec = importlib.util.spec_from_file_location('w3_frozen_analytic_control', OLD)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
EXTENT = np.array([3.4, 2.4, 1.7])
C = 343.0
RIDGES = (0.0003, 0.003, 0.03, 0.3)
WEIGHTS = (0.3, 1.0, 3.0)
PAIRS = tuple(itertools.product(WEIGHTS, RIDGES))
SCALES = (0.5, 1.0, 2.0, 4.0)
DIRECTIONS = np.vstack((np.eye(3), -np.eye(3)))
GEOMETRY = qmc.Sobol(3, scramble=True, seed=682).random_base2(10) * EXTENT
LAYOUTS = {kind: controls.nested_layout(EXTENT, 96, kind) for kind in ('sobol', 'lattice_maximin')}
LOSS_FACTORS = (0.001, 0.01, 0.05, 0.2, 1.0)
BOOTSTRAPS = 4000

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def nmse(pred, truth):
    den = np.vdot(truth, truth).real
    if not den > 0:
        raise ValueError('Nonpositive target-pressure-energy denominator')
    value = np.vdot(pred-truth, pred-truth).real / den
    if not np.isfinite(value):
        raise ValueError('Nonfinite target loss')
    return float(value)

def exact_interval(successes, n, alpha=0.025):
    """Two one-sided Clopper-Pearson endpoints; n independent fields, not directions."""
    if n <= 0 or successes < 0 or successes > n:
        raise ValueError('Invalid binomial denominator')
    return [0.0 if successes == 0 else float(beta.ppf(alpha, successes, n-successes+1)),
            1.0 if successes == n else float(beta.ppf(1-alpha, successes+1, n-successes))]

def interval(values, alpha=0.025, seed=20261028):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Complete finite field values required')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, alpha)),
                high=float(np.quantile(means, 1-alpha)), n=len(values))

def benefit_capture(unaware, source, policy):
    den = float(np.mean(np.asarray(unaware)-np.asarray(source)))
    numerator=float(np.mean(np.asarray(unaware)-np.asarray(policy)))
    return dict(numerator=numerator,denominator=den, value=None if den <= 0 else numerator/den,
                status='undefined_nonpositive_reference_benefit' if den <= 0 else 'defined')

@lru_cache(maxsize=8)
def modal_indices(cutoff):
    modes = np.array(list(itertools.product(range(cutoff+1), repeat=3)), int)
    kap = np.pi * modes / EXTENT
    norm = np.prod(np.where(modes == 0, 1.0, 2.0), axis=1) / np.prod(EXTENT)
    return modes, kap, norm

def modal_shape(points, cutoff):
    modes, kap, norm = modal_indices(cutoff)
    pts = np.asarray(points, float)
    return np.cos(pts[:, 0, None]*kap[None, :, 0]) * np.cos(pts[:, 1, None]*kap[None, :, 1]) * np.cos(pts[:, 2, None]*kap[None, :, 2]) * np.sqrt(norm)[None, :]

def modal_green(points, source, frequency, loss_factor, cutoff=18):
    """Galerkin point-source resolvent: (-Delta-k^2-i eta k^2)G=delta_s.

    The factor 4*pi matches the convention exp(i*k*r)/r. It cancels on field normalization.
    No source smoothing or singularity clipping is used. Cutoff diagnostics are mandatory.
    """
    if loss_factor <= 0:
        raise ValueError('Positive bulk damping required')
    _, kap, _ = modal_indices(cutoff)
    k = 2*np.pi*frequency/C
    coeff = modal_shape(np.asarray(source)[None, :], cutoff)[0] / (np.sum(kap**2, axis=1)-k*k-1j*loss_factor*k*k)
    return 4*np.pi*(modal_shape(points, cutoff) @ coeff)

def basis(points, nodes, frequency, c_multiplier=1.0):
    nodes = np.atleast_2d(nodes)
    d = cdist(points, nodes)
    g = cdist(GEOMETRY, nodes)
    if np.any(d <= 0) or np.any(g <= 0):
        raise ValueError('Sampled source singularity')
    norm = np.sqrt(np.mean(1/g**2, axis=0))
    k = 2*np.pi*frequency/(C*c_multiplier)
    return np.exp(1j*k*d)/d/norm[None, :]

def make_field(seed, frequency=250.0, count=96, model='analytic', loss_factor=0.05,
               mismatch='none', level=0.0, cutoff=18, target_count=256):
    """Shared physical truth, source-excluded target mask and unit calibration-RMS normalization."""
    pair = controls.make_pair(EXTENT, seed, LAYOUTS, n_eval=target_count, n_cal=512)
    x, q, s = LAYOUTS['sobol'][:count], pair['evaluation'], pair['source_inside']
    lam = C/frequency
    field, _, scale = controls.field_bundle(pair, frequency, 'inside')
    true_x = x.copy()
    if mismatch == 'jitter':
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8151]))
        true_x = x + level*lam*rng.normal(size=x.shape)
    if model == 'modal' or mismatch == 'reverberation':
        eta = loss_factor if model == 'modal' else 10.0**(-level)
        points = np.concatenate((true_x, q, pair['calibration']))
        pressure = modal_green(points, s, frequency, eta, cutoff)
        scale = np.sqrt(np.mean(np.abs(pressure[len(x)+len(q):])**2))
        clean, truth = pressure[:len(x)]/scale, pressure[len(x):len(x)+len(q)]/scale
    else:
        clean, truth = field(true_x)/scale, field(q)/scale
        if mismatch == 'image':
            image = s*np.array([-1, 1, 1])
            gscale = np.sqrt(np.mean(np.abs(controls.green(pair['calibration'], s, frequency))**2))
            def extra(p):
                return level*np.sqrt(.7)*pair['direct']*controls.green(p, image, frequency)/gscale/scale
            clean, truth = clean+extra(true_x), truth+extra(q)
        elif mismatch == 'diffuse':
            rng = np.random.default_rng(np.random.SeedSequence([seed, 8152]))
            directions = rng.normal(size=(8, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
            amps = rng.normal(size=8)+1j*rng.normal(size=8)
            def wave(p):
                return np.exp(1j*2*np.pi*frequency/C*p@directions.T)@amps
            fac = np.sqrt(level)/np.sqrt(np.mean(np.abs(wave(pair['calibration']))**2))
            clean, truth = clean+fac*wave(true_x), truth+fac*wave(q)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(frequency*10), count, 8153]))
    noise = (rng.normal(size=count)+1j*rng.normal(size=count))*np.sqrt(.01/2)
    y = clean+noise
    return dict(seed=seed, frequency=frequency, count=count, x=x, true_x=true_x, q=q, source=s,
                y=y, clean=clean, truth=truth, noise=noise, scale=float(scale), calibration=pair['calibration'],
                model=model, loss_factor=loss_factor, mismatch=mismatch, level=float(level), cutoff=cutoff,
                c_multiplier=1+level if mismatch == 'speed' else 1.0,
                input_hash=array_hash(x, true_x, q, s, y, truth), target_energy=float(np.vdot(truth, truth).real))

class Estimator:
    """Finite-menu Helmholtz/RBF kernels and efficient source evidence."""
    def __init__(self, frequency=250., count=96, c_multiplier=1., build_grid=True):
        self.frequency, self.count, self.c_multiplier = float(frequency), int(count), float(c_multiplier)
        self.x = LAYOUTS['sobol'][:count]
        self.k = 2*np.pi*frequency/(C*c_multiplier)
        self.h = np.sinc(self.k*cdist(self.x, self.x)/np.pi).astype(complex)
        self.pre = {}
        for ridge in RIDGES:
            fac = cho_factor(self.h+ridge*np.eye(count), lower=True)
            inv = cho_solve(fac, np.eye(count))
            self.pre[ridge] = dict(inv=inv, logdet=float(2*np.log(np.diag(fac[0]).real).sum()), factor=fac)
        self.unaware = []
        for scale in (None,)+SCALES:
            gram = self.h if scale is None else np.exp(-.5*(self.k*cdist(self.x, self.x)/scale)**2)
            for ridge in RIDGES:
                fac = cho_factor(gram+ridge*np.eye(count), lower=True)
                inv = cho_solve(fac, np.eye(count))
                self.unaware.append(dict(kind='helmholtz' if scale is None else 'rbf', scale=scale, ridge=ridge,
                                         gram=gram, inv=inv, logdet=float(2*np.log(np.diag(fac[0]).real).sum())))
        lam = C/frequency
        self.lower = .35*EXTENT-lam
        self.upper = .65*EXTENT+lam
        if not build_grid:
            return
        axes = [np.linspace(a, b, int(np.ceil((b-a)/(.3*lam)))+1) for a, b in zip(self.lower, self.upper)]
        self.grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        self.bgrid = basis(self.x, self.grid, frequency, c_multiplier)
        self.valid = cdist(self.x, self.grid).min(axis=0) > .05
        self.grid_pre = {r: p['inv']@self.bgrid for r, p in self.pre.items()}

    def source_evidence(self, y, nodes, fixed_pair=None, use_grid=False):
        b = self.bgrid if use_grid else basis(self.x, nodes, self.frequency, self.c_multiplier)
        valid = self.valid if use_grid else cdist(self.x, nodes).min(axis=0) > .05
        pairs = PAIRS if fixed_pair is None else (fixed_pair,)
        out = np.full((len(pairs), b.shape[1]), -np.inf)
        for i, (w, r) in enumerate(pairs):
            p = self.pre[r]; ib = self.grid_pre[r] if use_grid else p['inv']@b
            den = 1+w*np.real(np.sum(b.conj()*ib, axis=0))
            iy = p['inv']@y
            quadratic = float(np.vdot(y, iy).real)-w*np.abs(b.conj().T@iy)**2/den
            if np.any(quadratic <= 0) or np.any(den <= 0):
                raise ValueError('Nonpositive rank-one evidence quantity')
            out[i] = np.where(valid, -len(y)*np.log(quadratic)-p['logdet']-np.log(den), -np.inf)
        return out

    def localise(self, y):
        scores = self.source_evidence(y, self.grid, use_grid=True)
        best = sorted([(float(scores[i].max()), i, int(scores[i].argmax())) for i in range(len(PAIRS))], reverse=True)[:3]
        choices = []
        for _, i, j in best:
            pair = PAIRS[i]
            def objective(theta):
                if np.any(theta < self.lower) or np.any(theta > self.upper):
                    return 1e20+float(np.sum(np.maximum(self.lower-theta, 0)+np.maximum(theta-self.upper, 0)))*1e10
                ev = self.source_evidence(y, theta[None, :], pair)[0, 0]
                return -float(ev) if np.isfinite(ev) else 1e20
            result = minimize(objective, self.grid[j], method='Nelder-Mead',
                              options=dict(maxfev=240, xatol=1e-4, fatol=1e-8))
            choices.append(dict(theta=result.x, pair=pair, evidence=-float(result.fun),
                                converged=bool(result.success), nfev=int(result.nfev)))
        return max(choices, key=lambda c:c['evidence'])

    def score_matrix(self, gram, cross, ridge, field, blocked=False):
        fac = cho_factor(gram+ridge*np.eye(self.count), lower=True)
        inv = cho_solve(fac, np.eye(self.count)); a = inv@field['y']
        pred = cross@a
        evidence = -self.count*np.log(np.vdot(field['y'], a).real)-2*np.log(np.diag(fac[0]).real).sum()
        result = dict(loss=nmse(pred, field['truth']), evidence=float(evidence),
                      press=float(np.sum(np.abs(a/np.diag(inv).real)**2)/np.vdot(field['y'], field['y']).real),
                      prediction=pred)
        if blocked:
            total = 0.
            for held in np.array_split(np.argsort(self.x[:, 0]), 4):
                train = np.setdiff1d(np.arange(self.count), held)
                aa = cho_solve(cho_factor(gram[np.ix_(train, train)]+ridge*np.eye(len(train)), lower=True), field['y'][train])
                total += np.sum(np.abs(field['y'][held]-gram[np.ix_(held, train)]@aa)**2)
            result['blocked'] = float(total/np.vdot(field['y'], field['y']).real)
        return result

    def unaware_scores(self, field, blocked=False):
        qd = cdist(field['q'], self.x)
        scores = []
        for option in self.unaware:
            cross = np.sinc(self.k*qd/np.pi) if option['scale'] is None else np.exp(-.5*(self.k*qd/option['scale'])**2)
            if blocked:
                item = self.score_matrix(option['gram'], cross, option['ridge'], field, True)
            else:
                a = option['inv']@field['y']; pred = cross@a
                item = dict(loss=nmse(pred, field['truth']), prediction=pred,
                            evidence=float(-self.count*np.log(np.vdot(field['y'], a).real)-option['logdet']),
                            press=float(np.sum(np.abs(a/np.diag(option['inv']).real)**2)/np.vdot(field['y'], field['y']).real))
            scores.append(dict(item, kind=option['kind'], ridge=option['ridge'], scale=option['scale']))
        return scores

    def source_scores(self, field, theta, blocked=False):
        b = basis(self.x, theta, self.frequency, self.c_multiplier)[:, 0]
        bq = basis(field['q'], theta, self.frequency, self.c_multiplier)[:, 0]
        hx = np.sinc(self.k*cdist(field['q'], self.x)/np.pi)
        out = []
        for w, r in PAIRS:
            if blocked:
                item = self.score_matrix(self.h+w*np.outer(b,b.conj()), hx+w*np.outer(bq,b.conj()), r, field, True)
            else:
                p=self.pre[r]; ib=p['inv']@b; den=1/w+np.vdot(b,ib).real
                a=p['inv']@field['y']-ib*np.vdot(b,p['inv']@field['y'])/den
                invdiag=np.diag(p['inv']).real-np.abs(ib)**2/den
                pred=hx@a+w*bq*np.vdot(b,a)
                item=dict(loss=nmse(pred, field['truth']), prediction=pred,
                          evidence=float(-self.count*np.log(np.vdot(field['y'],a).real)-p['logdet']-np.log(w*den)),
                          press=float(np.sum(np.abs(a/invdiag)**2)/np.vdot(field['y'],field['y']).real))
            out.append(dict(item, weight=w, ridge=r))
        return out

@lru_cache(maxsize=12)
def estimator(frequency=250., count=96, c_multiplier=1.):
    return Estimator(frequency, count, c_multiplier)

def best(scores, metric='evidence', kind=None):
    seq = [s for s in scores if kind is None or s.get('kind')==kind]
    return min(seq, key=lambda s:s[metric]) if metric in ('press','blocked','loss') else max(seq,key=lambda s:s[metric])

def point_pipeline(field, selectors=False):
    est=estimator(field['frequency'],field['count'],field['c_multiplier'])
    loc=est.localise(field['y'])
    unaware=est.unaware_scores(field, selectors)
    source=est.source_scores(field, loc['theta'], selectors)
    rbf=best(unaware, kind='rbf'); sl=best(source)
    exact=best(est.source_scores(field,field['source']))
    result=dict(theta=loc['theta'], localization_wavelengths=float(np.linalg.norm(loc['theta']-field['source'])*field['frequency']/C),
                loss_sl=sl['loss'],loss_rbf=rbf['loss'],loss_exact=exact['loss'],source_evidence=sl['evidence'],
                rbf_evidence=rbf['evidence'],evidence_advantage=(sl['evidence']-rbf['evidence'])/field['count'],
                point_converged=loc['converged'],point_nfev=loc['nfev'],selected_weight=sl['weight'],selected_ridge=sl['ridge'],
                oracle_unaware_loss=min(s['loss'] for s in unaware),source_prediction=sl['prediction'],rbf_prediction=rbf['prediction'])
    if selectors:
        for metric in ('press','blocked','evidence'):
            u=best(unaware,metric); selected=best(unaware+source,metric)
            admitted=any(selected is s for s in source)
            result['loss_'+metric]=selected['loss'];result['admitted_'+metric]=int(admitted)
            result['harm_oracle_'+metric]=int(admitted and selected['loss']>result['oracle_unaware_loss'])
            result['harm_rbf_'+metric]=int(admitted and selected['loss']>rbf['loss'])
            result['loss_unaware_'+metric]=u['loss']
    return result

def joint_pipeline(field):
    """Joint Gaussian evidence fit, followed by a separately labelled robust coordinate/c fit.

    The robust fit uses fixed whitening and scale from the joint Gaussian solution; it does
    not optimise evidence. All final priors use the same 12 weight/ridge candidates.
    """
    nominal=field['c_multiplier']; f=field['frequency']; n=field['count']; y=field['y']
    candidates=[]
    for factor in (.7,.85,1.,1.15,1.3):
        e=estimator(f,n,nominal*factor); scores=e.source_evidence(y,e.grid,use_grid=True)
        i,j=np.unravel_index(np.argmax(scores),scores.shape)
        candidates.append((float(scores[i,j]),e.grid[j],PAIRS[i],nominal*factor))
    _,theta,pair,cm=max(candidates,key=lambda v:v[0])
    lower,upper=estimator(f,n,nominal).lower,estimator(f,n,nominal).upper
    def objective(v):
        if np.any(v[:3]<lower) or np.any(v[:3]>upper) or not .6*nominal<=v[3]<=1.4*nominal or np.min(np.linalg.norm(field['x']-v[:3],axis=1))<=.05:
            return 1e20
        k=2*np.pi*f/(C*v[3]); h=np.sinc(k*cdist(field['x'],field['x'])/np.pi).astype(complex)
        b=basis(field['x'],v[:3],f,v[3])[:,0]; w,r=pair
        fac=cho_factor(h+r*np.eye(n),lower=True); iy=cho_solve(fac,y); ib=cho_solve(fac,b)
        den=1+w*np.vdot(b,ib).real; quad=np.vdot(y,iy).real-w*np.abs(np.vdot(b,iy))**2/den
        return float(n*np.log(quad)+2*np.log(np.diag(fac[0]).real).sum()+np.log(den))
    fitted=minimize(objective,np.r_[theta,cm],method='Nelder-Mead',options=dict(maxfev=280,xatol=1e-4,fatol=1e-8))
    v=fitted.x; e=Estimator(f,n,v[3],False); joint=best(e.source_scores(field,v[:3]))
    w,r=pair; fac=e.pre[r]['factor']; L=np.tril(fac[0]); yy=solve_triangular(L,y,lower=True)
    scale=np.sqrt(np.mean(np.abs(yy)**2)); delta=1.5
    def robust_objective(t):
        if np.any(t[:3]<lower) or np.any(t[:3]>upper) or not .6*nominal<=t[3]<=1.4*nominal or np.min(np.linalg.norm(field['x']-t[:3],axis=1))<=.05:
            return 1e20
        b=basis(field['x'],t[:3],f,t[3])[:,0]; z=solve_triangular(L,b,lower=True)
        a=np.vdot(z,yy)/(np.vdot(z,z).real+1/w)
        for _ in range(6):
            u=np.abs(yy-a*z)/scale; weights=np.minimum(1.,delta/np.maximum(u,1e-15))
            a=np.vdot(z,weights*yy)/(np.vdot(z,weights*z).real+1/w)
        u=np.abs(yy-a*z)/scale
        return float(np.sum(np.where(u<=delta,.5*u*u,delta*u-.5*delta*delta))+np.abs(a)**2/(2*w*scale*scale))
    robust=minimize(robust_objective,v,method='Nelder-Mead',options=dict(maxfev=240,xatol=1e-4,fatol=1e-8))
    z=robust.x; er=Estimator(f,n,z[3],False); robust_scores=best(er.source_scores(field,z[:3]))
    return dict(loss_joint=joint['loss'],loss_robust=robust_scores['loss'],joint_c_multiplier=float(v[3]),
                robust_c_multiplier=float(z[3]),joint_error_wavelengths=float(np.linalg.norm(v[:3]-field['source'])*f/C),
                robust_error_wavelengths=float(np.linalg.norm(z[:3]-field['source'])*f/C),joint_theta=v[:3],robust_theta=z[:3],
                joint_converged=bool(fitted.success),robust_converged=bool(robust.success))

def integrated(field, center, centered_on_estimate=False):
    """Finite, model-based local coordinate distribution; empirical Bayes, not calibrated uncertainty.

    A fixed 5^3-node cube at 0.1 wavelength increments covers +/-0.2 wavelengths per axis.
    The same support is used at every controlled error; no true coordinate sets its weights.
    """
    est=estimator(field['frequency'],field['count'],field['c_multiplier']); lam=C/field['frequency']
    nodes=center+np.array(list(itertools.product((-.2,-.1,0.,.1,.2),repeat=3)))*lam
    point=best(est.source_scores(field,center)); pair=(point['weight'],point['ridge'])
    ev=est.source_evidence(field['y'],nodes,pair)[0]
    p=np.exp(ev-logsumexp(ev))
    bx=basis(field['x'],nodes,field['frequency'],field['c_multiplier'])
    bq=basis(field['q'],nodes,field['frequency'],field['c_multiplier'])
    gs=(bx*p)@bx.conj().T; cs=(bq*p)@bx.conj().T
    hx=np.sinc(est.k*cdist(field['q'],field['x'])/np.pi)
    scores=[]
    for w,r in PAIRS:
        scores.append(dict(est.score_matrix(est.h+w*gs,hx+w*cs,r,field),weight=w,ridge=r))
    chosen=best(scores)
    return dict(point=point['loss'],integrated=chosen['loss'],effective_nodes=float(1/np.sum(p*p)),
                posterior_mean=p@nodes,posterior_rms_m=float(np.sqrt(np.sum(p*np.sum((nodes-p@nodes)**2,axis=1)))),
                max_probability=float(p.max()),point_prediction=point['prediction'],integrated_prediction=chosen['prediction'])
