"""New modules for AC-W4-1 and AC-W4-2. Frozen science.py is imported, never edited.

Contents: a factorised (memory-light) rigid-box modal point-source solution that reproduces
science.modal_green, an independent image-source sum for rigid or partially reflecting walls,
wall-proximal field construction, a physics-aware estimator that adds the six first-order rigid-wall
images of the candidate coordinate, and a split-half coordinate-agreement feature.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
ROOT = RECORD.parents[2]
FINISH = ROOT / 'archive/2026-10-28-finish/acoustic-estimator-robustness'
sys.path.insert(0, str(FINISH / 'common'))
import science as s  # noqa: E402  (frozen module)

EXTENT, C = s.EXTENT, s.C
MIC_CLEARANCE = 0.12      # metres, as the registered source draw and target mask
SOURCE_GUARD = 0.05       # metres, minimum microphone distance to any candidate node (as science.Estimator.valid)


# ---------------------------------------------------------------- forward models

@lru_cache(maxsize=16)
def _axis_tables(cutoff):
    idx = np.arange(cutoff + 1)
    kap = [np.pi * idx / L for L in EXTENT]
    amp = [np.sqrt(np.where(idx == 0, 1.0, 2.0)) / L ** 0.5 for L in EXTENT]   # sqrt(norm) factorises per axis
    return kap, amp


def modal_green_fast(points, source, frequency, loss_factor, cutoff=18, chunk=256):
    """Same quantity as science.modal_green, evaluated through the separable tensor structure."""
    if loss_factor <= 0:
        raise ValueError('Positive bulk damping required')
    kap, amp = _axis_tables(cutoff)
    k = 2 * np.pi * frequency / C
    source = np.asarray(source, float)
    den = (kap[0][:, None, None] ** 2 + kap[1][None, :, None] ** 2 + kap[2][None, None, :] ** 2) - k * k - 1j * loss_factor * k * k
    sx, sy, sz = (np.cos(kap[a] * source[a]) * amp[a] for a in range(3))
    coeff = sx[:, None, None] * sy[None, :, None] * sz[None, None, :] / den
    pts = np.asarray(points, float)
    out = np.empty(len(pts), complex)
    n1 = cutoff + 1
    for lo in range(0, len(pts), chunk):
        p = pts[lo:lo + chunk]
        cx = np.cos(p[:, 0, None] * kap[0][None, :]) * amp[0][None, :]
        cy = np.cos(p[:, 1, None] * kap[1][None, :]) * amp[1][None, :]
        cz = np.cos(p[:, 2, None] * kap[2][None, :]) * amp[2][None, :]
        t = cx @ coeff.reshape(n1, n1 * n1)                  # (P, n1*n1)
        t = t.reshape(len(p), n1, n1)
        t = np.einsum('pjk,pj->pk', t, cy)
        out[lo:lo + chunk] = np.einsum('pk,pk->p', t, cz)
    return 4 * np.pi * out


def complex_wavenumber(frequency, loss_factor):
    return 2 * np.pi * frequency / C * np.sqrt(1 + 1j * loss_factor)


def _axis_images(x0, length, max_reflections):
    """Image positions and wall-reflection counts along one axis, up to a reflection count."""
    m = np.arange(-(max_reflections // 2 + 1), max_reflections // 2 + 2)
    pos = np.concatenate((2 * m * length + x0, 2 * m * length - x0))
    refl = np.concatenate((np.abs(2 * m), np.abs(2 * m - 1)))
    keep = refl <= max_reflections
    return pos[keep], refl[keep]


def image_green(points, source, frequency, loss_factor=0.0, reflection=1.0, max_reflections=20, chunk_pts=None):
    """Image-source sum for a box with wall reflection coefficient `reflection` on every wall.

    Returns 4*pi*G with the convention exp(i k r)/r. loss_factor>0 uses the complex wavenumber
    k*sqrt(1+i*loss_factor), exactly the modal resolvent's bulk damping. reflection=1 is the rigid box.
    All images whose total wall-reflection count is at most max_reflections are summed.
    """
    kc = complex_wavenumber(frequency, loss_factor)
    pts = np.asarray(points, float)
    px, rx = _axis_images(source[0], EXTENT[0], max_reflections)
    py, ry = _axis_images(source[1], EXTENT[1], max_reflections)
    pz, rz = _axis_images(source[2], EXTENT[2], max_reflections)
    out = np.zeros(len(pts), complex)
    for ix in range(len(px)):
        for iy in range(len(py)):
            left = max_reflections - rx[ix] - ry[iy]
            if left < 0:
                continue
            sel = rz <= left
            nodes = np.stack(np.meshgrid([px[ix]], [py[iy]], pz[sel], indexing='ij'), axis=-1).reshape(-1, 3)
            weight = reflection ** (rx[ix] + ry[iy] + rz[sel])
            d = cdist(pts, nodes)
            if np.any(d <= 0):
                raise ValueError('Evaluation point coincides with an image source')
            out += (weight[None, :] * np.exp(1j * kc * d) / d).sum(axis=1)
    return out


# ---------------------------------------------------------------- fields

def draw_source(seed, wall_distance_wavelengths, frequency=250.0):
    """Registered draw of the two in-plane coordinates; z is set by distance to the nearest z wall.

    wall_distance_wavelengths None reproduces the registered z draw. Side (z=0 or z=Lz) follows seed parity.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9241]))
    mics = np.concatenate(list(s.LAYOUTS.values()))
    lam = C / frequency
    for _ in range(10000):
        src = (.35 + .3 * rng.random(3)) * EXTENT
        if wall_distance_wavelengths is not None:
            d = wall_distance_wavelengths * lam
            src[2] = d if seed % 2 == 0 else EXTENT[2] - d
        if np.min(np.linalg.norm(mics - src, axis=1)) >= MIC_CLEARANCE:
            return src
    raise RuntimeError('No feasible source')


def draw_points(seed, source, n, salt):
    rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
    kept = []
    while sum(map(len, kept)) < n:
        p = rng.random((n, 3)) * EXTENT
        kept.append(p[np.linalg.norm(p - source, axis=1) >= MIC_CLEARANCE])
    return np.concatenate(kept)[:n]


def build_field(seed, wall_distance_wavelengths, kind, frequency=250.0, count=96, loss_factor=0.05,
                reflection=1.0, cutoff=180, max_reflections=40):
    """Wall-proximal field. kind 'modal' (rigid box, bulk damping) or 'image' (wall reflection coefficient)."""
    src = draw_source(seed, wall_distance_wavelengths, frequency)
    x = s.LAYOUTS['sobol'][:count]
    q = draw_points(seed, src, 256, 9243)
    cal = draw_points(seed, src, 512, 9244)
    pts = np.concatenate((x, q, cal))
    if kind == 'modal':
        pressure = modal_green_fast(pts, src, frequency, loss_factor, cutoff)
    elif kind == 'image':
        pressure = image_green(pts, src, frequency, 0.0, reflection, max_reflections)
    else:
        raise ValueError(kind)
    scale = np.sqrt(np.mean(np.abs(pressure[len(x) + len(q):]) ** 2))
    clean, truth = pressure[:len(x)] / scale, pressure[len(x):len(x) + len(q)] / scale
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(frequency * 10), count, 8153]))
    noise = (rng.normal(size=count) + 1j * rng.normal(size=count)) * np.sqrt(.01 / 2)
    y = clean + noise
    return dict(seed=seed, frequency=frequency, count=count, x=x, true_x=x.copy(), q=q, source=src, y=y, clean=clean,
                truth=truth, noise=noise, scale=float(scale), calibration=cal, model=kind, loss_factor=float(loss_factor),
                mismatch='none', level=0.0, cutoff=cutoff, c_multiplier=1.0, reflection=float(reflection),
                wall_distance_wavelengths=None if wall_distance_wavelengths is None else float(wall_distance_wavelengths),
                nearest_wall_wavelengths=float(min(src[2], EXTENT[2] - src[2]) * frequency / C),
                input_hash=s.array_hash(x, x, q, src, y, truth), target_energy=float(np.vdot(truth, truth).real))


def cutoff_change(seed, wall_distance_wavelengths, loss_factor, low, high, frequency=250.0):
    """Relative L2 change of the pressure at microphones and targets between two cutoffs, fixed normalisation."""
    src = draw_source(seed, wall_distance_wavelengths, frequency)
    x = s.LAYOUTS['sobol'][:96]
    q = draw_points(seed, src, 256, 9243)
    cal = draw_points(seed, src, 512, 9244)
    pts = np.concatenate((x, q, cal))
    a = modal_green_fast(pts, src, frequency, loss_factor, low)
    b = modal_green_fast(pts, src, frequency, loss_factor, high)
    scale = np.sqrt(np.mean(np.abs(b[len(x) + len(q):]) ** 2))
    n = len(x) + len(q)
    return float(np.linalg.norm(a[:n] / scale - b[:n] / scale) / np.linalg.norm(b[:n] / scale))


# ---------------------------------------------------------------- physics-aware estimator

def reflections(theta):
    """The six first-order rigid-wall images of one coordinate in the known box."""
    out = []
    for a in range(3):
        for wall in (0.0, EXTENT[a]):
            r = np.array(theta, float)
            r[a] = 2 * wall - r[a]
            out.append(r)
    return np.array(out)


@lru_cache(maxsize=None)
def _geometry_cache():
    return s.GEOMETRY


def wall_basis(points, nodes, frequency, c_multiplier=1.0):
    """Source plus its six first-order wall images with tied unit weights, geometry-only rms normalisation.

    Columns are indexed like science.basis. Points and nodes are in the known box coordinates.
    """
    nodes = np.atleast_2d(nodes)
    k = 2 * np.pi * frequency / (C * c_multiplier)

    def raw(p, node):
        total = np.zeros(len(p), complex)
        for n in np.vstack((node[None, :], reflections(node))):
            d = np.linalg.norm(p - n, axis=1)
            if np.any(d <= 0):
                raise ValueError('Sampled source singularity')
            total += np.exp(1j * k * d) / d
        return total
    cols, norms = [], []
    for node in nodes:
        gm = raw(s.GEOMETRY, node)
        norms.append(np.sqrt(np.mean(np.abs(gm) ** 2)))
        cols.append(raw(np.asarray(points, float), node))
    return np.array(cols).T / np.array(norms)[None, :]


class WallEstimator(s.Estimator):
    """science.Estimator whose source basis is the wall-aware basis; candidate coordinates stay inside the known box."""
    MARGIN = 0.02  # metres inside each wall

    def __init__(self, frequency=250., count=96, c_multiplier=1.):
        super().__init__(frequency, count, c_multiplier, build_grid=False)
        lam = C / frequency
        self.lower = np.full(3, self.MARGIN)
        self.upper = EXTENT - self.MARGIN
        axes = [np.linspace(a, b, int(np.ceil((b - a) / (.3 * lam))) + 1) for a, b in zip(self.lower, self.upper)]
        self.grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        self.bgrid = wall_basis(self.x, self.grid, frequency, c_multiplier)
        self.valid = cdist(self.x, self.grid).min(axis=0) > SOURCE_GUARD
        self.grid_pre = {r: p['inv'] @ self.bgrid for r, p in self.pre.items()}

    def source_evidence(self, y, nodes, fixed_pair=None, use_grid=False):
        nodes = np.atleast_2d(nodes)
        b = self.bgrid if use_grid else wall_basis(self.x, nodes, self.frequency, self.c_multiplier)
        valid = self.valid if use_grid else cdist(self.x, nodes).min(axis=0) > SOURCE_GUARD
        pairs = s.PAIRS if fixed_pair is None else (fixed_pair,)
        out = np.full((len(pairs), b.shape[1]), -np.inf)
        for i, (w, r) in enumerate(pairs):
            p = self.pre[r]
            ib = self.grid_pre[r] if use_grid else p['inv'] @ b
            den = 1 + w * np.real(np.sum(b.conj() * ib, axis=0))
            iy = p['inv'] @ y
            quadratic = float(np.vdot(y, iy).real) - w * np.abs(b.conj().T @ iy) ** 2 / den
            if np.any(quadratic <= 0) or np.any(den <= 0):
                raise ValueError('Nonpositive rank-one evidence quantity')
            out[i] = np.where(valid, -len(y) * np.log(quadratic) - p['logdet'] - np.log(den), -np.inf)
        return out

    def localise(self, y, extra_starts=()):
        scores = self.source_evidence(y, self.grid, use_grid=True)
        best = sorted([(float(scores[i].max()), i, int(scores[i].argmax())) for i in range(len(s.PAIRS))], reverse=True)[:3]
        choices = []
        starts = [(s.PAIRS[i], self.grid[j]) for _, i, j in best]
        # Pair for the extra starts: the best-scoring pair at the nearest grid node to the start.
        for st in extra_starts:
            st = np.clip(np.asarray(st, float), self.lower, self.upper)
            j = int(np.argmin(np.linalg.norm(self.grid - st, axis=1)))
            starts.append((s.PAIRS[int(np.argmax(scores[:, j]))], st))
        for pair, start in starts:
            def objective(theta, pair=pair):
                if np.any(theta < self.lower) or np.any(theta > self.upper):
                    return 1e20 + float(np.sum(np.maximum(self.lower - theta, 0) + np.maximum(theta - self.upper, 0))) * 1e10
                ev = self.source_evidence(y, theta[None, :], pair)[0, 0]
                return -float(ev) if np.isfinite(ev) else 1e20
            result = minimize(objective, start, method='Nelder-Mead', options=dict(maxfev=240, xatol=1e-4, fatol=1e-8))
            choices.append(dict(theta=result.x, pair=pair, evidence=-float(result.fun),
                                converged=bool(result.success), nfev=int(result.nfev)))
        return max(choices, key=lambda c: c['evidence'])

    def source_scores(self, field, theta, blocked=False):
        if blocked:
            raise NotImplementedError('Blocked scoring is not used by the wall study')
        b = wall_basis(self.x, theta, self.frequency, self.c_multiplier)[:, 0]
        bq = wall_basis(field['q'], theta, self.frequency, self.c_multiplier)[:, 0]
        hx = np.sinc(self.k * cdist(field['q'], self.x) / np.pi)
        out = []
        for w, r in s.PAIRS:
            p = self.pre[r]
            ib = p['inv'] @ b
            den = 1 / w + np.vdot(b, ib).real
            a = p['inv'] @ field['y'] - ib * np.vdot(b, p['inv'] @ field['y']) / den
            invdiag = np.diag(p['inv']).real - np.abs(ib) ** 2 / den
            pred = hx @ a + w * bq * np.vdot(b, a)
            item = dict(loss=s.nmse(pred, field['truth']), prediction=pred,
                        evidence=float(-self.count * np.log(np.vdot(field['y'], a).real) - p['logdet'] - np.log(w * den)),
                        press=float(np.sum(np.abs(a / invdiag) ** 2) / np.vdot(field['y'], field['y']).real))
            out.append(dict(item, weight=w, ridge=r))
        return out


@lru_cache(maxsize=4)
def wall_estimator(frequency=250., count=96, c_multiplier=1.):
    return WallEstimator(frequency, count, c_multiplier)


def wall_pipeline(field, plain_theta=None):
    """Physics-aware comparator: microphone data and known box geometry only (the true source is used only to score)."""
    est = wall_estimator(field['frequency'], field['count'], field['c_multiplier'])
    extra = () if plain_theta is None else (plain_theta,)
    loc = est.localise(field['y'], extra)
    sc = s.best(est.source_scores(field, loc['theta']))
    ex = s.best(est.source_scores(field, np.clip(field['source'], est.lower, est.upper)))
    return dict(theta_wall=loc['theta'], loss_wall=sc['loss'], loss_wall_exact=ex['loss'],
                localization_wall_wavelengths=float(np.linalg.norm(loc['theta'] - field['source']) * field['frequency'] / C),
                wall_converged=loc['converged'], wall_nfev=loc['nfev'], wall_evidence=sc['evidence'],
                wall_selected_weight=sc['weight'], wall_selected_ridge=sc['ridge'], wall_prediction=sc['prediction'])


def study_a_pipeline(field):
    """Plain registered pipeline plus the physics-aware comparator on identical data."""
    res = s.point_pipeline(field)
    res.update(wall_pipeline(field, res['theta']))
    return res


# ---------------------------------------------------------------- split-half agreement

class SubsetEstimator(s.Estimator):
    """science.Estimator on an arbitrary subset of the 96 fixed microphones (all arrays remain registered)."""
    def __init__(self, indices, frequency=250.):
        idx = np.asarray(indices, int)
        count = len(idx)
        s_layout = s.LAYOUTS['sobol'][:96][idx]
        saved = s.LAYOUTS['sobol']
        # Build with the registered constructor on a temporary layout view, restoring afterwards.
        try:
            s.LAYOUTS['sobol'] = np.concatenate((s_layout, saved[count:]))   # first `count` rows are the subset
            super().__init__(frequency, count, 1.0, build_grid=True)
        finally:
            s.LAYOUTS['sobol'] = saved
        self.indices = idx


def split_plan(n_splits, seed, count=96):
    """Fixed random bisections of the microphone index set: list of (half_a, half_b), sorted index arrays."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    plan = []
    for _ in range(n_splits):
        perm = rng.permutation(count)
        plan.append((np.sort(perm[:count // 2]), np.sort(perm[count // 2:])))
    return plan


@lru_cache(maxsize=4)
def half_estimators(n_splits, seed):
    return [(SubsetEstimator(a), SubsetEstimator(b), a, b) for a, b in split_plan(n_splits, seed)]


def agreement_features(field, n_splits, seed):
    """Distances (wavelengths) between coordinates localised from the two halves of each split."""
    out = []
    for ea, eb, a, b in half_estimators(n_splits, seed):
        ta = ea.localise(field['y'][a])['theta']
        tb = eb.localise(field['y'][b])['theta']
        out.append(float(np.linalg.norm(ta - tb) * field['frequency'] / C))
    return out
