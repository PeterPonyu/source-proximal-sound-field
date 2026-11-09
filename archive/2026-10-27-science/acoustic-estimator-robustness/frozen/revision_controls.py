"""Paired source relocation controls; no test outcomes enter model selection."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def nested_layout(extent, count=96, kind='sobol'):
    extent=np.asarray(extent)
    if kind=='sobol':
        return qmc.Sobol(3,scramble=True,seed=681).random_base2(7)[:count]*extent
    if kind!='lattice_maximin': raise ValueError(kind)
    candidate=np.stack(np.meshgrid(np.linspace(0,1,9),np.linspace(0,1,7),np.linspace(0,1,5),indexing='ij'),axis=-1).reshape(-1,3)*extent
    selected=[int(np.argmin(np.linalg.norm(candidate-extent/2,axis=1)))]; distance=np.full(len(candidate),np.inf)
    for _ in range(1,count):
        distance=np.minimum(distance,np.linalg.norm(candidate-candidate[selected[-1]],axis=1));distance[selected]=-1
        selected.append(int(np.argmax(distance)))
    return candidate[selected]


def green(x,source,frequency):
    radius=np.linalg.norm(np.asarray(x)-source,axis=1)
    if np.any(radius<=0): raise ValueError('Source singularity sampled')
    return np.exp(1j*2*np.pi*frequency/343*radius)/radius


def make_pair(extent, seed, layouts, n_eval=256, n_cal=512):
    """Shared random inputs for relocation; calibration/test use disjoint seeds."""
    extent=np.asarray(extent);rng=np.random.default_rng(np.random.SeedSequence([seed,9241]))
    microphones=np.concatenate(list(layouts.values()))
    for _ in range(10000):
        source=(.35+.3*rng.random(3))*extent
        if np.min(np.linalg.norm(microphones-source,axis=1))>=.12: break
    else: raise RuntimeError('No feasible source')
    def sample(n):
        result=[]
        while sum(map(len,result))<n:
            points=rng.random((n,3))*extent
            result.append(points[np.linalg.norm(points-source,axis=1)>=.12])
        return np.concatenate(result)[:n]
    q=sample(n_eval);cal=sample(n_cal)
    direction=rng.normal(size=(16,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    amplitude=(rng.normal(size=16)+1j*rng.normal(size=16))/np.sqrt(32)
    direct=(rng.normal()+1j*rng.normal())/np.sqrt(2)
    return dict(source_inside=source,source_outside=source+np.array([1.2*extent[0],0,0]),evaluation=q,calibration=cal,direction=direction,amplitude=amplitude,direct=direct)


def field_bundle(pair, frequency, location):
    source=pair['source_'+location];k=2*np.pi*frequency/343;cal=pair['calibration']
    gscale=np.sqrt(np.mean(np.abs(green(cal,source,frequency))**2))
    background_scale=np.sqrt(np.mean(np.abs(np.exp(1j*k*cal@pair['direction'].T)@pair['amplitude'])**2))
    def basis(x): return green(x,source,frequency)/gscale
    def field(x):
        diffuse=np.exp(1j*k*np.asarray(x)@pair['direction'].T)@pair['amplitude']/background_scale
        return np.sqrt(.7)*pair['direct']*basis(x)+np.sqrt(.3)*diffuse
    scale=np.sqrt(np.mean(np.abs(field(cal))**2))
    return field,basis,scale


def covariance(x,y,frequency,kind,scale=1.,basis=None,weight=1.):
    k=2*np.pi*frequency/343;distance=cdist(x,y)
    if kind=='rbf': return np.exp(-.5*(k*distance/scale)**2)
    result=np.sinc(k*distance/np.pi).astype(complex)
    if kind=='oracle':
        if basis is None: raise ValueError('Source basis required')
        result+=weight*np.outer(basis(x),basis(y).conj())
    elif kind!='helmholtz': raise ValueError(kind)
    return result


def predict(x,y,q,frequency,kind,ridge=.01,scale=1.,basis=None,weight=1.):
    gram=covariance(x,x,frequency,kind,scale,basis,weight)+ridge*np.eye(len(x))
    return covariance(q,x,frequency,kind,scale,basis,weight)@cho_solve(cho_factor(gram,lower=True),y)


def nmse(prediction,truth):
    return float(np.sum(np.abs(prediction-truth)**2)/np.sum(np.abs(truth)**2))


def candidates(method):
    if method=='rbf_tuned': return [dict(kind='rbf',ridge=r,scale=s) for s in (.5,1.,2.,4.) for r in (.003,.01,.1)]
    if method=='helmholtz_tuned': return [dict(kind='helmholtz',ridge=r) for r in (.003,.01,.1)]
    if method=='oracle_green_tuned': return [dict(kind='oracle',ridge=r,weight=w) for w in (.3,1.,3.) for r in (.003,.01,.1)]
    raise ValueError(method)


def select_configuration(losses,options):
    values=np.asarray(losses)
    if values.ndim!=2 or values.shape[1]!=len(options): raise ValueError('Realization by option loss matrix required')
    if not np.isfinite(values).all(): raise ValueError('Nonfinite calibration loss')
    index=int(np.argmin(values.mean(axis=0)))
    return options[index].copy(),index


def paired_interval(values,seed=721,resamples=10000):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(resamples,len(values)))].mean(axis=1)
    low,high=np.quantile(means,[.025,.975])
    return dict(mean=float(values.mean()),low=float(low),high=float(high),n=len(values))
