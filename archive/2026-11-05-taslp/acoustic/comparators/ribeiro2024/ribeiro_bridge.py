"""Python wrapper around bridge.jl (Ribeiro et al. 2024 adaptive physics-constrained kernel).

predict(x, y, q, f, c, seed) -> complex ndarray (N,)
Configure paths with env vars (defaults are the vast.ai box layout):
  RIBEIRO_JULIA  julia binary          RIBEIRO_ENV    project dir (Project.toml/Manifest.toml)
  RIBEIRO_DEPOT  JULIA_DEPOT_PATH      PCNK_DIR       clone of PhysicsConstrainedNeuralKernel
  RIBEIRO_THREADS  Julia threads (default 8)
device="cpu" is the only supported value (default).
"""
import os, subprocess, tempfile
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_BASE = Path(os.environ.get("RIBEIRO_BASE", "/root/taslp/ribeiro2024"))


def predict(x, y, q, f, c, seed, timeout=None, return_info=False, device="cpu", cuda_device=None):
    if device != "cpu":
        raise NotImplementedError("only device='cpu' is supported: GPU training/evaluation of this library was not made to work (see README)")
    x = np.asarray(x, float); q = np.asarray(q, float); y = np.asarray(y, complex)
    assert x.ndim == 2 and x.shape[1] == 3 and q.ndim == 2 and q.shape[1] == 3 and y.shape == (x.shape[0],)
    env = dict(os.environ)
    env["JULIA_DEPOT_PATH"] = os.environ.get("RIBEIRO_DEPOT", str(_BASE / "julia-depot"))
    env["PCNK_DIR"] = os.environ.get("PCNK_DIR", str(_BASE / "PhysicsConstrainedNeuralKernel"))
    thr = os.environ.get("RIBEIRO_THREADS", "8")
    env["JULIA_NUM_THREADS"] = thr; env["OPENBLAS_NUM_THREADS"] = thr
    julia = os.environ.get("RIBEIRO_JULIA", str(_BASE / "julia-1.11.9" / "bin" / "julia"))
    proj = os.environ.get("RIBEIRO_ENV", str(_BASE / "env"))
    with tempfile.TemporaryDirectory() as td:
        fin, fout = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(fin, x=x, y=y, q=q, f=np.array([f], float), c=np.array([c], float), seed=np.array([seed], np.int64))
        r = subprocess.run([julia, f"--project={proj}", str(_HERE / "bridge.jl"), fin, fout],
                           env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("bridge.jl failed:\n" + r.stderr[-3000:])
        with np.load(fout) as o:
            p = np.asarray(o["p"], complex)
            info = {"fit_seconds": float(o["fit_seconds"][0]), "train_residual": float(o["train_residual"][0])}
    return (p, info) if return_info else p
