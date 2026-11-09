"""CPU bridge test: single plane wave at 500 Hz, M random mics in a 1.2 x 1.2 x 1.05 m box, 200 random targets.
usage: python3 -I -B test_bridge.py [M=96] [threads=8]"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = int(sys.argv[1]) if len(sys.argv) > 1 else 96
os.environ["RIBEIRO_THREADS"] = sys.argv[2] if len(sys.argv) > 2 else "8"
import numpy as np, ribeiro_bridge as rb
rng = np.random.default_rng(0)
f, c = 500.0, 343.0; k = 2*np.pi*f/c; box = np.array([1.2, 1.2, 1.05])
x = rng.uniform(0, 1, (M, 3))*box; q = rng.uniform(0, 1, (200, 3))*box
d = np.array([0.6, 0.48, 0.64]); d /= np.linalg.norm(d)
pw = lambda r: np.exp(-1j*k*(r @ d))
t = time.time(); p, info = rb.predict(x, pw(x), q, f, c, seed=1, return_info=True, device="cpu"); dt = time.time()-t
ref = pw(q); nmse = np.sum(abs(p-ref)**2)/np.sum(abs(ref)**2)
print(f"M={M}: NMSE = {nmse:.3e} ({10*np.log10(nmse):.2f} dB), wall = {dt:.0f} s (training {info['fit_seconds']:.0f} s), train residual = {info['train_residual']:.2e}", flush=True)
