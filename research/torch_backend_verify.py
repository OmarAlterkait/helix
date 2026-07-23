import numpy as np, pywt, torch
from helix.core import backend
from helix.core.wavelet import sparsify, reconstruct, ThresholdSpec

rng = np.random.default_rng(0)
x = rng.standard_normal((8, 1024)).astype(np.float32)
x[:, 200:260] += 30*np.exp(-0.5*((np.arange(60)-30)/6)**2)

# 1) raw torch _wavedec vs pywt.wavedec
from helix.core import wavelet_ops_torch as wt
xt = torch.as_tensor(x)
lev = min(6, pywt.dwt_max_level(1024, pywt.Wavelet('coif3').dec_len))
tc = wt._wavedec(xt, 'coif3', 6)
pc = pywt.wavedec(x, 'coif3', level=lev, mode='periodization', axis=-1)
print(f"level used: torch={len(tc)-1} pywt={len(pc)-1}")
for i,(a,b) in enumerate(zip(tc, pc)):
    a = a.numpy()
    print(f"  band {i}: len torch={a.shape[-1]} pywt={b.shape[-1]}  maxdiff={np.abs(a-b).max():.2e}")

# 2) full sparsify: numpy vs torch should now be coefficient-identical
spec = ThresholdSpec("universal", "hard", scale=1.2)
backend.set_backend("numpy")
rn = sparsify(x, wavelet="coif3", level=6, mode="periodization", threshold=spec)
backend.set_backend("torch")
rt = sparsify(x, wavelet="coif3", level=6, mode="periodization", threshold=spec)
backend.set_backend("numpy")
print(f"\nsparsify n_kept: numpy={rn.n_kept} torch={rt.n_kept}  n_total numpy={rn.n_total} torch={rt.n_total}")
md = max(np.abs(np.asarray(a)-np.asarray(b)).max() for a,b in zip(rn.coeffs, rt.coeffs))
print(f"coeff maxdiff numpy vs torch = {md:.2e}")

# 3) reconstruct agreement
recn = np.asarray(reconstruct(rn, 1024))
backend.set_backend("torch")
rect = np.asarray(reconstruct(rt, 1024))
backend.set_backend("numpy")
print(f"reconstruct maxdiff numpy vs torch = {np.abs(recn-rect).max():.2e}")
