import sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.wirecell_sp import search_high_res
x = np.arange(80.)
s = 3000*np.exp(-(x-20)**2/8) + 1500*np.exp(-(x-45)**2/18) + 400*np.exp(-(x-60)**2/4)
print("peaks", np.round(search_high_res(s.tolist(), 2.0, 0.1, 200), 2))
rng = np.random.default_rng(0); n = s + rng.normal(0, 50, len(s))
print("noisy peaks", np.round(search_high_res(n.tolist(), 2.0, 0.1, 200), 2))
