import sys, numpy as np
sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import common as C
from helix.tpc.config import DetectorConfig
CFG=DetectorConfig(group_size=64,mask_threshold_nsigma=3.0,num_time_steps=C.N_TICKS,temporal_dilation_ticks=11,n_passes=3)
pt='U'; ev=C.train_test_events(n_train=24,n_test=24)[1][4:5]
clean=C.load_clean(ev,pt)[0]
intr=C.make_noisy(clean[None],pt,seed=11,coherent=False)[0]   # clean+intrinsic (floor)
coh =C.make_noisy(clean[None],pt,seed=11,coherent=True)[0]    # +coherent
import importlib
from helix.core import backend as B
def rem(bk):
    B.set_backend(bk); from helix.tpc.coherent import remove_coherent
    return np.asarray(remove_coherent(coh,CFG,sigma_per_wire=None))
rnp=rem('numpy'); rjx=rem('jax')
sig=np.abs(clean)>0
def pw(x):
    d=(x-clean).astype(np.float64); d[sig]=np.nan
    return np.sqrt(np.nanmean(d**2,axis=1))
bands=[(0,500),(500,1000),(1000,1500),(1500,1900),(1900,1920),(1920,1969)]
print(f"{pt} per-wire OFF-signal residual RMS by wire band (last group = 1920-1969, 49 wires):")
print(f"{'band':12s}"+ "".join(f"{f'{lo}-{hi}':>11s}" for lo,hi in bands))
for lab,a in [('intrinsic flr',intr),('coherent raw',coh),('removed numpy',rnp),('removed jax',rjx)]:
    r=pw(a); print(f"{lab:12s}"+"".join(f"{np.nanmean(r[lo:hi]):11.2f}" for lo,hi in bands))
# is the high-wire residual COHERENT (correlated across wires in a group) or white?
def coh_frac(x, w0=1920, w1=1969):
    d=(x-clean); m=~sig
    # group common-mode = mean across the group's non-signal wires per tick; coherent power vs total
    g=d[w0:w1]; mm=m[w0:w1]
    cm=np.where(mm,g,np.nan); cm=np.nanmean(cm,axis=0)            # per-tick common mode
    tot=np.nanmean(np.where(mm,g,np.nan)**2)
    return float(np.nanmean(cm**2)/max(tot,1e-9))
print(f"\ncoherent fraction in last-group residual (cm_power/total): "
      f"coherent-raw {coh_frac(coh):.2f} | removed-numpy {coh_frac(rnp):.2f} | removed-jax {coh_frac(rjx):.2f}")
print(f"(intrinsic-only last group coherent frac: {coh_frac(intr):.2f}  <- white baseline)")
