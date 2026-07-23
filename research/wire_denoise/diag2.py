import sys, numpy as np
sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import common as C
from helix.core import backend as B
from helix.tpc.config import DetectorConfig
CFG=DetectorConfig(group_size=64,mask_threshold_nsigma=3.0,num_time_steps=C.N_TICKS,temporal_dilation_ticks=11,n_passes=3)
pt='U'; ev=C.train_test_events(n_train=24,n_test=24)[1][4:5]
clean=C.load_clean(ev,pt)[0]
intr=C.make_noisy(clean[None],pt,seed=11,coherent=False)[0]
coh =C.make_noisy(clean[None],pt,seed=11,coherent=True)[0]
B.set_backend('numpy'); from helix.tpc.coherent import remove_coherent
removed=np.asarray(remove_coherent(coh,CFG,sigma_per_wire=None))
true_coh = coh - intr            # ~ the coherent component (+/- rounding)
left_coh = removed - intr        # leftover coherent after removal (removed should == intr if perfect)
sig=np.abs(clean)>0; m=~sig
def cm_rms(arr, lo, hi):          # per-tick common-mode RMS over non-signal wires in [lo,hi)
    a=arr[lo:hi]; mm=m[lo:hi]
    cm=np.where(mm,a,np.nan); cm=np.nanmean(cm,axis=0)
    return float(np.sqrt(np.nanmean(cm**2)))
for lab,(lo,hi) in [('full group #2 (128-192)',(128,192)),('full group #20 (1280-1344)',(1280,1344)),
                    ('LAST partial group (1920-1969, 49 wires)',(1920,1969))]:
    t=cm_rms(true_coh,lo,hi); r=cm_rms(left_coh,lo,hi)
    print(f"{lab:46s}: coherent rms {t:.2f} -> leftover {r:.2f}  ({100*(1-r/t):.0f}% removed)  n_wires={hi-lo}")
print(f"\n=> alpha factor for partial group = n/group_size = 49/64 = {49/64:.3f}; full group ~64/64=1.0")
