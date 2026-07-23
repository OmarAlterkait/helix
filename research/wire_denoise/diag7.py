import sys, numpy as np, pywt
sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import common as C
from helix.core import backend as B; B.set_backend('numpy')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig
CFG=DetectorConfig(group_size=64,mask_threshold_nsigma=3.0,num_time_steps=C.N_TICKS,temporal_dilation_ticks=11,n_passes=3)
def rem(stack): return np.stack([np.asarray(remove_coherent(stack[i],CFG,sigma_per_wire=None)) for i in range(stack.shape[0])])
# EXACT original (758da4d) TPC suppression: coif3 L4, default mode, per-band MAD sigma, threshold ALL bands, kappa=1
def orig_sparsify(img, wavelet='coif3', level=4, kappa=1.0, include_approx=True):
    co=pywt.wavedec(img, wavelet, level=level, axis=1)            # default (symmetric) mode
    nk=0; out=[]
    for i,c in enumerate(co):
        if i==0 and not include_approx: out.append(c); nk+=c.size; continue
        sg=np.median(np.abs(c).ravel())/0.6745                     # GLOBAL per-band MAD sigma
        t=kappa*sg*np.sqrt(2*np.log(max(c.shape[-1],2)))
        cc=c*(np.abs(c)>=t); out.append(cc); nk+=int(np.count_nonzero(cc))
    rec=pywt.waverec(out, wavelet, axis=1)[:, :C.N_TICKS]
    return rec, nk
for pt in ['Y','U','V']:
    ev=C.train_test_events(n_train=24,n_test=24)[1][4:8]
    clean=C.load_clean(ev,pt); removed=rem(C.make_noisy(clean,pt,seed=11,coherent=True))
    recs=[]; nks=[]
    for i in range(len(ev)):
        r,nk=orig_sparsify(removed[i]); recs.append(r); nks.append(nk)
    R=np.stack(recs); f0,nrm=C.aggregate(clean,R); sig=np.abs(clean)>0; bias=float((R-clean)[sig].mean())
    print(f"[{pt}] ORIGINAL TPC suppression (coif3 L4, per-band sigma, threshold-all incl approx, k=1):")
    print(f"     n_kept = {np.mean(nks):,.0f} coeffs/plane   F0={f0:.4f}  nrms={nrm:.2f}  bias={bias:+.3f}")
