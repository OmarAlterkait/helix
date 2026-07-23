import sys, numpy as np, pywt
sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import common as C
from helix.core import backend as B; B.set_backend('numpy')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig
CFG=DetectorConfig(group_size=64,mask_threshold_nsigma=3.0,num_time_steps=C.N_TICKS,temporal_dilation_ticks=11,n_passes=3)
pt='U'; ev=C.train_test_events(n_train=24,n_test=24)[1][4:5]
clean=C.load_clean(ev,pt)[0]; nw=clean.shape[0]
def rem(img): return np.asarray(remove_coherent(img,CFG,sigma_per_wire=None))
noisy = rem(C.make_noisy(clean[None],pt,seed=11,coherent=True)[0])         # signal+noise, removed
noise = rem(C.make_noisy(np.zeros_like(clean)[None],pt,seed=11,coherent=True)[0])  # noise ONLY, removed
w,lv,dk='bior4.4',8,2.0
def counts(img, perband_noise=None):
    co=pywt.wavedec(img,w,mode='periodization',level=lv,axis=-1)
    cn=pywt.wavedec(noise,w,mode='periodization',level=lv,axis=-1) if perband_noise is None else perband_noise
    sg_fine=np.median(np.abs(co[-1]),axis=-1,keepdims=True)/0.6745
    per=[]; nk=0
    for b in range(1,len(co)):
        if perband_noise is not None:  # TRUE per-band noise sigma (from noise-only band MAD)
            sg=np.median(np.abs(cn[b]),axis=-1,keepdims=True)/0.6745
        else:
            sg=sg_fine
        t=dk*sg*np.sqrt(2*np.log(max(co[b].shape[-1],2)))
        k=int(np.count_nonzero(np.where(np.abs(co[b])>=t,co[b],0))); nk+=k; per.append((co[b].shape[-1],k))
    return nk, per
nk_fine,per_fine=counts(noisy)
nkn_fine,pern_fine=counts(noise)
# per-band noise sigma (true noise level per band, from noise-only)
cn=pywt.wavedec(noise,w,mode='periodization',level=lv,axis=-1)
nk_pb,_=counts(noisy,perband_noise=cn); nkn_pb,_=counts(noise,perband_noise=cn)
print(f"U bior4.4 L8 dk=2, DETAIL bands only (approx excluded):")
print(f"  finest-sigma VisuShrink : NOISY total detail survivors={nk_fine:,}  | NOISE-ONLY survivors={nkn_fine:,}")
print(f"  per-band-noise-sigma    : NOISY total detail survivors={nk_pb:,}  | NOISE-ONLY survivors={nkn_pb:,}")
print(f"\n  per-band breakdown (band_len, NOISE-ONLY survivors) finest-sigma:")
for (L,k) in pern_fine: print(f"    band N={L:>5}: noise survivors={k:>7,}  ({100*k/(L*nw):.1f}% of band)")
