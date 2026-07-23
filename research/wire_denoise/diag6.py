import sys, numpy as np, pywt
sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import common as C
from helix.core import backend as B; B.set_backend('numpy')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig
CFG=DetectorConfig(group_size=64,mask_threshold_nsigma=3.0,num_time_steps=C.N_TICKS,temporal_dilation_ticks=11,n_passes=3)
def rem(stack): return np.stack([np.asarray(remove_coherent(stack[i],CFG,sigma_per_wire=None)) for i in range(stack.shape[0])])
def hard(x,t): return np.where(np.abs(x)>=t,x,0.)
for pt in ['Y','U','V']:
    ev=C.train_test_events(n_train=24,n_test=24)[1][4:8]
    clean=C.load_clean(ev,pt)
    noisy=rem(C.make_noisy(clean,pt,seed=11,coherent=True))
    noise=rem(C.make_noisy(np.zeros_like(clean),pt,seed=11,coherent=True))
    w,lv,dk='bior4.4',8,2.0
    def run(perband):
        recs=[]; nk=0
        for i in range(len(ev)):
            co=pywt.wavedec(noisy[i],w,mode='periodization',level=lv,axis=-1)
            cn=pywt.wavedec(noise[i],w,mode='periodization',level=lv,axis=-1)
            sgf=np.median(np.abs(co[-1]),axis=-1,keepdims=True)/0.6745
            cc=[]
            for b in range(len(co)):
                sg=(np.median(np.abs(cn[b]),axis=-1,keepdims=True)/0.6745) if perband else sgf
                t=dk*sg*np.sqrt(2*np.log(max(co[b].shape[-1],2)))
                bb=hard(co[b],t); cc.append(bb); nk+=int(np.count_nonzero(bb))
            recs.append(pywt.waverec(cc,w,mode='periodization',axis=-1)[:,:C.N_TICKS])
        R=np.stack(recs); f0,nr=C.aggregate(clean,R); sig=np.abs(clean)>0
        return nk/len(ev), f0, nr, float((R-clean)[sig].mean())
    nf,ff,rf,bf=run(False); nb,fb,rb,bb=run(True)
    print(f"[{pt}] bior4.4 L8 dk2 (approx ALSO thresholded the same way):")
    print(f"   single finest-sigma : {nf:>8,.0f} coeffs/plane  F0={ff:.4f} nrms={rf:.2f} bias={bf:+.3f}")
    print(f"   per-band noise-sigma: {nb:>8,.0f} coeffs/plane  F0={fb:.4f} nrms={rb:.2f} bias={bb:+.3f}   ({nf/nb:.1f}x fewer)")
