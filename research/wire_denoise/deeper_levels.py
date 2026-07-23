"""Does going to deeper DWT levels help the R-D (high compression at high F0)?
VisuShrink-hard, several (wavelet, max-level), Stage A intrinsic."""
import numpy as np, pywt, common as C
COMBOS=[('coif3',7),('db8',8),('sym4',9),('db4',9),('db2',10),('haar',12)]
KAP=[1,1.5,2,3,4,6,8]; NT=6
def rd(noisy,clean,w,lv,k):
    rec=np.empty_like(noisy); nk=nt=0
    for i in range(noisy.shape[0]):
        co=pywt.wavedec(noisy[i],w,mode='periodization',level=lv,axis=-1)
        sg=np.median(np.abs(co[-1]),axis=-1,keepdims=True)/0.6745
        nk+=co[0].size; nt+=co[0].size
        for b in range(1,len(co)):
            t=k*sg*np.sqrt(2*np.log(max(co[b].shape[-1],2)))
            co[b]=np.where(np.abs(co[b])>=t,co[b],0.0); nk+=int(np.count_nonzero(co[b])); nt+=co[b].size
        rec[i]=pywt.waverec(co,w,mode='periodization',axis=-1)[:,:C.N_TICKS]
    f0,nr=C.aggregate(clean,rec); sig=np.abs(clean)>0
    return dict(w=w,lv=lv,k=k,f0=f0,nr=nr,comp=nt/max(nk,1),bias=float((rec-clean)[sig].mean()))
for pt in ['Y','U']:
    clean=C.load_clean(C.train_test_events()[1][:NT],pt); noisy=C.make_noisy(clean,pt,seed=1,coherent=False)
    print(f"=== {pt} ===  (best F0 near each compression target; w/lv that achieves it)")
    allp=[]
    for w,lv in COMBOS:
        for k in KAP: allp.append(rd(noisy,clean,w,lv,k))
    for ct in [50,100,200,400]:
        cand=[p for p in allp if abs(np.log(p['comp']/ct))<0.35]
        if not cand: print(f"  ~{ct}x: (none reached)"); continue
        b=max(cand,key=lambda p:p['f0'])
        print(f"  ~{ct:4d}x: {b['w']:5s} L{b['lv']:2d} k{b['k']:<3g} comp={b['comp']:5.0f}x F0={b['f0']:.3f} nrms={b['nr']:.2f} bias={b['bias']:+.2f}")
