import numpy as np, pywt, common as C
pt='U'; ev=C.train_test_events(n_train=24,n_test=24)[1][4:5]
clean=C.load_clean(ev,pt)[0]; noisy=C.make_noisy(clean[None],pt,seed=11,coherent=False)[0]
nw=clean.shape[0]
def decomp(w,lv,k,thr_approx=False):
    co=pywt.wavedec(noisy,w,mode='periodization',level=lv,axis=-1)
    sg=np.median(np.abs(co[-1]),axis=-1,keepdims=True)/0.6745
    approx=co[0]
    if thr_approx:
        t=k*sg*np.sqrt(2*np.log(max(approx.shape[-1],2))); ak=int(np.count_nonzero(np.where(np.abs(approx)>=t,approx,0)))
    else:
        ak=approx.size
    dk=0
    for b in range(1,len(co)):
        t=k*sg*np.sqrt(2*np.log(max(co[b].shape[-1],2))); dk+=int(np.count_nonzero(np.where(np.abs(co[b])>=t,co[b],0)))
    return ak,dk,approx.size
print(f"{pt} plane, {nw} wires x {C.N_TICKS} ticks. signal occupancy {100*(np.abs(clean)>0).mean():.2f}%")
print(f"{'config':22s} {'approx_band':>12s} {'approx_kept':>12s} {'detail_kept':>12s} {'TOTAL':>10s}")
for w,lv,k in [('coif3',4,1.0),('coif3',8,1.0),('bior4.4',8,2.0)]:
    ak,dk,asz=decomp(w,lv,k,thr_approx=False)
    print(f"{w} L{lv} k{k} approx-KEPT  : band={asz:>10,} kept={ak:>10,} det={dk:>10,} TOT={ak+dk:>9,}")
    akt,dkt,_=decomp(w,lv,k,thr_approx=True)
    print(f"{w} L{lv} k{k} approx-THRESH: band={asz:>10,} kept={akt:>10,} det={dkt:>10,} TOT={akt+dkt:>9,}")
