import numpy as np, pywt, common as C
for pt in ['Y','U','V']:
    ev=C.train_test_events(n_train=24,n_test=24)[1][4:8]   # 4 events
    clean=C.load_clean(ev,pt)
    nws=clean.shape[1]
    # coeffs of CLEAN truth (no noise): how many nonzero? (empty wires -> 0)
    tot=[]
    for w,lv in [('coif3',4),('bior4.4',8)]:
        nk=0
        for i in range(clean.shape[0]):
            co=pywt.wavedec(clean[i],w,mode='periodization',level=lv,axis=-1)
            for c in co: nk+=int(np.count_nonzero(c))
        tot.append((w,lv,nk/clean.shape[0]))
    occ=100*(np.abs(clean)>0).mean()
    sigwires=np.mean([np.sum(np.abs(clean[i]).max(1)>0) for i in range(clean.shape[0])])
    print(f"[{pt}] occ {occ:.2f}%, ~{sigwires:.0f}/{nws} wires have signal, {np.abs(clean).sum()/clean.shape[0]:.0f} total |ADC|/event")
    for w,lv,n in tot:
        print(f"     CLEAN-truth nonzero coeffs ({w} L{lv}): {n:,.0f} /event")
