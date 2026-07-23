"""Confirm the learned-removal win is robust: more test events x seeds."""
import sys; sys.path.insert(0,'/sdf/group/neutrino/omara/helix')
import numpy as np, torch
import common as C, baselines as Bl
from group_removal import CoherentNet, learned_remove, BEST_DWT
from stage_b import coherent_remove
KAP={'Y':1.5,'U':1.5,'V':1.5}; NEV=10; SEEDS=[11,12,13]
_,test=C.train_test_events(n_train=24,n_test=14)
for pt in ['Y','U','V']:
    clean=C.load_clean(test[:NEV],pt)
    net=CoherentNet().to('cuda'); net.load_state_dict(torch.load(f'artifacts/grpnet_{pt}.pt',weights_only=True))
    w,lv=BEST_DWT[pt]; k=KAP[pt]
    dl=[]
    for s in SEEDS:
        noisy=C.make_noisy(clean,pt,seed=s,coherent=True)
        hx=coherent_remove(noisy,pt); lr=learned_remove(noisy,net)
        fh=Bl.dwt_rd_point(hx,clean,w,lv,k); fl=Bl.dwt_rd_point(lr,clean,w,lv,k)
        dl.append(fl['f0']-fh['f0'])
        print(f"  {pt} seed{s}: helix F0={fh['f0']:.4f} (c{fh['compression']:.0f}x) | learned F0={fl['f0']:.4f} (c{fl['compression']:.0f}x) | d={fl['f0']-fh['f0']:+.4f}")
    print(f"[{pt}] learned-helix F0 over {len(SEEDS)} seeds x {NEV} ev: mean {np.mean(dl):+.4f} +/- {np.std(dl):.4f}")
