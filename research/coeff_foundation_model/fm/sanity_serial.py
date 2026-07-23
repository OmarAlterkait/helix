import sys, glob, torch
sys.path.insert(0,'.')
import data as D; D.init_pipeline_cpu()
from model import FMModel, losses_fused
from model_serial import SerialFMModel
from train import make_mask
dev='cuda'
ck=torch.load('ckpt_nll_base_fix.pt', map_location=dev)
kw=dict(n_wirefeat=1, d=ck['d'], blocks=ck['blocks'], dec_blocks=ck['dec_blocks'],
        heads=ck['d']//ck.get('head_dim',64), film=tuple(ck['film'].split(',')), nll=ck['nll'],
        cond=ck['cond'], dec_mode=ck['dec_mode'], mup=ck['mup'], d_base=ck['d_base'], wire_rope=ck['wire_rope'])
full=FMModel(ck['n_slot'],4,6,**kw).to(dev); full.load_state_dict(ck['model']); full.eval()
for split in (False, True):
    m=SerialFMModel(ck['n_slot'],4,6,rope_split=split,gp=1024,gd=2048,**kw).to(dev)
    miss=m.load_state_dict(ck['model'], strict=False)
    print(f"rope_split={split}: missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    m.eval()
    B=D.get_cached(sorted(glob.glob('../artifacts/fm_cache_tpc/ev_*.npz'))[3], device=dev)
    mk=make_mask(B,'random',0.75,1)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        if not split:  # only once: full-attn reference
            of,mf,lf=full(B,mk); bce_f,val_f=losses_fused(of,mf,lf,B,mk); print(f"  FULL-ATTN loss: bce={float(bce_f):.3f} val={float(val_f):.3f}")
        o,mu,lv=m(B,mk); bce,val=losses_fused(o,mu,lv,B,mk)
        fe=m.encode_layers(B,{12})[12]
    print(f"  SERIAL(split={split}) loss: bce={float(bce):.3f} val={float(val):.3f}  encode finite={torch.isfinite(fe).all().item()} shape={tuple(fe.shape)}")
