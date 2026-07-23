import torch, glob
import data as D; D.init_pipeline_cpu()
from model import FMModel
dev="cuda"
def rankme(z):
    z=(z-z.mean(0)).float(); s=torch.linalg.svdvals(z); p=s/(s.sum()+1e-12)
    return float(torch.exp(-(p*torch.log(p+1e-12)).sum()))
paths=sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"), key=lambda p:int("".join(filter(str.isdigit,p.split("/")[-1]))))[30000:30020]
for name,ck,nll in [("MAE-MSE  sc_obj_mse",512,False),("MAE-NLL  sc_obj_nll",512,True),
                    ("JEPA     sc_jepa_d512",512,False)]:
    fn={"MAE-MSE  sc_obj_mse":"ckpt_sc_obj_mse.pt","MAE-NLL  sc_obj_nll":"ckpt_sc_obj_nll.pt",
        "JEPA     sc_jepa_d512":"ckpt_sc_jepa_d512.pt"}[name]
    m=FMModel(128,4,6,d=512,blocks=12,dec_blocks=4,heads=4,cond="film",dec_mode="cross",nll=nll).to(dev)
    ck_sd=torch.load(fn,map_location=dev)
    # JEPA: probe the EMA TEACHER (the encoder the objective actually shapes), not the student
    m.load_state_dict(ck_sd["teacher"] if ("jepa" in fn and "teacher" in ck_sd) else ck_sd["model"]); m.eval()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        fz=torch.cat([m.encode(D.get_cached(p,device=dev)).float() for p in paths])
    print(f"  {name}: RankMe = {rankme(fz):.1f}/512", flush=True)
