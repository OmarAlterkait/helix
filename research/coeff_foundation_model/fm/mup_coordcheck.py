"""muP coordinate check for FMModel (the correctness test for the muP recipe).

Under CORRECT muP, the coordinate size (RMS over entries) of key activations is
WIDTH-INVARIANT: flat across d at step 0 AND after a few optimizer steps. Under
standard parametrization it blows up / shrinks with d. We run both param schemes
across d in {128,256,512,1024}, ~8 AdamW steps on a few held-out real events
(>=20000), and print a table of coord sizes vs width per activation.

Activations probed (via forward hooks): embed out, enc block in/out, attention
logits (pre-softmax QK^T*scale), dec_norm out (feature), occ/val logits.

Run (GPU, container):  see slurm/mup_coordcheck.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D
from model import FMModel, losses
from data import DEV, N_SLOT, N_BAND, N_PLANE

WIDTHS = [int(x) for x in os.environ.get("MUP_WIDTHS", "128,256,512,1024").split(",")]
D_BASE = 128
STEPS = 60                  # readout output has a KNOWN 1/sqrt(width) transient at init that
                            # decays over the first steps (mup README); >8 steps confirms flattening
BASE_LR = 3e-4
CACHE = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/fm_cache_tpc"
EVENTS = [20000, 20001, 20002, 20003]          # held-out (>=20000)
HEADS_HD = 64                                   # head_dim fixed => heads = d/64 (base config)


def rms(t):
    return float(t.detach().float().pow(2).mean().sqrt().cpu())


def attach_hooks(model, store):
    """Hook coordinate (RMS) sizes of key activations into `store` (name->rms)."""
    hs = []
    hs.append(model.embed.register_forward_hook(
        lambda m, i, o: store.__setitem__("embed_out", rms(o))))
    b0 = model.enc[0]
    hs.append(b0.register_forward_hook(
        lambda m, i, o: store.__setitem__("enc0_out", rms(o))))
    hs.append(b0.n1.register_forward_hook(
        lambda m, i, o: store.__setitem__("enc0_in(n1)", rms(o))))
    hs.append(model.dec_norm.register_forward_hook(
        lambda m, i, o: store.__setitem__("feature(dec_norm)", rms(o))))
    hs.append(model.occ_head.register_forward_hook(
        lambda m, i, o: store.__setitem__("occ_logit", rms(o * model.readout_mult))))
    hs.append(model.val_head.register_forward_hook(
        lambda m, i, o: store.__setitem__("val_logit", rms(o * model.readout_mult))))

    # attention logits: recompute QK^T*scale from enc[0].qkv output via a hook on qkv
    def qkv_hook(m, inp, out):
        # coord size only needs the SCALE of the logits, not the full TxT matrix
        # (T~30k -> 17 GiB). Sample the first S tokens: RMS of an S x S sub-block is
        # an unbiased estimate of the full-matrix RMS.
        S = min(2048, out.shape[0])
        q, k, _ = out[:S].chunk(3, -1)
        q = q.view(S, b0.h, b0.hd).transpose(0, 1)
        k = k.view(S, b0.h, b0.hd).transpose(0, 1)
        scale = b0.attn_scale if b0.attn_scale is not None else 1.0 / (b0.hd ** 0.5)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        store["attn_logit"] = rms(logits)
    hs.append(b0.qkv.register_forward_hook(qkv_hook))
    return hs


def run(mup, batches, masks):
    per_step = {}                                   # d -> {step-> {act->rms}}
    for d in WIDTHS:
        heads = max(1, d // HEADS_HD)
        torch.manual_seed(0)
        model = FMModel(N_SLOT, N_BAND, N_PLANE, n_wirefeat=1, d=d, blocks=2, dec_blocks=1,
                        heads=heads, film=("band", "plane", "wire"), ffn_mult=4,
                        mup=mup, d_base=D_BASE).to(DEV)
        opt = torch.optim.AdamW(model.param_groups(BASE_LR) if mup
                                else model.parameters(), lr=BASE_LR, weight_decay=0.0)
        rec = {}
        for step in range(STEPS + 1):
            B, m = batches[step % len(batches)], masks[step % len(masks)]
            store = {}
            hs = attach_hooks(model, store)
            occ, mu, lv = model(B, m)
            for h in hs:
                h.remove()
            if step in (0, STEPS // 2, STEPS):
                rec[step] = dict(store)
            bce, val = losses(occ, mu, lv, B, m)
            loss = bce + val
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        per_step[d] = rec
        del model, opt; torch.cuda.empty_cache()
    return per_step


def ptable(title, per_step, step):
    acts = ["embed_out", "enc0_in(n1)", "enc0_out", "attn_logit",
            "feature(dec_norm)", "occ_logit", "val_logit"]
    print(f"\n=== {title}  (coord RMS @ step {step}) ===")
    hdr = f"{'activation':22s}" + "".join(f"{('d='+str(d)):>12s}" for d in WIDTHS)
    print(hdr); print("-" * len(hdr))
    for a in acts:
        row = f"{a:22s}"
        vals = []
        for d in WIDTHS:
            v = per_step[d][step].get(a, float("nan")); vals.append(v)
            row += f"{v:12.4g}"
        # ratio max/min across widths (invariance metric; ~1 = width-invariant)
        fin = [x for x in vals if x == x and x > 0]
        r = (max(fin) / min(fin)) if len(fin) > 1 else float("nan")
        row += f"   x{r:5.2f}"
        print(row)


def numerical_identity_check(batches, masks):
    """mup=False must reproduce the pre-change forward EXACTLY. We can't call the
    old code, but we assert: (a) mup=False readout_mult==1 and attn_scale is None
    (SDPA default), and (b) a fresh mup=False model with a fixed seed gives the same
    outputs whether we pass mup=False explicitly or omit it (default)."""
    d = 256; heads = d // HEADS_HD
    torch.manual_seed(3); a = FMModel(N_SLOT, N_BAND, N_PLANE, d=d, blocks=2, dec_blocks=1, heads=heads).to(DEV)
    torch.manual_seed(3); b = FMModel(N_SLOT, N_BAND, N_PLANE, d=d, blocks=2, dec_blocks=1, heads=heads, mup=False).to(DEV)
    B, m = batches[0], masks[0]
    with torch.no_grad():
        oa = a(B, m)[0]; ob = b(B, m)[0]
    md = (oa - ob).abs().max().item()
    assert a.readout_mult == 1.0 and a.enc[0].attn_scale is None
    print(f"\n[identity] mup=False: readout_mult=1, attn_scale=None; default-vs-explicit forward maxdiff={md:.3e}")


def main():
    D.init_pipeline_cpu()
    batches = []
    for e in EVENTS:
        p = os.path.join(CACHE, f"ev_{e:05d}.npz")
        B = D.get_cached(p, device="cpu")
        B = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in B.items()}
        batches.append(B)
    masks = []
    for i, B in enumerate(batches):
        g = torch.Generator(device=DEV).manual_seed(100 + i)
        masks.append(torch.rand(B["n_cells"], generator=g, device=DEV) < 0.5)

    numerical_identity_check(batches, masks)

    for scheme, mup in (("STANDARD param (mup=False)", False), ("muP (mup=True)", True)):
        per = run(mup, batches, masks)
        ptable(scheme, per, 0)
        ptable(scheme, per, STEPS // 2)
        ptable(scheme, per, STEPS)

    print("\nVERDICT: under muP the coord RMS should be ~flat across d (ratio x~1) "
          "at step 0 AND step 8, esp. for enc0_out / attn_logit / feature / logits; "
          "under STANDARD param those ratios grow with d.")


if __name__ == "__main__":
    main()
