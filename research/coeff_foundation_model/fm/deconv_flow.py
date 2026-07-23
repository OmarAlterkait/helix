"""Deconvolution with a per-token FLOW-MATCHING head (vs the NLL/MSE mean head).

A/B partner of the NLL deconv (ckpt_dc_ref): identical d/init/data, only the head
differs. Tests whether modeling the conditional DISTRIBUTION recovers the
fluctuations the mean blurs away. Scored by var(pred)/var(true) (-> 1 if
fluctuations preserved), CRPS (energy form from samples), and R2 of the sample
mean (point accuracy). RoPE held at the OLD band so the MAE init matches.

Run: python deconv_flow.py --events 6000 --steps 16000 --init mae --mae_ckpt ckpt_dual44M.pt
"""
import sys, os, math, time, argparse, numpy as np, torch
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import DEV, N_SLOT, N_BAND
from model import FMModel
from flow_head import FlowHead, flow_loss, flow_sample

VB = ["A4", "D4", "D3", "D2"]
OLD_LAM = (2 * math.pi, 47120.0)        # match the NLL baseline's base=10000 RoPE


def move(B, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}


def load_b(it):
    return D.get_cached_charge(*it) if isinstance(it, tuple) else it


@torch.no_grad()
def band_scales(batches):
    sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
    for it in batches:
        B = load_b(it); c = B["target_charge"].cpu().numpy(); bd = B["band_id"][B["cell"]].cpu().numpy()
        np.add.at(sq, bd, c ** 2); np.add.at(n, bd, 1)
    return np.sqrt(sq / np.maximum(n, 1)) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--dec_blocks", type=int, default=4)
    ap.add_argument("--init", default="mae", choices=["scratch", "mae"])
    ap.add_argument("--mae_ckpt", default="ckpt_dual44M.pt")
    ap.add_argument("--flow_steps", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt", default="ckpt_dc_flow.pt")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))

    model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=args.d, blocks=args.blocks,
                    dec_blocks=args.dec_blocks, nll=False, cond="film",
                    lam_t=OLD_LAM, lam_w=OLD_LAM).to(DEV)
    if args.init == "mae":
        sd = torch.load(os.path.join(here, args.mae_ckpt), map_location=DEV)["model"]
        msd = model.state_dict(); keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(keep, strict=False)
        print(f"MAE init: loaded {len(keep)}/{len(sd)} tensors", flush=True)
    head = FlowHead(args.d, N_SLOT).to(DEV)
    npar = (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"deconv-FLOW d={args.d} init={args.init} params={npar:.1f}M (head {sum(p.numel() for p in head.parameters())/1e6:.1f}M)", flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    def lr_at(s):
        if args.warmup and s < args.warmup:
            return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in have]
    lazy = len(paths) > 500
    allb = paths if lazy else [move(D.get_cached_charge(*p), "cpu") for p in paths]
    nte = max(1, int(len(allb) * 0.2)); test_b, train_b = allb[:nte], allb[nte:]
    S = torch.tensor(band_scales(train_b[:200]), dtype=torch.float32, device=DEV)
    print(f"{'lazy' if lazy else 'staged'} train={len(train_b)} test={len(test_b)} S={S.cpu().numpy().round(1)}", flush=True)

    def dense_target(B):
        """y1 (n_cells, N_SLOT) = asinh(charge/S) at active slots, 0 else; + slot_mask."""
        n = int(B["n_cells"])
        y1 = torch.zeros(n, N_SLOT, device=DEV); sm = torch.zeros(n, N_SLOT, dtype=torch.bool, device=DEV)
        tgt = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]])
        y1[B["cell"], B["slot"]] = tgt; sm[B["cell"], B["slot"]] = True
        return y1, sm

    @torch.no_grad()
    def evaluate(batches):
        model.eval(); head.eval()
        P = [[] for _ in range(N_BAND)]; T = [[] for _ in range(N_BAND)]
        for it in batches:
            B = move(load_b(it), DEV)
            m = torch.zeros(int(B["n_cells"]), dtype=torch.bool, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.forward_feat(B, m).float()
            sv = flow_sample(head, z, N_SLOT, steps=args.flow_steps, k=args.samples)  # (k,n,slot)
            s_act = sv[:, B["cell"], B["slot"]].cpu().numpy()                          # (k, n_active)
            t = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]]).cpu().numpy()
            bd = B["band_id"][B["cell"]].cpu().numpy()
            for b in range(N_BAND):
                mm = bd == b; P[b].append(s_act[:, mm]); T[b].append(t[mm])
        model.train(); head.train()
        out = {}
        for b in range(N_BAND):
            s = np.concatenate(P[b], axis=1); t = np.concatenate(T[b])           # s (k, n), t (n)
            mean = s.mean(0)
            r2 = 1 - ((mean - t) ** 2).mean() / (t.var() + 1e-9)
            vr = s.reshape(-1).var() / (t.var() + 1e-9)                          # predictive marginal var
            crps = (np.abs(s - t[None]).mean()
                    - 0.5 * np.abs(s[:, None] - s[None, :]).mean(axis=(0, 1)).mean())
            out[b] = dict(r2=r2, vr=vr, crps=float(crps))
        return out

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        B = move(load_b(train_b[step % len(train_b)]), DEV)
        m = torch.zeros(int(B["n_cells"]), dtype=torch.bool, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.forward_feat(B, m).float()
        y1, sm = dense_target(B)
        loss = flow_loss(head, z, y1, sm)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0); opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step}: floss {float(loss):.4f} lr={lr_at(step):.1e} {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
        if step % args.eval_every == 0:
            M = evaluate(test_b[:60])
            ov_r2 = np.mean([M[b]["r2"] for b in range(N_BAND)]); ov_vr = np.mean([M[b]["vr"] for b in range(N_BAND)])
            print(f"   [eval {step}] R2 ov={ov_r2*100:.1f}% varRatio={ov_vr:.3f} | "
                  + " ".join(f"{VB[b]}(r2={M[b]['r2']*100:.0f},vr={M[b]['vr']:.2f},crps={M[b]['crps']:.3f})" for b in range(N_BAND)), flush=True)
            torch.save(dict(model=model.state_dict(), head=head.state_dict(), step=step), os.path.join(here, args.ckpt))
    print("done")


if __name__ == "__main__":
    main()
