"""DIRECT-supervised deconvolution with a Perceiver (latent bottleneck) + flow head.
No MAE pretraining -- trains the charge regression from scratch to see the
architecture's task ceiling, cheaply. Scored by R2(sample-mean), var-ratio (->1
= fluctuations preserved), CRPS, per band.

Run: python deconv_perceiver.py --events 6000 --M 2048 --depth 24 --steps 16000
"""
import sys, os, math, time, argparse, numpy as np, torch
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import DEV, N_SLOT, N_BAND
from perceiver_model import PerceiverDeconv
from flow_head import FlowHead, flow_loss, flow_sample
VB = ["A4", "D4", "D3", "D2"]


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
    ap.add_argument("--events", type=int, default=6000); ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=24); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=4); ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--ckpt", default="ckpt_perc.pt")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))

    model = PerceiverDeconv(N_SLOT, N_BAND, 6, d=args.d, M=args.M, depth=args.depth, heads=args.heads).to(DEV)
    head = FlowHead(args.d, N_SLOT).to(DEV)
    npar = (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"Perceiver-deconv d={args.d} M={args.M} depth={args.depth} heads={args.heads} params={npar:.1f}M", flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in have]
    lazy = len(paths) > 500
    allb = paths if lazy else [move(D.get_cached_charge(*p), "cpu") for p in paths]
    nte = max(1, int(len(allb) * 0.2)); test_b, train_b = allb[:nte], allb[nte:]
    S = torch.tensor(band_scales(train_b[:200]), dtype=torch.float32, device=DEV)
    print(f"train={len(train_b)} test={len(test_b)} S={S.cpu().numpy().round(1)}", flush=True)

    def dense_target(B):
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.forward_feat(B).float()
            sv = flow_sample(head, z, N_SLOT, steps=args.flow_steps, k=args.samples)
            s_act = sv[:, B["cell"], B["slot"]].cpu().numpy()
            t = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]]).cpu().numpy()
            bd = B["band_id"][B["cell"]].cpu().numpy()
            for b in range(N_BAND):
                mm = bd == b; P[b].append(s_act[:, mm]); T[b].append(t[mm])
        model.train(); head.train()
        out = {}
        for b in range(N_BAND):
            s = np.concatenate(P[b], axis=1); t = np.concatenate(T[b])
            mean = s.mean(0); r2 = 1 - ((mean - t) ** 2).mean() / (t.var() + 1e-9)
            vr = s.reshape(-1).var() / (t.var() + 1e-9)
            crps = float(np.abs(s - t[None]).mean() - 0.5 * np.abs(s[:, None] - s[None, :]).mean())
            out[b] = dict(r2=r2, vr=vr, crps=crps)
        return out

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        B = move(load_b(train_b[step % len(train_b)]), DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.forward_feat(B).float()
        y1, sm = dense_target(B)
        loss = flow_loss(head, z, y1, sm)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0); opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step}: floss {float(loss):.4f} lr={lr_at(step):.1e} {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
        if step % args.eval_every == 0:
            M = evaluate(test_b[:50]); ov_r2 = np.mean([M[b]["r2"] for b in range(N_BAND)]); ov_vr = np.mean([M[b]["vr"] for b in range(N_BAND)])
            print(f"   [eval {step}] R2 ov={ov_r2*100:.1f}% varRatio={ov_vr:.3f} | "
                  + " ".join(f"{VB[b]}(r2={M[b]['r2']*100:.0f},vr={M[b]['vr']:.2f})" for b in range(N_BAND)), flush=True)
            torch.save(dict(model=model.state_dict(), head=head.state_dict(), step=step,
                            d=args.d, M=args.M, depth=args.depth), os.path.join(here, args.ckpt))
    print("done")


if __name__ == "__main__":
    main()
