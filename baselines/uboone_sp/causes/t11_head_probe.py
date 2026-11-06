#!/usr/bin/env python3
"""T11: head vs encoder. Retrain ONLY a charge head on M7dc's FROZEN encoder and score bright cells.

    CUDA_VISIBLE_DEVICES=k scripts/helix_run.sh python baselines/uboone_sp/causes/t11_head_probe.py --tag P2 --head fresh --out <dir>

Probes (one property changed at a time; same data stream, training rows and support as M7dc; loss = the hurdle
charge term only, MSE of y = log1p(q/Q0) on rows with q > 25 e-, so the presence BCE no longer dominates):
  --head m7dc    M7dc's own CellDecoder, continued                         (P1)
  --head fresh   a freshly initialised CellDecoder of M7dc's shape          (P2)
  --amp          + the covering tokens' amplitudes sinh(value) / AMP_S beside arcsinh(value) in the head's input (P3)
  --proj/--layers a larger decoder                                         (P4)
  --nbr-raw      + the raw (arcsinh value, occupancy) slots of all 3x3 neighbourhood tokens per band in the head's
                 direct input, not only the covering token's (P5): the coefficients an exact inversion needs (T1f)
  --nbr-amp      with --nbr-raw: + those slots' amplitudes sinh(value) / AMP_S (linear in the measured charge)
  --no-feat      the encoder features replaced by zeros: the head sees only its direct inputs (no encoder at all)
  --min-q Q      the charge loss on rows with truth >= Q only (default: every row above the support, as M7dc)
Resumable (<out>/<tag>_state.pt every 500 steps). --converge: at the end, T9's first-order convergence test restricted
to the head (gradient signal / noise over held-out training events, line search along the Adam-preconditioned mean).
Training: AdamW (0.9, 0.95), wd 0.05, clip 1.0, cosine to 2%, --accum events per step, bf16 autocast (as M7dc).
Score (fp32): the 48 floor test events, training-label complete truth, hw68 / median of y - log1p(T/Q0) per plane
and truth bin, at step 0 and at the end.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
Q0 = 2043.2578108909702
AMP_S = 30.0
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--head", choices=["m7dc", "fresh"], required=True)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--nbr-raw", action="store_true")
    ap.add_argument("--nbr-amp", action="store_true")
    ap.add_argument("--no-feat", action="store_true")
    ap.add_argument("--min-q", type=float, default=None)
    ap.add_argument("--encoder", choices=["m7dc", "pretrained"], default="m7dc",
                    help="frozen features from M7dc's fine-tuned encoder or the pretrained FM artifact")
    ap.add_argument("--layer", type=int, default=None, help="encoder block whose output the head reads (default: last)")
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--conv-n", type=int, default=128)
    ap.add_argument("--conv-m", type=int, default=32)
    ap.add_argument("--proj", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.data.denoise import DenoiseEvents, split_events
    from helix.model.denoise import CellDecoder, build_denoise
    from helix.probe import resolution as R_
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]; args = meta["args"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    for p in model.parameters():
        p.requires_grad_(False)
    fm = model.fm
    if a.encoder == "pretrained":
        from helix.model.artifact import load as load_art
        from helix.model.fm import build_fm
        art = load_art("/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1")
        fm = build_fm({**art.arch, **ov}); fm.load_state_dict(art.state_dict); fm.to(dev).eval()
        for p in fm.parameters():
            p.requires_grad_(False)
    L_enc = len(fm.enc)
    if a.layer is not None and not 1 <= a.layer <= L_enc:
        raise SystemExit(f"--layer must be in 1..{L_enc}")

    def encode(B):
        if a.no_feat:
            return torch.zeros(B["inp"].shape[0], fm.d, device=B["inp"].device)
        if a.layer is None or a.layer == L_enc:
            return fm.encode(B)
        return fm.encode_layers(B, [a.layer])[a.layer]
    hk = dict(meta["head_kw"]); hk.pop("kind", None)
    m7 = model.head

    class AmpDecoder(CellDecoder):
        """CellDecoder whose direct input also carries the covering tokens' amplitudes sinh(value) / AMP_S."""
        def __init__(self, *args_, **kw):
            n_bands = kw.get("n_bands", 4); n_slot = kw.get("n_slot", 128)
            n_aux = 13 * n_bands + n_bands * n_slot          # extra amplitude slots ride in front of the raw block
            super().__init__(*args_, n_aux=n_aux, **kw)
            self.n_slot_ = n_slot

        def forward(self, feats, idx, aux, nbr=None, B=None):
            r = B["inp"][idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).to(B["inp"].dtype)    # (n, nb, n_slot)
            amp = (torch.sinh(r.float()) / AMP_S).to(aux.dtype)
            return super().forward(feats, idx, torch.cat([aux, amp.flatten(1)], 1), nbr=nbr, B=B)

    class NbrRawDecoder(CellDecoder):
        """CellDecoder whose direct input also carries the raw slots of every 3x3 neighbourhood token per band."""
        def __init__(self, *args_, **kw):
            n_bands = kw.get("n_bands", 4); n_slot = kw.get("n_slot", 128); n_nbr = kw.get("n_nbr", 9)
            super().__init__(*args_, n_aux=13 * n_bands + n_bands * n_nbr * (3 if a.nbr_amp else 2) * n_slot, **kw)

        def forward(self, feats, idx, aux, nbr=None, B=None):
            parts = [B["inp"], B["occ"].to(B["inp"].dtype)]
            if a.nbr_amp:
                parts.append((torch.sinh(B["inp"].float()) / AMP_S).to(B["inp"].dtype))
            raw = torch.cat(parts, -1)
            rn = raw[nbr.clamp(min=0)] * (nbr >= 0).unsqueeze(-1).to(raw.dtype)            # (n, nb, 9, 2 n_slot)
            return super().forward(feats, idx, torch.cat([aux, rn.flatten(1).to(aux.dtype)], 1), nbr=nbr, B=B)

    if a.head == "m7dc":
        head = m7
        for p in head.parameters():
            p.requires_grad_(True)
    else:
        kw = dict(n_bands=m7.n_bands, proj=a.proj or m7.out.in_features, layers=a.layers or len(m7.blocks),
                  presence=False)
        cls = NbrRawDecoder if a.nbr_raw else (AmpDecoder if a.amp else CellDecoder)
        head = cls(fm.d, **kw).to(dev)
    head.train()
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[{a.tag}] head {type(head).__name__} params {n_par/1e6:.2f}M", flush=True)

    def head_y(feats, idx, aux, nbr, Bt):
        o = head(feats, idx, aux, nbr, B=Bt)
        return (o[0] if isinstance(o, tuple) else o).float()

    runs = meta["runs"]
    train = DenoiseEvents(CORPUS, runs, CT, "train", cov_per_sig=args["cov_per_sig"], win_per_event=args["win_per_event"],
                          near_any_per_sig=args["near_any_per_sig"], win_near_per_event=args["win_near_per_event"],
                          neighbors=True, support_min_q=args["support_min_q"])
    y_sup = float(np.log1p(args["support_min_q"] / Q0))
    if a.min_q is not None:
        y_sup = max(y_sup, float(np.log1p(a.min_q / Q0)) - 1e-9)       # loss rows: truth >= min_q

    START = [0]
    state_path = os.path.join(a.out, f"{a.tag}_state.pt")
    if os.path.exists(state_path):
        START[0] = int(torch.load(state_path, map_location="cpu", weights_only=False)["step"])

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
            rng = np.random.default_rng((sum(map(ord, a.tag)), START[0], wid))
            while True:
                yield train[int(rng.integers(len(train)))]

    loader = iter(torch.utils.data.DataLoader(Stream(), batch_size=None, num_workers=a.workers, prefetch_factor=2,
                                              persistent_workers=True))

    def to_dev(it):
        B = {k: torch.as_tensor(v).to(dev) if isinstance(v, (np.ndarray, torch.Tensor)) else v for k, v in it["B"].items()}
        B["n_cells"] = B["plane_id"].shape[0]
        return B, torch.as_tensor(it["idx"]).to(dev), torch.as_tensor(it["aux"]).to(dev), torch.as_tensor(it["y"]).to(dev), \
            torch.as_tensor(it["nbr"]).to(dev)

    # ---- scoring on the floor test events (cached inputs; encoder features recomputed each time, fp32)
    ds0 = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds0.cfg
    test = []
    for ev in range(260, 308):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
        tkeys, tq, _ = ds0._truth(run, path, pos)
        sel = tq >= 4e3
        test.append((path, pos, tkeys[sel].astype(np.int64), tq[sel].astype(np.float64)))

    @torch.no_grad()
    def score():
        head.eval()
        E, P, T = [], [], []
        for path, pos, keys, tq in test:
            _, B, bl = ds0.tokens(path, pos)
            idx, aux = R_.cell_inputs(keys, B["cell_key"], bl, cfg)
            ok = (idx >= 0).any(1)
            nbr = R_.cell_neighbors(keys[ok], B["cell_key"], bl, cfg)
            Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            feats = encode(Bt)
            y = head_y(feats, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), torch.as_tensor(nbr).to(dev), Bt)
            g, _, _ = R_.unkey(keys[ok])
            E.append(y.cpu().numpy() - np.log1p(tq[ok] / Q0)); P.append(g % 3); T.append(tq[ok])
        E, P, T = map(np.concatenate, (E, P, T))
        head.train()
        r = {}
        for p in range(3):
            for lo, hi in BINS:
                m = (P == p) & (T >= lo) & (T < hi)
                a16, a50, a84 = np.percentile(E[m], [16, 50, 84])
                r[f"{'UVY'[p]}_{lo:.0f}"] = dict(hw68=float((a84 - a16) / 2), med=float(a50), n=int(m.sum()))
        return r

    os.makedirs(a.out, exist_ok=True)
    res = dict(tag=a.tag, args=vars(a), n_params=n_par, scores={})
    if START[0] == 0 and a.head == "m7dc":
        res["scores"]["0"] = score(); print(f"[{a.tag}] step 0", {k: round(v["hw68"], 4) for k, v in res["scores"]["0"].items()}, flush=True)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def converge(head, opt, n=128, m=32):
        """T9 restricted to the head: per-event gradients of the training loss (charge-only), split into bright rows
        (y >= log1p(64k/Q0)) and the rest; SNR at the training batch (accum events); then an fp32 line search on m
        other training events along the Adam-preconditioned mean (sqrt(v) floored at its median), step sized for a
        first-order decrease c * L0."""
        params = [p for p in head.parameters() if p.requires_grad]
        vs = [opt.state[p]["exp_avg_sq"].double() for p in params]
        vflat = torch.cat([v.reshape(-1) for v in vs]).sqrt()
        pre = 1.0 / torch.maximum(vflat, vflat.median())
        y_br = float(np.log1p(64e3 / Q0))
        rng = np.random.default_rng(99)
        evs = [train[int(i)] for i in rng.integers(len(train), size=n + m)]
        head.eval()

        def parts(it, fp32=False):
            B, idx, aux, y, nbr = to_dev(it)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
                feats = encode(B)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
                p = head_y(feats, idx, aux, nbr, B)
            pos = y > y_sup; br = y >= y_br; npos = pos.sum().clamp(min=1)
            se = (p - y) ** 2
            return (se * br).sum() / npos, (se * (pos & ~br)).sum() / npos

        flat = lambda gs: torch.cat([g.reshape(-1) for g in gs]).double()
        nP = sum(p.numel() for p in params)
        Sb = torch.zeros(nP, device=dev, dtype=torch.float64); So = torch.zeros_like(Sb)
        qb = qo = qt = 0.0
        for it in evs[:n]:
            rb, ro = parts(it)
            gb = flat(torch.autograd.grad(rb, params, retain_graph=True)); go = flat(torch.autograd.grad(ro, params))
            Sb += gb; So += go; qb += float(gb @ gb); qo += float(go @ go); qt += float((gb + go) @ (gb + go))

        def snr(S, q):
            mean = S / n; m2 = float(mean @ mean); tr = (q - n * m2) / (n - 1); g2 = m2 - tr / n
            return dict(G2=g2, trSigma=tr, SNR_B=g2 / (tr / a.accum))

        out = dict(bright=snr(Sb, qb), other=snr(So, qo), total=snr(Sb + So, qt),
                   cos_bright_other=float((Sb @ So) / (Sb.norm() * So.norm())))
        theta0 = [p.detach().clone() for p in params]

        def set_theta(step):
            off = 0
            with torch.no_grad():
                for p, t0_ in zip(params, theta0):
                    k = p.numel(); p.copy_(t0_ - step[off:off + k].view_as(p).to(p.dtype)); off += k

        @torch.no_grad()
        def evalm():
            t = np.zeros(2)
            for it in evs[n:]:
                rb, ro = parts(it, fp32=True); t += [float(rb), float(ro)]
            t /= m
            return dict(bright=t[0], rest=t[1], total=t[0] + t[1])

        base = evalm(); out["line"] = {"base": base}
        for dname, G in (("total", (Sb + So) / n), ("bright", Sb / n)):
            D = G * pre; L0 = base[dname]; gd = float(G @ D); out["line"][dname] = {}
            for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                set_theta(D * (c * L0 / gd)); r = evalm(); r["pred"] = -c * L0
                out["line"][dname][str(c)] = r
        set_theta(torch.zeros(nP, device=dev, dtype=torch.float64))
        head.train()
        print(f"[{a.tag}] converge", json.dumps(out), flush=True)
        return out

    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min((s - a.warmup) / max(1, a.steps - a.warmup), 1.0)))

    if START[0]:
        st = torch.load(state_path, map_location=dev, weights_only=False)
        head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); res["scores"] = st["scores"]
        print(f"[{a.tag}] resumed at step {START[0]}", flush=True)
    t0, acc = time.time(), 0.0
    for step in range(START[0], a.steps):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            B, idx, aux, y, nbr = to_dev(next(loader))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = encode(B)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = head_y(feats, idx, aux, nbr, B)
            pos = y > y_sup
            loss = ((p - y) ** 2 * pos).sum() / pos.sum().clamp(min=1) / a.accum
            loss.backward(); acc += float(loss)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"[{a.tag}] step {step + 1}/{a.steps} loss {acc / 100:.5f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time() - t0) / 100:.2f}s/step", flush=True)
            t0, acc = time.time(), 0.0
        if (step + 1) % 1000 == 0 or step + 1 == a.steps:
            res["scores"][str(step + 1)] = score()
            print(f"[{a.tag}] step {step + 1}", {k: round(v["hw68"], 4) for k, v in res["scores"][str(step + 1)].items()}, flush=True)
            json.dump(res, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1)
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step + 1, scores=res["scores"]), state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    torch.save(head.state_dict(), os.path.join(a.out, f"{a.tag}_head.pt"))
    if a.converge:
        res["converge"] = converge(head, opt, n=a.conv_n, m=a.conv_m)
        json.dump(res, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
