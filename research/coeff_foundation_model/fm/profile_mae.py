"""Profile the REAL MAE training step end-to-end (mask 0.75, encoder-drop): per-phase ms,
throughput, peak mem, and a torch.profiler kernel breakdown. The MAE regime differs from the
all-visible deconv: encoder sees only (1-mask) tokens, decoder sees all.

Phases: data(np.load+assemble, CPU) | h2d | mask | fwd | loss | bwd | opt
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D
from data import DEV
from model import FMModel, losses
from train import make_mask, move


def sync(): torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--dec_blocks", type=int, default=2); ap.add_argument("--mask", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--nev", type=int, default=12)
    ap.add_argument("--profiler", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); D.init_pipeline_cpu()
    model = FMModel(D.N_SLOT, D.N_BAND, D.N_PLANE, n_wirefeat=1, d=args.d, blocks=args.blocks,
                    dec_blocks=args.dec_blocks).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    npar = sum(p.numel() for p in model.parameters())
    files = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:args.nev]
    print(f"MAE profile: d={args.d} enc={args.blocks} dec={args.dec_blocks} mask={args.mask} "
          f"params={npar/1e6:.1f}M  (encoder sees ~{(1-args.mask)*100:.0f}% of tokens)")

    ph = {k: [] for k in ("data", "h2d", "mask", "fwd", "loss", "bwd", "opt")}
    ntok = []; nvis = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        f = files[step % len(files)]
        t = time.time(); Bc = D.get_cached(f, device="cpu"); ph["data"].append(time.time() - t)   # CPU assemble
        t = time.time(); B = move(Bc, DEV); sync(); ph["h2d"].append(time.time() - t)
        g = torch.Generator(device=DEV).manual_seed(step)
        t = time.time(); m = make_mask(B, "random", args.mask, 1, g); sync(); ph["mask"].append(time.time() - t)
        ntok.append(int(B["n_cells"])); nvis.append(int((~m).sum()))
        opt.zero_grad(set_to_none=True)
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m); sync(); ph["fwd"].append(time.time() - t)
            t = time.time(); bce, val = losses(occ, mu, lv, B, m); loss = bce + val; sync(); ph["loss"].append(time.time() - t)
        t = time.time(); loss.backward(); sync(); ph["bwd"].append(time.time() - t)
        t = time.time(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sync(); ph["opt"].append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 1e9
    warm = 5
    tot = sum(np.mean(ph[k][warm:]) for k in ph)
    print(f"\n  mean tokens/event {int(np.mean(ntok))}  visible(enc) {int(np.mean(nvis))}  peak {peak:.1f} GB")
    print(f"\n  {'phase':>6} {'ms':>7} {'%':>6}")
    for k in ("data", "h2d", "mask", "fwd", "loss", "bwd", "opt"):
        ms = np.mean(ph[k][warm:]) * 1000
        print(f"  {k:>6} {ms:>7.1f} {ms/(tot*1000)*100:>5.1f}%")
    gpu = tot - np.mean(ph["data"][warm:])    # everything except CPU data-assembly
    print(f"  {'TOTAL':>6} {tot*1000:>7.1f}")
    print(f"\n  GPU-bound step (excl data): {gpu*1000:.0f} ms -> {1/gpu:.1f} events/s/GPU")
    print(f"  full step (incl serial data): {tot*1000:.0f} ms -> {1/tot:.1f} events/s/GPU")
    print(f"  data is {np.mean(ph['data'][warm:])/tot*100:.0f}% of serial step -> {'HIDE with workers' if np.mean(ph['data'][warm:])>gpu*0.3 else 'minor'}")

    if args.profiler:
        from torch.profiler import profile, ProfilerActivity
        B = move(D.get_cached(files[0], device="cpu"), DEV)
        g = torch.Generator(device=DEV).manual_seed(1); m = make_mask(B, "random", args.mask, 1, g)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m)
            (bce + val).backward(); opt.step()
        sync()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m)
                (bce + val).backward(); opt.step()
            sync()
        print("\n=== top CUDA kernels ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
