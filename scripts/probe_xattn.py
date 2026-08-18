"""Event-level CROSS-ATTENTION probe for the along-wire coordinate u.

Port of ``research/coeff_foundation_model/fm/pb_xattn.py``, at patch granularity.

Why this exists, in the reference's words: the per-token readout *"only sees the
pixel's OWN plane's 4 tokens -> it under-credits if the FM triangulated but
stored u NON-LOCALLY (in the V/Y tokens / cross-plane relations)."* Here a query
anchored at the target's geometry attends over ALL of the event's tokens, every
plane and both volumes, and predicts u.

This is the only remaining readout that can see cross-plane structure. The two
that came before it cannot:

  * ``mlp``        gathers only the target patch's own 4 band tokens.
  * ``triangulate`` forms partner context by band-pooling then slab-MEANING, and
    the reference documents that this destroys it — RoPE-encoded wire features do
    not average. Its selective replacement (``probe_3d_select``) was built, run,
    and did NOT rescue it (0.108/0.134 against a 0.624 ceiling).

Arms, matching the reference's controls:

  ``trained``  attention over the trained encoder's tokens
  ``random``   attention over RANDOM-INIT tokens. The decisive control: the
               query geometry and the additive key-position are present in BOTH,
               so anything above this arm is LEARNED CONTENT, not geometry the
               attention could have reconstructed by itself.
  ``raw``      attention over the input coefficient tokens.

Scored with the same ``fisher_r`` as the other probes — within-(event, plane)
Pearson r on out-of-fold predictions — so the numbers sit beside them.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _xprobe(d, in_q, in_k, in_kg, nh=8, no_attn=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class XProbe(nn.Module):
        """One multi-head cross-attention layer, then an MLP to a scalar.

        Verbatim in structure from ``pb_xattn.XProbe``: a geometry MLP makes the
        query, keys carry an ADDITIVE positional term from each cell's geometry,
        values are a plain projection. The additive key-position is what lets the
        random-init control still resolve geometry — which is exactly why that
        control is the meaningful floor.
        """

        def __init__(self):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(in_q, d), nn.GELU(), nn.Linear(d, d))
            self.k = nn.Linear(in_k, d)
            self.v = nn.Linear(in_k, d)
            # Consumes the KEY GEOMETRY (plane one-hot + wire + drift t),
            # not the feature vector — sizing this to in_k silently builds a
            # 512-wide layer for an 8-wide input and dies at the first matmul.
            self.pos = nn.Linear(in_kg, d)
            self.nh, self.hd = nh, d // nh
            self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, qg, H, kg):
            if no_attn:
                # SANITY ARM: query geometry only, attention bypassed. Must reach
                # what a plain MLP on the same 12 columns reaches (~0.10-0.13).
                # If it does not, the harness is not training and no other number
                # from it means anything — which is exactly what happened on the
                # first run, where every arm sat at ~0.00.
                return self.out(self.q(qg)).squeeze(-1)
            Q, K, V = self.q(qg), self.k(H) + self.pos(kg), self.v(H)
            nq, nc = Q.shape[0], H.shape[0]
            Q = Q.view(nq, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
            K = K.view(nc, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
            V = V.view(nc, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
            # Memory-efficient kernel: the explicit (nh, Nq, Nc) attention matrix
            # is ~6k x 30k x 8 = 1.4e9 entries per event and will not fit.
            ctx = F.scaled_dot_product_attention(Q, K, V)
            ctx = ctx.squeeze(0).transpose(0, 1).reshape(nq, -1)
            return self.out(ctx).squeeze(-1)

    return XProbe()


def _fit(events, folds, epochs, seed, d, device, lr=2e-3, wd=1e-2,
         patience=6, no_attn=False, qbatch=512):
    """Event-grouped CV. Returns out-of-fold predictions aligned to the input."""
    import torch
    from helix.probe.metrics import fisher_r

    ev_ids = np.array([e["event"] for e in events])
    uniq = np.unique(ev_ids)
    rng = np.random.RandomState(12345)          # same fold seed as fit_probe
    fold_of = {e: i % folds for i, e in enumerate(rng.permutation(uniq))}
    oof = [np.zeros_like(e["y"]) for e in events]

    for f in range(folds):
        tr = [i for i, e in enumerate(events) if fold_of[e["event"]] != f]
        te = [i for i, e in enumerate(events) if fold_of[e["event"]] == f]
        if not te:
            continue
        # A validation slice carved from TRAIN events only, for early stopping.
        rs = np.random.RandomState(seed * 1000 + f)
        va = set(rs.choice(tr, max(1, int(0.2 * len(tr))), replace=False).tolist())
        tr = [i for i in tr if i not in va]
        va = sorted(va)

        torch.manual_seed(seed * 100 + f)
        m = _xprobe(d, events[0]["qg"].shape[1], events[0]["H"].shape[1],
                    events[0]["kg"].shape[1], no_attn=no_attn).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        # Standardise the target on TRAIN only.
        ytr = np.concatenate([events[i]["y"] for i in tr])
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-6)
        best, bad, bstate = -1e9, 0, None
        # One optimizer step per EVENT gave 48 x 25 = 1200 steps total and the
        # probe never left its initialisation — every arm scored ~0.00, below
        # even the geometry-only floor. Chunk the queries and shuffle the chunks
        # across events so this is ordinary minibatch SGD; attention still sees
        # every key, only the query block is split.
        chunks = [(i, sl) for i in tr
                  for sl in np.array_split(np.arange(len(events[i]["y"])),
                                           max(1, len(events[i]["y"]) // qbatch))]
        for ep in range(epochs):
            m.train()
            for ci in np.random.RandomState(ep).permutation(len(chunks)):
                i, sl = chunks[ci]
                e = events[i]
                opt.zero_grad(set_to_none=True)
                p = m(e["qg"][sl], e["H"], e["kg"])
                loss = torch.nn.functional.mse_loss(
                    p, (torch.as_tensor(e["y"][sl], device=device) - mu) / sd)
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                py = [m(events[i]["qg"], events[i]["H"], events[i]["kg"]).cpu().numpy()
                      for i in va]
            r, _, _ = fisher_r(np.concatenate([events[i]["y"] for i in va]),
                               np.concatenate(py) * sd + mu,
                               np.concatenate([events[i]["ev_col"] for i in va]),
                               np.concatenate([events[i]["plane"] for i in va]))
            r = -1e9 if not np.isfinite(r) else r
            if r > best:
                best, bad = r, 0
                bstate = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if bstate is not None:
            m.load_state_dict(bstate)
        m.eval()
        with torch.no_grad():
            for i in te:
                oof[i] = (m(events[i]["qg"], events[i]["H"],
                            events[i]["kg"]).cpu().numpy() * sd + mu)
        del m
        torch.cuda.empty_cache()
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--max-events", type=int, default=60)
    ap.add_argument("--cap", type=int, default=3000, help="patches/event")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="geo,trained,random",
                    help="`geo` is the SANITY arm — query geometry, no "
                         "attention. It must reach ~0.10-0.13 or the "
                         "harness is not training and nothing else "
                         "from this run is quotable.")
    ap.add_argument("--qbatch", type=int, default=512)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--out", default="probe_xattn.jsonl")
    a = ap.parse_args()

    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, assemble, to_fm, pixel_cells
    from helix.model.checkpoint import patch_config_from_checkpoint
    from helix.probe.alongwire import u_of
    from helix.probe.features import features_at_layer, load_probe_model
    from helix.probe.metrics import fisher_r
    from helix.probe.patches import patch_rows
    from run_probe import _load_truth, _position_of

    cfg, aw, pix, offs, ident, stale = _load_truth(a.truth, a.corpus, strict=False)
    pcfg = patch_config_from_checkpoint(a.checkpoint) or PatchConfig()
    n_ev = min(a.max_events, len(ident))
    models = {}
    models["trained"], meta = load_probe_model(a.checkpoint, weights=a.weights)
    if "random" in a.arms.split(","):
        models["random"], _ = load_probe_model(a.checkpoint, random_init=True,
                                               random_seed=0)
    dev = next(models["trained"].parameters()).device
    print(f"{a.tag}: {n_ev} events, layer {a.layer}, arms {a.arms}", flush=True)

    prepped = []
    for i in range(n_ev):
        run, src, ev = ident[i]
        tag = src.replace("sim_wire_sensor_", "").replace(".h5", "")
        shard = os.path.join(a.corpus, f"{a.dataset_name}_coeff_{tag}.h5")
        with h5py.File(shard, "r") as f:
            c = f["config"]
            gids, nw = c["gids"][:], c["n_wires"][:]
            bl, ns = c["band_lengths"][:], c["norm_sigma"][:]
        ce = read_coeff_event(shard, _position_of(shard, ev))
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value,
                       gids=gids, n_wires=nw, band_lengths=bl, norm_sigma=ns, cfg=pcfg)
        B = to_fm(tok)
        lo, hi = offs[i], offs[i + 1]
        P = {k: v[lo:hi] for k, v in pix.items()}
        pc = pixel_cells(P["gid"], P["wire"], P["tick"], bl, pcfg)
        keys = tok["cell_key"]
        rows = np.searchsorted(keys, pc)
        rows = np.where((rows < len(keys)) &
                        (keys[np.clip(rows, 0, len(keys) - 1)] == pc), rows, -1)
        u = u_of(P["gid"].astype(np.int64), P["b1"], aw)
        pr = patch_rows(rows, P["gid"].astype(np.int64), P["wire"], P["tick"],
                        P["qtot"], P["ftop"], u,
                        dom_threshold=float(cfg.get("dom_threshold", 0.5)))
        if not len(pr["y"]):
            continue
        sel = np.arange(len(pr["y"]))
        if len(sel) > a.cap:
            sel = np.sort(np.random.RandomState(i).choice(sel, a.cap, replace=False))
        # KEY geometry, per cell: plane one-hot + wire + physical drift time. This
        # is what makes the random-init control a real floor rather than a zero.
        kg = np.concatenate([
            np.eye(6, dtype=np.float32)[tok["cell_gid"] % 6],
            (tok["cell_wire"] / 2000.0).astype(np.float32)[:, None],
            (tok["cell_t"] / 4321.0).astype(np.float32)[:, None],
        ], 1)
        prepped.append(dict(B=B, kg=kg, qg=pr["geo"][sel].astype(np.float32),
                            y=pr["y"][sel].astype(np.float32),
                            plane=pr["plane"][sel], event=i,
                            ev_col=np.full(len(sel), i)))
        if (i + 1) % 20 == 0:
            print(f"  prepped {i+1}/{n_ev}", flush=True)

    rows_out = []
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        events = []
        for e in prepped:
            Bt = {k: (torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v)
                  for k, v in e["B"].items()}
            Bt.setdefault("n_cells", Bt["plane_id"].shape[0])
            if arm == "geo":
                H = torch.zeros(1, 1, device=dev)
            elif arm == "raw":
                H = torch.as_tensor(e["B"]["inp"]).float().to(dev)
            else:
                H = features_at_layer(models[arm], Bt, a.layer).float()
            events.append(dict(H=H.detach(),
                               kg=torch.as_tensor(e["kg"]).to(dev),
                               qg=torch.as_tensor(e["qg"]).to(dev),
                               y=e["y"], plane=e["plane"], event=e["event"],
                               ev_col=e["ev_col"]))
        oof = _fit(events, a.folds, a.epochs, a.seed, a.d, dev,
                   no_attn=(arm == "geo"), qbatch=a.qbatch)
        y = np.concatenate([e["y"] for e in events])
        r, rs, minfo = fisher_r(y, np.concatenate(oof),
                                np.concatenate([e["ev_col"] for e in events]),
                                np.concatenate([e["plane"] for e in events]))
        row = dict(tag=a.tag, arm=arm, probe="xattn", layer=a.layer,
                   fisher_r=round(float(r), 5),
                   per_group_r_std=round(float(np.std(rs)), 4),
                   n_patch=int(len(y)), n_events=len(events), cap=a.cap,
                   d=a.d, folds=a.folds, epochs=a.epochs, seed=a.seed,
                   checkpoint=os.path.abspath(a.checkpoint),
                   corpus=os.path.abspath(a.corpus),
                   weights=meta.get("weights"))
        rows_out.append(row)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"  [{arm}] fisher_r {r:.5f}  ({len(y):,} patches)", flush=True)
        del events
        torch.cuda.empty_cache()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
