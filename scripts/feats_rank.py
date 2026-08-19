"""RankMe: label-free effective rank of encoder features, per layer.

Port of ``research/coeff_foundation_model/fm/feats_rank.py``. ``exp(entropy of
the normalised singular values)`` of the centred feature matrix — a
representation-health number that needs **no target, no probe head, no folds and
no design matrix**.

That is the whole point of having it. Every failure this codebase hit while
comparing two checkpoints came from the probe's apparatus rather than from the
representation: the score depends on how many columns the design has (3.8x for
identical information), on which layer is read (k30 peaks at 8, not 12), on
whether the weights are raw or EMA (2x), and on which checkpoint of a flat-LR
run you happen to sample (+-0.1). RankMe has none of those knobs.

It measures collapse, not usefulness — a representation can be full-rank and
useless. Read it as a necessary condition and alongside the probe, which is how
the reference used it ("complements the layer-wise probe").

Usage::

    python3 scripts/feats_rank.py \\
        --checkpoint /sdf/data/neutrino/omara/exp/export/coeff-fm-train-r1-ema \\
        --corpus /sdf/data/neutrino/omara/coeff_tpc_r1/run_0027575715 \\
        --truth  .../truth/probe_truth_probe.h5 --tag r1-ema --layers 3,6,8,11,12
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rankme(X, exact=False):
    """``exp(-sum p log p)`` over ``p = s / sum(s)``, ``s`` = singular values.

    Reference (``fm/feats_rank.py``) centres, then calls ``svdvals`` directly.
    At our scale that is intractable: 50 events x ~30k cells is N ~ 1.5M rows,
    and ``svdvals`` on (N, 512) costs O(N d^2). The singular values of ``X`` are
    the square roots of the eigenvalues of the Gram matrix ``X^T X`` (512x512),
    which is the same quantity for a fraction of the cost — verified against
    ``svdvals`` in ``tests/test_feats_rank.py``. Pass ``exact=True`` to use
    ``svdvals`` directly on small inputs.
    """
    import torch
    X = torch.as_tensor(X).float()
    if exact:
        Xc = X - X.mean(0, keepdim=True)
        return _entropy_rank(torch.linalg.svdvals(Xc))
    return rank_from_gram((X.T @ X).double(), X.sum(0).double(), X.shape[0])


def _entropy_rank(s):
    """``exp(-sum p log p)`` over ``p = s / sum(s)``. The definition itself."""
    import torch
    p = s / (s.sum() + 1e-9)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def rank_from_gram(gram, colsum, n):
    """RankMe from a STREAMED Gram matrix: ``gram = X^T X``, ``colsum = X.sum(0)``.

    The seam that lets `main` and `rankme` be the same computation. `main` must
    stream — N ~ 1.5M rows x 512 is ~3 GB of features it never needs at once —
    so it accumulates the Gram batch by batch, and it used to finish the job with
    its own inline copy of the centring, eigendecomposition and entropy. That
    copy was the shipped path; `rankme` was the tested one. Verified equal here
    instead of assumed.

    Centring happens in Gram space: ``cov = X^T X - n mu mu^T``, which is exact
    and needs only the column sums, not a second pass over X.
    """
    import torch
    mu = colsum / n
    ev = torch.linalg.eigvalsh(gram - n * torch.outer(mu, mu))
    # Tiny negatives are round-off on a PSD matrix, not signal.
    return _entropy_rank(torch.sqrt(torch.clamp(ev, min=0.0)).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="3,6,8,11,12")
    ap.add_argument("--max-events", type=int, default=50)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--out", default="feats_rank.jsonl")
    a = ap.parse_args()

    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, assemble, to_fm
    from helix.model.checkpoint import patch_config_from_checkpoint
    from helix.probe.features import features_at_layer, load_probe_model
    from run_probe import _load_truth, _position_of

    cfg, aw, pix, offs, ident, stale = _load_truth(a.truth, a.corpus, strict=False)
    pcfg = patch_config_from_checkpoint(a.checkpoint) or PatchConfig()
    layers = [int(x) for x in a.layers.split(",")]
    n_ev = min(a.max_events, len(ident))
    model, meta = load_probe_model(a.checkpoint, weights=a.weights)
    dev = next(model.parameters()).device
    print(f"{a.tag}: {n_ev} events, layers {layers}, {os.path.basename(a.checkpoint)}",
          flush=True)

    batches = []
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
        batches.append(to_fm(tok))

    res = {"tag": a.tag, "checkpoint": os.path.abspath(a.checkpoint),
           "corpus": os.path.abspath(a.corpus), "weights": meta.get("weights"),
           "n_events": n_ev, "rank": {}}
    for L in layers:
        # Accumulate the Gram matrix and the column sums instead of holding every
        # feature row: N ~ 1.5M x 512 float32 is ~3 GB, and it is not needed.
        d = None
        g = None
        colsum = None
        n = 0
        for B in batches:
            Bt = {k: (torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v)
                  for k, v in B.items()}
            Bt.setdefault("n_cells", Bt["plane_id"].shape[0])
            f = features_at_layer(model, Bt, L).float()
            if g is None:
                d = f.shape[1]
                g = torch.zeros(d, d, dtype=torch.float64, device=f.device)
                colsum = torch.zeros(d, dtype=torch.float64, device=f.device)
            g += (f.T @ f).double()
            colsum += f.sum(0).double()
            n += f.shape[0]
            del f
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        rm = rank_from_gram(g, colsum, n)
        res["rank"][str(L)] = round(rm, 3)
        print(f"[{a.tag}] layer {L:>2}: RankMe={rm:8.2f}  (d={d}, N={n:,})", flush=True)

    with open(a.out, "a") as fh:
        fh.write(json.dumps(res, sort_keys=True) + "\n")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
