"""Freeze the FM model + tokenizer outputs, so research/ can be deleted.

`tests/test_model_fm.py` proves helix.model reproduces the research
implementation by importing `model_serial` and diffing against it. That makes
the guarantee depend on the very code we are about to remove: delete research/
and the proof goes with it.

This re-anchors the guarantee. Run it while research/ still exists — it verifies
research == helix ONE more time and, only if that holds, writes the outputs to a
golden file. Afterwards the parity test compares against the golden and no
longer imports research at all.

Two goldens, matching what pimm will depend on:

  model      m113's head outputs (occ_logit / val / logvar) and encode(), on a
             seeded input batch. The batch is regenerated from its seed rather
             than stored, so the file stays small.
  tokenizer  helix.tokenize.assemble + to_fm on pinned REAL coeff events from
             the corpus. Nothing pins the token layout today (patch geometry,
             centroid cell_t, arcsinh normalisation), and pimm's CoeffTokenize
             will consume exactly this.

Format follows `research/goldens/capture.py`: `dtype|shape|sha256[:16]` per
array, so the golden is a few kB of JSON rather than tens of MB of tensors.

Usage::

    python tools/capture_fm_golden.py --write          # capture (needs research/)
    python tools/capture_fm_golden.py --check          # verify (needs neither)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, os.pardir, "tests", "goldens_fm.json")
RESEARCH = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm"
CKPT = os.path.join(RESEARCH, "ckpt_clean160cat_m113_snap1000000.pt")
# The DURABLE anchor: a self-contained converted checkpoint (config + weights +
# inlined bin edges) outside git and outside research/. `--check` reads this and
# nothing else, so the guarantee outlives the research tree. The raw research
# checkpoint is only consulted when capturing.
ARCHIVE = "/sdf/data/neutrino/omara/archive/fm_m113_converted.pt"
CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"

BATCH_SEED = 20260806          # regenerate the input batch, don't store it
PINNED_EVENTS = (0, 1)         # events of shard 0000 used for the tokenizer golden


def dig(a):
    """`dtype|shape|sha256[:16]` — the research goldens format."""
    a = np.asarray(a)
    return f"{a.dtype}|{a.shape}|{hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}"


# --------------------------------------------------------------------------
# the seeded input batch — defined here so capture and check agree exactly
# --------------------------------------------------------------------------

def make_batch(n_cells, n_slot, n_band, n_plane, seed=BATCH_SEED):
    import torch
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.3).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)
    return dict(
        band_id=torch.randint(0, n_band, (n_cells,), generator=g),
        plane_id=torch.randint(0, n_plane, (n_cells,), generator=g),
        t_phys=torch.randn(n_cells, generator=g) * 500,
        wire_pos=r(n_cells) * 1900,
        wirefeat=r(n_cells, 1),
        inp=torch.randn(n_cells, n_slot, generator=g),
        occ=occ, valid=(r(n_cells, n_slot) < 0.9), tgt=tgt,
        target=tgt[cell, slot], cell=cell, slot=slot, n_cells=n_cells,
    )


def model_outputs(model, cfg):
    """Head outputs + encode features for the pinned batch."""
    import torch
    B = make_batch(300, cfg["n_slot"], cfg["n_band"], cfg["n_plane"])
    mask = torch.zeros(300, dtype=torch.bool)
    mask[::2] = True                                  # deterministic, not drawn
    model.eval()
    with torch.no_grad():
        occ, val, logvar = model.raw_heads(B, mask)
        enc = model.encode(B)
    out = {"occ_logit": dig(occ.numpy()), "val": dig(val.numpy()),
           "encode": dig(enc.numpy())}
    out["logvar"] = "None" if logvar is None else dig(logvar.numpy())
    return out


def tokenizer_outputs():
    """helix.tokenize on pinned REAL corpus events."""
    import glob
    from helix.core.coeff_io import read_coeff_event
    from helix.tokenize import assemble, to_fm, PatchConfig
    import h5py

    shard = sorted(glob.glob(os.path.join(CORPUS, "sim_wire_coeff_0000.h5")))
    if not shard:
        return None
    shard = shard[0]
    clean = shard.replace("_coeff_", "_coeff_clean_")
    with h5py.File(shard, "r") as f:
        cfg_g = f["config"]
        gids = cfg_g["gids"][:]
        n_wires = cfg_g["n_wires"][:]
        band_lengths = cfg_g["band_lengths"][:]
        norm_sigma = cfg_g["norm_sigma"][:]

    out = {}
    for ev in PINNED_EVENTS:
        ce = read_coeff_event(shard, ev)
        cc = read_coeff_event(clean, ev, coords_from=shard)
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value,
                       gids=gids, n_wires=n_wires, band_lengths=band_lengths,
                       norm_sigma=norm_sigma, cfg=PatchConfig(),
                       value_clean=cc.value)
        fm = to_fm(tok)
        out[f"ev{ev}"] = {k: dig(v) for k, v in sorted(fm.items())
                          if isinstance(v, np.ndarray)}
        out[f"ev{ev}"]["n_cells"] = int(tok["n_cells"])
    return out


def load_anchor():
    """The converted checkpoint the golden is anchored to.

    Prefers the archived self-contained artifact; falls back to converting the
    research checkpoint (capture-time only). Raises rather than returning None —
    a missing anchor must be loud, since the whole point is that the guarantee
    cannot evaporate quietly."""
    import torch
    if os.path.exists(ARCHIVE):
        b = torch.load(ARCHIVE, map_location="cpu", weights_only=False)
        return b, ARCHIVE
    if os.path.exists(CKPT):
        from tools.convert_fm_ckpt import convert
        return convert(CKPT, None), CKPT
    raise SystemExit(
        f"no anchor checkpoint found.\n"
        f"  looked for: {ARCHIVE}\n"
        f"          and: {CKPT}\n"
        f"The golden verifies helix.model against real trained weights; without "
        f"them there is nothing to verify. Restore the archived converted "
        f"checkpoint (tools/convert_fm_ckpt.py writes it).")


def build(verify_against_research):
    """Compute both goldens. When `verify_against_research`, assert the helix
    model still matches the research implementation before freezing."""
    import torch
    # CPU only, SINGLE THREADED. Pinning the device is not enough: CPU float
    # reductions are split across intra-op threads, so the sum order — and the
    # last bits of the result — depend on how many threads torch chose, which
    # depends on machine load. Measured on this model: 1 thread, 2 threads and
    # 4+ threads each give a DIFFERENT digest. An unpinned golden therefore
    # "fails" whenever the node is busy, which is exactly the intermittent
    # mismatch this had before. One thread is reproducible anywhere.
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(HERE, os.pardir))
    from helix.model import build_fm

    blob, src = load_anchor()
    cfg, sd = blob["config"], blob["state_dict"]
    if isinstance(cfg.get("film"), list):
        cfg = dict(cfg, film=tuple(cfg["film"]))
    model = build_fm(cfg, serial=True).cpu()
    model.load_state_dict(sd, strict=True)

    got = {"model": model_outputs(model, cfg),
           "config": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in sorted(cfg.items())},
           "ckpt_digest": blob["provenance"]["state_digest"],
           "anchor": os.path.basename(src),
           "device": "cpu",
           "threads": 1,
           "torch": torch.__version__,
           "batch_seed": BATCH_SEED}

    if verify_against_research:
        sys.path.insert(0, RESEARCH)
        from model_serial import SerialFMModel as Ref
        ref_kw = dict(cfg)
        ref_kw.pop("n_wirefeat", None)
        ref = Ref(**{k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in ref_kw.items()})
        ref.load_state_dict(sd, strict=True)
        ref_out = model_outputs(_AsHeads(ref), cfg)
        bad = [k for k in got["model"] if got["model"][k] != ref_out[k]]
        if bad:
            raise SystemExit(
                f"REFUSING to freeze: helix and research disagree on {bad}. "
                f"The golden would enshrine a divergence.")
        got["verified_against_research"] = True

    tok = tokenizer_outputs()
    if tok is not None:
        got["tokenizer"] = tok
    return got


class _AsHeads:
    """Adapt the research model (whose `forward` IS the head call) to the
    `raw_heads`/`encode` surface `model_outputs` uses."""

    def __init__(self, m):
        self._m = m

    def raw_heads(self, B, mask):
        return self._m(B, mask)

    def encode(self, B):
        return self._m.encode(B)

    def eval(self):
        self._m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="capture and write the golden")
    ap.add_argument("--check", action="store_true", help="verify against the golden")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the research cross-check when capturing (only if "
                         "research/ is already gone — the golden is then unwitnessed)")
    a = ap.parse_args(argv)
    if not (a.write or a.check):
        ap.error("pass --write or --check")

    got = build(verify_against_research=a.write and not a.no_verify
                and os.path.exists(CKPT))

    if a.write:
        with open(GOLDEN, "w") as f:
            json.dump(got, f, indent=1, sort_keys=True)
        print(f"wrote {GOLDEN}")
        print(f"  verified against research: {got.get('verified_against_research', False)}")
        print(f"  tokenizer events: {sorted(got.get('tokenizer', {}))}")
        return 0

    with open(GOLDEN) as f:
        want = json.load(f)
    diffs = []
    for env in ("device", "threads", "torch"):
        if want.get(env) != got.get(env):
            _why = {"device": "  (bit-exact digests are device-dependent)",
                    "threads": "  (CPU reduction order depends on thread count)",
                    "torch": "  (a torch upgrade can change numerics)"}
            diffs.append(f"{env}: {got.get(env)!r} != golden {want.get(env)!r}"
                         + _why.get(env, ""))
    for section in ("model", "tokenizer"):
        if section not in want:
            continue
        _cmp(want[section], got.get(section, {}), section, diffs)
    if diffs:
        print(f"GOLDEN MISMATCH ({len(diffs)}):")
        for d in diffs:
            print(f"  - {d}")
        return 1
    print("goldens OK")
    return 0


def _cmp(want, got, path, diffs):
    for k, v in want.items():
        if isinstance(v, dict):
            _cmp(v, got.get(k, {}), f"{path}/{k}", diffs)
        elif got.get(k) != v:
            diffs.append(f"{path}/{k}: {got.get(k)!r} != {v!r}")


if __name__ == "__main__":
    raise SystemExit(main())
