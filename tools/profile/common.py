"""Shared machinery for the helix FM profiling suite.

Everything here loads REAL corpus events through the production tokenizer, so
shapes and densities are not invented. No pimm import: the model, the tokenizer
and the corpus reader are all reachable from helix alone.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager

import numpy as np
import torch

# ---------------------------------------------------------------------------
# batch construction
# ---------------------------------------------------------------------------

#: keys that live in the CELL row-space (n_cells rows)
CELL_KEYS = ("occ", "inp", "tgt", "valid", "dead", "band_id", "plane_id",
             "cell_key", "t_phys", "wire_pos", "cell_wb", "cell_tb", "wirefeat")
#: keys that live in the ACTIVE-COEFFICIENT row-space, indexing cells by `cell`
ACT_KEYS = ("band", "val", "target", "cell", "slot")


def load_events(n=8, corpus=None, cell_t="grid_center", start=0):
    """-> list of per-event dicts of torch CPU tensors, exactly as CoeffCollect
    would emit them (minus n_cells, which the model derives)."""
    from helix.data import CoeffTPCDataset
    from helix.model.tokenize import CoeffTokenize
    from helix.paths import root

    corpus = corpus or str(root("HELIX_CORPUS"))
    ds = CoeffTPCDataset(data_root=corpus, dataset_name="sim_wire",
                         modalities=("coeff", "coeff_clean"), transform=None)
    tok = CoeffTokenize(part="coeff", clean_part="coeff_clean",
                        cfg=dict(cell_t=cell_t), fm_names=True)
    out, raw_times, tok_times = [], [], []
    for i in range(start, start + n):
        t0 = time.perf_counter()
        sample = ds.get_data(i)
        t1 = time.perf_counter()
        sample = tok(sample)
        t2 = time.perf_counter()
        raw_times.append(t1 - t0); tok_times.append(t2 - t1)
        sub = sample["coeff"]
        b = {}
        for k, v in sub.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                b[k] = torch.from_numpy(np.ascontiguousarray(v))
        b["_name"] = sample.get("name", f"ev{i}")
        out.append(b)
    return out, dict(read_s=raw_times, tokenize_s=tok_times)


def to_device(b, dev, non_blocking=False):
    return {k: (v.to(dev, non_blocking=non_blocking) if torch.is_tensor(v) else v)
            for k, v in b.items()}


def pack(events):
    """Concatenate K event batches into one token set, rebasing `cell`.

    NOTE: this is the *memory/throughput* pack. The production model has no
    event separation, so a packed batch is only valid for measurement unless the
    attention is made event-aware. `offset` is emitted so an event-aware forward
    can use it.
    """
    if len(events) == 1:
        b = dict(events[0])
        b["offset"] = torch.tensor([b["plane_id"].shape[0]], device=b["plane_id"].device)
        return b
    out = {}
    counts = [e["plane_id"].shape[0] for e in events]
    base = np.cumsum([0] + counts[:-1])
    for k in events[0]:
        if k.startswith("_") or not torch.is_tensor(events[0][k]):
            continue
        if k == "cell":
            out[k] = torch.cat([e[k] + int(base[i]) for i, e in enumerate(events)])
        elif k in ("cell_key",):
            out[k] = torch.cat([e[k] for e in events])
        else:
            out[k] = torch.cat([e[k] for e in events])
    dev = out["plane_id"].device
    out["offset"] = torch.tensor(np.cumsum(counts), device=dev)
    return out


def synth_from(event, n_cells):
    """A synthetic batch of exactly `n_cells` cells, built by tiling a real one.

    Preserves dtypes, slot occupancy density and the plane/band label mix; cell
    coordinates repeat, which is irrelevant to cost.
    """
    N0 = event["plane_id"].shape[0]
    reps = -(-n_cells // N0)
    out = {}
    for k in CELL_KEYS:
        if k not in event:
            continue
        v = event[k]
        out[k] = v.repeat(*([reps] + [1] * (v.dim() - 1)))[:n_cells].contiguous()
    # active rows: keep only rows whose cell survives the truncation
    cell = event["cell"]
    cells, vals = [], {k: [] for k in ACT_KEYS if k in event}
    for r in range(reps):
        off = r * N0
        keep = (cell + off) < n_cells
        if not bool(keep.any()):
            break
        cells.append(cell[keep] + off)
        for k in vals:
            if k == "cell":
                continue
            vals[k].append(event[k][keep])
    out["cell"] = torch.cat(cells)
    for k, parts in vals.items():
        if k != "cell":
            out[k] = torch.cat(parts)
    return out


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

M113 = dict(type="Coeff-FM", n_slot=128, n_band=4, n_plane=6, d=512, blocks=12,
            dec_blocks=4, heads=8, n_bins=128, dec_mode="cross", mup=True,
            d_base=128, serial=True, rope_split=False, gp=1024, gd=2048,
            plane_frac=0.1, mask_mode="random", mask_ratio=0.75, n_planes=1,
            plane_mode="plane")


def build(overrides=None, bins=True, device="cuda"):
    from helix.model.fm import build_fm
    cfg = dict(M113)
    cfg.pop("type")
    cfg.update(overrides or {})
    m = build_fm(cfg)
    if m.n_bins > 0 and bins:
        nb = m.bin_edges.shape[0]
        K = m.n_bins
        edges = np.stack([np.linspace(-6, 6, K + 1) for _ in range(nb)])
        m.set_bins(edges)
    return m.to(device)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# timing / memory
# ---------------------------------------------------------------------------

@contextmanager
def peak_mem():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    d = {}
    yield d
    torch.cuda.synchronize()
    d["base_MiB"] = base / 2**20
    d["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    d["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
    d["delta_MiB"] = (torch.cuda.max_memory_allocated() - base) / 2**20


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return dict(ms_mean=statistics.mean(ts), ms_med=statistics.median(ts),
                ms_min=min(ts), ms_max=max(ts), n=iters)


class ModuleProfiler:
    """Per-submodule wall time and allocator delta, via forward hooks.

    Generic on purpose: it mirrors no forward, so it cannot drift from one.
    Times are CUDA-event based and the totals are inclusive (a parent's time
    contains its children's), so read the LEAF rows for attribution.
    """

    def __init__(self, model):
        self.model = model
        self.rows = {}
        self._h = []
        self._stack = []

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if name == "":
                continue
            self._h.append(mod.register_forward_pre_hook(self._pre(name)))
            self._h.append(mod.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []

    def _pre(self, name):
        def f(mod, inp):
            e = torch.cuda.Event(True); e.record()
            self._stack.append((name, e, torch.cuda.memory_allocated()))
        return f

    def _post(self, name):
        def f(mod, inp, out):
            e = torch.cuda.Event(True); e.record()
            nm, s, m0 = self._stack.pop()
            r = self.rows.setdefault(nm, dict(type=type(mod).__name__, calls=0,
                                              ms=0.0, alloc_MiB=0.0, _ev=[]))
            r["calls"] += 1
            r["_ev"].append((s, e))
            r["alloc_MiB"] += (torch.cuda.memory_allocated() - m0) / 2**20
        return f

    def finish(self):
        torch.cuda.synchronize()
        for r in self.rows.values():
            r["ms"] = sum(s.elapsed_time(e) for s, e in r["_ev"])
            del r["_ev"]
        return self.rows


def leaf_rollup(rows):
    """Aggregate hook rows by module TYPE over leaf modules only."""
    names = set(rows)
    leaves = {n: r for n, r in rows.items()
              if not any(o != n and o.startswith(n + ".") for o in names)}
    agg = {}
    for n, r in leaves.items():
        a = agg.setdefault(r["type"], dict(calls=0, ms=0.0, alloc_MiB=0.0))
        a["calls"] += r["calls"]; a["ms"] += r["ms"]; a["alloc_MiB"] += r["alloc_MiB"]
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]["ms"]))


def gpu_info():
    if not torch.cuda.is_available():
        return dict(name="cpu-only", torch=torch.__version__)
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, sm=f"{p.major}.{p.minor}", total_GiB=p.total_memory / 2**30,
                smcount=p.multi_processor_count, torch=torch.__version__)


def prof_out():
    """Where profiling results go: $PROF_OUT, else <HELIX_EXP>/profiling/out.

    Resolved through helix.paths like every other root, so it is right on any
    site and names no one's directory. No site and no $PROF_OUT fails loudly.
    """
    env = os.environ.get("PROF_OUT")
    if env:
        return env
    from helix.paths import exp
    return str(exp("profiling", "out"))


def emit(tag, payload):
    out = prof_out()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{tag}.json")
    payload = dict(payload)
    payload["_gpu"] = gpu_info()
    payload["_jobid"] = os.environ.get("SLURM_JOB_ID")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"\n[emit] {path}")
    return path
