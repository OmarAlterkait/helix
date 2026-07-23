"""On-the-fly noisy+clean coefficient events (Phase 2 production pattern).

Noise is a LOAD-TIME TRANSFORM, mirroring production (pimm_data adds forward
noise on GPU; helix runs the batched GPU DWT), with the pimm-data loading
pattern: **DataLoader workers parallelize the CPU h5 read**, the GPU stage
runs post-collate in the main process. Per event —
  worker (CPU): read doraemon chunks (clean truth) -> flat array + lengths
  main (GPU):  noisy = clean + N(0, 2.6) on valid samples
               per-chunk sigma = db1 finest MAD (NaN-pad + nanmedian — the
               GPU twin of helix.optical.io.chunk_noise_sigma)
               batched coif3 L10 DWT of noisy AND clean (helix
               wavelet_ops_torch, pywt-exact), event-common pad
               support: |c_noisy| >= 1.2*sigma*sqrt(2 ln N_b), A10 kept
  -> packed rows: noisy value (input) + clean value (denoise target)

Noise seed is per-(event, packer-seed): deterministic for eval (fixed packer
seed), fresh per epoch for training (seed varies). Nothing is written to disk.

Use: OnflyPacker(event_keys(N), budget=..., seed=..., sort_size=...) — a
drop-in for star_model.Packer.
"""
import sys

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401  (must precede h5py in doraemon_optical)
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import doraemon_optical as dop
from helix.core.wavelet_ops_torch import _wavedec
from star_model import build_event_struct, pack_event, NBANDS, DEV

NOISE_STD = 2.6
KAPPA = 1.2
LEVEL = 10
WORKERS = 20


def event_keys(n_events=300):
    return list(dop.iter_events(n_events))


class _EventDS(Dataset):
    """CPU side (runs in workers): h5 read only."""

    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        path, ek = self.keys[i]
        ec = dop.read_event_chunks(path, ek)
        return dict(flat=np.concatenate(ec.chunks).astype(np.float32),
                    lengths=ec.lengths.astype(np.int64), idx=i)


@torch.no_grad()
def gpu_event(flat, cl, noise_seed):
    """GPU stage: noise -> sigma -> DWT(noisy, clean) -> threshold -> struct."""
    n = len(cl)
    L = int(np.ceil(cl.max() / (1 << LEVEL)) * (1 << LEVEL))
    X = torch.zeros(n, L, device=DEV)
    ar = torch.arange(L, device=DEV)
    lens = torch.as_tensor(cl, device=DEV)
    m = ar[None, :] < lens[:, None]
    X[m] = torch.as_tensor(flat, device=DEV)
    g = torch.Generator(device=DEV)
    g.manual_seed(int(noise_seed) % (2**31))
    noisy = X + torch.randn(X.shape, generator=g, device=DEV) * NOISE_STD * m

    d = (noisy[:, 0::2] - noisy[:, 1::2]) / np.sqrt(2.0)
    pair_ok = (torch.arange(L // 2, device=DEV)[None, :] * 2 + 1) < lens[:, None]
    dd = torch.where(pair_ok, d.abs(), torch.full_like(d, float("nan")))
    sigma = torch.nanmedian(dd, dim=1).values / 0.6745

    cn = _wavedec(noisy, "coif3", LEVEL)
    cc = _wavedec(X, "coif3", LEVEL)

    bands, chunks, idxs, vals, vals_c = [], [], [], [], []
    for i in range(NBANDS):                       # 0=A10 .. 9=D2 (D1 dropped)
        j = LEVEL if i == 0 else LEVEL - i + 1
        nb = cn[i].shape[-1]
        valid = (torch.arange(nb, device=DEV)[None, :] << j) < lens[:, None]
        if i == 0:
            act = valid                            # A10 kept untouched
        else:
            t = KAPPA * sigma[:, None] * float(np.sqrt(2.0 * np.log(max(nb, 2))))
            act = (cn[i].abs() >= t) & valid
        ch, ix = act.nonzero(as_tuple=True)
        bands.append(np.full(len(ch), i, np.int64))
        chunks.append(ch.cpu().numpy().astype(np.int64))
        idxs.append(ix.cpu().numpy().astype(np.int64))
        vals.append(cn[i][act].cpu().numpy().astype(np.float32))
        vals_c.append(cc[i][act].cpu().numpy().astype(np.float32))

    return build_event_struct(
        np.concatenate(bands), np.concatenate(idxs), np.concatenate(vals),
        np.concatenate(chunks), cl, val_clean=np.concatenate(vals_c))


def prep_onfly(key, seed=0):
    """Single-event convenience (no workers) — smoke tests."""
    path, ek = key
    ec = dop.read_event_chunks(path, ek)
    return gpu_event(np.concatenate(ec.chunks).astype(np.float32),
                     ec.lengths.astype(np.int64),
                     zlib.crc32(f"{path}:{ek}".encode()) ^ seed)


class OnflyPacker:
    """Drop-in for star_model.Packer: worker-prefetched reads + GPU transform."""

    def __init__(self, keys, budget=60000, seed=0, sort_size=False,
                 workers=WORKERS, prep=None):
        self.keys, self.budget = keys, budget
        self.seed, self.sort_size = seed, sort_size
        self.workers = min(workers, len(keys))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)
        dl = DataLoader(_EventDS(self.keys), batch_size=None, shuffle=True,
                        generator=g, num_workers=self.workers,
                        prefetch_factor=2 if self.workers else None,
                        collate_fn=lambda x: x)
        for item in dl:
            key = self.keys[item["idx"]]
            ev = gpu_event(item["flat"], np.asarray(item["lengths"]),
                           zlib.crc32(f"{key[0]}:{key[1]}".encode()) ^ self.seed)
            import star_model as _sm
            yield from pack_event(ev, rng, self.budget, self.sort_size,
                                  assemble_fn=_sm.Packer.assemble_fn)
