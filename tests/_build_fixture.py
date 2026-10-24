"""Plane images from a function through :func:`helix.build.build`, for tests: a source with no
reader, so a test controls the input exactly."""
from pathlib import Path

import h5py

from helix.build import build
from helix.core.coeff_io import read_coeff_event, read_coeff_shard


class PlanesSource:
    """``events``: ints, or (source_file, event[, noise_seed]); ``plane_fn(event) -> (noisy, clean)``
    as ``{gid: (wires, ticks)}`` images (``clean`` None or {} for no clean target)."""

    systems = ()

    def __init__(self, events, plane_fn, *, run="", noise=None, provenance=None):
        self.events_, self.plane_fn, self.run = list(events), plane_fn, run
        self.attrs = dict(noise=noise, provenance=provenance)

    def __len__(self):
        return len(self.events_)

    def bind(self, chain):
        pass

    def shard_attrs(self):
        return dict(self.attrs)

    def events(self, indices, workers):
        for k in indices:
            e = self.events_[k]
            src, ev, seed = (e[0], e[1], e[2] if len(e) > 2 else -1) if isinstance(e, (tuple, list)) else ("", e, -1)
            noisy, clean = self.plane_fn(ev)
            ident = dict(run=self.run, source_file=src, event=int(ev))
            if seed >= 0:
                ident["noise_seed"] = int(seed)
            yield dict(ident=ident, planes=noisy, clean=clean or None)


def build_corpus(events, plane_fn, chain, out_dir, *, dataset_name="cx", run="", norm_sigma=None,
                 with_clean=True, noise=None, provenance=None):
    """Build one shard ``out_dir/<dataset_name>_coeff_0000.h5`` (+ clean); returns the events
    read back (noisy, clean) and the shard's norm_sigma."""
    fn = plane_fn if with_clean else (lambda ev: (plane_fn(ev)[0], None))
    src = PlanesSource(events, fn, run=run, noise=noise, provenance=provenance)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    build(src, str(Path(out_dir) / dataset_name), indices=list(range(len(src))), chain=chain,
          events_per_shard=len(src), norm_sigma=norm_sigma)
    p = Path(out_dir) / f"{dataset_name}_coeff_0000.h5"
    noisy = read_coeff_shard(p)
    pc = Path(out_dir) / f"{dataset_name}_coeff_clean_0000.h5"
    clean = [read_coeff_event(pc, i, coords_from=p) for i in range(len(noisy))] if pc.exists() else []
    with h5py.File(p, "r") as f:
        norm = f["config"]["norm_sigma"][:]
    return noisy, clean, norm
