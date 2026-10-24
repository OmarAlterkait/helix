"""Build from JAXTPC sensor files (one run per directory): the planes are densified, noised (the
colored incoherent + coherent model, :mod:`helix.tpc.noise`, seeded per event) and digitized on the
device; the noise-free planes are the clean target. Chain: :func:`helix.tpc.chain.sim_config`.

    python scripts/build_sim.py --input RUN/sim_wire_sensor_0000.h5 --output OUT/RUN/sim_wire --device cuda
    python scripts/build_sim.py plan OUT RUN_DIR...        one task per sensor file, shard = file number

Options and the shards: :mod:`helix.build`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from helix.build import main


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class SimSource:
    """JAXTPC sensor files of one run: noisy planes (on the device) and the clean target."""

    systems = ()
    dataset = "sim_wire"
    pattern = "*_sensor_*.h5"

    def __init__(self, files, device, optical=True, geom="cubic_wireplane_geometry.json", npz=None):
        from helix.tpc.geometry import _resolve, load_plane_registry
        from helix.tpc.io import list_events, sensor_layout
        from helix.paths import packaged, resolve
        self.files, self.device = list(files), device
        self.layouts = {f: sensor_layout(f) for f in self.files}
        nw = {json.dumps(np.asarray(lay.num_wires).tolist()) for lay in self.layouts.values()}
        if any(lay.num_wires is None for lay in self.layouts.values()) or len(nw) != 1:
            raise SystemExit("sim: every input file needs the same /config/num_wires layout")
        self.num_wires = self.layouts[self.files[0]].num_wires
        self.run = Path(self.files[0]).parent.name
        self.index = [(f, e) for f in self.files for e in list_events(f)]
        if npz is None:                      # JAXTPC's spectrum when a checkout is declared, else helix's copy
            jx, _ = resolve("HELIX_JAXTPC_ROOT")
            cand = Path(jx) / "config" / "noise_spectrum.npz" if jx else None
            npz = str(cand) if cand and cand.exists() else str(packaged("noise_spectrum.npz"))
        z = np.load(npz, allow_pickle=True)
        self.spectrum = (z["spectrum_freqs_hz"], z["spectrum_shape"])
        self.reg = load_plane_registry(geom)
        try:
            geom_path = str(_resolve(geom))
        except Exception:
            geom_path = geom
        self.npz, self.geom_path = npz, geom_path

    def __len__(self):
        return len(self.index)

    @staticmethod
    def file_shard(path):
        """``..._sensor_0042.h5`` -> shard 42."""
        return int(Path(path).stem.rsplit("_", 1)[1])

    def chain_config(self, overrides):
        from helix.tpc.chain import sim_config
        return sim_config(self.num_wires, overrides)

    def bind(self, chain):
        self.group_size = int(chain.cfg["readout"]["group_size"])

    def seed(self, src, ev):
        """The event's noise seed: run, file and event, so no two events share a realisation."""
        key = f"{self.run}/{Path(src).name}/ev{ev}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & 0xFFFFFFFF

    def events(self, indices, workers):
        import torch
        from pimm_data.dense_ops import densify
        from helix.tpc import noise
        from helix.tpc.io import read_sensor_event_coo
        dev = self.device or "cpu"
        for k in indices:
            path, ev = self.index[k]
            layout = self.layouts[path]
            seed = self.seed(path, ev)
            w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
            for label, (w, t, v, _, _) in read_sensor_event_coo(path, ev, layout).items():
                gid = layout.plane_gid(label)
                gmap[gid] = label
                w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
                t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
                v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
                p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
            wire = torch.cat(w_l)
            grids = densify(wire, torch.cat(t_l), torch.cat(v_l), torch.cat(p_l),
                            torch.tensor([wire.numel()], dtype=torch.long, device=dev), self.reg)
            clean = {int(g): x[0].clone() for g, x in grids.items()}
            grids = noise.add_intrinsic_noise(grids, self.reg, seeds=[seed], incoherent=True, coherent=True,
                                                  series_spectrum=self.spectrum, group_size=self.group_size)
            grids = noise.digitize(grids, {int(g): layout.pedestal(gmap[int(g)]) for g in grids}, n_bits=12)
            yield dict(ident=dict(run=self.run, source_file=Path(path).name, event=int(ev), noise_seed=seed),
                       planes={int(g): x[0] for g, x in grids.items()}, clean=clean)

    def shard_attrs(self):
        """What the coeff shards record about the input beyond the chain: the noise and its inputs."""
        noise = dict(kind="colored", incoherent=True, coherent=True, group_size=self.group_size,
                     spectrum=os.path.basename(self.npz), mode="serial", backend="torch")
        prov = dict(geom=os.path.basename(self.geom_path), geom_sha256=_sha256(self.geom_path),
                    spectrum=os.path.basename(self.npz), spectrum_sha256=_sha256(self.npz),
                    seed_formula="blake2b(run/source_file/ev{n})")
        return dict(noise=noise, provenance=prov)


if __name__ == "__main__":
    main(SimSource)
