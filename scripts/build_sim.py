"""Build from a JAXTPC wire production campaign (:mod:`helix.tpc.io`): the planes of each sensor
entry are densified, noised (the colored incoherent + coherent model, :mod:`helix.tpc.noise`, seeded
per event) and digitized on the device; the noise-free planes are written only with ``--keep-clean``.
Chain: :func:`helix.tpc.chain.sim_config`.

One corpus run per campaign group directory (``<root>/j00X000``, 1000 jobs), one build task per
task's sensor file. An event is named by its campaign identity: ``/ident`` run = the group, source_file
= the sensor file's name, event = the campaign-unique ``event_id``. Shard k holds campaign events
k * events_per_shard .. (k + 1) * events_per_shard - 1, so a shard's number says which events it holds.

    python scripts/build_sim.py --input GROUP/TASK/sensor/sim_wire_sensor_TASK.h5 --output OUT/GROUP/sim_wire \\
        --device cuda --events-per-shard 200 --shard-start K
    python scripts/build_sim.py plan OUT GROUP_DIR...        one task per sensor file

Options and the shards: :mod:`helix.build`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
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
    """The sensor files of one campaign group: noisy planes (on the device) and the noise-free planes."""

    systems = ()
    dataset = "sim_wire"
    pattern = "*/sensor/*_sensor_*.h5"          # in a group directory: TASK/sensor/FILE

    def __init__(self, files, device, optical=True, geom="cubic_wireplane_geometry.json", npz=None):
        from helix.paths import packaged, resolve
        from helix.tpc import io
        from helix.tpc.geometry import _resolve, load_plane_registry
        self.files, self.device = list(files), device
        self.layouts = {f: io.sensor_layout(f) for f in self.files}
        lay = self.layouts[self.files[0]]
        for f, other in self.layouts.items():
            if (not np.array_equal(other.num_wires, lay.num_wires) or other.num_time_steps != lay.num_time_steps
                    or other.pedestals != lay.pedestals or other.campaign != lay.campaign):
                raise SystemExit(f"sim: {f} has another layout or campaign than {self.files[0]}")
        groups = {Path(f).parents[2].name for f in self.files}
        if len(groups) != 1:
            raise SystemExit(f"sim: the input files span groups {sorted(groups)}; one corpus run is one group")
        self.run, self.campaign, self.num_wires = groups.pop(), lay.campaign, lay.num_wires
        self.index, self.commits = [], set()
        for f in self.files:
            with h5py.File(f, "r") as h:
                self.index += [(f, pos, int(e)) for pos, e in enumerate(io.event_ids(h))]
                self.commits.add(str(h["config"].attrs["git_commit"]))
        if npz is None:                      # JAXTPC's spectrum when a checkout is declared, else helix's copy
            jx, _ = resolve("HELIX_JAXTPC_ROOT")
            cand = Path(jx) / "config" / "noise_spectrum.npz" if jx else None
            npz = str(cand) if cand and cand.exists() else str(packaged("noise_spectrum.npz"))
        z = np.load(npz, allow_pickle=True)
        self.spectrum = (z["spectrum_freqs_hz"], z["spectrum_shape"])
        self.reg = load_plane_registry(geom)
        for label in lay.plane_labels:       # the noise model's planes are the file's planes
            e = self.reg.get(lay.plane_gid(label))
            if e is None or (e["n_wires"], e["n_ticks"]) != (lay.n_wires(label), lay.num_time_steps):
                raise SystemExit(f"sim: {geom} plane {label} is {e and (e['n_wires'], e['n_ticks'])}, "
                                 f"the sensor file's {(lay.n_wires(label), lay.num_time_steps)}")
        try:
            geom_path = str(_resolve(geom))
        except Exception:
            geom_path = geom
        self.npz, self.geom_path = npz, geom_path

    def __len__(self):
        return len(self.index)

    @staticmethod
    def file_plan(path, events_per_shard):
        """``(events, first shard)`` of a sensor file: its events are a contiguous run of campaign
        ids starting at a multiple of ``events_per_shard``, so shard k is events k*eps .. (k+1)*eps-1."""
        from helix.tpc import io
        with h5py.File(path, "r") as f:
            io.require_format(f)
            ids = io.event_ids(f)
        if not ids.size or ids[0] % events_per_shard or (np.diff(ids) != 1).any():
            raise ValueError(f"{path}: events {ids[:1].tolist()}..{ids[-1:].tolist()} are not a contiguous run "
                             f"starting at a multiple of {events_per_shard}; shards are numbered by event id")
        return len(ids), int(ids[0]) // events_per_shard

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
        from helix.tpc import io, noise
        dev = self.device or "cpu"
        handles = {}
        try:
            for k in indices:
                path, pos, ev = self.index[k]
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                layout = self.layouts[path]
                seed = self.seed(path, ev)
                w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
                for label, (w, t, v) in io.read_event_coo(io.entry(handles[path], pos, ev), layout).items():
                    gid = layout.plane_gid(label)
                    gmap[gid] = label
                    w_l.append(torch.as_tensor(w, dtype=torch.long, device=dev))
                    t_l.append(torch.as_tensor(t, dtype=torch.long, device=dev))
                    v_l.append(torch.as_tensor(v, dtype=torch.float32, device=dev))
                    p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
                wire = torch.cat(w_l)
                grids = densify(wire, torch.cat(t_l), torch.cat(v_l), torch.cat(p_l),
                                torch.tensor([wire.numel()], dtype=torch.long, device=dev), self.reg)
                clean = {int(g): x[0].clone() for g, x in grids.items()}
                grids = noise.add_intrinsic_noise(grids, self.reg, seeds=[seed], incoherent=True, coherent=True,
                                                  series_spectrum=self.spectrum, group_size=self.group_size)
                grids = noise.digitize(grids, {int(g): layout.pedestal(gmap[int(g)]) for g in grids}, n_bits=12)
                yield dict(ident=dict(run=self.run, source_file=Path(path).name, event=ev, noise_seed=seed),
                           planes={int(g): x[0] for g, x in grids.items()}, clean=clean)
        finally:
            for h in handles.values():
                h.close()

    def shard_attrs(self):
        """What the coeff shards record about the input beyond the chain: the noise and its inputs."""
        noise = dict(kind="colored", incoherent=True, coherent=True, group_size=self.group_size,
                     spectrum=os.path.basename(self.npz))
        prov = dict(geom=os.path.basename(self.geom_path), geom_sha256=_sha256(self.geom_path),
                    spectrum=os.path.basename(self.npz), spectrum_sha256=_sha256(self.npz),
                    seed_formula="blake2b(run/source_file/ev{event})",
                    sim=dict(campaign=self.campaign, jaxtpc=sorted(self.commits)))
        return dict(noise=noise, provenance=prov)


if __name__ == "__main__":
    main(SimSource)
