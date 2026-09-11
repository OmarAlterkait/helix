"""Data-loader recipe — JAXTPC sensor DENSE path (densify + add noise on GPU).

CAMPAIGN.md (JAXTPC): the sensor readout is loaded SPARSE in the worker, moved to
the GPU, then densified to per-plane images and given fresh intrinsic noise
(JAXTPC production omits coherent noise by default, so it's added at load). This is
the only recipe that exercises the post-collate dense ops.

It is ONE `transform` list (map -> reduce -> map). `Collect` is the reduce
boundary: everything up to it runs per-event in the DataLoader workers (sparse —
only sparse crosses PCIe); everything after runs per-batch **after collate**, with
`ToDevice` as the device step so `Densify`/`AddNoise`/`Digitize` (the single
dispatching dense ops — no `Batch*`, no policy dict) run on the GPU and the grids
are born on-device. The dataset splits the list at `Collect`: the head is
`dataset.transform` (per-event), the tail is `dataset.batch_transform`, which a
trainer runs post-collate (e.g. `on_after_batch_transfer`).

Batch after the head + collate:  sensor_wire/time/value/plane_gid (ΣM,), sensor_offset.
After the tail (batch_transform): sensor_dense = {plane_gid: (B, W, T)} (noisy, digitized).
"""

# Moved from pimm-data with the transforms it drives. AddNoise and Digitize are
# helix's (the LArTPC forward model); Densify stays pimm-data's (going to dense
# is a data-layer operation). A recipe that needs helix's physics is a helix
# recipe applied to jaxtpc data -- and jaxtpc IS a LArTPC simulator, so the
# pairing is natural rather than a boundary violation.
custom_imports = dict(imports=["helix.data.transforms"], allow_failed_imports=False)

import os

from pimm_data.geometry import load_plane_registry

# real wire data: /sdf/data/neutrino/doraemon/wire_test_00_00_02
# (split='run_0027575715', dataset_name='sim_wire'; type-first layout sensor/run_*/)
_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")
# per-plane geometry (n_wires/n_ticks/wire_lengths) for densify — config-derived;
# must match the data's planes. (A future nicety: let the dataset inject its own
# plane_geometry() when geom is omitted, so even this isn't needed in the config.)
_geom = load_plane_registry("cubic_wireplane_geometry.json")

transform = [
    # ---- worker / per-event (sparse; only sparse crosses PCIe) ----
    dict(type="Collect", parts={
        "sensor": dict(keys=("wire", "time", "value", "plane_gid"))}),
    # ---- post-collate / per-batch (the dataset splits here -> batch_transform) ----
    dict(type="ToDevice", device="cuda"),                       # the device step
    dict(type="Densify",  geom=_geom, modality="sensor"),       # COO -> {gid:(B,W,T)} on GPU
    dict(type="AddNoise", geom=_geom, modality="sensor", coherent=True),
    dict(type="Digitize", geom=_geom, modality="sensor", n_bits=12),
]

data = dict(
    train=dict(type="JAXTPCDataset", data_root=_data_root, split="train",
               dataset_name="sim", modalities=("sensor",),
               transform=transform, max_len=-1),
    val=dict(type="JAXTPCDataset", data_root=_data_root, split="val",
             dataset_name="sim", modalities=("sensor",),
             transform=transform, max_len=1000),
)
# model = dict(...)  # placeholder — a dense/CNN model over sensor_dense.
