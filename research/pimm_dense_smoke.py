"""Smoke test of the pimm-data dense GPU path + the pimm run_step flow, on real
doraemon events, using the config-derived geometry registry.

  load sparse (JAXTPCDataset+Collect) -> collate -> [main process / on device]
  densify -> add_intrinsic_noise (coherent+incoherent) -> digitize  (born-on-GPU)
"""
from copy import deepcopy
import numpy as np
import torch
from pimm_data import (JAXTPCDataset, collate_fn, load_plane_registry,
                       build_sensor_gpu_stages, apply_batch_transforms)

DATA_ROOT = '/sdf/home/o/omara/data/omara/doraemon'
SPLIT = 'run_0026628546'
B = 2

reg = load_plane_registry('cubic_wireplane_geometry.json')
ds = JAXTPCDataset(
    data_root=DATA_ROOT, split=SPLIT, modalities=('sensor',), max_len=B,
    transform=[dict(type='Collect', stream='sensor',
                    keys=('wire', 'time', 'value', 'plane_gid'))])
batch = collate_fn([ds[i] for i in range(B)])   # what the DataLoader yields (SPARSE, CPU)
print(f"sparse batch keys={sorted(batch)}  hits={batch['wire'].numel()}  "
      f"device={batch['wire'].device}  ~{batch['wire'].numel()*12/1e6:.1f} MB COO\n")

stages = build_sensor_gpu_stages(reg, coherent=True, incoherent=True, n_bits=12)
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

for dev in devices:
    out = apply_batch_transforms(deepcopy(batch), stages, device=dev,
                                 base_seed=0, epoch=0, rank=0)
    grids = out['sensor_dense']
    print(f"[device={dev}] sensor_dense planes={sorted(grids)}")
    for gid in sorted(grids):
        g = grids[gid]
        exp = (B, reg[gid]['n_wires'], reg[gid]['n_ticks'])
        assert tuple(g.shape) == exp, f"plane {gid} shape {tuple(g.shape)} != {exp}"
        assert g.device.type == dev
        integral = bool(torch.allclose(g, torch.round(g)))      # digitized
        off_rms = float(g[g != 0].float().abs().std()) if (g != 0).any() else 0.0
        print(f"   gid {gid} ({reg[gid]['label']}): {tuple(g.shape)} on {g.device}, "
              f"integer={integral}, nonzero-std~{off_rms:.1f}")
    assert batch['wire'].device.type == 'cpu', "sparse stays CPU (born-on-device)"
    print()

# noise actually applied? compare clean-only densify vs noised
clean_only = apply_batch_transforms(deepcopy(batch),
                                    build_sensor_gpu_stages(reg, coherent=False,
                                                            incoherent=False, digitize=False),
                                    device='cpu')['sensor_dense']
noised = apply_batch_transforms(deepcopy(batch),
                                build_sensor_gpu_stages(reg, coherent=True, incoherent=True,
                                                        digitize=False),
                                device='cpu', base_seed=0)['sensor_dense']
g0 = sorted(reg)[0]
diff = float((noised[g0] - clean_only[g0]).abs().mean())
print(f"noise applied (mean|noised-clean| on gid {g0}) = {diff:.3f} ADC  (>0 => noise present)\n")

# --- pimm run_step flow (mirrors engines/train.py:run_step divert) ---
input_dict = collate_fn([ds[i] for i in range(B)])
gpu_transforms = stages                       # built once in Trainer.before_train
input_dict = apply_batch_transforms(input_dict, gpu_transforms, device=devices[-1],
                                    base_seed=0, epoch=0, rank=0)  # rank=comm.get_rank()
model_input = input_dict['sensor_dense']
print("pimm run_step: model receives input_dict['sensor_dense'] =",
      {gid: tuple(t.shape) for gid, t in sorted(model_input.items())})
print("SMOKE OK")
