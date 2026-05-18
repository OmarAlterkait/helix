"""HDF5 I/O for JAXTPC production sensor files and processed output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import h5py

from helix.config import DetectorConfig
from helix.wavelet import SparseResult


def read_sensor_plane(
    path: str | Path,
    event_idx: int,
    plane_label: str,
    num_time_steps: int = 2701,
    pedestal: int = 0,
) -> np.ndarray:
    """Read one plane from a JAXTPC production sensor file.

    Returns (n_wires, n_ticks) float32 pedestal-subtracted image.
    """
    with h5py.File(path, "r") as f:
        evt = f[f"event_{event_idx}"]
        grp = evt[plane_label]
        wire_indices = grp["wire"][:]
        time_indices = grp["time"][:]
        values = grp["values"][:].astype(np.float32)
        n_wires = int(grp.attrs.get("n_wires", wire_indices.max() + 1))

    image = np.full((n_wires, num_time_steps), float(pedestal), dtype=np.float32)
    image[wire_indices, time_indices] = values
    image -= pedestal
    return image


def read_sensor_event(
    path: str | Path,
    event_idx: int,
    config: DetectorConfig,
) -> dict[str, np.ndarray]:
    """Read all planes for one event."""
    planes = {}
    with h5py.File(path, "r") as f:
        evt = f[f"event_{event_idx}"]
        available = list(evt.keys())

    for label in config.plane_labels:
        if label in available:
            pt = label.split("_")[-1] if "_" in label else label
            ped = config.pedestals.get(pt, 0)
            planes[label] = read_sensor_plane(
                path, event_idx, label, config.num_time_steps, ped)
    return planes


def count_events(path: str | Path) -> int:
    """Count number of events in a sensor file."""
    with h5py.File(path, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("event_"))


def write_processed(
    path: str | Path,
    event_idx: int,
    planes: dict[str, SparseResult],
    config: DetectorConfig,
) -> None:
    """Write processed sparse results to HDF5."""
    with h5py.File(path, "a") as f:
        evt = f.require_group(f"event_{event_idx}")
        evt.attrs["wavelet"] = config.wavelet
        evt.attrs["dwt_level"] = config.dwt_level
        evt.attrs["threshold_kappa"] = config.threshold_kappa

        for label, result in planes.items():
            grp = evt.require_group(label)
            grp.attrs["n_kept"] = result.n_kept
            grp.attrs["n_total"] = result.n_total
            grp.attrs["sparsity"] = result.sparsity
            grp.attrs["sigma_per_band"] = result.sigma_per_band

            for i, c in enumerate(result.coeffs):
                band_name = "cA" if i == 0 else f"cD_{len(result.coeffs) - i}"
                if band_name in grp:
                    del grp[band_name]
                nz = np.nonzero(c)
                if len(nz[0]) > 0:
                    bg = grp.require_group(band_name)
                    bg["wire"] = nz[0].astype(np.int32)
                    bg["coeff_idx"] = nz[1].astype(np.int32)
                    bg["values"] = c[nz].astype(np.float32)
                    bg.attrs["shape"] = c.shape
