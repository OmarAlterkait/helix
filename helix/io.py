"""HDF5 I/O for JAXTPC production sensor files and processed output.

Readout geometry is extracted from the file itself — no external config needed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from helix.config import DetectorConfig
from helix.wavelet import SparseResult


def config_from_file(path: str | Path, **overrides) -> DetectorConfig:
    """Build a DetectorConfig by reading geometry from a sensor HDF5 file.

    Extracts plane labels, num_time_steps, and pedestals from the file
    attributes. Algorithm parameters use defaults unless overridden.

    Parameters
    ----------
    path : str or Path
        Path to a JAXTPC production sensor file.
    **overrides
        Any DetectorConfig field to override (e.g. beta=0.20).
    """
    with h5py.File(path, "r") as f:
        evt_keys = sorted(k for k in f.keys() if k.startswith("event_"))
        if not evt_keys:
            raise ValueError(f"No events found in {path}")
        evt = f[evt_keys[0]]

        plane_labels = tuple(k for k in evt.keys() if isinstance(evt[k], h5py.Group))

        num_time_steps = int(evt.attrs.get("num_time_steps", 2701))

        pedestals = {}
        for label in plane_labels:
            grp = evt[label]
            pt = label.split("_")[-1] if "_" in label else label
            pedestals[pt] = int(grp.attrs.get("pedestal", 0))

    kwargs: dict[str, Any] = dict(
        plane_labels=plane_labels,
        num_time_steps=num_time_steps,
        pedestals=pedestals,
    )
    kwargs.update(overrides)
    return DetectorConfig(**kwargs)


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
        available = set(evt.keys())

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
