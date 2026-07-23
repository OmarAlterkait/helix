"""Loader for the doraemon optical sensor files (label_N schema).

Data: /sdf/data/neutrino/doraemon/optical_test_00_00_02/sensor/ — 100 files x
200 events, NOISE-FREE (baseline_noise_std=0; noise to be added at load in a
later phase), 164 global channels (pmt_id 0-163), SER 5 us, pedestal 29490.3.

Schema per event: event_NNN/label_K/{adc, offsets, pmt_id, t0_ns, pe_counts,
tpc_de, tpc_labels, tpc_n_photons, tpc_pdg, tpc_positions, tpc_t_step}.

SEMANTICS (verified 2026-06-11): chunks are PER-INTERACTION waveform
contributions — the same PMT can hold time-OVERLAPPING chunks under different
labels (285 overlapping same-pmt pairs in file0/event_001). This is NOT the
summed detector readout; for the operator-discrimination phase each chunk is
treated as an independent clean signal. pe_counts is per-(label, channel), so
chunk-level truth = pe_counts[label][pmt_id] (chunks of one channel within one
label share it).

helix.optical.io reads the east/west schema and cannot read these files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import h5py

DATA_DIR = "/sdf/data/neutrino/doraemon/optical_test_00_00_02/sensor"
PEDESTAL = 29490.3  # /config pedestal (same as light_output)


def list_files(data_dir: str = DATA_DIR) -> list[str]:
    return sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")
    )


def list_events(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("event_"))


def read_config(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return dict(f["config"].attrs)


@dataclass
class DoraemonEventChunks:
    """Pedestal-subtracted per-interaction chunks of one event, all labels.

    chunks  : list of 1-D float32 (variable length), clean signal (no noise)
    pmt_id  : (n,) global channel 0..163
    t0_ns   : (n,) chunk start time
    label   : (n,) interaction-label index (int, parsed from 'label_K')
    pe      : (n,) true PE count of (label, channel) — chunk-level truth
    lengths : (n,)
    """
    chunks: list
    pmt_id: np.ndarray
    t0_ns: np.ndarray
    label: np.ndarray
    pe: np.ndarray
    lengths: np.ndarray


def read_event_chunks(path: str, event_key: str,
                      pedestal: float = PEDESTAL) -> DoraemonEventChunks:
    chunks, pmts, t0s, labs, pes, lens = [], [], [], [], [], []
    with h5py.File(path, "r") as f:
        evt = f[event_key]
        for lk in sorted(evt.keys()):
            if not lk.startswith("label_"):
                continue
            lab = int(lk.split("_", 1)[1])
            g = evt[lk]
            adc = g["adc"][:].astype(np.float32) - pedestal
            offsets = g["offsets"][:]
            pmt_id = g["pmt_id"][:]
            t0_ns = g["t0_ns"][:]
            pe_counts = g["pe_counts"][:]
            for k in range(len(pmt_id)):
                c = adc[offsets[k]:offsets[k + 1]]
                chunks.append(c)
                pmts.append(int(pmt_id[k]))
                t0s.append(float(t0_ns[k]))
                labs.append(lab)
                pes.append(int(pe_counts[int(pmt_id[k])]))
                lens.append(len(c))
    return DoraemonEventChunks(
        chunks=chunks,
        pmt_id=np.asarray(pmts, np.int16),
        t0_ns=np.asarray(t0s, np.float64),
        label=np.asarray(labs, np.int16),
        pe=np.asarray(pes, np.int32),
        lengths=np.asarray(lens, np.int32),
    )


def iter_events(n_events: int, data_dir: str = DATA_DIR,
                per_file: int | None = None):
    """Yield (path, event_key) spread across files to avoid single-file bias.

    per_file: events taken from each file (default: ceil(n_events / n_files)).
    """
    files = list_files(data_dir)
    if per_file is None:
        per_file = max(1, -(-n_events // len(files)))
    done = 0
    for path in files:
        for ek in list_events(path)[:per_file]:
            yield path, ek
            done += 1
            if done >= n_events:
                return
