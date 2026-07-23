"""Bidirectional spot-check: forward coherent injection (pimm-data) must be
undone by the inverse coherent filter (helix), and the loop must close on
`group_size` alone — independent of the group size, with no 1/sqrt(N) escape.

Run: python research/bidirectional_spotcheck.py
"""
import numpy as np

from pimm_data.noise import coherent_noise, incoherent_noise
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

N_WIRES, N_TICKS = 256, 2701
COH_RMS = 2.5
rng = np.random.default_rng(0)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


print(f"{'group_size':>10} | {'injected coh RMS':>16} | {'residual RMS':>12} "
      f"| {'signal err':>10}")
print("-" * 60)

# A planted "event": a few wires carry large negative-going pulses (real signal
# the filter must PRESERVE while killing the shared coherent waveform).
clean = np.zeros((N_WIRES, N_TICKS), dtype=np.float32)
sig_wires = [10, 11, 12, 130, 131]
for w in sig_wires:
    clean[w, 1000:1040] = -80.0  # bright prompt, well above noise

for g in (32, 64, 128):
    coh = coherent_noise(N_WIRES, N_TICKS, np.random.default_rng(1),
                         group_size=g, rms_adc=COH_RMS)
    inc = incoherent_noise((N_WIRES, N_TICKS), 2.33, np.random.default_rng(2))
    noisy = clean + coh + inc

    cfg = DetectorConfig(group_size=g)
    assert cfg.group_size == g, "forward/inverse group_size must match"
    cleaned = remove_coherent(noisy, cfg)
    cleaned = np.asarray(cleaned)

    # residual coherent on the *noise-only* wires (exclude planted signal)
    mask = np.ones(N_WIRES, bool)
    mask[sig_wires] = False
    resid_coh = rms(cleaned[mask] - inc[mask])      # what coherent survived removal
    sig_err = rms(cleaned[sig_wires] - clean[sig_wires] - inc[sig_wires])

    print(f"{g:>10} | {rms(coh):>16.3f} | {resid_coh:>12.3f} | {sig_err:>10.3f}")

print()
print("Expected: residual coherent RMS << injected (~2.5) and roughly constant")
print("across group_size (no 1/sqrt(N)); signal error stays small (preserved).")
