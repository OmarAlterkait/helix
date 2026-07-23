"""Is coherent structure actually removed, or is the residual just incoherent
grain (which remove_coherent is NOT meant to touch)?

Inject coherent and incoherent SEPARATELY on a real plane, run remove_coherent,
and measure the *coherent content* of each stage = RMS of the per-group
cross-wire mean (over off-signal wires). Coherent noise is shared within a
64-wire group so it survives that mean; incoherent is suppressed ~1/sqrt(64).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
from pimm_data.noise import coherent_noise, incoherent_noise, digitize
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
import _tpc_common as C

GS = 64
label = "volume_0_Y"
clean, nt, wl, ped = C.load_clean(0, label)
nw = clean.shape[0]
rng = np.random.default_rng(1000)
coh = coherent_noise(nw, nt, rng, group_size=GS, rms_adc=2.5)
inc = incoherent_noise((nw, nt), wl, rng)
noisy = digitize(clean + coh + inc, ped)

sigw = DetectorConfig().wire_sigma_intrinsic(wl)        # true per-wire sigma (production passes this)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def coherent_content(x, sig_wire):
    """RMS of per-group cross-wire mean over OFF-signal wires (coherent survives,
    incoherent ~1/sqrt(group)). x: (nw,nt)."""
    vals = []
    for g0 in range(0, nw, GS):
        sl = slice(g0, min(g0 + GS, nw))
        keep = ~sig_wire[sl]
        if keep.sum() >= 8:
            vals.append(x[sl][keep].mean(0))           # (nt,) shared estimate
    return rms(np.stack(vals)) if vals else float("nan")


sig_wire = (np.abs(clean) > 0).any(1)
cfg = DetectorConfig(group_size=GS, num_time_steps=nt, plane_labels=(label,))
cleaned_est = np.asarray(remove_coherent(noisy, cfg))               # sigma MAD-estimated
cleaned_tru = np.asarray(remove_coherent(noisy, cfg, sigma_per_wire=sigw))  # production sigma

print(f"plane {label}: nw={nw} nt={nt}, {sig_wire.sum()} signal wires\n")
print(f"{'stage':<34}{'full RMS':>10}{'coherent content':>18}")
print("-" * 62)
print(f"{'injected coherent (coh)':<34}{rms(coh):>10.3f}{coherent_content(coh, sig_wire):>18.3f}")
print(f"{'injected incoherent (inc)':<34}{rms(inc):>10.3f}{coherent_content(inc, sig_wire):>18.3f}")
print(f"{'noisy noise (coh+inc, digitized)':<34}{rms(noisy-clean):>10.3f}{coherent_content(noisy-clean, sig_wire):>18.3f}")
print(f"{'after remove (sigma=MAD)':<34}{rms(cleaned_est-clean):>10.3f}{coherent_content(cleaned_est-clean, sig_wire):>18.3f}")
print(f"{'after remove (sigma=per-wire)':<34}{rms(cleaned_tru-clean):>10.3f}{coherent_content(cleaned_tru-clean, sig_wire):>18.3f}")
print(f"\nincoherent-only floor (no coherent injected, for reference):")
nz = digitize(clean + inc, ped)
cz = np.asarray(remove_coherent(nz, cfg, sigma_per_wire=sigw))
print(f"{'after remove, inc-only':<34}{rms(cz-clean):>10.3f}{coherent_content(cz-clean, sig_wire):>18.3f}")
