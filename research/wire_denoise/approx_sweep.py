"""Approx-band thresholding study. Fixed wavelet = bior4.4 L8 (our best). Detail
bands: VisuShrink-hard, sweep detail kappa. Approx band: sweep DIFFERENT thresholding
structures (the approx holds the pulse low-freq + the colored-noise low-freq, so a
detail-tuned rule is likely wrong for it). Coherent + helix removal (production).

Approx strategies:
  keep           : keep full approx (helix legacy)
  visu_fine       : Ka * sigma_finest * sqrt(2 ln Na)        (uses finest-detail noise)
  visu_selfglob   : Ka * sigma_approx * sqrt(2 ln Na)        (global approx-band noise, robust)
  abs             : |approx| >= T  (absolute ADC)
  topk            : keep top fraction per wire
  wire_gated      : keep approx ONLY on wires that have >=1 surviving detail coeff (else 0)
Reports F0, n_kept, bias, noise per combo (streamed)."""
import os, sys, json, time
import numpy as np, pywt
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import common as C
from helix.core.backend import set_backend; set_backend('jax')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

CFG = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=C.N_TICKS,
                     temporal_dilation_ticks=11, n_passes=3)
WAV = {'Y': ('bior4.4', 8), 'U': ('bior4.4', 8), 'V': ('bior4.4', 8)}
DKAPPA = [1.0, 1.5, 2.0, 2.5]
N_TEST = 4
SEED = 1
JSONL = 'artifacts/approx_sweep.jsonl'


def hard(x, t):
    return np.where(np.abs(x) >= t, x, 0.0)


def approx_variants(approx, sg_fine, Na, sig_wire):
    """List of (strategy, param, thresholded_approx)."""
    sga = float(np.median(np.abs(approx)) / 0.6745)          # robust global approx-band noise sigma
    lf = np.sqrt(2 * np.log(max(Na, 2)))
    out = [('keep', -1, approx.copy())]
    for Ka in [1, 2, 3, 4, 6]:
        out.append(('visu_fine', Ka, hard(approx, Ka * sg_fine * lf)))
    for Ka in [1, 2, 3, 4, 6]:
        out.append(('visu_selfglob', Ka, hard(approx, Ka * sga * lf)))
    for T in [2, 5, 10, 20, 40]:
        out.append(('abs', T, hard(approx, T)))
    for f in [0.3, 0.1, 0.05, 0.02, 0.01]:
        n = approx.shape[1]; kk = max(1, int(f * n))
        thr = np.partition(np.abs(approx), n - kk, axis=1)[:, n - kk][:, None]
        out.append(('topk', f, np.where(np.abs(approx) >= thr, approx, 0.0)))
    g = approx.copy(); g[~sig_wire] = 0.0
    out.append(('wire_gated', -1, g))
    return out, sga


def helix_remove(noisy):
    return np.stack([np.asarray(remove_coherent(noisy[i], CFG, sigma_per_wire=None))
                     for i in range(noisy.shape[0])])


def run():
    fh = open(JSONL, 'w', buffering=1)
    for pt in ['Y', 'U', 'V']:
        w, lv = WAV[pt]
        clean = C.load_clean(C.train_test_events(n_train=24, n_test=24)[1][4:4 + N_TEST], pt)
        removed = helix_remove(C.make_noisy(clean, pt, seed=SEED, coherent=True))
        sig = np.abs(clean) > 0
        cos = [pywt.wavedec(removed[i], w, mode='periodization', level=lv, axis=-1) for i in range(N_TEST)]
        sgf = [np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745 for co in cos]
        Na = cos[0][0].shape[-1]
        ratio = float(np.median([np.median(np.abs(co[0])) / 0.6745 for co in cos]) /
                      np.median([s.mean() for s in sgf]))
        print(f"[{pt}] {w} L{lv}  Na={Na}/wire  sigma_approx/sigma_fine ~ {ratio:.1f}", flush=True)
        for dk in DKAPPA:
            # threshold detail bands once per (event, dk); record survivor wire mask
            det = []; det_nk = np.zeros(N_TEST, int); sigwire = []
            for i, co in enumerate(cos):
                bands = []; surv = np.zeros(clean.shape[1], bool)
                for b in range(1, len(co)):
                    t = dk * sgf[i] * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
                    bb = hard(co[b], t); bands.append(bb)
                    det_nk[i] += int(np.count_nonzero(bb)); surv |= (np.count_nonzero(bb, axis=1) > 0)
                det.append(bands); sigwire.append(surv)
            for i, co in enumerate(cos):
                variants, sga = approx_variants(co[0], sgf[i], Na, sigwire[i])
                if i == 0:
                    allrec = {key: [] for key in [(s, p) for s, p, _ in variants]}
                    allnk = {key: 0 for key in allrec}
                for s, p, ath in variants:
                    rec = pywt.waverec([ath] + det[i], w, mode='periodization', axis=-1)[:, :C.N_TICKS]
                    allrec[(s, p)].append(rec)
                    allnk[(s, p)] += int(np.count_nonzero(ath)) + int(det_nk[i])
            for (s, p), recs in allrec.items():
                R = np.stack(recs); f0, nr = C.aggregate(clean, R)
                rec_pt = dict(plane=pt, w=w, lv=lv, dkappa=dk, approx=s, aparam=p,
                              f0=round(f0, 5), noise_rms=round(nr, 4),
                              bias=round(float((R - clean)[sig].mean()), 4),
                              n_kept=int(allnk[(s, p)]),
                              compression=round(N_TEST * clean.shape[1] * clean.shape[2] / max(allnk[(s, p)], 1), 2))
                fh.write(json.dumps(rec_pt) + '\n'); fh.flush()
            print(f"  [{pt}] dk={dk} done", flush=True)
    fh.close()
    analyze()


def analyze():
    pts = [json.loads(l) for l in open(JSONL) if l.strip()]
    for pt in ['Y', 'U', 'V']:
        P = [p for p in pts if p['plane'] == pt]
        if not P:
            continue
        keep = [p for p in P if p['approx'] == 'keep']
        # F0 reference = best keep-approx F0 across detail-kappa
        f0ref = max(p['f0'] for p in keep)
        print(f"\n[{pt}] keep-approx best F0={f0ref:.4f}; coeffs at that point="
              f"{min(p['n_kept'] for p in keep if p['f0']>=f0ref-1e-9):,}")
        # for each approx strategy, the leanest combo with F0 >= f0ref-0.002 and |bias|<0.15
        for s in ['keep', 'visu_fine', 'visu_selfglob', 'abs', 'topk', 'wire_gated']:
            cand = [p for p in P if p['approx'] == s and p['f0'] >= f0ref - 0.002 and abs(p['bias']) < 0.15]
            if not cand:
                print(f"   {s:14s}: (no combo within F0 tol & bias)"); continue
            b = min(cand, key=lambda p: p['n_kept'])
            print(f"   {s:14s}: n_kept={b['n_kept']:>8,} comp={b['compression']:6.1f}x F0={b['f0']:.4f} "
                  f"bias={b['bias']:+.3f} (dk={b['dkappa']}, aparam={b['aparam']})")


if __name__ == '__main__':
    run() if not (len(sys.argv) > 1 and sys.argv[1] == 'analyze') else analyze()
