"""Thorough wavelet/level/kappa sweep -> Pareto frontier (compression vs F0),
Stage A intrinsic, VisuShrink-hard (unbiased keep).

Robust: streams EACH combo's result to artifacts/pareto.jsonl (append+flush) as
it is computed, so nothing is ever lost and progress is watchable. Re-running
RESUMES (skips combos already in the jsonl). `python pareto.py analyze` reads the
jsonl and writes the consolidated pareto.json (no recompute).
"""
import os, sys, json, time
import numpy as np, pywt
import common as C

WAVELETS = ['haar', 'db2', 'db4', 'db6', 'db8', 'db10', 'db12',
            'sym4', 'sym6', 'sym8', 'sym12',
            'coif1', 'coif2', 'coif3', 'coif4', 'coif5',
            'bior1.3', 'bior2.2', 'bior2.4', 'bior2.6', 'bior3.5', 'bior4.4', 'bior6.8',
            'rbio2.2', 'rbio4.4', 'dmey']
KAPPAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
N_TEST = 4
SEED = 1
PLANES = ['Y', 'U', 'V']
# STAGE A = intrinsic noise only; STAGE B = coherent noise + HELIX multi-pass removal (production)
STAGE = os.environ.get('STAGE', 'A').upper()
# THRESH_APPROX: VisuShrink-threshold the approximation band too (matches production that
# does NOT keep the full approx; empty/noise wires -> ~0 approx). Default off (helix legacy).
THRESH_APPROX = os.environ.get('THRESH_APPROX', '0') == '1'
_tag = ('' if STAGE == 'A' else f'_{STAGE}') + ('_ta' if THRESH_APPROX else '')
JSONL = f'artifacts/pareto{_tag}.jsonl'
JSON_OUT = f'artifacts/pareto{_tag}.json'

_HELIX_CFG = None
def _helix_remove(noisy_stack):
    """Production helix multi-pass coherent removal (jax), per event."""
    global _HELIX_CFG
    if _HELIX_CFG is None:
        sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
        from helix.core.backend import set_backend; set_backend('jax')
        from helix.tpc.config import DetectorConfig
        _HELIX_CFG = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0,
                                    num_time_steps=C.N_TICKS, temporal_dilation_ticks=11, n_passes=3)
    from helix.tpc.coherent import remove_coherent
    out = np.empty_like(noisy_stack)
    for i in range(noisy_stack.shape[0]):
        out[i] = np.asarray(remove_coherent(noisy_stack[i], _HELIX_CFG, sigma_per_wire=None))
    return out


def sweep_input(pt, clean):
    """Images fed into the wavelet sweep. A: intrinsic-noisy. B: coherent-noisy
    then HELIX multi-pass coherent removal (production front-end)."""
    if STAGE == 'A':
        return C.make_noisy(clean, pt, seed=SEED, coherent=False)
    noisy = C.make_noisy(clean, pt, seed=SEED, coherent=True)
    return _helix_remove(noisy)


def levels_for(w):
    mx = pywt.dwt_max_level(C.N_TICKS, pywt.Wavelet(w).dec_len)
    return list(range(1, mx + 1))                      # full depth range, shallow -> max


def _key(p):
    return (p['plane'], p['w'], p['lv'], p['k'])


def load_points(path=JSONL):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass                            # tolerate a torn last line
    return pts


def _shard_wavelets():
    """SHARD='i/n' -> this process handles wavelets with index % n == i (parallel shards
    append to the same jsonl; atomic line appends + resume dedup make that safe)."""
    spec = os.environ.get('SHARD', '0/1')
    i, n = (int(x) for x in spec.split('/'))
    return [w for j, w in enumerate(WAVELETS) if j % n == i], spec


def sweep():
    my_wavelets, spec = _shard_wavelets()
    done = {_key(p) for p in load_points()}
    print(f"shard {spec}: wavelets {my_wavelets} | resume: {len(done)} combos already done", flush=True)
    fh = open(JSONL, 'a', buffering=1)                 # line-buffered append
    total_combos = sum(len(levels_for(w)) for w in WAVELETS) * len(KAPPAS) * len(PLANES)
    n_done = len(done)
    for pt in PLANES:
        clean = C.load_clean(C.train_test_events()[1][:N_TEST], pt)
        noisy = sweep_input(pt, clean)                 # Stage A: intrinsic | Stage B: coherent+helix-removed
        sig = np.abs(clean) > 0
        for w in my_wavelets:
            for lv in levels_for(w):
                if all((pt, w, lv, k) in done for k in KAPPAS):
                    continue                            # whole (plane,w,lv) already done
                t0 = time.perf_counter()
                cos = [pywt.wavedec(noisy[i], w, mode='periodization', level=lv, axis=-1)
                       for i in range(noisy.shape[0])]
                sgs = [np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745 for co in cos]
                for k in KAPPAS:
                    if (pt, w, lv, k) in done:
                        continue
                    rec = np.empty_like(noisy); nk = nt = 0
                    for i, co in enumerate(cos):
                        b0 = 0 if THRESH_APPROX else 1    # also threshold the approx band?
                        cc = []
                        if not THRESH_APPROX:
                            cc = [co[0]]; nk += co[0].size; nt += co[0].size
                        for b in range(b0, len(co)):
                            t = k * sgs[i] * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
                            bb = np.where(np.abs(co[b]) >= t, co[b], 0.0); cc.append(bb)
                            nk += int(np.count_nonzero(bb)); nt += bb.size
                        rec[i] = pywt.waverec(cc, w, mode='periodization', axis=-1)[:, :C.N_TICKS]
                    f0, nr = C.aggregate(clean, rec)
                    p = dict(plane=pt, w=w, lv=lv, k=k, f0=round(f0, 5),
                             noise_rms=round(nr, 4), bias=round(float((rec - clean)[sig].mean()), 4),
                             compression=round(nt / max(nk, 1), 2), n_kept=nk)
                    fh.write(json.dumps(p) + '\n'); fh.flush()
                    done.add((pt, w, lv, k)); n_done += 1
                print(f"  [{pt}] {w} L{lv:<2d} done ({time.perf_counter()-t0:.1f}s)  "
                      f"{n_done}/{total_combos}", flush=True)
        print(f"[{pt}] plane complete -> {JSONL}", flush=True)
    fh.close()
    if spec.split('/')[1] == '1':                      # only auto-analyze when NOT sharded
        analyze()
    else:
        print(f"shard {spec} done; run `STAGE={STAGE} python pareto.py analyze` after all shards.", flush=True)


def pareto(points, bias_max=None):
    P = [p for p in points if (bias_max is None or abs(p['bias']) <= bias_max)]
    P = sorted(P, key=lambda p: (-p['compression'], -p['f0']))
    front, best = [], -1
    for p in P:
        if p['f0'] > best + 1e-9:
            front.append(p); best = p['f0']
    return sorted(front, key=lambda p: p['compression'])


def select_best(points, f0_tol=0.002, coeff_tol=1.5, bias_max=0.3):
    """User's rule: best F0, fewest coefficients; a DEEPER-level config is fine if
    its coeffs are <= coeff_tol x the leanest best-F0 config (and bias acceptable)."""
    P = [p for p in points if abs(p['bias']) <= bias_max]
    if not P:
        P = points
    f0max = max(p['f0'] for p in P)
    near = [p for p in P if p['f0'] >= f0max - f0_tol]          # best-F0 plateau
    lean = min(near, key=lambda p: p['n_kept'])                 # fewest coefficients
    cap = coeff_tol * lean['n_kept']
    allowed = [p for p in near if p['n_kept'] <= cap]
    deep = max(allowed, key=lambda p: (p['lv'], -p['n_kept']))  # deepest level within tolerance
    return dict(f0max=round(f0max, 4), leanest=lean, recommended=deep)


def analyze():
    pts = load_points()
    res = {}
    for pt in PLANES:
        pp = [p for p in pts if p['plane'] == pt]
        if not pp:
            continue
        fr = pareto(pp)
        sel = select_best(pp)
        res[pt] = dict(n_combos=len(pp), points=pp, pareto=fr, select=sel)
        print(f"\n[{pt}] {len(pp)} combos | best-F0 plateau max={sel['f0max']}")
        ln, rc = sel['leanest'], sel['recommended']
        print(f"   leanest@bestF0 : {ln['w']} L{ln['lv']} k{ln['k']:g} -> "
              f"comp={ln['compression']:.1f}x F0={ln['f0']:.4f} bias={ln['bias']:+.2f} nkept={ln['n_kept']}")
        print(f"   RECOMMENDED    : {rc['w']} L{rc['lv']} k{rc['k']:g} -> "
              f"comp={rc['compression']:.1f}x F0={rc['f0']:.4f} bias={rc['bias']:+.2f} nkept={rc['n_kept']} "
              f"({rc['n_kept']/ln['n_kept']:.2f}x lean coeffs, deeper level)")
        print(f"   Pareto front ({len(fr)} pts):")
        for p in fr:
            print(f"     comp={p['compression']:7.1f}x F0={p['f0']:.4f} nrms={p['noise_rms']:.2f} "
                  f"bias={p['bias']:+.2f}  [{p['w']} L{p['lv']} k{p['k']:g}]")
    json.dump(res, open(JSON_OUT, 'w'))
    print(f'\nsaved {JSON_OUT}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        sweep()
