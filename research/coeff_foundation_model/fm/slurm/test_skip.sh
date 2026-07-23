#!/bin/bash
#SBATCH --job-name=fm_test_skip
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=8 --mem=64G --time=00:20:00
#SBATCH --output=slurm_logs/test_skip_%j.out --error=slurm_logs/test_skip_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 -c "
import sys; sys.path.insert(0,'.')
import data as D
D.init_pipeline_cpu()                      # <-- the piece my first test omitted
missing='../artifacts/fm_cache_tpc/ev_31082.npz'   # now renamed -> FileNotFoundError
corrupt='../artifacts/fm_cache_tpc/ev_31082.npz.corrupt'
good='../artifacts/fm_cache_tpc/ev_31083.npz'
print('baseline good file loads:', D.get_cached(good, device='cpu')['inp'].shape[0], 'tokens')
for name,bad in [('missing(renamed)',missing),('corrupt(truncated)',corrupt)]:
    ds = D.CachedTPC([bad, good])
    try: print(f'{name}: SKIP OK -> {ds[0][\"inp\"].shape[0]} tokens')
    except Exception as e: print(f'{name}: SKIP FAILED -> {type(e).__name__}')
"
