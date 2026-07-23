#!/bin/bash
#SBATCH --job-name=fm_babysit20k
#SBATCH --partition=milano --account=mli:nu-ml-dev --qos=preemptable
#SBATCH --cpus-per-task=1 --mem=2G --time=4-00:00:00 --signal=B:TERM@120
#SBATCH --output=slurm_logs/babysit_job_%j.out --error=slurm_logs/babysit_job_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
# self-perpetuating: on preemption/timeout, spawn a successor (unless the run is complete)
resub(){ [ -f ckpt_dscale600b4_20k.pt.done ] || sbatch slurm/babysit_job.sh; exit 0; }
trap resub TERM
ACCT=default INTERVAL=600 ./slurm/train.sh babysit configs/dscale600b4_20k.yaml &
wait $!
