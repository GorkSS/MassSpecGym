#!/bin/bash
# Submit DeepSets + Fourier baselines for MassSpecGym v1 and v1.5.
# Usage: ./submit_v1.5_baselines.sh [v1|v1.5|both]
set -euo pipefail

WHICH="${1:-both}"

train_dir="/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts"
outdir="${train_dir}/submissions"
mkdir -p "${outdir}"
outfile="${outdir}/submissions.csv"

submit_one() {
  local variant="$1"
  local job_key
  job_key=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10 ; echo '')
  local job_id
  job_id=$(sbatch \
    --account="project_465002061" \
    --partition="small-g" \
    --nodes=1 \
    --gpus-per-node=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=7 \
    --mem=64G \
    --time="06:00:00" \
    --output="${outdir}/${job_key}_stdout.txt" \
    --error="${outdir}/${job_key}_errout.txt" \
    --job-name="expmisc002_${variant}" \
    --export="ALL,job_key=${job_key},VARIANT=${variant}" \
    "${train_dir}/run_v1.5_baseline.sh" \
    | awk '{print $4}')
  echo "$(date),${job_id},${job_key},expmisc002_${variant}" >> "${outfile}"
  echo "submitted ${variant} → job_id=${job_id} job_key=${job_key}"
}

case "$WHICH" in
  v1)    submit_one v1 ;;
  v1.5)  submit_one v1.5 ;;
  both)  submit_one v1; submit_one v1.5 ;;
  *)     echo "Usage: $0 [v1|v1.5|both]"; exit 1 ;;
esac
