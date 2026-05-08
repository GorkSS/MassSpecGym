#!/bin/bash
# DeepSets + Fourier baseline for MassSpecGym v1.5 validation.
# Submitted via ./submit.sh run_v1.5_baseline.sh 06:00 1 with VARIANT=v1 or VARIANT=v1.5.

set -euo pipefail

echo "job_key \"${job_key}\""
echo "VARIANT \"${VARIANT:-v1.5}\""

cd /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym
module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate

cd scripts

if [[ "${VARIANT:-v1.5}" == "v1" ]]; then
  RUN_NAME="expmisc002_v1_baseline"
  DATASET_PTH="../data/MassSpecGym.tsv"
  CANDIDATES_PTH="../data/MassSpecGym_retrieval_candidates_mass.json"
elif [[ "${VARIANT:-v1.5}" == "v1.5" ]]; then
  RUN_NAME="expmisc002_v1.5_baseline"
  DATASET_PTH="../data/MassSpecGym1.5.tsv"
  CANDIDATES_PTH="../data/MassSpecGym1.5_retrieval_candidates_mass.json"
else
  echo "Unknown VARIANT='${VARIANT}'"
  exit 1
fi

srun --export=ALL --preserve-env python3 run.py \
    --job_key="${job_key}" \
    --run_name="${RUN_NAME}" \
    --wandb_entity_name=roman-bushuiev \
    --task=retrieval \
    --model=deepsets \
    --devices=1 \
    --seed=0 \
    --log_only_loss_at_stages="train,val" \
    --val_check_interval=0.5 \
    --batch_size=16 \
    --lr=0.0003 \
    --hidden_channels=128 \
    --num_layers_per_mlp=4 \
    --dropout=0.1 \
    --max_epochs=50 \
    --dataset_pth="${DATASET_PTH}" \
    --candidates_pth="${CANDIDATES_PTH}"
