#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=rdkit_canon_v1.5
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts/fixes/rdkit_canon_v1.5_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts/fixes/rdkit_canon_v1.5_%j.err

set -euo pipefail

module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

cd /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym

python -u scripts/fixes/rdkit_canon_massspecgym.py
