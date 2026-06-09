#!/bin/bash
# Run A2 inference on all 1,830 CT studies (train+val+test combined) so we can
# feed CT predictions into the unified Stage-2 Cox using v0.0.7 outcomes.
#
#SBATCH --job-name=a2_ct_all_infer
#SBATCH --output=/home/bensh240/crohn_project/Phase9_V21/logs/a2_ct_all_infer_%j.out
#SBATCH --error=/home/bensh240/crohn_project/Phase9_V21/logs/a2_ct_all_infer_%j.err
#SBATCH --partition=long
#SBATCH --exclude=argus01,argus02,Alsx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "=== a2 CT-all inference on $(hostname) at $(date) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crohn_vlm
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

# zombie-GPU defense (same as nested CV)
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0} 2>/dev/null | head -1 | tr -d " ")
if [[ -n "$USED_MB" && "$USED_MB" -gt 2000 ]]; then
  echo "ZOMBIE_GPU: $USED_MB MiB used by other procs; requeue."
  scontrol requeue $SLURM_JOB_ID 2>&1 || true
  sleep 120; exit 75
fi

CKPT=/synology-data/users/bensh240/Phase9_V21/checkpoints/v21c_a2_film/best_model.pt
SCALER=/synology-data/users/bensh240/Phase9_V21/checkpoints/v21c_a2_film/clinical_scaler.joblib
DATA=/synology-data/users/bensh240/Phase9_V21/data/ct_all.csv
OUT=/synology-data/users/bensh240/Phase9_V21/predictions/a2_ct_all.csv

cd /home/bensh240/crohn_project/Phase9_V21
python -u inference/run_inference_v21.py \
  --ckpt   ${CKPT} \
  --data-csv ${DATA} \
  --modality ct \
  --scaler ${SCALER} \
  --output ${OUT}
echo "=== done at $(date) ==="
wc -l ${OUT}
