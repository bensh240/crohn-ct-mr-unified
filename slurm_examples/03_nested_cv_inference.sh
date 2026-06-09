#!/bin/bash
# Run inference for each (fold, variant) after the training array completes.
# 15 tasks (5 folds * 3 variants), 1 GPU each, fast (~minutes).
# Submit with: sbatch --dependency=afterok:<training_jobid> submit_nestedcv_infer.sh
#
#SBATCH --job-name=v21_ncv_infer
#SBATCH --output=/home/bensh240/crohn_project/Phase9_V21/logs/ncv_infer_%A_%a.out
#SBATCH --error=/home/bensh240/crohn_project/Phase9_V21/logs/ncv_infer_%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crohn_vlm
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

K=${SLURM_ARRAY_TASK_ID}
FOLD=$(( K / 3 ))
VAR_ID=$(( K % 3 ))
case ${VAR_ID} in
  0) VARIANT=a1 ;;
  1) VARIANT=a2 ;;
  2) VARIANT=a3 ;;
esac
CKPT=/synology-data/users/bensh240/Phase9_V21/checkpoints/nestedcv/fold_${FOLD}/${VARIANT}/best_model.pt
SCALER=/synology-data/users/bensh240/Phase9_V21/checkpoints/nestedcv/fold_${FOLD}/${VARIANT}/clinical_scaler.joblib
TEST_CSV=/synology-data/users/bensh240/Phase9_V21/data/cv5/fold_${FOLD}/test.csv
OUT_DIR=/synology-data/users/bensh240/Phase9_V21/predictions/nestedcv/fold_${FOLD}
mkdir -p ${OUT_DIR}

cd /home/bensh240/crohn_project/Phase9_V21
for MOD in mr ct; do
  python -u inference/run_inference_v21.py \
    --ckpt ${CKPT} \
    --data-csv ${TEST_CSV} \
    --modality ${MOD} \
    --scaler ${SCALER} \
    --output ${OUT_DIR}/${VARIANT}_${MOD}_test.csv
done
echo "=== inference fold ${FOLD} ${VARIANT} done at $(date) ==="
