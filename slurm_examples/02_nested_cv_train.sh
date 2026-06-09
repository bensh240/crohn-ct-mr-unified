#!/bin/bash
# Nested CV: 5 patient-level outer folds x 3 conditioning variants (A1=token, A2=film, A3=dsbn).
# Array index k = fold * 3 + variant_id, k = 0..14.
# Throttle to 4 concurrent on argus03 (4 GPUs).
#
# Expected wall: ~12 h per run; 4 concurrent -> ~4 waves of 12 h = ~48 h.
# Submit: sbatch /home/bensh240/crohn_project/Phase9_V21/training/submit_nested_cv.sh
#
#SBATCH --job-name=v21_nestedcv
#SBATCH --output=/home/bensh240/crohn_project/Phase9_V21/logs/nestedcv_%A_%a.out
#SBATCH --error=/home/bensh240/crohn_project/Phase9_V21/logs/nestedcv_%A_%a.err
#SBATCH --partition=long
#SBATCH --exclude=argus01,argus02,Alsx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-14%4
#SBATCH --requeue

set -euo pipefail
echo "=== nested-CV task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crohn_vlm
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=4
echo "SLURM-assigned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Defensive zombie-GPU check: refuse to run if the assigned GPU is already
# being used by another process beyond a small idle threshold (zombie processes
# from other users' crashed jobs are a known issue on this cluster, esp. on
# argus04 GPU 0). Explicitly scontrol-requeue and exit 75 -> SLURM puts the
# task back in the queue and the throttle will (eventually) land it on a
# clean GPU. We cap retries via a marker file to avoid an infinite loop.
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0} 2>/dev/null | head -1 | tr -d " ")
if [[ -n "$USED_MB" && "$USED_MB" -gt 2000 ]]; then
  # use shared FS so the counter survives requeues across argus03/argus04
  mkdir -p /synology-data/users/bensh240/Phase9_V21/state 2>/dev/null
  RETRY_FILE=/synology-data/users/bensh240/Phase9_V21/state/nestedcv_retry_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
  TRIES=$(cat ${RETRY_FILE} 2>/dev/null || echo 0)
  TRIES=$((TRIES + 1))
  echo ${TRIES} > ${RETRY_FILE}
  # cap = 200 attempts, ~5 min apart -> up to ~16 h waiting per task.
  # Cluster may be saturated for hours; this gives the scheduler time to
  # free a clean GPU before we truly fail.
  if [[ ${TRIES} -le 200 ]]; then
    echo "ZOMBIE_GPU on $(hostname):${CUDA_VISIBLE_DEVICES} (${USED_MB} MiB used by other procs); requeue attempt ${TRIES}/200"
    scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} 2>&1 || true
    sleep 300   # 5 min: avoid hot-loop and give other queued tasks a chance
    exit 75
  else
    echo "ZOMBIE_GPU: ${TRIES} retries exhausted; failing for human review."
    exit 99
  fi
fi
echo "GPU is free (${USED_MB:-unknown} MiB used by other procs). Proceeding."
if [ -z "${HF_TOKEN:-}" ] && [ ! -f ~/.cache/huggingface/token ]; then
  echo "ERROR: no HF token (DINOv3 is gated)"; exit 1
fi

K=${SLURM_ARRAY_TASK_ID}
FOLD=$(( K / 3 ))
VAR_ID=$(( K % 3 ))
case ${VAR_ID} in
  0) CONDITIONING=token ; VARIANT=a1 ;;
  1) CONDITIONING=film  ; VARIANT=a2 ;;
  2) CONDITIONING=dsbn  ; VARIANT=a3 ;;
esac
DATA_DIR=/synology-data/users/bensh240/Phase9_V21/data/cv5/fold_${FOLD}
RUN_NAME=nestedcv/fold_${FOLD}/${VARIANT}
WARM=/synology-data/users/bensh240/Phase8_DINOv3_V20/checkpoints/v20_dinov3/best_model.pt

echo "FOLD=${FOLD}  VARIANT=${VARIANT}  CONDITIONING=${CONDITIONING}"
echo "DATA_DIR=${DATA_DIR}"
echo "RUN_NAME=${RUN_NAME}"
if [ ! -f "${DATA_DIR}/train.csv" ] || [ ! -f "${DATA_DIR}/val.csv" ]; then
  echo "ERROR: missing splits at ${DATA_DIR}"; exit 1
fi

cd /home/bensh240/crohn_project/Phase9_V21
python -u training/train_mil_v21.py \
  --arch v21c --conditioning ${CONDITIONING} --modality both \
  --warm-start ${WARM} \
  --data-dir ${DATA_DIR} \
  --run-name ${RUN_NAME}

echo "=== done task ${SLURM_ARRAY_TASK_ID} at $(date) ==="
