#!/bin/bash
#SBATCH --job-name=v21c_a2_film
#SBATCH --output=/home/bensh240/crohn_project/Phase9_V21/logs/v21c_a2_film_%j.out
#SBATCH --error=/home/bensh240/crohn_project/Phase9_V21/logs/v21c_a2_film_%j.err
#SBATCH --partition=long
#SBATCH --nodelist=argus03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=14-00:00:00

echo "=== A2: V21-C token + FiLM (no DSBN) ==="
echo "Start: $(date) | Node: $(hostname) | Job: ${SLURM_JOB_ID}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crohn_vlm
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}   # node-local: avoid GLIBC mismatch across argus03/04
# let SLURM assign the GPU (do not override CVD)
export OMP_NUM_THREADS=4
echo "SLURM-assigned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then echo "ERROR: no HF token (DINOv3 gated)"; exit 1; fi
cd /home/bensh240/crohn_project/Phase9_V21

python -u training/train_mil_v21.py --arch v21c --conditioning film  --modality both \
    --warm-start /synology-data/users/bensh240/Phase8_DINOv3_V20/checkpoints/v20_dinov3/best_model.pt \
    --run-name v21c_a2_film

echo "Exit: $? | End: $(date)"
