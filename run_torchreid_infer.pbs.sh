#!/bin/bash -l
#PBS -N torchreid_infer
#PBS -l walltime=02:00:00
#PBS -l mem=16gb
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

echo '================================================'
WORKSPACE="${PBS_O_WORKDIR:-$(pwd)}"
echo "Submission directory = ${WORKSPACE}"
echo '================================================'
cd "${WORKSPACE}"

if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

echo "Working directory is now: $(pwd)"

echo '=========='
echo 'Fixed job configuration'
echo '=========='
CONDA_ENV_NAME="torchreid"
WEIGHTS_PATH="osnet_ain_x1_0.pth"
INPUT_PATH="test_data"
OUTPUT_PATH="outputs/test_data_reid_embeddings.pt"
MODEL_NAME="osnet_ain_x1_0"
DEVICE="cuda"
BATCH_SIZE="32"
INPUT_MODE="tracklets"
SAVE_FRAME_EMBEDDINGS="1"
NORMALIZE_EMBEDDINGS="1"
GPU_INDEX="0"

echo '=========='
echo 'Load CUDA modules'
echo '=========='
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
echo "Activating conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

echo '=========='
echo 'Environment diagnostics'
echo '=========='
nvidia-smi || true
which python

python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
EOF

echo '=========='
echo 'Installation reminder'
echo '=========='
echo 'If the environment is not prepared yet, install with:'
echo '  conda create -n torchreid python=3.10 -y'
echo '  conda activate torchreid'
echo '  pip install -r requirements.txt'
echo '  pip install torch torchvision'
echo '  python setup.py develop'

echo '=========='
echo 'Prepare inference arguments'
echo '=========='

if [ ! -s "tools/infer_reid_embeddings.py" ]; then
  echo "ERROR: tools/infer_reid_embeddings.py was not found in $(pwd)."
  exit 1
fi

if [ ! -s "${WEIGHTS_PATH}" ]; then
  echo "ERROR: checkpoint not found at ${WEIGHTS_PATH}"
  echo "Hint: update WEIGHTS_PATH in this script to your downloaded checkpoint."
  exit 1
fi

if [ ! -e "${INPUT_PATH}" ]; then
  echo "ERROR: INPUT_PATH does not exist: ${INPUT_PATH}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "Using GPU index: ${CUDA_VISIBLE_DEVICES}"
echo "Model name: ${MODEL_NAME}"
echo "Checkpoint: ${WEIGHTS_PATH}"
echo "Input path: ${INPUT_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Input mode: ${INPUT_MODE}"
echo "Batch size: ${BATCH_SIZE}"

echo '========================='
echo 'Running Torchreid inference'
echo '========================='
date

EXTRA_ARGS=()
if [ "${SAVE_FRAME_EMBEDDINGS}" = "1" ]; then
  EXTRA_ARGS+=(--save-frame-embeddings)
fi
if [ "${NORMALIZE_EMBEDDINGS}" = "1" ]; then
  EXTRA_ARGS+=(--normalize)
fi

python tools/infer_reid_embeddings.py \
  "${INPUT_PATH}" \
  --weights "${WEIGHTS_PATH}" \
  --output "${OUTPUT_PATH}" \
  --model-name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --input-mode "${INPUT_MODE}" \
  "${EXTRA_ARGS[@]}"

echo '========================='
echo 'Done.'
echo '========================='
date
