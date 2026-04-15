#!/bin/bash
#SBATCH --job-name=qwen3_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
#SBATCH --partition=b200x24
#SBATCH --output=qwen3_serve_%j.log

# Load environment
source qwen3-vl-env/bin/activate

# Get the head node IP
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500

# Model Path
MODEL_PATH="./Qwen3-VL-235B"

# Serving parameters
# --tensor-parallel-size 8: Distribute the 235B model across 8 GPUs
# --trust-remote-code: Required for many VLM implementations
# --max-model-len: Adjust based on context requirements (native is 256k)
# --gpu-memory-utilization: Default is 0.9, can be tuned

vllm serve $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --port 8000
