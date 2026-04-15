#!/bin/bash
#SBATCH --job-name=install_vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=b200x24

source qwen3-vl-env/bin/activate
pip install --upgrade pip
pip install vllm --no-cache-dir
pip list | grep vllm
