# Qwen3-VL-235B Deployment on NVIDIA GB200 (Blackwell)

This repository provides the complete configuration, scripts, and documentation for deploying the **Qwen3-VL-235B-A22B-Instruct** model on an NVIDIA GB200 (Blackwell) cluster using **vLLM**.

## 🚀 Overview

**Qwen3-VL-235B-A22B-Instruct** is a state-of-the-art Vision-Language Model (VLM) with a Mixture-of-Experts (MoE) architecture. 
- **Total Parameters:** 236B
- **Active Parameters:** 22B
- **Context Length:** 32K (Configured) to 256K (Native)
- **Target Hardware:** NVIDIA GB200 / B200 (Blackwell)

This deployment utilizes **Tensor Parallelism (TP=8)** to distribute the 472GB (BF16) model across 8x B200 GPUs, providing high-throughput inference for complex vision and video reasoning tasks.

## 🛠 Prerequisites

- **Compute:** 1x Node with 8x NVIDIA B200 (180GB HBM3e each).
- **Storage:** ~500GB free space for model weights.
- **Software:** 
  - Python 3.10+
  - Slurm Workload Manager
  - CUDA 12.4+ / Driver 550+

## 📥 Installation & Setup

### 1. Environment Preparation
```bash
python3 -m venv qwen3-vl-env
source qwen3-vl-env/bin/activate
pip install --upgrade pip huggingface_hub vllm==0.19.0
```

### 2. Model Download
Use the provided `download_model.py` script for a stable, resumable download of the 439GB weights.
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
    local_dir="Qwen3-VL-235B",
    local_dir_use_symlinks=False,
    resume_download=True
)
```

### 3. GPU-Optimized Installation
If installing on a cluster, ensure `vLLM` is compiled/installed on a GPU-enabled node to correctly target the Blackwell architecture.
```bash
sbatch install_vllm.slurm
```

## 🛰 Serving the Model

Launch the OpenAI-compatible API server using Slurm. The configuration is optimized for the 1.44TB VRAM available on a GB200 node.

```bash
sbatch serve_qwen3.slurm
```

### Key Configurations:
- `--tensor-parallel-size 8`: Distributes the model across all 8 GPUs.
- `--gpu-memory-utilization 0.95`: Allocates 95% of HBM3e for weights and KV cache.
- `--trust-remote-code`: Required for Qwen3-VL MoE architecture.

## 🤖 Inference Examples

### Using the Python Client (`query_qwen3.py`)
A clean, modular Python example is provided in the repository to demonstrate text and vision capabilities.
```bash
source qwen3-vl-env/bin/activate
python3 query_qwen3.py
```

### Text + Image Query (Vision)
```bash
curl http://<node_ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Qwen3-VL-235B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the contents of this image."},
          {"type": "image_url", "image_url": {"url": "https://example.com/sample.jpg"}}
        ]
      }
    ]
  }'
```

## 📊 Performance Metrics (Blackwell)
- **Download Time:** ~22 minutes (708 MB/s).
- **Model Load Time:** ~15 minutes (TP=8 distribution).
- **VRAM Usage:** ~1.4 TB (95% utilization).
- **Serving Stability:** Configured for 7-day continuous uptime.

## 📝 License
This deployment logic is provided under the MIT License. Model weights are subject to the [Qwen Research License](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/blob/main/LICENSE).
