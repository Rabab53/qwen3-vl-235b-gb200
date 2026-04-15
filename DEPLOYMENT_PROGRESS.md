# Technical Deployment Report: Qwen3-VL-235B-A22B-Instruct

This report serves as the definitive documentation for the deployment of the Qwen3-VL-235B model on the NVIDIA GB200 (Blackwell) cluster.

## 1. System Architecture

### Hardware Configuration (Compute Node: x24b200v4)
- **GPUs:** 8x NVIDIA B200 (Blackwell Architecture).
- **VRAM:** 180 GB HBM3e per GPU (Total: 1.44 TB).
- **Interconnect:** NVLink (High-bandwidth communication between all 8 GPUs).
- **System Memory:** 3.9 TB RAM.

### Software Stack
- **OS:** Ubuntu/Debian based Linux.
- **Runtime:** Python 3.10 Virtual Environment (`qwen3-vl-env`).
- **Inference Engine:** vLLM v0.19.0 (optimized for Blackwell).
- **Model Storage:** 439 GB (Safetensors) located in `/home/ralomairy_tahakom_com/Qwen3-VL-235B`.

---

## 2. Implementation Details

### A. Environment Creation
The environment was isolated to prevent dependency conflicts with system-level packages.
```bash
python3 -m venv qwen3-vl-env
source qwen3-vl-env/bin/activate
pip install --upgrade pip huggingface_hub
```

### B. Model Download Strategy (`download_model.py`)
Standard CLI tools were bypassed in favor of a Python script to handle the massive 439GB payload with better stability and resumption support.
```python
from huggingface_hub import snapshot_download
repo_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
local_dir = "Qwen3-VL-235B"
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8
)
```

### C. GPU-Accelerated Installation (`install_vllm.slurm`)
vLLM was installed on a compute node to ensure proper detection of the Blackwell architecture and CUDA 12.x drivers.
```bash
#!/bin/bash
#SBATCH --partition=b200x24 --gres=gpu:1 --time=01:00:00
source qwen3-vl-env/bin/activate
pip install vllm==0.19.0 --no-cache-dir
```

### D. Serving Configuration (`serve_qwen3.slurm`)
The model is served across all 8 GPUs using **Tensor Parallelism**.
```bash
#!/bin/bash
#SBATCH --job-name=qwen3_serve
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --partition=b200x24
#SBATCH --output=qwen3_serve_%j.log

source qwen3-vl-env/bin/activate

vllm serve ./Qwen3-VL-235B \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --port 8000
```
**Rationale:**
- `--tensor-parallel-size 8`: Required to fit the 472GB (BF16) model. Weights are split across 8 GPUs.
- `--gpu-memory-utilization 0.95`: Maximizes the use of 1.44TB VRAM for high-throughput serving and long context.

---

## 3. Operations & Access

### Accessing the API
The server runs on the compute node (e.g., `x24b200v4-nodes3-0`). To access it from the login node or your local machine:

**From Login Node:**
```bash
curl http://x24b200v4-nodes3-0:8000/v1/models
```

**From Local Machine (SSH Tunnel):**
```bash
ssh -L 8000:x24b200v4-nodes3-0:8000 <user>@<login_node_ip>
```

### Sample Inference (Vision Task)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Qwen3-VL-235B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail."},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

---

## 4. Current Deployment Status
- **Model Weights:** Verified (439 GB).
- **Environment:** Ready (vLLM 0.19.0).
- **Server Status:** **RESTARTING** (Job ID: 1118 - **Extended to 1 Week**).
- **Endpoint:** `http://x24b200v4-nodes3-0:8000/v1`
- **VRAM Usage:** 1.4 TB / 1.44 TB (Optimized for context/throughput).

## 5. Deployment Log Summary
- **Download Time:** 22 minutes (708 MB/s).
- **Environment Setup:** 5 minutes.
- **Model Loading:** ~15 minutes (Weight distribution + CUDA warm-up).
- **Total Deployment Time:** < 1 hour.
