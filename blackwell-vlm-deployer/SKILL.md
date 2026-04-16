---
name: blackwell-vlm-deployer
description: Deployment, optimization, and serving of Vision-Language Models (VLMs) and Large Language Models (LLMs) on NVIDIA Blackwell GB200/B200 hardware. Use when Gemini CLI needs to assess hardware capacity, estimate VRAM needs, configure multi-GPU serving (vLLM/TP), or setup interactive web interfaces for large models like Qwen3-VL.
---

# Blackwell VLM Deployer

## Overview

This skill enables rapid and efficient deployment of large-scale models (200B+ parameters) on NVIDIA Blackwell (B200/GB200) clusters. It focuses on maximizing performance through optimized VRAM management, tensor parallelism, and high-throughput serving.

## Workflow Decision Tree

1. **Hardware Assessment**: Confirm GPU availability and VRAM capacity (e.g., 8x B200 180GB).
2. **VRAM Estimation**: Use `estimate_vram.py` to determine if the model fits in BF16 or requires quantization.
3. **Deployment Strategy**:
   - For **Latency**: Focus on INT4 quantization and optimized TP=8.
   - For **Throughput**: Focus on continuous batching and aggressive KV cache allocation.
4. **Implementation**: Generate Slurm scripts for weight download, vLLM serving, and Web UI setup.

## Core Capabilities

### 1. VRAM Estimation
Use the `scripts/estimate_vram.py` utility to calculate memory requirements.
```bash
python3 scripts/estimate_vram.py --params 235 --precision bf16
```

### 2. Optimized vLLM Serving
Configure `vllm serve` with Blackwell-specific parameters:
- `--tensor-parallel-size 8`: Standard for 8-GPU nodes.
- `--gpu-memory-utilization 0.95`: Maximizes HBM3e usage.
- `--trust-remote-code`: Required for Qwen3-VL MoE.

### 3. Interactive Web Interface
A Flask-based UI template is available in `assets/web_app/`. It supports:
- Real-time chat via OpenAI-compatible API.
- Local image uploads for vision analysis.
- Markdown rendering for formatted model responses.

## Guidelines & Best Practices

- **Latency Optimization**: See [latency-guidelines.md](references/latency-guidelines.md) for minimizing single-request response time.
- **Throughput Maximization**: See [throughput-guidelines.md](references/throughput-guidelines.md) for handling high concurrent request volumes.

## Resources

- **scripts/estimate_vram.py**: Precision-aware VRAM calculator.
- **references/latency-guidelines.md**: Best practices for low-latency serving.
- **references/throughput-guidelines.md**: Best practices for high-throughput serving.
- **assets/web_app/**: Boilerplate code for a chat/vision web interface.
