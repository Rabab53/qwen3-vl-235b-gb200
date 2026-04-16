# Guidelines for Maximizing Model Throughput

When deploying models for applications requiring high request volumes, such as serving many users simultaneously, optimizing for **throughput** is paramount. Throughput refers to the number of requests a model can process per unit of time. This requires a different focus than low-latency, single-request optimization.

## Batching Strategies: The Core of Throughput

*   **Continuous Batching:**
    *   **Purpose:** Dynamically groups incoming requests together to keep the GPUs maximally utilized. This avoids wasting computation time on requests that finish early. vLLM's PagedAttention and continuous batching are highly effective for this.
    *   **Impact:** Significantly increases throughput compared to static batching, especially for variable request lengths.

*   **Max Batch Size:**
    *   **Definition:** The maximum number of sequences that can be processed together in a single forward pass.
    *   **Tuning:** Experimenting with the `max_num_batched_tokens` (in vLLM) or similar parameters can help find the optimal balance between batch size and VRAM usage for your specific hardware. A larger batch size generally leads to higher throughput, up to the VRAM limit.

## Quantization: The Throughput Multiplier

*   **Benefit for Throughput:** Quantization (especially to INT8 or INT4) is a primary strategy for boosting throughput.
    *   **Reduced VRAM:** Lower VRAM requirements mean more requests can fit into the KV cache and memory, allowing for larger batch sizes.
    *   **Faster Computation:** Operations on lower-precision data (INT8/INT4) are often faster on modern hardware, reducing computation time per request.

## Model Precision and Throughput

*   **FP16/BF16:** Offers the best accuracy but requires more VRAM and can be slower than quantized models, potentially limiting batch size and thus throughput.
*   **INT8 / INT4:** Significantly boosts throughput by reducing VRAM footprint and computation time. This is often the preferred choice when maximizing requests per second is the goal.

## Hardware Allocation and Parallelism

*   **Tensor Parallelism (TP):**
    *   **Role:** Essential for fitting very large models into GPU memory and distributing computation.
    *   **Throughput Impact:** A well-configured TP strategy ensures all GPUs are utilized, but excessive TP can introduce communication overhead that might limit scaling.
*   **Data Parallelism (DP) / Pipeline Parallelism (PP):**
    *   For extreme throughput needs on multi-node clusters, DP (running multiple copies of the model across nodes) or PP (splitting the model layers across nodes) can be employed. This is more complex and usually reserved for very high-demand scenarios.

## Inference Framework Choice

*   **vLLM:** Highly recommended for throughput optimization due to its advanced features like PagedAttention, continuous batching, and optimized CUDA kernels, specifically designed for LLM serving.
*   **Other Frameworks:** Frameworks like Hugging Face `transformers` (with optimizations) or specialized inference servers (like NVIDIA Triton) can also be used, but vLLM often leads in throughput for conversational LLMs.

## System Resources

*   **CPU:** While GPUs do the heavy lifting, a strong CPU is needed for managing requests, pre-processing data, and handling the overhead of batching and scheduling.
*   **RAM:** Sufficient system RAM is crucial for loading model weights initially, managing data, and supporting multiple processes.
*   **Network I/O:** For API endpoints, fast network speeds are essential to ingest requests and return responses quickly.

---

### Applying to Qwen3.5-397B-A17B for High Throughput:

*   **Recommendation:** For maximum throughput, especially with interactive chat, **4-bit quantization** is highly recommended. This would drastically reduce VRAM needs (~200GB for weights), leaving substantial room (~1.2 TB) for a very large KV cache and enabling massive batch sizes for high request concurrency.
*   **vLLM Configuration:** Use `tensor_parallel_size=8` to leverage all GPUs. Tune `max_num_batched_tokens` to the highest value that fits within VRAM after accounting for weights and essential overheads. Experiment with `gpu_memory_utilization` to maximize memory usage for batching.

---
This guide focuses on maximizing the number of requests processed over time.
