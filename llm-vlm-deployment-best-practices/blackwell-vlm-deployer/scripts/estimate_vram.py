import sys
import argparse

def estimate_vram(params_billions, precision_bytes, kv_cache_percent=0.20):
    """
    Estimates VRAM requirements for LLM/VLM inference.
    """
    # Base weight size
    weight_vram_gb = params_billions * precision_bytes
    
    # Total with overhead (KV Cache, activations, etc.)
    total_vram_gb = weight_vram_gb * (1 + kv_cache_percent)
    
    return weight_vram_gb, total_vram_gb

def main():
    parser = argparse.ArgumentParser(description="Estimate VRAM for LLM/VLM Deployment")
    parser.add_argument("--params", type=float, required=True, help="Total parameters in billions (e.g., 235, 397)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "int8", "int4"], help="Model precision")
    parser.add_argument("--kv_overhead", type=float, default=0.20, help="Estimated overhead for KV Cache and activations (default 0.20)")

    args = parser.parse_args()

    precision_map = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }

    bytes_per_param = precision_map[args.precision]
    weights, total = estimate_vram(args.params, bytes_per_param, args.kv_overhead)

    print(f"\n--- VRAM Estimation Results ---")
    print(f"Model Parameters: {args.params}B")
    print(f"Precision: {args.precision.upper()} ({bytes_per_param} bytes/param)")
    print(f"Weights VRAM: {weights:.2f} GB")
    print(f"Estimated Total VRAM (with {args.kv_overhead*100:.0f}% overhead): {total:.2f} GB")
    print(f"-------------------------------\n")

if __name__ == "__main__":
    main()
