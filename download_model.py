from huggingface_hub import snapshot_download
import os

repo_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
local_dir = "Qwen3-VL-235B"

print(f"Starting download of {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8
)
print("Download complete.")
