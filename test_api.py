import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

subprocess.run([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/models/gemma-3",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.7",
    "--max-model-len", "4096"
])