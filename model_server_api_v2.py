import os
import torch
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Check available GPUs and select free ones
def get_available_gpus(num_gpus_needed=4):
    """Find available GPUs on the system"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs available: {total_gpus}")
    
    # Try to find free GPUs or use the first available ones
    available_gpus = list(range(min(num_gpus_needed, total_gpus)))
    
    # Set CUDA_VISIBLE_DEVICES to use specific GPUs
    gpu_ids = ",".join(map(str, available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    print(f"Using GPUs: {gpu_ids}")
    return len(available_gpus)

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to your model
MAX_MODEL_LEN = 4096
TENSOR_PARALLEL_SIZE = get_available_gpus(num_gpus_needed=4)

# Initialize vLLM model
print(f"Loading model: {MODEL_NAME}")
print(f"Using tensor parallelism with {TENSOR_PARALLEL_SIZE} GPUs")

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)

# Initialize FastAPI
app = FastAPI(title="vLLM Multi-GPU API Server")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stop: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    finish_reason: str

@app.get("/")
async def root():
    return {
        "message": "vLLM Multi-GPU API Server",
        "model": MODEL_NAME,
        "gpus": TENSOR_PARALLEL_SIZE,
        "endpoints": ["/generate"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text completion using vLLM
    
    Parameters:
    - prompt: Input text prompt
    - max_tokens: Maximum number of tokens to generate
    - temperature: Sampling temperature (0.0 to 2.0)
    - top_p: Nucleus sampling parameter
    - top_k: Top-k sampling parameter
    - stop: List of stop sequences
    """
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop
        )
        
        # Generate
        outputs = llm.generate([request.prompt], sampling_params)
        
        # Extract result
        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            finish_reason=finish_reason
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "gpus": TENSOR_PARALLEL_SIZE
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )