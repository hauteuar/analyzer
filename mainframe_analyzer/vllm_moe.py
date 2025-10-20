from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import argparse

# Configuration
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # MoE model
MAX_CONTEXT_LENGTH = 32768  # Extended context window
GPU_MEMORY_UTILIZATION = 0.9
TENSOR_PARALLEL_SIZE = 1  # Adjust based on your GPUs

# Initialize FastAPI app
app = FastAPI(title="vLLM MoE API Server")
llm_engine = None

# Request/Response models for /generate endpoint
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    n: int = 1

class GenerateResponse(BaseModel):
    text: List[str]
    prompt: str
    finished: bool

def create_vllm_engine(model_name, max_model_len, tensor_parallel_size):
    """Initialize vLLM engine with MoE model and extended context"""
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="half",  # Use float16 for efficiency
        enforce_eager=False,  # Enable CUDA graph for better performance
    )
    return llm

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Native vLLM /generate endpoint - faster than OpenAI-compatible API"""
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            n=request.n,
        )
        
        # Generate using vLLM engine
        outputs = llm_engine.generate(request.prompt, sampling_params)
        
        # Extract generated text
        generated_texts = [output.outputs[i].text for output in outputs for i in range(len(output.outputs))]
        finished = all(output.outputs[0].finish_reason is not None for output in outputs)
        
        return GenerateResponse(
            text=generated_texts,
            prompt=request.prompt,
            finished=finished
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "max_context_length": MAX_CONTEXT_LENGTH
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "vLLM MoE API Server with native /generate endpoint",
        "model": MODEL_NAME,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "endpoints": ["/generate", "/health"]
    }

def main():
    global llm_engine, MODEL_NAME, MAX_CONTEXT_LENGTH
    
    parser = argparse.ArgumentParser(description="vLLM MoE API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--max-model-len", type=int, default=MAX_CONTEXT_LENGTH, 
                       help="Maximum context length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    MAX_CONTEXT_LENGTH = args.max_model_len
    
    print(f"Starting vLLM server with MoE model: {args.model}")
    print(f"Max context length: {args.max_model_len}")
    print(f"Initializing LLM engine...")
    
    # Initialize the vLLM engine
    llm_engine = create_vllm_engine(
        args.model,
        args.max_model_len,
        args.tensor_parallel_size
    )
    
    print(f"Server will be available at http://{args.host}:{args.port}")
    print(f"Native /generate endpoint: http://{args.host}:{args.port}/generate")
    
    # Start FastAPI server with uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()