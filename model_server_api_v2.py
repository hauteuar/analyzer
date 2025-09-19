#!/usr/bin/env python3
"""
vLLM Multi-GPU API Server for OpenAI gpt-oss-20b
Serves OpenAI's new open-source gpt-oss-20b model using vLLM with automatic GPU detection

Requirements:
pip install vllm>=0.6.0 fastapi uvicorn torch pydantic

Note: gpt-oss-20b requires:
- At least 16GB GPU memory (uses MoE architecture with 3.6B active parameters)
- vLLM version 0.6.0 or later for proper MoE support
- The model uses "harmony" response format - handled automatically by vLLM
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"  # OpenAI's new open-source 20B parameter model (3.6B active)
# Alternative: "openai/gpt-oss-120b" for the larger 120B model (requires 80GB+ GPU)
HOST = "0.0.0.0"
PORT = 8100
MAX_MODEL_LEN = 128000  # gpt-oss supports up to 128k context length
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512

class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: Optional[int] = Field(MAX_TOKENS, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(TEMPERATURE, description="Sampling temperature (0.0 to 2.0)")
    top_p: Optional[float] = Field(TOP_P, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.0, description="Repetition penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Stream response")
    reasoning_effort: Optional[str] = Field("medium", description="Reasoning effort: low, medium, or high")

class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    prompt: str
    model: str
    finish_reason: str
    usage: Dict[str, int]

class GPUManager:
    """Manages GPU detection and allocation"""
    
    @staticmethod
    def get_available_gpus() -> List[int]:
        """Get list of available GPU IDs"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Running on CPU.")
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for i in range(gpu_count):
            try:
                # Test GPU availability
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_gpus.append(i)
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
            except Exception as e:
                logger.warning(f"GPU {i} not available: {e}")
        
        return available_gpus

class VLLMServer:
    """vLLM server wrapper"""
    
    def __init__(self, model_name: str, gpu_id: Optional[int] = None):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.engine = None
        
    async def initialize(self):
        """Initialize the vLLM engine"""
        try:
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=1,  # Use single GPU
                gpu_memory_utilization=0.9,
                max_model_len=MAX_MODEL_LEN,
                dtype="float16" if self.gpu_id is not None else "float32",
                trust_remote_code=True,
                enforce_eager=False,
            )
            
            # Set GPU device if available
            if self.gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                logger.info(f"Using GPU: {self.gpu_id}")
            else:
                logger.info("Using CPU")
            
            # Initialize async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"Model {self.model_name} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text using the model"""
        if not self.engine:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop,
            )
            
            # Generate text
            results = []
            async for result in self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=None
            ):
                results.append(result)
            
            if not results:
                raise HTTPException(status_code=500, detail="No results generated")
            
            # Get the final result
            final_result = results[-1]
            output = final_result.outputs[0]
            
            # Create response
            response = GenerateResponse(
                generated_text=output.text,
                prompt=request.prompt,
                model=self.model_name,
                finish_reason=output.finish_reason,
                usage={
                    "prompt_tokens": len(final_result.prompt_token_ids),
                    "completion_tokens": len(output.token_ids),
                    "total_tokens": len(final_result.prompt_token_ids) + len(output.token_ids)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Global server instance
vllm_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global vllm_server
    
    # Startup
    logger.info("Starting vLLM API Server...")
    
    # Get available GPUs and select the first one
    available_gpus = GPUManager.get_available_gpus()
    selected_gpu = available_gpus[0] if available_gpus else None
    
    if selected_gpu is not None:
        logger.info(f"Selected GPU {selected_gpu} out of available GPUs: {available_gpus}")
    else:
        logger.warning("No GPUs available, running on CPU")
    
    # Initialize server
    vllm_server = VLLMServer(MODEL_NAME, selected_gpu)
    await vllm_server.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down vLLM API Server...")
    vllm_server = None

# Create FastAPI app
app = FastAPI(
    title="vLLM Multi-GPU API Server",
    description="Serves open-source GPT models using vLLM with GPU support",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "vLLM API Server",
        "model": MODEL_NAME,
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health",
            "info": "GET /info"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    global vllm_server
    if vllm_server and vllm_server.engine:
        return {"status": "healthy", "model": MODEL_NAME}
    else:
        raise HTTPException(status_code=503, detail="Server not ready")

@app.get("/info")
async def info():
    """Get server and model information"""
    available_gpus = GPUManager.get_available_gpus()
    selected_gpu = available_gpus[0] if available_gpus else None
    
    return {
        "model": MODEL_NAME,
        "available_gpus": available_gpus,
        "selected_gpu": selected_gpu,
        "gpu_count": len(available_gpus),
        "max_model_len": MAX_MODEL_LEN,
        "default_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model"""
    global vllm_server
    
    if not vllm_server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        response = await vllm_server.generate(request)
        logger.info("Text generation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

def main():
    """Main function to run the server"""
    logger.info(f"Starting vLLM API Server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_NAME}")
    
    # Check for available GPUs
    available_gpus = GPUManager.get_available_gpus()
    if available_gpus:
        logger.info(f"Available GPUs: {available_gpus}")
        logger.info(f"Will use GPU: {available_gpus[0]}")
    else:
        logger.warning("No GPUs detected, will run on CPU")
    
    # Run server
    uvicorn.run(
        "vllm_api_server:app",  # Change this to your actual filename
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
        workers=1
    )

if __name__ == "__main__":
    main()