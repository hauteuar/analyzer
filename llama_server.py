#!/usr/bin/env python3
"""
llama.cpp API Server for OpenAI gpt-oss-20b
Uses llama-cpp-python instead of vLLM for serving the model

Requirements:
pip install llama-cpp-python[server] fastapi uvicorn pydantic huggingface-hub

For GPU support:
pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from llama_cpp import Llama, ChatCompletionMessage
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "ggml-org/gpt-oss-20b-GGUF"
MODEL_FILE = "gpt-oss-20b-mxfp4.gguf"  # Pre-quantized MXFP4 version
HOST = "0.0.0.0"
PORT = 8100
MAX_CONTEXT = 16384  # Adjust based on your memory
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512

class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: Optional[int] = Field(MAX_TOKENS, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(TEMPERATURE, description="Sampling temperature (0.0 to 2.0)")
    top_p: Optional[float] = Field(TOP_P, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
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
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_gpus.append(i)
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
            except Exception as e:
                logger.warning(f"GPU {i} not available: {e}")
        
        return available_gpus

class LlamaCppServer:
    """llama.cpp server wrapper"""
    
    def __init__(self, model_name: str, model_file: str, gpu_layers: int = -1):
        self.model_name = model_name
        self.model_file = model_file
        self.gpu_layers = gpu_layers
        self.llama = None
        self.model_path = None
        
    async def download_model(self):
        """Download model from Hugging Face if not present"""
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        self.model_path = models_dir / self.model_file
        
        if not self.model_path.exists():
            logger.info(f"Downloading model {self.model_name}/{self.model_file}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=self.model_file,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False
                )
                self.model_path = Path(downloaded_path)
                logger.info(f"Model downloaded to: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise
        else:
            logger.info(f"Using existing model at: {self.model_path}")
    
    async def initialize(self):
        """Initialize the llama.cpp model"""
        try:
            # Download model if needed
            await self.download_model()
            
            # Determine GPU layers
            available_gpus = GPUManager.get_available_gpus()
            if available_gpus and self.gpu_layers != 0:
                # Use all GPU layers if GPU is available
                n_gpu_layers = self.gpu_layers if self.gpu_layers > 0 else -1
                logger.info(f"Using GPU with {n_gpu_layers} layers offloaded")
            else:
                n_gpu_layers = 0
                logger.info("Using CPU only")
            
            # Initialize llama.cpp
            logger.info(f"Loading model from {self.model_path}...")
            self.llama = Llama(
                model_path=str(self.model_path),
                n_ctx=MAX_CONTEXT,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                n_threads=os.cpu_count() // 2,  # Use half of available threads
                n_batch=512,
                f16_kv=True,  # Use f16 for key/value cache
                use_mmap=True,  # Memory map the model
                use_mlock=False,  # Don't lock memory
            )
            
            logger.info(f"Model {self.model_name} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error("Make sure you have llama-cpp-python installed:")
            logger.error("pip install llama-cpp-python[server]")
            raise
    
    def format_chat_prompt(self, prompt: str, reasoning_effort: str = "medium") -> str:
        """Format prompt with reasoning effort for chat completion"""
        reasoning_instructions = {
            "low": "Provide a quick, concise response.",
            "medium": "Think through this step by step and provide a detailed response.",
            "high": "Carefully analyze this question, consider multiple perspectives, and provide a comprehensive, well-reasoned response with detailed chain-of-thought."
        }
        
        instruction = reasoning_instructions.get(reasoning_effort, reasoning_instructions["medium"])
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. {instruction}"},
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text using llama.cpp"""
        if not self.llama:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        try:
            # Format as chat completion
            messages = self.format_chat_prompt(request.prompt, request.reasoning_effort)
            
            # Prepare stop sequences
            stop_sequences = request.stop or []
            
            # Generate response using chat completion
            response = self.llama.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repetition_penalty,
                stop=stop_sequences,
                stream=False,
            )
            
            # Extract the generated text
            generated_text = response['choices'][0]['message']['content']
            finish_reason = response['choices'][0]['finish_reason']
            
            # Calculate usage
            usage = response.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            
            # Create response
            result = GenerateResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                model=self.model_name,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Global server instance
llamacpp_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llamacpp_server
    
    # Startup
    logger.info("Starting llama.cpp API Server...")
    
    # Get available GPUs
    available_gpus = GPUManager.get_available_gpus()
    gpu_layers = -1 if available_gpus else 0  # Use all GPU layers if available, else CPU
    
    if available_gpus:
        logger.info(f"Found {len(available_gpus)} GPUs, will offload all layers to GPU")
    else:
        logger.warning("No GPUs available, running on CPU only")
    
    # Initialize server
    llamacpp_server = LlamaCppServer(MODEL_NAME, MODEL_FILE, gpu_layers)
    await llamacpp_server.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down llama.cpp API Server...")
    if llamacpp_server and llamacpp_server.llama:
        del llamacpp_server.llama
    llamacpp_server = None

# Create FastAPI app
app = FastAPI(
    title="llama.cpp gpt-oss-20b API Server",
    description="Serves OpenAI's gpt-oss-20b model using llama.cpp",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "llama.cpp gpt-oss-20b API Server",
        "model": MODEL_NAME,
        "model_file": MODEL_FILE,
        "endpoints": {
            "generate": "POST /generate",
            "chat": "POST /v1/chat/completions (OpenAI compatible)",
            "health": "GET /health",
            "info": "GET /info"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    global llamacpp_server
    if llamacpp_server and llamacpp_server.llama:
        return {"status": "healthy", "model": MODEL_NAME}
    else:
        raise HTTPException(status_code=503, detail="Server not ready")

@app.get("/info")
async def info():
    """Get server and model information"""
    available_gpus = GPUManager.get_available_gpus()
    
    return {
        "model": MODEL_NAME,
        "model_file": MODEL_FILE,
        "available_gpus": available_gpus,
        "gpu_layers": -1 if available_gpus else 0,
        "gpu_count": len(available_gpus),
        "max_context": MAX_CONTEXT,
        "backend": "llama.cpp",
        "default_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS
        },
        "reasoning_efforts": ["low", "medium", "high"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model"""
    global llamacpp_server
    
    if not llamacpp_server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        response = await llamacpp_server.generate(request)
        logger.info("Text generation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint"""
    global llamacpp_server
    
    if not llamacpp_server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Extract messages and parameters
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")
        
        # Get the last user message as prompt
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Convert to our request format
        generate_request = GenerateRequest(
            prompt=user_message,
            max_tokens=request.get("max_tokens", MAX_TOKENS),
            temperature=request.get("temperature", TEMPERATURE),
            top_p=request.get("top_p", TOP_P),
            stop=request.get("stop"),
        )
        
        # Generate response
        result = await llamacpp_server.generate(generate_request)
        
        # Format as OpenAI response
        return {
            "id": "chatcmpl-llamacpp",
            "object": "chat.completion", 
            "created": int(asyncio.get_event_loop().time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.generated_text
                },
                "finish_reason": result.finish_reason
            }],
            "usage": result.usage
        }
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

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
    logger.info(f"Starting llama.cpp API Server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_NAME}/{MODEL_FILE}")
    
    # Check for available GPUs
    available_gpus = GPUManager.get_available_gpus()
    if available_gpus:
        logger.info(f"Available GPUs: {available_gpus}")
        logger.info("Will offload all layers to GPU")
    else:
        logger.warning("No GPUs detected, will run on CPU")
    
    # Print memory requirements
    logger.info("Memory requirements:")
    logger.info("- CPU only: ~20GB RAM")
    logger.info("- GPU: 16GB VRAM (all layers)")
    logger.info("- Mixed: 8GB VRAM + 12GB RAM")
    
    # Run server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
        workers=1
    )

if __name__ == "__main__":
    main()