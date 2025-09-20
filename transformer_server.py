#!/usr/bin/env python3
"""
Air-gapped Transformers API Server for OpenAI gpt-oss-20b
Pure transformers implementation for offline deployment with multi-GPU support

Requirements (install offline):
pip install torch transformers accelerate fastapi uvicorn pydantic

Model Path: ./models/gpt-oss-20b/
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "./models/gpt-oss-20b"  # Local model directory
HOST = "0.0.0.0"
PORT = 8100
MAX_CONTEXT = 32768  # gpt-oss supports up to 128k, but start conservative
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512

class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: Optional[int] = Field(MAX_TOKENS, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(TEMPERATURE, description="Sampling temperature (0.0 to 2.0)")
    top_p: Optional[float] = Field(TOP_P, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.05, description="Repetition penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    reasoning_effort: Optional[str] = Field("medium", description="Reasoning effort: low, medium, or high")

class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    prompt: str
    model: str
    finish_reason: str
    usage: Dict[str, int]

class GPUManager:
    """Manages GPU detection and memory optimization"""
    
    @staticmethod
    def get_gpu_info():
        """Get detailed GPU information"""
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}
        
        gpu_count = torch.cuda.device_count()
        devices = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)  # GB
            
            devices.append({
                "id": i,
                "name": props.name,
                "memory_total_gb": round(memory_total, 1),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "count": gpu_count,
            "devices": devices
        }
    
    @staticmethod
    def setup_device_map(gpu_count: int):
        """Setup optimal device mapping for multi-GPU"""
        if gpu_count == 0:
            return "cpu"
        elif gpu_count == 1:
            return "cuda:0"
        else:
            # Auto-balance across all GPUs
            return "auto"

class TransformersServer:
    """Pure transformers server for gpt-oss-20b"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.gpu_info = None
        
    def validate_model_path(self):
        """Validate that model files exist"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        required_files = ["config.json", "pytorch_model.bin"]  # Minimum required
        missing_files = []
        
        for file in required_files:
            if not (self.model_path / file).exists():
                # Check for alternative formats
                alternatives = {
                    "pytorch_model.bin": ["model.safetensors", "pytorch_model-00001-of-*.bin"]
                }
                
                if file in alternatives:
                    found = False
                    for alt_pattern in alternatives[file]:
                        if list(self.model_path.glob(alt_pattern)):
                            found = True
                            break
                    if not found:
                        missing_files.append(file)
                else:
                    missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        logger.info(f"Model validation passed: {self.model_path}")
    
    async def initialize(self):
        """Initialize the model and tokenizer"""
        try:
            # Validate model files
            self.validate_model_path()
            
            # Get GPU information
            self.gpu_info = GPUManager.get_gpu_info()
            logger.info(f"GPU Info: {self.gpu_info}")
            
            # Setup device configuration
            if self.gpu_info["available"]:
                device_map = GPUManager.setup_device_map(self.gpu_info["count"])
                torch_dtype = torch.float16
                logger.info(f"Using device_map: {device_map}")
                
                # Calculate total GPU memory
                total_gpu_memory = sum(dev["memory_total_gb"] for dev in self.gpu_info["devices"])
                logger.info(f"Total GPU Memory: {total_gpu_memory:.1f} GB")
                
                if total_gpu_memory < 20:
                    logger.warning("Low GPU memory detected. Model may not fit entirely on GPU.")
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                logger.warning("No GPU detected. Using CPU (will be slow)")
            
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,  # Force offline mode
                use_fast=True,
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Load model with optimal settings
            logger.info("Loading model... (this may take several minutes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=True,  # Force offline mode
                low_cpu_mem_usage=True,
                # Optimization settings
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                use_cache=True,
            )
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model device map: {getattr(self.model, 'hf_device_map', 'Not available')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error("Common issues:")
            logger.error("1. Model not downloaded to correct path")
            logger.error("2. Insufficient GPU memory")
            logger.error("3. Missing dependencies: pip install accelerate")
            raise
    
    def format_chat_prompt(self, prompt: str, reasoning_effort: str = "medium") -> str:
        """Format prompt with reasoning effort and proper chat template"""
        reasoning_instructions = {
            "low": "Provide a quick, direct response without extensive reasoning.",
            "medium": "Think through this step by step and provide a clear, well-reasoned response.",
            "high": "Analyze this question carefully. Consider multiple perspectives, think through each step of your reasoning, and provide a comprehensive response with detailed explanation of your thought process."
        }
        
        instruction = reasoning_instructions.get(reasoning_effort, reasoning_instructions["medium"])
        
        # Try to use model's chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": f"You are a helpful assistant. {instruction}"},
                {"role": "user", "content": prompt}
            ]
            
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted
            except Exception:
                logger.warning("Chat template failed, using fallback format")
        
        # Fallback format
        return f"System: You are a helpful assistant. {instruction}\n\nUser: {prompt}\n\nAssistant:"
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text using the model"""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        try:
            # Format prompt
            formatted_prompt = self.format_chat_prompt(request.prompt, request.reasoning_effort)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT - request.max_tokens,
                padding=False
            )
            
            # Move to appropriate device
            if torch.cuda.is_available() and self.gpu_info["available"]:
                # For multi-GPU, the model handles device placement
                if isinstance(self.model.device, torch.device):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # Prepare generation arguments
            generation_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Add stop sequences if provided
            if request.stop:
                stop_token_ids = []
                for stop_seq in request.stop:
                    stop_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                    stop_token_ids.extend(stop_ids)
                if stop_token_ids:
                    generation_kwargs["eos_token_id"] = stop_token_ids
            
            # Generate text
            logger.info(f"Generating {request.max_tokens} tokens...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Determine finish reason
            finish_reason = "stop"
            if len(generated_tokens) >= request.max_tokens:
                finish_reason = "length"
            
            # Calculate token usage
            completion_tokens = len(generated_tokens)
            prompt_tokens = input_length
            total_tokens = prompt_tokens + completion_tokens
            
            response = GenerateResponse(
                generated_text=generated_text.strip(),
                prompt=request.prompt,
                model="gpt-oss-20b",
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            logger.info(f"Generated {completion_tokens} tokens successfully")
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            raise HTTPException(
                status_code=507, 
                detail="GPU out of memory. Try reducing max_tokens or context length."
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Global server instance
transformers_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global transformers_server
    
    # Startup
    logger.info("Starting Transformers API Server for gpt-oss-20b...")
    
    # Check if model path exists
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Please download the model first using the download script")
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    
    # Initialize server
    transformers_server = TransformersServer(MODEL_PATH)
    await transformers_server.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Transformers API Server...")
    if transformers_server and transformers_server.model:
        del transformers_server.model
        del transformers_server.tokenizer
        torch.cuda.empty_cache()
    transformers_server = None

# Create FastAPI app
app = FastAPI(
    title="Air-gapped gpt-oss-20b Server",
    description="Pure transformers implementation for offline gpt-oss-20b deployment",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "Air-gapped gpt-oss-20b API Server",
        "model": "gpt-oss-20b",
        "backend": "transformers",
        "model_path": MODEL_PATH,
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
    global transformers_server
    if transformers_server and transformers_server.model:
        return {"status": "healthy", "model": "gpt-oss-20b", "backend": "transformers"}
    else:
        raise HTTPException(status_code=503, detail="Server not ready")

@app.get("/info")
async def info():
    """Get server and model information"""
    global transformers_server
    
    info_data = {
        "model": "gpt-oss-20b",
        "model_path": MODEL_PATH,
        "backend": "transformers",
        "max_context": MAX_CONTEXT,
        "default_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS
        },
        "reasoning_efforts": ["low", "medium", "high"]
    }
    
    if transformers_server:
        info_data["gpu_info"] = transformers_server.gpu_info
    
    return info_data

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model"""
    global transformers_server
    
    if not transformers_server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:100]}...")
        response = await transformers_server.generate(request)
        logger.info("Text generation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint"""
    global transformers_server
    
    if not transformers_server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Extract parameters
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Convert to generate request
        generate_request = GenerateRequest(
            prompt=user_message,
            max_tokens=request.get("max_tokens", MAX_TOKENS),
            temperature=request.get("temperature", TEMPERATURE),
            top_p=request.get("top_p", TOP_P),
            stop=request.get("stop"),
        )
        
        # Generate response
        result = await transformers_server.generate(generate_request)
        
        # Format as OpenAI response
        return {
            "id": "chatcmpl-transformers",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": "gpt-oss-20b",
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
    logger.info(f"Starting Air-gapped Transformers API Server on {HOST}:{PORT}")
    logger.info(f"Model Path: {MODEL_PATH}")
    
    # Check model existence
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Run the download script first!")
        return
    
    # Display GPU info
    gpu_info = GPUManager.get_gpu_info()
    if gpu_info["available"]:
        logger.info(f"Found {gpu_info['count']} GPUs:")
        for device in gpu_info["devices"]:
            logger.info(f"  GPU {device['id']}: {device['name']} ({device['memory_total_gb']} GB)")
    else:
        logger.warning("No GPUs detected - will use CPU (very slow)")
    
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