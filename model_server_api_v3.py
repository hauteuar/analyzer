import os
import torch
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
import base64
from io import BytesIO

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

# Configuration - Choose your model
# Gemma 3 models (MULTIMODAL - text + images):
MODEL_NAME = "google/gemma-3-12b-it"  # 12B instruction-tuned (recommended)
# MODEL_NAME = "google/gemma-3-4b-it"   # 4B instruction-tuned (smaller, faster)
# MODEL_NAME = "google/gemma-3-27b-it"  # 27B instruction-tuned (best quality)

# Other multimodal models:
# MODEL_NAME = "google/paligemma-3b-mix-448"
# MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
# MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Text-only models:
# MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "google/gemma-3-1b-it"  # Text-only 1B variant

MAX_MODEL_LEN = 4096
TENSOR_PARALLEL_SIZE = get_available_gpus(num_gpus_needed=4)

# Check if model is multimodal
MULTIMODAL_MODELS = [
    "paligemma", "llava", "qwen2-vl", "phi-3-vision", 
    "internvl", "cogvlm", "fuyu", "blip"
]
IS_MULTIMODAL = any(model_type in MODEL_NAME.lower() for model_type in MULTIMODAL_MODELS)

# Initialize vLLM model
print(f"Loading model: {MODEL_NAME}")
print(f"Model type: {'Multimodal' if IS_MULTIMODAL else 'Text-only'}")
print(f"Using tensor parallelism with {TENSOR_PARALLEL_SIZE} GPUs")

llm_kwargs = {
    "model": MODEL_NAME,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "max_model_len": MAX_MODEL_LEN,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": True
}

# Add multimodal-specific parameters
if IS_MULTIMODAL:
    llm_kwargs["max_num_seqs"] = 5  # Reduce batch size for multimodal
    llm_kwargs["limit_mm_per_prompt"] = {"image": 10}  # Max images per prompt

llm = LLM(**llm_kwargs)

# Initialize FastAPI
app = FastAPI(title="vLLM Multi-GPU Multimodal API Server")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stop: Optional[List[str]] = None
    images: Optional[List[str]] = None  # Base64 encoded images or URLs

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    finish_reason: str
    model_type: str

@app.get("/")
async def root():
    return {
        "message": "vLLM Multi-GPU Multimodal API Server",
        "model": MODEL_NAME,
        "model_type": "multimodal" if IS_MULTIMODAL else "text-only",
        "gpus": TENSOR_PARALLEL_SIZE,
        "endpoints": ["/generate", "/generate_with_image"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text completion using vLLM (supports both text and multimodal)
    
    Parameters:
    - prompt: Input text prompt
    - max_tokens: Maximum number of tokens to generate
    - temperature: Sampling temperature (0.0 to 2.0)
    - top_p: Nucleus sampling parameter
    - top_k: Top-k sampling parameter
    - stop: List of stop sequences
    - images: List of base64 encoded images or image URLs (for multimodal models)
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
        
        # Prepare inputs based on model type
        if IS_MULTIMODAL and request.images:
            # For multimodal models with images
            # vLLM expects images in the prompt as special tokens or URLs
            inputs = {
                "prompt": request.prompt,
                "multi_modal_data": {
                    "image": request.images  # Can be URLs or base64
                }
            }
            outputs = llm.generate(inputs, sampling_params)
        else:
            # For text-only generation
            outputs = llm.generate([request.prompt], sampling_params)
        
        # Extract result
        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            finish_reason=finish_reason,
            model_type="multimodal" if IS_MULTIMODAL else "text-only"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_with_image")
async def generate_with_image(
    prompt: str,
    image: UploadFile = File(...),
    max_tokens: int = 512,
    temperature: float = 0.7
):
    """
    Generate text from image and prompt (multimodal only)
    Upload image file directly
    """
    if not IS_MULTIMODAL:
        raise HTTPException(
            status_code=400, 
            detail="This endpoint requires a multimodal model"
        )
    
    try:
        # Read and encode image
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Generate with image
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [image_url]
            }
        }
        
        outputs = llm.generate(inputs, sampling_params)
        output = outputs[0]
        
        return {
            "generated_text": output.outputs[0].text,
            "prompt": prompt,
            "finish_reason": output.outputs[0].finish_reason,
            "model_type": "multimodal"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_type": "multimodal" if IS_MULTIMODAL else "text-only",
        "gpus": TENSOR_PARALLEL_SIZE
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": MODEL_NAME,
        "is_multimodal": IS_MULTIMODAL,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "supported_features": {
            "text_generation": True,
            "image_input": IS_MULTIMODAL,
            "batch_processing": True
        }
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )