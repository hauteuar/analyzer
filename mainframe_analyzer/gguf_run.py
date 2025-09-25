import os
import subprocess
from pathlib import Path

def get_gpu_info():
    """Get GPU memory information"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free',
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                index, name, total, used, free = line.split(', ')
                gpus.append({
                    'index': int(index),
                    'name': name.strip(),
                    'total_mb': int(total),
                    'used_mb': int(used),
                    'free_mb': int(free),
                    'free_ratio': int(free) / int(total)
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def find_best_gpu(gpus, memory_threshold=0.7):
    """Find GPU with most free memory above threshold"""
    suitable = [gpu for gpu in gpus if gpu['free_ratio'] >= memory_threshold]
    return max(suitable, key=lambda x: x['free_mb']) if suitable else None

def print_gpu_status(gpus):
    """Print GPU status"""
    print("\n=== GPU Status ===")
    for gpu in gpus:
        free_gb = gpu['free_mb'] / 1024
        total_gb = gpu['total_mb'] / 1024
        used_gb = gpu['used_mb'] / 1024
        
        status = "✓ Available" if gpu['free_ratio'] >= 0.7 else "✗ Busy"
        
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Memory: {used_gb:.1f}GB/{total_gb:.1f}GB used, {free_gb:.1f}GB free ({gpu['free_ratio']*100:.1f}% free)")
        print(f"  Status: {status}")
        print()

def run_gguf_with_python(model_path, prompt="Hello! How are you today?", **kwargs):
    """Run GGUF model using llama-cpp-python library directly"""
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Get GPU info and select best one
    gpus = get_gpu_info()
    print_gpu_status(gpus)
    
    best_gpu = find_best_gpu(gpus)
    if best_gpu:
        gpu_id = best_gpu['index']
        print(f"✅ Selected GPU {gpu_id}: {best_gpu['name']}")
        print(f"   Available memory: {best_gpu['free_mb']/1024:.1f}GB")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        print("⚠️  No suitable GPU found, using CPU")
        gpu_id = None
    
    # Try to import llama_cpp
    try:
        from llama_cpp import Llama
        print("✅ Found llama_cpp library")
    except ImportError:
        print("❌ llama-cpp-python not found. Installing...")
        try:
            import subprocess
            subprocess.run([
                'pip', 'install', 'llama-cpp-python',
                '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cu121'
            ], check=True)
            from llama_cpp import Llama
            print("✅ llama-cpp-python installed and imported")
        except Exception as e:
            print(f"❌ Failed to install llama-cpp-python: {e}")
            return None
    
    # Estimate memory requirements
    file_size_gb = model_path.stat().st_size / (1024**3)
    print(f"Model file size: {file_size_gb:.1f}GB")
    
    # Model parameters
    model_params = {
        'model_path': str(model_path),
        'n_ctx': kwargs.get('context_length', 2048),
        'verbose': kwargs.get('verbose', True),
        'n_batch': kwargs.get('n_batch', 512),
    }
    
    # GPU settings
    if gpu_id is not None and best_gpu:
        # Use GPU - set layers based on available memory
        available_gb = best_gpu['free_mb'] / 1024
        if available_gb > file_size_gb * 1.5:
            model_params['n_gpu_layers'] = -1  # Use all layers
        else:
            # Use partial GPU layers
            model_params['n_gpu_layers'] = max(1, int(available_gb / file_size_gb * 30))
        
        print(f"Using {model_params['n_gpu_layers']} GPU layers")
    else:
        model_params['n_gpu_layers'] = 0  # CPU only
        print("Using CPU only")
    
    print(f"\n🚀 Loading model: {model_path.name}")
    print("This may take a moment...")
    
    try:
        # Load the model
        llm = Llama(**model_params)
        print("✅ Model loaded successfully!")
        
        # Test generation
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 60)
        
        generation_params = {
            'prompt': prompt,
            'max_tokens': kwargs.get('max_tokens', 200),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'echo': False,
            'stop': kwargs.get('stop', None),
            'stream': kwargs.get('stream', False)
        }
        
        print("🤖 Response:")
        if generation_params['stream']:
            # Streaming output
            for output in llm(**generation_params):
                print(output['choices'][0]['text'], end='', flush=True)
            print()  # New line at end
        else:
            # Single output
            response = llm(**generation_params)
            output_text = response['choices'][0]['text'].strip()
            print(output_text)
        
        print("\n✅ Generation completed!")
        return llm
        
    except Exception as e:
        print(f"❌ Error loading/running model: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Provide helpful suggestions based on error
        if "CUDA" in str(e):
            print("💡 Try: Set n_gpu_layers=0 to use CPU only")
        elif "memory" in str(e).lower():
            print("💡 Try: Use a smaller model or reduce context_length")
        elif "file" in str(e).lower():
            print("💡 Check: Model file path is correct and readable")
        
        return None

def interactive_mode(model_path, **kwargs):
    """Interactive chat mode"""
    llm = run_gguf_with_python(model_path, "Hello", **kwargs)
    
    if not llm:
        print("❌ Could not load model for interactive mode")
        return
    
    print("\n" + "="*60)
    print("🚀 Interactive Chat Mode")
    print("Type your message and press Enter")
    print("Commands: 'quit', 'exit', 'clear', or 'help'")
    print("="*60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared")
                continue
            elif user_input.lower() == 'help':
                print("📋 Commands:")
                print("  quit/exit - End conversation")
                print("  clear - Clear conversation history")
                print("  help - Show this help")
                continue
            elif not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append(f"Human: {user_input}")
            
            # Create context from recent history (last 3 exchanges)
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            context = "\n".join(recent_history) + f"\nAssistant:"
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Generate response
            response = llm(
                prompt=context,
                max_tokens=kwargs.get('max_tokens', 200),
                temperature=kwargs.get('temperature', 0.7),
                stop=kwargs.get('stop', ['\nHuman:', '\n\nHuman:']),
                echo=False
            )
            
            output = response['choices'][0]['text'].strip()
            print(output)
            
            # Add to history
            conversation_history.append(f"Assistant: {output}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")

# Main execution
if __name__ == "__main__":
    # ⚠️  UPDATE THIS PATH TO YOUR GGUF FILE
    MODEL_PATH = "/path/to/your/model.gguf"
    
    print("🚀 GGUF Model Runner")
    print("="*60)
    
    # Example 1: Single prompt
    print("\n📋 Single Prompt Mode:")
    run_gguf_with_python(
        MODEL_PATH,
        prompt="Explain what large language models are in 3 sentences.",
        max_tokens=150,
        temperature=0.7,
        context_length=2048,
        verbose=False
    )
    
    # Example 2: Interactive mode (uncomment to use)
    # print("\n🗣️  Interactive Mode:")
    # interactive_mode(
    #     MODEL_PATH,
    #     max_tokens=200,
    #     temperature=0.8,
    #     context_length=2048
    # )
    
    # Example 3: Streaming mode (uncomment to use)
    # print("\n📡 Streaming Mode:")
    # run_gguf_with_python(
    #     MODEL_PATH,
    #     prompt="Write a short story about AI:",
    #     max_tokens=300,
    #     temperature=0.8,
    #     stream=True
    # )