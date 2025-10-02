import os
import subprocess
import json
import time
from pathlib import Path

class GPUManager:
    def __init__(self, memory_threshold=0.7):
        """
        Initialize GPU manager
        memory_threshold: Use GPU if it has more than this fraction of memory free
        """
        self.memory_threshold = memory_threshold
        self.available_gpus = self.get_gpu_info()
    
    def get_gpu_info(self):
        """Get GPU memory information"""
        try:
            # Run nvidia-smi to get GPU info
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
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error getting GPU info: {e}")
            print("Make sure nvidia-smi is available and you have NVIDIA GPUs")
            return []
    
    def find_best_gpu(self, required_memory_gb=None):
        """
        Find the best GPU based on free memory
        required_memory_gb: Minimum memory required in GB
        """
        if not self.available_gpus:
            return None
        
        # Filter GPUs that meet memory threshold
        suitable_gpus = []
        for gpu in self.available_gpus:
            free_gb = gpu['free_mb'] / 1024
            
            # Check if GPU has enough free memory
            if gpu['free_ratio'] >= self.memory_threshold:
                if required_memory_gb is None or free_gb >= required_memory_gb:
                    suitable_gpus.append(gpu)
        
        if not suitable_gpus:
            return None
        
        # Return GPU with most free memory
        return max(suitable_gpus, key=lambda x: x['free_mb'])
    
    def print_gpu_status(self):
        """Print current GPU status"""
        print("\n=== GPU Status ===")
        for gpu in self.available_gpus:
            free_gb = gpu['free_mb'] / 1024
            total_gb = gpu['total_mb'] / 1024
            used_gb = gpu['used_mb'] / 1024
            
            status = "✓ Available" if gpu['free_ratio'] >= self.memory_threshold else "✗ Busy"
            
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  Memory: {used_gb:.1f}GB/{total_gb:.1f}GB used, {free_gb:.1f}GB free ({gpu['free_ratio']*100:.1f}% free)")
            print(f"  Status: {status}")
            print()

class GGUFRunner:
    def __init__(self, model_path, gpu_manager=None):
        self.model_path = Path(model_path)
        self.gpu_manager = gpu_manager or GPUManager()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def estimate_model_memory(self):
        """Estimate memory requirements based on model file size"""
        file_size_gb = self.model_path.stat().st_size / (1024**3)
        # Rule of thumb: model needs ~1.2x its file size in GPU memory
        estimated_memory = file_size_gb * 1.2
        return estimated_memory
    
    def run_with_llama_cpp_python(self, gpu_id, **kwargs):
        """Run model using llama-cpp-python library"""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("llama-cpp-python not found. Install with: pip install llama-cpp-python")
            return None
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"Loading model on GPU {gpu_id}...")
        
        # Default parameters - you can modify these
        model_params = {
            'model_path': str(self.model_path),
            'n_gpu_layers': -1,  # Use all GPU layers
            'n_ctx': kwargs.get('context_length', 2048),
            'verbose': False
        }
        
        try:
            llm = Llama(**model_params)
            print(f"✓ Model loaded successfully on GPU {gpu_id}")
            return llm
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return None
    
    def run_with_llama_cpp_binary(self, gpu_id, prompt="Hello, how are you?", **kwargs):
        """Run model using llama.cpp binary (if available)"""
        
        # Try to find llama-cli or main binary
        possible_binaries = ['llama-cli', './llama-cli', './main', 'main']
        llama_binary = None
        
        for binary in possible_binaries:
            try:
                subprocess.run([binary, '--help'], capture_output=True, check=True)
                llama_binary = binary
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if not llama_binary:
            print("llama.cpp binary not found. Please build llama.cpp or install llama-cpp-python")
            return None
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [
            llama_binary,
            '-m', str(self.model_path),
            '-p', prompt,
            '-ngl', '-1',  # Use all GPU layers
            '-n', str(kwargs.get('max_tokens', 100)),
            '--temp', str(kwargs.get('temperature', 0.7))
        ]
        
        print(f"Running model on GPU {gpu_id}...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            print("Output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return result
        except subprocess.TimeoutExpired:
            print("Model execution timed out")
            return None
        except Exception as e:
            print(f"Error running model: {e}")
            return None
    
    def run(self, prompt="Hello, how are you?", method="auto", **kwargs):
        """
        Main method to run the model
        method: 'auto', 'python', or 'binary'
        """
        # Show GPU status
        self.gpu_manager.print_gpu_status()
        
        # Estimate memory requirements
        estimated_memory = self.estimate_model_memory()
        print(f"Estimated model memory requirement: {estimated_memory:.1f}GB")
        
        # Find best GPU
        best_gpu = self.gpu_manager.find_best_gpu(estimated_memory)
        
        if not best_gpu:
            print("❌ No suitable GPU found!")
            print(f"Need GPU with >70% free memory and >{estimated_memory:.1f}GB available")
            return None
        
        print(f"✅ Selected GPU {best_gpu['index']}: {best_gpu['name']}")
        print(f"   Available memory: {best_gpu['free_mb']/1024:.1f}GB")
        
        # Choose method
        if method == "auto":
            try:
                import llama_cpp
                method = "python"
            except ImportError:
                method = "binary"
        
        # Run the model
        if method == "python":
            return self.run_with_llama_cpp_python(best_gpu['index'], **kwargs)
        else:
            return self.run_with_llama_cpp_binary(best_gpu['index'], prompt, **kwargs)

# Example usage
def main():
    # Configuration
    MODEL_PATH = "path/to/your/model.gguf"  # Replace with your GGUF file path
    PROMPT = "Explain quantum computing in simple terms."
    
    try:
        # Create runner
        runner = GGUFRunner(MODEL_PATH)
        
        # Run the model
        result = runner.run(
            prompt=PROMPT,
            method="auto",  # or "python" or "binary"
            max_tokens=200,
            temperature=0.7,
            context_length=2048
        )
        
        if result:
            print("\n✅ Model execution completed successfully!")
            
            # If using Python method, you can continue the conversation
            if hasattr(result, '__call__'):  # It's a Llama object
                print("\nYou can now chat with the model:")
                while True:
                    user_input = input("\nYou: ")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    response = result(user_input, max_tokens=200)
                    print(f"Assistant: {response['choices'][0]['text']}")
        else:
            print("❌ Failed to run model")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    