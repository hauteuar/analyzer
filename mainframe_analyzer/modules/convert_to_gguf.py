import gguf
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

def convert_hf_to_gguf(model_path, output_path):
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "llama")
    
    # Add tokenizer
    print("Adding tokenizer...")
    tokens = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        tokens.append(token.encode('utf-8') if token else b'')
    
    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    
    # Add model metadata
    print("Adding metadata...")
    config = model.config
    gguf_writer.add_name("converted_model")
    gguf_writer.add_description("Model converted from HuggingFace")
    gguf_writer.add_architecture("llama")
    gguf_writer.add_context_length(getattr(config, 'max_position_embeddings', 2048))
    gguf_writer.add_embedding_length(config.hidden_size)
    gguf_writer.add_block_count(config.num_hidden_layers)
    gguf_writer.add_feed_forward_length(config.intermediate_size)
    gguf_writer.add_head_count(config.num_attention_heads)
    
    # Add model weights
    print("Converting model weights...")
    for name, param in model.named_parameters():
        print(f"Processing: {name}")
        # Convert tensor to numpy array
        tensor_data = param.detach().cpu().numpy().astype(np.float16)
        gguf_writer.add_tensor(name, tensor_data)
    
    # Write the file
    print("Writing GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    print(f"Conversion complete! Output saved to: {output_path}")

# Usage example
if __name__ == "__main__":
    model_path = "/path/to/your/model"  # Replace with your model path
    output_path = "converted_model.gguf"  # Output filename
    
    convert_hf_to_gguf(model_path, output_path)