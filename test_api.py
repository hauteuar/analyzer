import os
import asyncio
from vllm.entrypoints.openai.api_server import run_server
from argparse import Namespace

# Pick GPU(s) here
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # use GPU ID 2 only

async def main():
    args = Namespace(
        # Model args
        model="/models/gemma-3",
        tokenizer=None,
        skip_tokenizer_init=False,
        revision=None,
        code_revision=None,
        tokenizer_revision=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="auto",
        dtype="auto",
        kv_cache_dtype="auto",
        quantization_param_path=None,
        
        # Server args
        host="0.0.0.0",
        port=8000,
        uvicorn_log_level="info",
        allow_credentials=False,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        lora_modules=None,
        prompt_adapters=None,
        chat_template=None,
        response_role="assistant",
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        root_path=None,
        middleware=[],
        return_tokens_as_token_ids=False,
        disable_log_stats=False,
        max_log_len=None,
        
        # Engine args
        served_model_name=None,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_batched_tokens=None,
        max_num_seqs=256,
        max_paddings=256,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=False,
        max_log_len=None,
        seed=0,
        swap_space=4,
        disable_custom_all_reduce=False,
        
        # Additional common args
        block_size=16,
        enable_prefix_caching=False,
        disable_sliding_window=False,
        use_v2_block_manager=False,
        num_lookahead_slots=0,
        enable_chunked_prefill=None,
        speculative_model=None,
        num_speculative_tokens=None,
        speculative_draft_tensor_parallel_size=None,
        speculative_max_model_len=None,
        speculative_disable_by_batch_size=None,
        ngram_prompt_lookup_max=None,
        ngram_prompt_lookup_min=None,
        model_loader_extra_config=None,
        preemption_mode=None,
        
        # Distributed args
        distributed_executor_backend=None,
        otlp_traces_endpoint=None,
    )
    
    await run_server(args)

if __name__ == "__main__":
    asyncio.run(main())