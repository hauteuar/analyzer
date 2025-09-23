import os
import asyncio
from vllm.entrypoints.openai.api_server import run_server
from argparse import Namespace

# Pick GPU(s) here
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # use GPU ID 2 only

async def main():
    # Create args as a Namespace object with all required parameters
    args = Namespace(
        model="/models/gemma-3",
        host="0.0.0.0",
        port=8000,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        # Add other common defaults
        uvicorn_log_level="info",
        allow_credentials=False,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        served_model_name=None,
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
    )
    
    await run_server(args)

if __name__ == "__main__":
    asyncio.run(main())