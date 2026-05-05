# polyollama

Spin up multiple Ollama servers and fan out parallel inference across them — with optional NVIDIA MPS support for true GPU concurrency.

## Performance

Using multiple servers with MPS enabled yields up to **~2.9x throughput** over single-server sequential inference.

## Install

```bash
pip install "polyollama[inference]"   # includes langchain-ollama
pip install "polyollama[all]"         # + datasets, ipywidgets
```

Requires Python ≥ 3.12 and a running [Ollama](https://ollama.com) instance.

## Usage

### Pool only (infrastructure layer)

```python
from polyollama import OllamaPool

with OllamaPool(ports=[11435, 11436, 11437], num_parallel=2) as pool:
    print(pool.urls)   # ["http://127.0.0.1:11435", ...]
```

### Parallel batch inference

```python
import asyncio
from polyollama import OllamaPool
from polyollama.inference import parallel_batch_inference

async def main():
    pool = OllamaPool(ports=[11435, 11436, 11437], num_parallel=2)
    await pool.start_async()
    try:
        results = await parallel_batch_inference(
            urls=pool.urls,
            query_list=[{"question": q} for q in questions],
            prompt="Answer politely: {question}",
            model="gemma2:2b",
            model_kwargs={"temperature": 0},
        )
    finally:
        pool.stop()

asyncio.run(main())
```

### With NVIDIA MPS (best performance)

Servers must be started **inside** the `MPSContext` block so their CUDA contexts connect to the MPS daemon.

```python
from polyollama import OllamaPool, MPSContext
from polyollama.inference import parallel_batch_inference

async def main():
    with MPSContext(gpu_id=0):
        pool = OllamaPool(ports=[11435, 11436, 11437], num_parallel=4)
        await pool.start_async()
        try:
            results = await parallel_batch_inference(
                urls=pool.urls,
                query_list=[{"question": q} for q in questions],
                prompt="Answer politely: {question}",
                model="gemma2:2b",
                model_kwargs={"temperature": 0, "num_ctx": 2048},
            )
        finally:
            pool.stop()
```

> **VRAM note:** with many servers and high `num_parallel`, reduce `num_ctx` to limit KV-cache size (e.g. `num_ctx=2048`). Default is 8k which can OOM at scale.

## Architecture

```
OllamaPool   (pool.py)      — spawns/stops OllamaServer processes, exposes URLs
OllamaServer (server.py)    — wraps a single `ollama serve` process
MPSContext   (mps.py)       — starts/stops the nvidia-cuda-mps-control daemon
parallel_batch_inference     — splits query list into chunks, one chunk per server
parallel_inference           — round-robin fan-out across servers
```

## MPS requirements

- NVIDIA GPU (Compute Capability ≥ 3.5)
- `nvidia-cuda-mps-control` in PATH (comes with the NVIDIA driver)
- Do **not** start servers before entering `MPSContext` — they will bypass the daemon
