# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-03-10

### Added
- `OllamaPool` class for managing multiple Ollama server processes and exposing their URLs.
- `MPSContext` class for starting and stopping the NVIDIA MPS daemon around server pools.
- `parallel_batch_inference` — splits a query list into chunks, one per server, for efficient throughput.
- `parallel_inference` — round-robin fan-out across servers with a concurrency semaphore.
- `num_parallel` parameter on `OllamaPool` / `OllamaServer` to set `OLLAMA_NUM_PARALLEL` per server.
- `polyollama/misc/example_utils.py` with dataset loading and baseline inference helpers.
- Optional dependency groups: `[inference]` (langchain-ollama) and `[all]`.

### Changed
- Separated infrastructure (`pool.py`) from inference (`inference.py`) into distinct layers.
- `parallel.py` rewritten as a backwards-compatible shim over the new API (emits `DeprecationWarning`).
- `__init__.py` updated to export `OllamaPool`, `MPSContext`, `parallel_inference`, and `parallel_batch_inference`.
- Version bumped to `0.2.0`.

### Fixed
- MPS servers must be started inside the `MPSContext` block; starting them before the daemon caused CUDA context conflicts and deadlocks.
- Default `OLLAMA_NUM_PARALLEL=1` was never set explicitly — now enforced via `OllamaServer`.
