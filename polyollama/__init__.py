from .server import OllamaServer
from .pool import OllamaPool
from .inference import parallel_inference, parallel_batch_inference
from .mps import MPSContext


__all__ = [
    # Infrastructure
    "OllamaServer",
    "OllamaPool",
    # Inference utilities
    "parallel_inference",
    "parallel_batch_inference",
    # GPU
    "MPSContext",
]


__version__ = "0.2.0"
