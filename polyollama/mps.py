"""
NVIDIA Multi-Process Service (MPS) management for polyollama.

MPS replaces CUDA's default time-sliced process scheduling with a single
daemon that submits CUDA kernels from all client processes *concurrently*
onto the same GPU.  On a single high-end GPU (e.g. RTX 5090) this allows
multiple Ollama servers to share Streaming Multiprocessors in parallel
instead of waiting for each other.

Requirements
------------
- NVIDIA GPU (Compute Capability >= 3.5)
- ``nvidia-cuda-mps-control`` in PATH  (comes with the NVIDIA driver)
- Exclusive or Default compute mode on the GPU
  (check: ``nvidia-smi -q | grep "Compute Mode"``)

Quickstart
----------
    from polyollama.mps import MPSContext
    from polyollama import parallel_distributed_inference

    with MPSContext(gpu_id=0) as mps:
        results = await parallel_distributed_inference(
            model="gemma2:2b",
            questions=questions,
            extra_ports=[11435, 11436, 11437],
        )
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

_DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"
_DEFAULT_LOG_DIR  = "/tmp/nvidia-mps-log"


class MPSContext:
    """
    Start the NVIDIA MPS control daemon on *gpu_id* and set the required
    environment variables so that every child process (Ollama server) that
    is spawned while this context is active automatically uses MPS.

    Parameters
    ----------
    gpu_id : int
        Index of the GPU to use (``CUDA_VISIBLE_DEVICES``).
    pipe_dir : str
        Directory for the MPS Unix-domain socket.  Defaults to
        ``/tmp/nvidia-mps``.
    log_dir : str
        Directory where the MPS daemon writes logs.  Defaults to
        ``/tmp/nvidia-mps-log``.
    thread_percentage : int | None
        ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` — how many SMs each MPS
        client may use (1–100).  Leave ``None`` to use the driver default
        (100 %, all SMs shared among clients).
    """

    def __init__(
        self,
        gpu_id: int = 0,
        pipe_dir: str = _DEFAULT_PIPE_DIR,
        log_dir: str = _DEFAULT_LOG_DIR,
        thread_percentage: Optional[int] = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.pipe_dir = pipe_dir
        self.log_dir = log_dir
        self.thread_percentage = thread_percentage
        self._daemon_started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "MPSContext":
        """Start the MPS daemon and export required env vars."""
        if not self.is_available():
            raise RuntimeError(
                "nvidia-cuda-mps-control not found in PATH. "
                "Make sure the NVIDIA driver is installed."
            )

        if self.is_running(self.pipe_dir):
            print(f"MPS daemon already running (pipe={self.pipe_dir}) — reusing.")
            self._inject_env()
            return self

        Path(self.pipe_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]    = str(self.gpu_id)
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"]  = self.log_dir
        if self.thread_percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.thread_percentage)

        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start MPS daemon: {result.stderr.strip() or result.stdout.strip()}"
            )

        # Give the daemon a moment to create the socket
        time.sleep(1.0)

        self._daemon_started = True
        self._inject_env()
        pct = f", thread_percentage={self.thread_percentage}%" if self.thread_percentage else ""
        print(f"MPS daemon started (GPU {self.gpu_id}{pct})")
        return self

    def stop(self) -> None:
        """Send 'quit' to the MPS daemon and clean up env vars."""
        if not self._daemon_started:
            return

        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir

        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            env=env,
            capture_output=True,
            text=True,
        )

        self._daemon_started = False
        self._cleanup_env()
        print("MPS daemon stopped.")

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MPSContext":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if ``nvidia-cuda-mps-control`` is in PATH."""
        return shutil.which("nvidia-cuda-mps-control") is not None

    @staticmethod
    def is_running(pipe_dir: str = _DEFAULT_PIPE_DIR) -> bool:
        """Probe an existing MPS daemon by sending a no-op command."""
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="get_default_active_thread_percentage\n",
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_env(self) -> None:
        """Set MPS env vars on the current process so children inherit them."""
        os.environ["CUDA_VISIBLE_DEVICES"]    = str(self.gpu_id)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        os.environ["CUDA_MPS_LOG_DIRECTORY"]  = self.log_dir
        if self.thread_percentage is not None:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.thread_percentage)

    @staticmethod
    def _cleanup_env() -> None:
        for key in (
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        ):
            os.environ.pop(key, None)
