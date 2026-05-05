"""
Manages multiple Ollama server processes, each listening on a distinct port.

Usage
-----
    from polyollama import OllamaServer

    # As a context manager (server is stopped automatically on exit)
    with OllamaServer(port=11435) as srv:
        url = srv.url          # e.g. "http://127.0.0.1:11435"
        ...

    # Manual lifecycle
    srv = OllamaServer(port=11435)
    srv.start()
    url = srv.url
    srv.stop()
"""

import os
import time
import subprocess
import urllib.request
import urllib.error
from typing import Optional


class OllamaServer:
    """Spawns an `ollama serve` process on *port* and exposes its base URL."""

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        startup_timeout: float = 30,
        poll_interval: float = 0.5,
        num_parallel: int = 1,
        num_ctx: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.num_parallel = num_parallel
        self.num_ctx = num_ctx
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL understood by the Ollama / LangChain client."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """True if the server process is currently running."""
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "OllamaServer":
        """Start the Ollama server process and wait until it is ready."""
        if self.is_running:
            return self

        # Set OLLAMA_HOST to ensure the server listens on the correct port
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)

        # Start the Ollama server process
        self._process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._wait_until_ready()
        return self

    def stop(self) -> None:
        """Terminate the Ollama server process."""
        if self._process is not None:
            self._process.terminate()
            # Wait for the process to exit, with a timeout to avoid hanging
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self) -> None:
        """Poll the server until it responds or the process exits, up to the startup timeout."""
        deadline = time.time() + self.startup_timeout
        # Poll the server until it responds or the process exits
        while time.time() < deadline:
            # Check if the process has exited unexpectedly
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Ollama process on port {self.port} terminated unexpectedly."
                )
            # Try to connect to the server's API endpoint
            try:
                with urllib.request.urlopen(f"{self.url}/api/tags", timeout=2):
                    return  # Server is ready
            except (urllib.error.URLError, OSError):
                time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Ollama server on port {self.port} did not become ready within {self.startup_timeout} seconds."
        )

    # ------------------------------------------------------------------
    # External helpers
    # ------------------------------------------------------------------
    @staticmethod
    def server_url(port: int, host: str = "127.0.0.1") -> str:
        """Construct the base URL for a server on the given host and port."""
        return f"http://{host}:{port}"
