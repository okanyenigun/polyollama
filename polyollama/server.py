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
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

_STARTUP_TIMEOUT = 30   # seconds to wait for a server to become ready
_POLL_INTERVAL  = 0.5   # seconds between readiness checks


class OllamaServer:
    """Spawns an `ollama serve` process on *port* and exposes its base URL."""

    def __init__(
        self,
        port: int,
        host: str = DEFAULT_HOST,
        startup_timeout: float = _STARTUP_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL understood by the Ollama / LangChain client."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "OllamaServer":
        """Start the Ollama server process and wait until it is ready."""
        if self.is_running:
            return self

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"

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
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Ollama process on port {self.port} exited unexpectedly."
                )
            try:
                with urllib.request.urlopen(
                    f"{self.url}/api/tags", timeout=2
                ):
                    return   # server responded → ready
            except (urllib.error.URLError, OSError):
                time.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"Ollama server on port {self.port} did not become ready "
            f"within {self.startup_timeout} seconds."
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def server_url(port: int, host: str = DEFAULT_HOST) -> str:
    """Return the base URL for an already-running Ollama server on *port*."""
    return f"http://{host}:{port}"
