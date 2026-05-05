"""
OllamaPool — manages the lifecycle of multiple OllamaServer instances
and exposes their base URLs for use by inference code or any HTTP client.
"""

import asyncio
from .server import OllamaServer


class OllamaPool:
    """
    Manages a group of OllamaServer processes and exposes their URLs.

    Usage — as a sync context manager (blocks until all servers are ready):
        with OllamaPool(ports=[11435, 11436]) as pool:
            do_something(pool.urls)

    Usage — async startup (concurrent, faster):
        pool = OllamaPool(ports=[11435, 11436])
        await pool.start_async()
        try:
            do_something(pool.urls)
        finally:
            pool.stop()

    Usage — manual:
        pool = OllamaPool(ports=[11435, 11436])
        pool.start()   # sequential, blocking
        pool.stop()
    """

    def __init__(
        self,
        ports: list[int],
        host: str = "127.0.0.1",
        num_parallel: int = 1,
        num_ctx: int | None = None,
    ):
        self._servers = [
            OllamaServer(port=p, host=host, num_parallel=num_parallel, num_ctx=num_ctx)
            for p in ports
        ]

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "OllamaPool":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def urls(self) -> list[str]:
        """Base URLs of all running servers in this pool."""
        return [srv.url for srv in self._servers]

    @property
    def servers(self) -> list[OllamaServer]:
        """The underlying OllamaServer instances (read-only access)."""
        return list(self._servers)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "OllamaPool":
        """Start all servers sequentially (blocking). Use start_async for concurrency."""
        for srv in self._servers:
            srv.start()
        return self

    async def start_async(self) -> "OllamaPool":
        """Start all servers concurrently (non-blocking, faster than start())."""
        await asyncio.gather(*[asyncio.to_thread(srv.start) for srv in self._servers])
        return self

    def stop(self) -> None:
        """Stop all servers."""
        for srv in self._servers:
            srv.stop()
