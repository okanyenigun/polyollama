"""
Async parallel inference across multiple Ollama servers.
"""

import asyncio
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from .server import OllamaServer, DEFAULT_URL


async def _async_inference(
    model: str,
    question: str,
    base_url: Optional[str] = None,
) -> dict:
    """Run a single async inference call and return a result dict."""
    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs: dict = dict(model=model, temperature=0, max_tokens=2048)
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    chain = prompt | llm

    start = time.time()
    response = await chain.ainvoke({"question": question})
    elapsed = time.time() - start

    url = base_url or DEFAULT_URL
    print(f"[{url}] Time taken: {elapsed:.2f}s")
    return {"url": url, "response": response, "elapsed": elapsed}


async def parallel_inference(
    model: str,
    question: str,
    extra_ports: list[int],
) -> list[dict]:
    """
    Start one additional Ollama server per port in *extra_ports*, then run
    *question* against all of them **and** the default server simultaneously.

    The default server (port 11434) is assumed to be already running.
    Extra servers are started concurrently, queried in parallel, then stopped.

    Returns a list of result dicts: {url, response, elapsed}.
    """
    servers = [OllamaServer(port=port) for port in extra_ports]

    print(f"Starting {len(servers)} extra server(s)...")
    wall_start = time.time()

    # Boot all extra servers concurrently (blocking start() pushed to threads)
    await asyncio.gather(*[asyncio.to_thread(srv.start) for srv in servers])
    print(f"All servers ready in {time.time() - wall_start:.2f}s. Running inference...")

    try:
        urls = [None] + [srv.url for srv in servers]  # None → default server
        inference_start = time.time()
        results = await asyncio.gather(
            *[
                _async_inference(model=model, question=question, base_url=url)
                for url in urls
            ]
        )
        total_elapsed = time.time() - inference_start
        print(f"Total parallel inference time: {total_elapsed:.2f}s")
    finally:
        for srv in servers:
            srv.stop()

    return list(results)


# ---------------------------------------------------------------------------
# Sequential inference over n questions
# ---------------------------------------------------------------------------

def sequential_inference(
    model: str,
    questions: list[str],
    base_url: Optional[str] = None,
) -> list[dict]:
    """
    Run *questions* one by one against a single server and return results.
    Prints per-question and total elapsed time.
    """
    from langchain_ollama import ChatOllama  # local import to keep top-level clean

    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs: dict = dict(model=model, temperature=0, max_tokens=2048)
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    chain = prompt | llm

    url = base_url or DEFAULT_URL
    results = []
    total_start = time.time()

    for i, question in enumerate(questions):
        start = time.time()
        response = chain.invoke({"question": question})
        elapsed = time.time() - start
        if i % 100 == 0:
            print(f"[{url}] Q{i+1}/{len(questions)} done in {elapsed:.2f}s")
        results.append({"url": url, "question": question, "response": response, "elapsed": elapsed})

    total_elapsed = time.time() - total_start
    print(f"Sequential total: {total_elapsed:.2f}s for {len(questions)} questions")
    return results


# ---------------------------------------------------------------------------
# Batch inference (LangChain .batch()) over n questions
# ---------------------------------------------------------------------------

def batch_inference(
    model: str,
    questions: list[str],
    batch_size: int = 6,
    base_url: Optional[str] = None,
) -> list[dict]:
    """
    Run *questions* through a single server using LangChain's `.batch()`,
    chunked into groups of *batch_size*. Prints per-batch and total time.
    """
    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs: dict = dict(model=model, temperature=0, max_tokens=2048)
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    chain = prompt | llm

    url = base_url or DEFAULT_URL
    results = []
    total_start = time.time()
    batches = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]

    for b_idx, batch in enumerate(batches):
        inputs = [{"question": q} for q in batch]
        start = time.time()
        responses = chain.batch(inputs)
        elapsed = time.time() - start
        if b_idx % 10 == 0:
            print(
                f"[{url}] Batch {b_idx+1}/{len(batches)} "
                f"({len(batch)} examples) done in {elapsed:.2f}s"
            )
        for question, response in zip(batch, responses):
            results.append({"url": url, "question": question, "response": response})

    total_elapsed = time.time() - total_start
    print(f"Batch total: {total_elapsed:.2f}s for {len(questions)} questions (batch_size={batch_size})")
    return results


# ---------------------------------------------------------------------------
# Parallel distributed inference across multiple servers
# ---------------------------------------------------------------------------

async def _async_batch_on_server(
    model: str,
    questions: list[str],
    base_url: Optional[str],
    server_idx: int,
    total_servers: int,
) -> list[dict]:
    """Run a chunk of questions on one server using async .abatch()."""
    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs: dict = dict(model=model, temperature=0, max_tokens=2048)
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    chain = prompt | llm

    url = base_url or DEFAULT_URL
    inputs = [{"question": q} for q in questions]

    start = time.time()
    responses = await chain.abatch(inputs)
    elapsed = time.time() - start
    print(
        f"[{url}] Server {server_idx+1}/{total_servers}: "
        f"{len(questions)} questions done in {elapsed:.2f}s"
    )
    return [
        {"url": url, "question": q, "response": r, "elapsed": elapsed}
        for q, r in zip(questions, responses)
    ]


async def parallel_distributed_inference(
    model: str,
    questions: list[str],
    extra_ports: list[int],
) -> list[dict]:
    """
    Distribute *questions* evenly across the default server + one server per
    port in *extra_ports*, running all servers concurrently with .abatch().

    Example: 24 questions across 4 servers → 6 questions per server.

    Returns a flat list of result dicts preserving original question order.
    """
    all_ports = [None] + extra_ports  # None → default server
    n_servers = len(all_ports)
    chunk_size = max(1, len(questions) // n_servers)
    chunks = [questions[i : i + chunk_size] for i in range(0, len(questions), chunk_size)]
    # Handle remainder by folding into last chunk
    if len(chunks) > n_servers:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    servers = [OllamaServer(port=p) for p in extra_ports]

    print(f"Starting {len(servers)} extra server(s)...")
    wall_start = time.time()
    await asyncio.gather(*[asyncio.to_thread(srv.start) for srv in servers])
    print(f"All servers ready in {time.time() - wall_start:.2f}s. Distributing {len(questions)} questions across {n_servers} servers...")

    try:
        urls = [None] + [srv.url for srv in servers]
        inference_start = time.time()
        nested = await asyncio.gather(
            *[
                _async_batch_on_server(
                    model=model,
                    questions=chunk,
                    base_url=url,
                    server_idx=i,
                    total_servers=n_servers,
                )
                for i, (url, chunk) in enumerate(zip(urls, chunks))
            ]
        )
        total_elapsed = time.time() - inference_start
        print(f"Total parallel distributed inference time: {total_elapsed:.2f}s")
    finally:
        for srv in servers:
            srv.stop()

    # Flatten while preserving per-server chunk order
    return [item for chunk_results in nested for item in chunk_results]
