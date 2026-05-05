"""
Async parallel inference utilities for multiple Ollama servers.

These functions accept a plain list of server URLs and distribute work across
them. They have no knowledge of how those servers were started — they work
equally with locally-spawned processes (via OllamaPool) or pre-existing
servers (e.g., Docker services).
"""

import asyncio
import math
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Dict, Any, Optional, Union


async def _async_inference(
    query: Dict[str, str],
    prompt: PromptTemplate,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single async inference call against one server."""
    kwargs = dict(model=model, **(model_kwargs or {}))
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    if parser:
        chain = prompt | llm | parser
    else:
        chain = prompt | llm

    if parser:
        query["format_instructions"] = parser.get_format_instructions()

    response = await chain.ainvoke(query)
    return {"url": base_url or "default", "response": response}


async def parallel_inference(
    urls: List[str],
    query: Union[List[Dict[str, str]], List[str]],
    prompt: str,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fan out *query* items across *urls* in round-robin fashion.

    Args:
        urls:         List of Ollama base URLs to distribute work across.
                      Include the primary server URL here if you want it used.
                      Example: ["http://127.0.0.1:11434", "http://127.0.0.1:11435"]
        query:        List of query dicts (or strings) to process.
        prompt:       Prompt template string. Use {variable} placeholders.
        model:        Ollama model name (e.g. "llama3:8b").
        parser:       Optional LangChain output parser.
        model_kwargs: Extra kwargs forwarded to ChatOllama (temperature, etc.).
        verbose:      Print progress every 100 items.

    Returns:
        List of {"url": str, "response": Any} dicts in completion order.
    """
    if not urls:
        raise ValueError("urls must not be empty.")

    if parser and "{format_instructions}" not in prompt:
        prompt += "\n\nFormat Instructions:\n{format_instructions}"

    prompt_template = PromptTemplate.from_template(prompt)
    total = len(query)
    counter = [0]
    semaphore = asyncio.Semaphore(len(urls))

    async def _tracked(item, url):
        async with semaphore:
            result = await _async_inference(
                query=item,
                prompt=prompt_template,
                model=model,
                parser=parser,
                model_kwargs=model_kwargs,
                base_url=url,
            )
        counter[0] += 1
        if verbose and (counter[0] % 100 == 0 or counter[0] == total):
            print(f"  Progress: {counter[0]}/{total} done")
        return result

    if verbose:
        print(f"Running inference across {len(urls)} server(s)...")

    results = await asyncio.gather(
        *[_tracked(item, urls[i % len(urls)]) for i, item in enumerate(query)]
    )

    if verbose:
        print("Inference complete.")

    return list(results)


async def _batch_on_server(
    chunk: Union[List[Dict[str, str]], List[str]],
    prompt: PromptTemplate,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    counter: Optional[List[int]] = None,
    total: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a batch of items on a single server."""
    kwargs = dict(model=model, **(model_kwargs or {}))
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    chain = prompt | llm | parser if parser else prompt | llm

    if parser:
        fmt = parser.get_format_instructions()
        for q in chunk:
            if isinstance(q, dict):
                q["format_instructions"] = fmt

    all_responses = []
    for i in range(0, len(chunk), 100):
        sub = chunk[i : i + 100]
        responses = await chain.abatch(sub)
        all_responses.extend(responses)
        if counter is not None:
            counter[0] += len(responses)
            if verbose:
                print(f"  Progress: {counter[0]}/{total} done")

    return {"responses": all_responses, "url": base_url or "default"}


async def parallel_batch_inference(
    urls: List[str],
    query_list: List[Dict[str, str]],
    prompt: str,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Split *query_list* into equal chunks and run each chunk on a separate server.

    This is more efficient than parallel_inference for large datasets because
    each server processes a contiguous slice (better cache/KV reuse) rather
    than interleaved items.

    Args:
        urls:        List of Ollama base URLs. One chunk per URL.
        query_list:  Full list of query dicts to process.
        prompt:      Prompt template string.
        model:       Ollama model name.
        parser:      Optional LangChain output parser.
        model_kwargs: Extra kwargs forwarded to ChatOllama.
        verbose:     Print progress.

    Returns:
        Flat list of {"url": str, "response": Any} dicts preserving chunk order.
    """
    if not urls:
        raise ValueError("urls must not be empty.")

    if parser and "{format_instructions}" not in prompt:
        prompt += "\n\nFormat Instructions:\n{format_instructions}"

    prompt_template = PromptTemplate.from_template(prompt)
    chunk_size = math.ceil(len(query_list) / len(urls))
    chunks = [
        query_list[i : i + chunk_size] for i in range(0, len(query_list), chunk_size)
    ]

    total = len(query_list)
    counter = [0]

    if verbose:
        print(
            f"Running batch inference: {len(chunks)} chunk(s) across {len(urls)} server(s)..."
        )

    nested = await asyncio.gather(
        *[
            _batch_on_server(
                chunk=chunk,
                prompt=prompt_template,
                model=model,
                parser=parser,
                model_kwargs=model_kwargs,
                base_url=url,
                counter=counter,
                total=total,
                verbose=verbose,
            )
            for url, chunk in zip(urls, chunks)
        ]
    )

    if verbose:
        print("Batch inference complete.")

    return [
        {"url": chunk_result["url"], "response": response}
        for chunk_result in nested
        for response in chunk_result["responses"]
    ]
