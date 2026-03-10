"""
Async parallel inference across multiple Ollama servers.
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Dict, Any, Optional, Union
from .server import OllamaServer


async def _async_inference(
    query: Dict[str, str],
    prompt: PromptTemplate,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single async inference call and return a result dict."""
    # Note: model_kwargs can include any ChatOllama init args except 'model' and 'base_url'
    kwargs = dict(model=model, **(model_kwargs or {}))
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    # If a parser is provided, chain it after the LLM. Otherwise just prompt → LLM.
    if parser:
        chain = prompt | llm | parser
    else:
        chain = prompt | llm
    # If the parser is used, we need to add format instructions to the query dict for each inference call.
    if parser:
        format_instructions = parser.get_format_instructions()
        query["format_instructions"] = format_instructions

    response = await chain.ainvoke(query)

    url = base_url or "default server"

    return {"url": url, "response": response}


async def parallel_inference(
    extra_ports: List[int],
    query: Dict[str, str],
    prompt: str,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Start one additional Ollama server per port in *extra_ports*, then run inference in parallel."""
    # Create OllamaServer instances for each extra port. The default server is assumed to be already running.
    servers = [OllamaServer(port=port) for port in extra_ports]
    if verbose:
        print(f"Starting {len(servers)} extra server(s)...")
    # Boot all extra servers concurrently (blocking start() pushed to threads)
    await asyncio.gather(*[asyncio.to_thread(srv.start) for srv in servers])
    if verbose:
        print("All servers ready. Running inference...")
    # If a parser is provided, ensure the prompt includes format instructions.
    if parser:
        if "{format_instructions}" not in prompt:
            prompt += "\n\nFormat Instructions:\n{format_instructions}"

    prompt = PromptTemplate.from_template(prompt)

    try:
        urls = [None] + [srv.url for srv in servers]  # None → default server

        results = await asyncio.gather(
            *[
                _async_inference(
                    query=query,
                    prompt=prompt,
                    model=model,
                    parser=parser,
                    model_kwargs=model_kwargs,
                    base_url=url,
                )
                for url in urls
            ]
        )
        if verbose:
            print("Inference complete.")
    finally:
        for srv in servers:
            srv.stop()

    return list(results)


async def _async_batch_on_server(
    chunks: Union[List[List[Dict[str, str]]], List[List[str]]],
    prompt: PromptTemplate,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a batch of inference calls on a single server and return the responses along with the server URL."""
    # Note: model_kwargs can include any ChatOllama init args except 'model' and 'base_url'
    kwargs = dict(model=model, **(model_kwargs or {}))
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOllama(**kwargs)
    # If a parser is provided, chain it after the LLM. Otherwise just prompt → LLM.
    if parser:
        chain = prompt | llm | parser
    else:
        chain = prompt | llm

    # If the parser is used, we need to add format instructions to each query dict for the batch inference calls.
    if parser:
        format_instructions = parser.get_format_instructions()
        for chunk in chunks:
            for query in chunk:
                query["format_instructions"] = format_instructions

    responses = await chain.abatch(chunks)

    return {"responses": responses, "url": base_url or "default server"}


async def parallel_batch_inference(
    extra_ports: list[int],
    query_list: List[Dict[str, str]],
    prompt: str,
    model: str,
    parser: Optional[BaseOutputParser] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """Start one additional Ollama server per port in *extra_ports*, then run batch inference in parallel."""
    all_ports = [None] + extra_ports  # None → default server
    n_servers = len(all_ports)
    # Split the query list into chunks for each server. The last chunk may be larger if the total number of queries isn't perfectly divisible by n_servers.
    chunk_size = max(1, len(query_list) // n_servers)
    chunks = [
        query_list[i : i + chunk_size] for i in range(0, len(query_list), chunk_size)
    ]

    if len(chunks) > n_servers:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    # Create OllamaServer instances for each extra port. The default server is assumed to be already running.
    servers = [OllamaServer(port=p) for p in extra_ports]

    if verbose:
        print(
            f"Starting {len(servers)} extra server(s), with {len(chunks)} chunk(s) of queries..."
        )

    if parser:
        if "{format_instructions}" not in prompt:
            prompt += "\n\nFormat Instructions:\n{format_instructions}"

    prompt = PromptTemplate.from_template(prompt)

    # Boot all extra servers concurrently (blocking start() pushed to threads)
    await asyncio.gather(*[asyncio.to_thread(srv.start) for srv in servers])

    if verbose:
        print("All servers ready. Running batch inference...")

    try:
        urls = [None] + [srv.url for srv in servers]
        nested = await asyncio.gather(
            *[
                _async_batch_on_server(
                    chunks=chunk,
                    prompt=prompt,
                    model=model,
                    parser=parser,
                    model_kwargs=model_kwargs,
                    base_url=url,
                )
                for i, (url, chunk) in enumerate(zip(urls, chunks))
            ]
        )
        if verbose:
            print("Batch inference complete.")
    finally:
        for srv in servers:
            srv.stop()

    # Flatten while preserving per-server chunk order
    return [item for chunk_results in nested for item in chunk_results]
