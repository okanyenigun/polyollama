"""
Async parallel inference across multiple Ollama servers.
"""

import asyncio
import math
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
    query: Union[List[Dict[str, str]], List[str]],
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

    total = len(query)
    counter = [0]
    urls = [None] + [srv.url for srv in servers]  # None → default server
    # Limit in-flight requests to avoid overwhelming servers (one slot per server)
    semaphore = asyncio.Semaphore(len(urls))

    async def _tracked(item, url):
        async with semaphore:
            result = await _async_inference(
                query=item,
                prompt=prompt,
                model=model,
                parser=parser,
                model_kwargs=model_kwargs,
                base_url=url,
            )
        counter[0] += 1
        if verbose and (counter[0] % 100 == 0 or counter[0] == total):
            print(f"  Progress: {counter[0]}/{total} done")
        return result

    try:
        results = await asyncio.gather(
            *[
                _tracked(item, urls[i % len(urls)])
                for i, item in enumerate(query)
            ]
        )
        if verbose:
            print("Inference complete.")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        for srv in servers:
            srv.stop()

    return list(results)


async def _async_batch_on_server(
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
    """Run a batch of inference calls on a single server and return the responses along with the server URL."""
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
        # Process the batch in sub-chunks to avoid overwhelming the server. Adjust the sub-chunk size as needed.
        sub = chunk[i : i + 100]
        responses = await chain.abatch(sub)
        all_responses.extend(responses)
        if counter is not None:
            counter[0] += len(responses)
            if verbose:
                print(f"  Progress: {counter[0]}/{total} done")

    return {"responses": all_responses, "url": base_url or "default server"}


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
    # Split the query list into at most n_servers chunks.
    chunk_size = math.ceil(len(query_list) / n_servers)
    chunks = [
        query_list[i : i + chunk_size] for i in range(0, len(query_list), chunk_size)
    ]

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

    total = len(query_list)
    counter = [0]

    try:
        urls = [None] + [srv.url for srv in servers]
        nested = await asyncio.gather(
            *[
                _async_batch_on_server(
                    chunk=chunk,
                    prompt=prompt,
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
    except Exception as e:
        print(f"Error during batch inference: {e}")
        raise
    finally:
        for srv in servers:
            srv.stop()

    # Flatten while preserving per-server chunk order
    return [
        {"url": chunk_result["url"], "response": response}
        for chunk_result in nested
        for response in chunk_result["responses"]
    ]
