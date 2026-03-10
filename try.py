import asyncio
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from datasets import load_dataset
from typing import Optional

from polyollama import (
    OllamaServer,
    parallel_inference,
    sequential_inference,
    batch_inference,
    parallel_distributed_inference,
    MPSContext,
)


def load(n: int):
    ds = load_dataset("HiTruong/movie_QA", "default")
    questions = [item["Question"] for item in ds["train_v1"]]
    return questions[:n]

def get_chain(model: str, base_url: Optional[str] = None):
    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs = dict(model=model, temperature=0, max_tokens=2048)

    if base_url:
        kwargs["base_url"] = base_url

    model = ChatOllama(**kwargs)
    chain = prompt | model
    return chain

def simple_inference(model:str, question: str, base_url: Optional[str] = None):
    start = time.time()
    chain = get_chain(model=model, base_url=base_url)
    response = chain.invoke(question)
    print(f"Time taken: {time.time() - start:.2f} seconds")
    return response

async def main():
    print("Starting Testing...")

    n = 3000
    model = "gemma2:2b"
    questions = load(n=n)

    print("\n=== Simple inference (1 question) ===")
    simple_inference(model=model, question=questions[0])

    # ------------------------------------------------------------------ #
    # 1. Sequential: 300 questions, one by one on the default server
    # ------------------------------------------------------------------ #
    print("\n=== Sequential inference (300 questions) ===")
    sequential_inference(model=model, questions=questions)

    # ------------------------------------------------------------------ #
    # 2. Batch: 300 questions in chunks of 6 on the default server
    # ------------------------------------------------------------------ #
    print("\n=== Batch inference (300 questions, batch_size=6) ===")
    batch_inference(model=model, questions=questions, batch_size=6)

    # ------------------------------------------------------------------ #
    # 3. Parallel distributed: 300 questions split across 4 servers
    #    Default (11434) + 11435 + 11436 + 11437  →  6 questions each
    # ------------------------------------------------------------------ #
    print("\n=== Parallel distributed inference (4 servers × 6 questions) ===")
    await parallel_distributed_inference(
        model=model,
        questions=questions,
        extra_ports=[11435, 11436, 11437, 11438, 11439],
    )

    # ------------------------------------------------------------------ #
    # 4. MPS parallel distributed: same 4 servers, but GPU SMs are shared
    #    concurrently via the NVIDIA MPS daemon — should be faster than
    #    the plain parallel run above on a single GPU.
    # ------------------------------------------------------------------ #
    print("\n=== MPS parallel distributed inference (4 servers × 6 questions) ===")
    with MPSContext(gpu_id=0) as mps:
        await parallel_distributed_inference(
            model=model,
            questions=questions,
            extra_ports=[11435, 11436, 11437, 11438, 11439],
        )

    print("\nTesting complete.")
    simple_inference(model=model, question=questions[0])

if __name__ == "__main__":
    asyncio.run(main())
