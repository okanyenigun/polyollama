import time
from typing import Optional, Dict, Any, List

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

def load_question_dataset(n: int) -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("HiTruong/movie_QA", "default")
    questions = [item["Question"] for item in ds["train_v1"]]
    return questions[:n]

def get_chain(model: str, base_url: Optional[str] = None) -> Any:
    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    prompt = PromptTemplate.from_template("Answer politely: {question}")
    kwargs = dict(model=model, temperature=0, max_tokens=2048)

    if base_url:
        kwargs["base_url"] = base_url

    model = ChatOllama(**kwargs)
    chain = prompt | model
    return chain


def simple_inference(model:str, question: str, base_url: Optional[str] = None) -> Any:
    start = time.time()

    chain = get_chain(model=model, base_url=base_url)
    response = chain.invoke(question)

    print(f"Time taken: {time.time() - start:.2f} seconds")
    return response


def sequential_inference(
    model: str,
    questions: List[str],
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run *questions* one by one against a single server and return results.
    Prints per-question and total elapsed time.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate

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


def batch_inference(
    model: str,
    questions: List[str],
    batch_size: int = 6,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run *questions* through a single server using LangChain's `.batch()`,
    chunked into groups of *batch_size*. Prints per-batch and total time.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate

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
