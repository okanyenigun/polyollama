import asyncio
import time
from polyollama import (
    parallel_inference,
    parallel_batch_inference,
    MPSContext,
)
from polyollama.misc.example_utils import (
    load_question_dataset,
    simple_inference,
    sequential_inference,
    batch_inference,
)


async def main():
    print("Starting QUESTION Benchmark Testing...")

    N = 3000
    BATCH_SIZE = 6
    MODEL = "gemma2:2b"
    MODEL_KWARGS = {"temperature": 0, "max_tokens": 2048}
    N_PORTS = 5
    EXTRA_PORTS = list(range(11435, 11435 + N_PORTS))
    PROMPT_TEMPLATE = "Answer politely: {question}"

    question_list = load_question_dataset(n=N)

    print("\n=== Smoke Simple inference (1 question) ===")
    response = simple_inference(model=MODEL, question=question_list[0])
    print(f"Response: {response.content[:20]}...")

    # ------------------------------------------------------------------ #
    # 1. Sequential: one by one on the default server
    # ------------------------------------------------------------------ #
    print("\n=== Sequential inference ===")
    responses = sequential_inference(model=MODEL, questions=question_list)
    print(
        f"Total responses received: {len(responses)}\nLast response: {responses[-1]['response'].content[:20]}..."
    )

    # ------------------------------------------------------------------ #
    # 2. Batch: in chunks on the default server
    # ------------------------------------------------------------------ #
    print("\n=== Batch inference ===")
    responses = batch_inference(
        model=MODEL, questions=question_list, batch_size=BATCH_SIZE
    )
    print(
        f"Total responses received: {len(responses)}\nLast response: {responses[-1]['response'].content[:20]}..."
    )

    # ------------------------------------------------------------------ #
    # 3. Parallel: in chunks on multiple servers
    # ------------------------------------------------------------------ #
    start = time.time()
    print("\n=== Parallel inference ===")
    responses = await parallel_inference(
        extra_ports=EXTRA_PORTS,
        query=question_list,
        prompt=PROMPT_TEMPLATE,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
    )
    print(f"Total time taken for parallel inference: {time.time() - start:.2f} seconds")
    print(
        f"Total responses received: {len(responses)}\nLast response: {responses[-1]['response'].content[:20]}..."
    )

    # ------------------------------------------------------------------ #
    # 4. Parallel distributed: split across servers
    # ------------------------------------------------------------------ #
    print("\n=== Parallel distributed inference ===")
    start = time.time()
    responses = await parallel_batch_inference(
        extra_ports=EXTRA_PORTS,
        query_list=question_list,
        prompt=PROMPT_TEMPLATE,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
    )
    print(
        f"Total time taken for parallel distributed inference: {time.time() - start:.2f} seconds"
    )
    print(
        f"Total responses received: {len(responses)}\nLast response: {responses[-1]['response'].content[:20]}..."
    )

    # ------------------------------------------------------------------ #
    # 5. MPS parallel distributed: same servers, but GPU SMs are shared
    #    concurrently via the NVIDIA MPS daemon — should be faster than
    #    the plain parallel run above on a single GPU.
    # ------------------------------------------------------------------ #
    print("\n=== MPS parallel distributed inference (4 servers × 6 questions) ===")
    start = time.time()
    with MPSContext(gpu_id=0) as mps:
        responses = await parallel_batch_inference(
            extra_ports=EXTRA_PORTS,
            query_list=question_list,
            prompt=PROMPT_TEMPLATE,
            model=MODEL,
            model_kwargs=MODEL_KWARGS,
        )
    print(
        f"Total time taken for MPS parallel distributed inference: {time.time() - start:.2f} seconds"
    )
    print(
        f"Total responses received: {len(responses)}\nLast response: {responses[-1]['response'].content[:20]}..."
    )

    print("\n=== Smoke Simple inference (1 question) ===")
    simple_inference(model=MODEL, question=question_list[0])

    print("\nTesting complete.")


if __name__ == "__main__":
    asyncio.run(main())
