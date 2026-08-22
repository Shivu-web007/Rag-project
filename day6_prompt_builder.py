from stage1_retrieval import get_stage1_candidates
from stage2_reranker import Stage2Reranker


# ==========================================================
# STAGE 3 — PROMPT BUILDER
# ==========================================================

def build_prompt(question, reranked_results):

    context_parts = []

    for i, result in enumerate(reranked_results, start=1):

        source = result.get("document", "Unknown")
        content = result.get("chunk", "")

        context_parts.append(
            f"[{i}] Source: {source}\n"
            f"Content: {content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the information
provided in the context below.

Do not use outside knowledge.
Do not make up information.

If the answer is not present in the context, say:

"Information not available in the provided documents."

Question:
{question}

Context:
{context}

Give a concise answer.

At the end, provide source citations.

Sources:
"""

    for result in reranked_results:

        source = result.get(
            "document",
            "Unknown"
        )

        prompt += f"[{source}] "

    return prompt


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STAGE 3 — PROMPT BUILDER")
    print("=" * 70)

    question = input(
        "\nAsk a question: "
    )

    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------

    print(
        "\nRunning Stage 1 Retrieval..."
    )

    candidates = get_stage1_candidates(
        question,
        top_k=20
    )

    print(
        f"Retrieved {len(candidates)} candidates."
    )


    # ------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------

    print(
        "\nRunning Stage 2 Reranking..."
    )

    reranker = Stage2Reranker()

    reranked_results = reranker.rerank(
        question,
        candidates,
        top_k=5
    )

    print(
        f"Reranked {len(reranked_results)} candidates."
    )


    # ------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------

    prompt = build_prompt(
        question,
        reranked_results
    )


    print("\n" + "=" * 70)
    print("GENERATED RAG PROMPT")
    print("=" * 70)

    print(prompt)

    print("\n" + "=" * 70)
    print("STAGE 3 PROMPT BUILDING COMPLETED")
    print("=" * 70)