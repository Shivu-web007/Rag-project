from stage1_retrieval import get_stage1_candidates
from stage2_reranker import Stage2Reranker
from day6_prompt_builder import build_prompt
from stage4_answer_generator import generate_local_answer


# ==========================================================
# MAIN RAG PIPELINE
# ==========================================================

def run_rag_pipeline(question):

    print("\n" + "=" * 70)
    print("COMPLETE RAG PIPELINE")
    print("=" * 70)

    print("\nQuestion:", question)

    # ------------------------------------------------------
    # STAGE 1 — RETRIEVAL
    # ------------------------------------------------------

    print("\n[1] STAGE 1 — VECTOR RETRIEVAL")

    candidates = get_stage1_candidates(
        question,
        top_k=20
    )

    print(
        f"Retrieved candidates: {len(candidates)}"
    )

    # ------------------------------------------------------
    # STAGE 2 — RERANKING
    # ------------------------------------------------------

    print("\n[2] STAGE 2 — CROSS-ENCODER RERANKING")

    reranker = Stage2Reranker()

    reranked_results = reranker.rerank(
        question,
        candidates,
        top_k=5
    )

    print(
        f"Reranked candidates: {len(reranked_results)}"
    )

    # ------------------------------------------------------
    # STAGE 3 — PROMPT BUILDER
    # ------------------------------------------------------

    print("\n[3] STAGE 3 — PROMPT BUILDING")

    prompt = build_prompt(
        question,
        reranked_results
    )

    print("RAG prompt built successfully.")

    # ------------------------------------------------------
    # STAGE 4 — FINAL ANSWER
    # ------------------------------------------------------

    print("\n[4] STAGE 4 — FINAL ANSWER")

    answer = generate_local_answer(
        question,
        reranked_results
    )

    # ------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RAG ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("COMPLETE RAG PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a question: "
    )

    run_rag_pipeline(
        question
    )