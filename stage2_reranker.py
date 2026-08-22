from sentence_transformers import CrossEncoder

from stage1_retrieval import get_stage1_candidates


# ==========================================================
# STAGE 2 — RERANKER
# ==========================================================

class Stage2Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print("Loading Cross-Encoder Reranker...")

        self.model = CrossEncoder(
            model_name
        )


    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:

            pairs.append(
                (
                    query,
                    candidate["chunk"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores
        ):

            result = candidate.copy()

            result["rerank_score"] = float(score)

            reranked.append(
                result
            )

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    query = "What is the onboarding process?"

    print("\n" + "=" * 70)
    print("STAGE 2 — RERANKING")
    print("=" * 70)

    print("\nQuestion:", query)

    # ------------------------------------------------------
    # Stage 1 Retrieval
    # ------------------------------------------------------

    print("\nRunning Stage 1 Retrieval...")

    raw_results = get_stage1_candidates(
        query,
        top_k=20
    )

    print(
        f"\nRaw candidates from Stage 1: {len(raw_results)}"
    )


    # ------------------------------------------------------
    # Display Stage 1 results
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1 RAW CANDIDATES")
    print("=" * 70)

    for index, candidate in enumerate(
        raw_results,
        start=1
    ):

        print(f"\nCandidate {index}")
        print(
            f"Distance Score : {candidate['score']:.4f}"
        )
        print(
            f"Document       : {candidate['document']}"
        )
        print(
            f"Chunk          : {candidate['chunk']}"
        )


    # ------------------------------------------------------
    # Duplicate Removal
    # ------------------------------------------------------

    unique_results = []

    seen_content = set()

    for candidate in raw_results:

        content = candidate["chunk"].strip()

        if content in seen_content:
            continue

        seen_content.add(content)

        unique_results.append(
            candidate
        )


    print("\n" + "=" * 70)
    print("DUPLICATE REMOVAL")
    print("=" * 70)

    print(
        f"Before duplicate removal : {len(raw_results)}"
    )

    print(
        f"After duplicate removal  : {len(unique_results)}"
    )


    # ------------------------------------------------------
    # Stage 2 Reranking
    # ------------------------------------------------------

    reranker = Stage2Reranker()

    reranked_results = reranker.rerank(
        query,
        unique_results,
        top_k=5
    )


    # ------------------------------------------------------
    # Display Reranked Results
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 2 — RERANKED TOP-5")
    print("=" * 70)

    for index, candidate in enumerate(
        reranked_results,
        start=1
    ):

        print(f"\nRank {index}")

        print(
            f"Rerank Score : {candidate['rerank_score']:.4f}"
        )

        print(
            f"Document     : {candidate['document']}"
        )

        print(
            f"Chunk        : {candidate['chunk']}"
        )

        print("-" * 70)


    # ------------------------------------------------------
    # Final Comparison
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW vs RERANKED COMPARISON")
    print("=" * 70)

    print(
        f"Raw candidates : {len(raw_results)}"
    )

    print(
        f"Reranked Top-5 : {len(reranked_results)}"
    )

    if reranked_results:

        best = reranked_results[0]

        print("\nBest Reranked Candidate:")

        print(
            best["chunk"]
        )

    print(
        "\n--- STAGE 2 RERANKING COMPLETED SUCCESSFULLY ---"
    )