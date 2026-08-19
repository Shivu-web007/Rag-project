import os
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Stage2Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print(f"Loading CrossEncoder Model: {model_name}")
        self.reranker = CrossEncoder(model_name)

    def rerank(
        self,
        query,
        candidate_tuples,
        top_k=5
    ):
        if not candidate_tuples:
            return []

        documents = [
            doc for doc, score in candidate_tuples
        ]

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        print(
            f"Scoring {len(pairs)} candidates..."
        )

        scores = self.reranker.predict(pairs)

        results = []

        for doc, score in zip(documents, scores):
            results.append(
                (doc, float(score))
            )

        # Higher CrossEncoder score = better
        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]


def display_raw_results(results):

    print("\n" + "=" * 65)
    print("RAW STAGE-1 RESULTS")
    print("=" * 65)

    for i, (doc, distance) in enumerate(
        results, 1
    ):
        print(
            f"{i}. Distance: {distance:.4f}"
        )
        print(
            f"   Source: {doc.metadata.get('source')}"
        )
        print(
            f"   Content: {doc.page_content}"
        )
        print("-" * 65)


def display_reranked_results(results):

    print("\n" + "=" * 65)
    print("RERANKED STAGE-2 RESULTS")
    print("=" * 65)

    for i, (doc, score) in enumerate(
        results, 1
    ):
        print(
            f"{i}. CrossEncoder Score: {score:.4f}"
        )
        print(
            f"   Source: {doc.metadata.get('source')}"
        )
        print(
            f"   Content: {doc.page_content}"
        )
        print("-" * 65)


if __name__ == "__main__":

    query = "What is the onboarding process?"

    print(
        "\n--- DAY 5 RERANKING TEST ---"
    )

    # Stage-1
    print(
        "\n1. Retrieving Top-20 candidates..."
    )

    raw_results = retrieve_candidates(
        query=query,
        top_k=20
    )

    print(
        f"Retrieved {len(raw_results)} candidates."
    )

    # Display raw results
    display_raw_results(raw_results)

    # Stage-2 CrossEncoder
    print(
        "\n2. Starting CrossEncoder Reranking..."
    )

    reranker = Stage2Reranker()

    reranked_results = reranker.rerank(
        query=query,
        candidate_tuples=raw_results,
        top_k=5
    )

    # Display reranked results
    display_reranked_results(
        reranked_results
    )

    # Comparison
    print("\n" + "=" * 65)
    print("RAW vs RERANKED COMPARISON")
    print("=" * 65)

    print(
        f"Raw candidates : {len(raw_results)}"
    )

    print(
        f"Reranked Top-K : {len(reranked_results)}"
    )

    if reranked_results:
        print(
            "\nBest Reranked Candidate:"
        )

        print(
            reranked_results[0][0].page_content
        )

    print(
        "\n--- DAY 5 Reranking Completed Successfully ---"
    )
