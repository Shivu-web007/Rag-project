import os
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def retrieve_candidates(query, top_k=20):

    print("2. Loading Embedding Model & Vector Store...")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings_model
    )

    print(
        f"3. Performing Vector Similarity Search for top-{top_k} candidates..."
    )

    results = vector_db.similarity_search_with_score(
        query,
        k=top_k
    )

    return results


class Stage2Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print(
            f"\n4. Loading CrossEncoder Model..."
        )

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
            doc for doc, distance in candidate_tuples
        ]

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        print(
            f"5. Scoring {len(pairs)} candidates..."
        )

        scores = self.reranker.predict(pairs)

        results = []

        for doc, score in zip(documents, scores):

            results.append(
                (doc, float(score))
            )

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
            f"{i}. Distance Score: {distance:.4f}"
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

    print(
        "\n1. Retrieving Top-20 candidates..."
    )

    raw_results = retrieve_candidates(
        query,
        top_k=20
    )

    print(
        f"Retrieved {len(raw_results)} candidate chunks."
    )

    display_raw_results(
        raw_results
    )

    reranker = Stage2Reranker()

    reranked_results = reranker.rerank(
        query,
        raw_results,
        top_k=5
    )

    display_reranked_results(
        reranked_results
    )

    print("\n" + "=" * 65)
    print("RAW vs RERANKED COMPARISON")
    print("=" * 65)

    print(
        f"Raw candidates : {len(raw_results)}"
    )

    print(
        f"Reranked Top-5 : {len(reranked_results)}"
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