from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


@lru_cache(maxsize=4)
def get_vector_db(vector_db_path="./chroma_db"):
    print("1. Loading Embedding Model & Vector Store...")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings_model,
    )

    return vector_db


def get_stage1_candidates(query, vector_db_path="./chroma_db", top_k=20):

    vector_db = get_vector_db(vector_db_path)

    print(
        f"2. Performing Vector Similarity Search for top-{top_k} candidates..."
    )

    # Retrieve extra results because duplicates may exist
    raw_k = max(top_k * 3, 50)

    results = vector_db.similarity_search_with_score(
        query,
        k=raw_k
    )

    unique_results = []
    seen_content = set()

    for doc, score in results:

        metadata = doc.metadata or {}

        content = doc.page_content.strip()

        # Remove exact duplicate chunks
        if content in seen_content:
            continue

        seen_content.add(content)

        candidate = {
            "chunk": content,
            "score": float(score),
            "document": (
                metadata.get("document")
                or metadata.get("source")
                or "N/A"
            ),
            "page": metadata.get("page", "N/A"),
            "section": metadata.get("section", "N/A"),
            "chunk_id": (
                metadata.get("chunk_id")
                or metadata.get("id")
                or "N/A"
            ),
        }

        unique_results.append(candidate)

        if len(unique_results) >= top_k:
            break

    return unique_results


if __name__ == "__main__":

    test_query = "What is the onboarding process?"

    print("\n============================================================")
    print("STAGE 1 — TOP 20 RETRIEVAL")
    print("============================================================")

    print(f"\nQuestion: {test_query}\n")

    candidates = get_stage1_candidates(
        test_query,
        top_k=20
    )

    print(
        f"Retrieved {len(candidates)} UNIQUE candidate chunks:\n"
    )

    for idx, candidate in enumerate(candidates, start=1):

        print(f"Candidate {idx}")
        print(f"Distance Score : {candidate['score']:.4f}")
        print(f"Document       : {candidate['document']}")
        print(f"Page           : {candidate['page']}")
        print(f"Section        : {candidate['section']}")
        print(f"Chunk ID        : {candidate['chunk_id']}")
        print(f"Chunk          : {candidate['chunk']}")
        print("-" * 60)

    print("\n============================================================")
    print(
        f"STAGE 1 COMPLETED — {len(candidates)} UNIQUE CANDIDATES"
    )
    print("============================================================")