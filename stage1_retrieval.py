import os
import unittest
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def get_stage1_candidates(query, vector_db_path="./chroma_db", top_k=20):
    print("1. Loading Embedding Model & Vector Store...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings_model)

    print(f"2. Performing Vector Similarity Search for top-{top_k} candidates...")
    results = vector_db.similarity_search_with_score(query, k=top_k)
    return results

if __name__ == "__main__":
    test_query = "What is the onboarding process?"
    print(f"--- Running Stage-1 Retrieval Test: \"{test_query}\" ---")
    
    candidates = get_stage1_candidates(test_query, top_k=20)
    
    print(f"\nRetrieved {len(candidates)} candidate chunks:\n")
    for idx, (doc, score) in enumerate(candidates, start=1):
        source_val = doc.metadata.get("source", "N/A")
        print(f"Candidate {idx} [Distance Score: {score:.4f}]:")
        print(f"Content: {doc.page_content}")
        print(f"Source: {source_val}\n")
    
    print("--- Stage-1 Retrieval Test Completed Successfully ---")
