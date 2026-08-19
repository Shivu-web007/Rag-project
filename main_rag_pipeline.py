from stage1_retrieval import get_stage1_candidates
from stage2_reranker import rerank_candidates
from day6_prompt_builder import build_rag_prompt

def run_rag_pipeline(user_query, top_k_stage1=20, top_k_stage2=3, rerank_method="cross_encoder"):
    # Edge Case 1: Empty or Whitespace-only Query
    if not user_query or not user_query.strip():
        return {
            "query": user_query,
            "error": "Query cannot be empty.",
            "prompt": None,
            "citations": {}
        }

    print("\n==========================================================")
    print(f" EXECUTE END-TO-END RAG PIPELINE: \"{user_query}\"")
    print("==========================================================")

    # 1. Stage-1 Retrieval
    print(f"\n[Stage 1] Retrieving Top-{top_k_stage1} Candidates from Vector DB...")
    stage1_candidates = get_stage1_candidates(user_query, top_k=top_k_stage1)
    
    # Edge Case 2: Deduplicate identical chunks
    seen_contents = set()
    unique_candidates = []
    for doc, score in stage1_candidates:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_candidates.append((doc, score))
            
    print(f" -> Retreived {len(stage1_candidates)} candidate chunks ({len(unique_candidates)} unique).")

    # Edge Case 3: Zero Candidates Retrieved
    if not unique_candidates:
        print(" [Warning] No relevant candidates found in Vector Store.")
        return {
            "query": user_query,
            "prompt": "Context unavailable. No documents matched your query.",
            "citations": {},
            "contexts": []
        }

    # 2. Stage-2 Reranking
    print(f"\n[Stage 2] Reranking Candidates using method: \"{rerank_method}\"...")
    reranked_docs = rerank_candidates(user_query, unique_candidates, top_k=top_k_stage2, method=rerank_method)
    print(f" -> Selected Top-{len(reranked_docs)} most relevant contexts.")

    # 3. Prompt Construction & Citations
    print("\n[Stage 3] Formatting Context & Building Grounded Prompt...")
    final_prompt, citation_map = build_rag_prompt(user_query, reranked_docs)

    print("\n==================== GROUNDED RAG PROMPT ====================")
    print(final_prompt)
    print("=============================================================")

    print("\n=================== CITATION AUDIT LOG =====================")
    for doc_id, meta in citation_map.items():
        source_name = meta.get("source", "Unknown")
        score_val = meta.get("relevance_score", 0.0)
        print(f" {doc_id} | Source: {source_name} | Score: {score_val:.4f}")
    print("=============================================================\n")

    return {
        "query": user_query,
        "prompt": final_prompt,
        "citations": citation_map,
        "contexts": reranked_docs
    }

if __name__ == "__main__":
    query = "What is the employee onboarding process?"
    run_rag_pipeline(query)
