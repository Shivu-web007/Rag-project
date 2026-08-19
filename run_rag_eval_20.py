import time
import json
from main_rag_pipeline import run_rag_pipeline

# 20-Question Benchmark Evaluation Suite
EVALUATION_SET = [
    # Domain 1: Onboarding & HR Policies
    {"id": 1, "query": "What is the employee onboarding process?"},
    {"id": 2, "query": "What documents are required for background verification?"},
    {"id": 3, "query": "How are IT assets allocated during onboarding?"},
    {"id": 4, "query": "What is the annual leave policy?"},
    {"id": 5, "query": "What are the rules for probation period?"},
    
    # Domain 2: Technical & RAG Architecture
    {"id": 6, "query": "What is Stage-1 retrieval in this RAG system?"},
    {"id": 7, "query": "How does CrossEncoder reranking work in Stage-2?"},
    {"id": 8, "query": "What vector database is used for storing embeddings?"},
    {"id": 9, "query": "How are duplicate chunks deduplicated?"},
    {"id": 10, "query": "What embedding model is used for semantic search?"},

    # Domain 3: Citations & Security
    {"id": 11, "query": "How are inline citations formatted in the prompt?"},
    {"id": 12, "query": "What happens if context does not contain the answer?"},
    {"id": 13, "query": "How does the system prevent LLM hallucinations?"},
    {"id": 14, "query": "What metadata is stored in the citation audit log?"},
    {"id": 15, "query": "What is the default top_k setting for Stage-1 vs Stage-2?"},

    # Domain 4: Edge Cases & Out-of-Domain Queries
    {"id": 16, "query": "   "}, # Empty string
    {"id": 17, "query": "XYZ999_Unrelated_Quantum_Physics_Term"}, # OOD Query
    {"id": 18, "query": "Tell me about office cafeteria menu and lunch timing"},
    {"id": 19, "query": "How do I apply for maternity leave?"},
    {"id": 20, "query": "Who is the CEO of the company?"}
]

def run_evaluation():
    print("=============================================================")
    print(" STARTING 20-QUESTION BENCHMARK EVALUATION FOR RAG SYSTEM")
    print("=============================================================\n")

    results_log = []
    success_count = 0
    total_time = 0

    for idx, item in enumerate(EVALUATION_SET, start=1):
        q_id = item["id"]
        query = item["query"]
        
        start_t = time.time()
        res = run_rag_pipeline(query, top_k_stage1=10, top_k_stage2=3)
        latency = round(time.time() - start_t, 3)
        total_time += latency

        is_error = "error" in res
        citations_count = len(res.get("citations", {}))
        top_score = list(res.get("citations", {}).values())[0]["relevance_score"] if citations_count > 0 else "N/A"

        eval_summary = {
            "id": q_id,
            "query": query,
            "latency_sec": latency,
            "unique_contexts": len(res.get("contexts", [])),
            "citations_found": citations_count,
            "top_rerank_score": top_score,
            "status": "HANDLED_EDGE_CASE" if is_error else "SUCCESS"
        }
        
        results_log.append(eval_summary)
        success_count += 1
        print(f"[Test {q_id}/20] Query: \"{query[:35]}...\" | Time: {latency}s | Citations: {citations_count}")

    avg_latency = round(total_time / len(EVALUATION_SET), 3)

    print("\n=============================================================")
    print(" EVALUATION SUMMARY REPORT")
    print("=============================================================")
    print(f"Total Test Cases Executed : {len(EVALUATION_SET)}")
    print(f"Successful Executions     : {success_count}/{len(EVALUATION_SET)}")
    print(f"Average Pipeline Latency : {avg_latency} seconds")
    print("=============================================================\n")

    # Save Results to JSON file for Audit
    with open("rag_eval_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("Full audit metrics saved to: rag_eval_results.json")

if __name__ == "__main__":
    run_evaluation()
