def build_rag_prompt(user_query, reranked_docs):
    """
    Reranked context se strict citation-tracked system prompt construct karta hai.
    """
    context_blocks = []
    citation_map = {}

    for idx, (doc, score) in enumerate(reranked_docs, start=1):
        doc_id = f"Doc {idx}"
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content.strip()

        # Build Context Block for LLM
        context_blocks.append(f"[{doc_id}] ({source})\n{content}")
        
        # Store metadata mapping for auditability
        citation_map[doc_id] = {
            "source": source,
            "relevance_score": float(score),
            "content_preview": content[:100] + "..."
        }

    formatted_context = "\n\n".join(context_blocks)

    system_prompt = f"""You are a precise and grounded AI Assistant. Answer the user question based ONLY on the provided Context below.

CRITICAL INSTRUCTIONS:
1. Every claim, fact, or statement you write MUST be immediately followed by its source citation bracket, e.g., [Doc 1] or [Doc 2].
2. If the answer cannot be found in the Context, respond strictly with: "I cannot answer based on the provided documents."
3. Do NOT use outside knowledge or hallucinate information.

CONTEXT:
----------------------------------------
{formatted_context}
----------------------------------------

USER QUESTION: {user_query}

STRUCTURED ANSWER (with inline citations):"""

    return system_prompt, citation_map

if __name__ == "__main__":
    from langchain_core.documents import Document
    sample_reranked = [
        (Document(page_content="Onboarding requires submitting background verification.", metadata={"source": "sop_01.pdf"}), 8.5),
        (Document(page_content="IT assets like laptop and email are provisioned on Day 1.", metadata={"source": "it_policy.pdf"}), 6.2)
    ]
    prompt, citations = build_rag_prompt("What happens during onboarding?", sample_reranked)
    print("--- GENERATED SYSTEM PROMPT ---")
    print(prompt)
    print("\n--- CITATION MAP ---")
    print(citations)
