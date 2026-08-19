from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def query_faiss(query, index_path="./faiss_db", top_k=5):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    
    results = vector_db.similarity_search_with_score(query, k=top_k)
    return results

if __name__ == "__main__":
    test_query = "What is the onboarding process?"
    print(f"--- FAISS Query Test: \"{test_query}\" ---")
    results = query_faiss(test_query)
    
    for idx, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "N/A")
        print(f"Rank {idx} [Score: {score:.4f}]: {doc.page_content} (Source: {src})")
