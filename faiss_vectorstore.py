from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_faiss_index(documents, save_path="./faiss_db"):
    print("1. Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("2. Generating Embeddings & Creating FAISS Index...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    print(f"3. Saving FAISS Index at: {save_path}")
    vector_db.save_local(save_path)
    return vector_db

if __name__ == "__main__":
    sample_docs = [
        Document(page_content="SOP-01: Employee onboarding requires background verification.", metadata={"source": "sop.pdf"}),
        Document(page_content="Proposal 2026: Multi-document RAG system with two-stage retrieval.", metadata={"source": "rag.pdf"})
    ]
    print("--- Starting FAISS Indexing ---")
    build_faiss_index(sample_docs)
    print("--- FAISS Indexing Completed ---")
