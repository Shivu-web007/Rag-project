from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def build_vector_store(chunks, persist_directory="./chroma_db"):
    print("1. Loading Embedding Model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("2. Generating Embeddings & Creating Chroma Vector Store...")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=persist_directory)
    
    print(f"3. Success! Vector Store saved at: {persist_directory}")
    return vector_db

if __name__ == "__main__":
    sample_chunks = [
        Document(page_content="SOP-01: Employee onboarding requires background verification.", metadata={"source": "sop.pdf"}),
        Document(page_content="Proposal 2026: Multi-document RAG system with two-stage retrieval.", metadata={"source": "rag.pdf"})
    ]
    print("--- Starting Vector Store Indexing ---")
    db = build_vector_store(sample_chunks)
    print("--- Completed Successfully ---")
