import os
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Disable parallel tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_day3_pipeline():
    print("--- Starting Vector Store Indexing ---")

    # 1. Load Embedding Model
    print("1. Loading Embedding Model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Prepare Sample Chunks & Metadata
    chunks_data = [
        {
            "text": "SOP-01: Employee onboarding requires background verification.",
            "source": "sop.pdf",
            "chunk_id": 0,
            "category": "HR"
        },
        {
            "text": "IT Asset Allocation: Laptops and security keys are issued on Day 1 after verification.",
            "source": "it_policy.pdf",
            "chunk_id": 1,
            "category": "IT"
        },
        {
            "text": "Leave Policy: Employees are entitled to 18 days of paid annual leave per calendar year.",
            "source": "hr_policy.pdf",
            "chunk_id": 2,
            "category": "HR"
        },
        {
            "text": "Proposal 2026: Multi-document RAG system with two-stage retrieval architecture.",
            "source": "rag.pdf",
            "chunk_id": 3,
            "category": "Tech"
        }
    ]

    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "category": item["category"]
            }
        )
        for item in chunks_data
    ]

    # 3. Generate Embeddings & Persist to ChromaDB
    persist_directory = "./chroma_db"
    print("2. Generating Embeddings & Creating Chroma Vector Store...")

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        ids=[str(uuid.uuid4()) for _ in documents]
    )

    print(f"3. Success! Vector Store saved at: {persist_directory}")
    print("--- Completed Successfully ---")

if __name__ == "__main__":
    run_day3_pipeline()