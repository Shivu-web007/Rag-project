from pathlib import Path

from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ==========================================================
# 1. Document folder
# ==========================================================

DOCUMENTS_DIR = Path(__file__).parent

print("Documents folder:", DOCUMENTS_DIR)


# ==========================================================
# 2. Read PDF and TXT files
# ==========================================================

documents = []

for file_path in DOCUMENTS_DIR.iterdir():

    if file_path.name.endswith("_backup.py"):
        continue

    # PDF
    if file_path.suffix.lower() == ".pdf":

        print(f"\nReading PDF: {file_path.name}")

        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""
            text = text.strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "document": file_path.name,
                            "source": file_path.name,
                            "page": page_number,
                            "section": "N/A",
                        },
                    )
                )

    # TXT
    elif file_path.suffix.lower() == ".txt":

        print(f"\nReading TXT: {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "document": file_path.name,
                        "source": file_path.name,
                        "page": "N/A",
                        "section": "N/A",
                    },
                )
            )


print("\nTotal documents loaded:", len(documents))


# ==========================================================
# 3. Create chunks
# ==========================================================

chunk_size = 500
overlap = 50

chunk_documents = []

for document in documents:

    text = document.page_content

    start = 0
    chunk_id = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            metadata = document.metadata.copy()
            metadata["chunk_id"] = chunk_id

            chunk_documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

            chunk_id += 1

        start = end - overlap


print("Total chunks:", len(chunk_documents))


if chunk_documents:
    print("\nFirst clean chunk:")
    print(chunk_documents[0].page_content)

print("Chunk size:", chunk_size)
print("Overlap:", overlap)


# ==========================================================
# 4. Create Embedding Model
# ==========================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ==========================================================
# 5. Store in ChromaDB
# ==========================================================

print("\nStoring documents in ChromaDB...")

vector_db = Chroma.from_documents(
    documents=chunk_documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print("\nStored in ChromaDB successfully!")


# ==========================================================
# 6. Test Search
# ==========================================================

question = input("\nAsk a question: ")

results = vector_db.similarity_search_with_score(
    question,
    k=3,
)

print("\nSearch Results:")

for index, (doc, score) in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Score:", score)
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Chunk ID:", doc.metadata.get("chunk_id"))
    print("Content:", doc.page_content)