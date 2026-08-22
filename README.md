# RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) system that retrieves relevant information from documents, reranks the retrieved results, builds a grounded prompt, and generates a concise answer with source citation.

## RAG Pipeline

Documents
|
v
Preprocessing and Chunking
|
v
HuggingFace Embeddings
|
v
ChromaDB Vector Store
|
v
Stage 1 - Vector Retrieval
|
v
Stage 2 - Cross-Encoder Reranking
|
v
Stage 3 - Prompt Builder
|
v
Stage 4 - Final Answer
|
v
Source Citation

## Features

- PDF and TXT document ingestion
- Text extraction using PyPDF
- Text chunking with overlap
- HuggingFace sentence embeddings
- Persistent ChromaDB vector store
- Top-K semantic retrieval
- Cross-Encoder reranking
- Grounded prompt generation
- Source citation
- Unknown-question handling
- End-to-end RAG pipeline

## Technologies

- Python
- PyPDF
- LangChain
- HuggingFace
- Sentence Transformers
- ChromaDB
- Cross-Encoder

## Project Structure

project/
|
+-- documents/
|   +-- preprocessing.py
|   +-- sample.pdf
|   +-- test_document.txt
|
+-- chroma_db/
+-- stage1_retrieval.py
+-- stage2_reranker.py
+-- day6_prompt_builder.py
+-- stage4_answer_generator.py
+-- main_rag_pipeline.py
+-- run_rag_eval_20.py
+-- requirements.txt
+-- README.md

## Installation

Create virtual environment:

python -m venv venv

Activate:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## Run

python .\main_rag_pipeline.py

## Example

Question:

What is the onboarding process?

Answer:

The employee onboarding process involves the following steps:

1. The employee submits the required documents.
2. HR verifies the employee information.
3. Background verification is completed.
4. IT creates the employee account.
5. The employee attends the onboarding session.

Source: test_document.txt

## Unknown Information Handling

If the requested information is not available in the documents, the system returns:

Information not available in the provided documents.

## Evaluation

The pipeline was tested for:

- Onboarding information
- Leave policy
- IT support
- Source citation
- Unknown information handling
- End-to-end pipeline execution

## Future Improvements

- Integrate an external LLM
- Improve semantic chunking
- Add automated evaluation metrics
- Add a web interface
- Add conversation history
- Support additional document formats

## Author

Shivani Kakpure
