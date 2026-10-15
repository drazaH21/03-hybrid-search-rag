import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

SAMPLE_DOCS = [
    Document(
        page_content="Iron Mountain's global revenue reporting pipeline processes automated P&L bridges using Python, BigQuery, and Looker. Data pipelines run daily at 02:00 UTC.",
        metadata={"source": "Data Engineering Handbook", "category": "architecture"}
    ),
    Document(
        page_content="Concession scoring engines utilize machine learning heuristics to score retention risks. Models are retrained bi-weekly using DuckDB and Polars for fast local processing.",
        metadata={"source": "AI Engineering Specs", "category": "ml-models"}
    ),
    Document(
        page_content="Project 3 Hybrid RAG architecture combines BM25 sparse keyword search with dense vector embeddings via sentence-transformers, followed by a Cross-Encoder reranking stage.",
        metadata={"source": "Advanced RAG Guidelines", "category": "architecture"}
    ),
    Document(
        page_content="Groq LPU inference accelerators provide ultra-low latency generation for agentic workflows using models like openai/gpt-oss-20b and llama-3.3-70b-versatile.",
        metadata={"source": "Infrastructure Documentation", "category": "llm-infra"}
    )
]

def create_vector_store(docs=SAMPLE_DOCS, persist_directory="./data/chroma_db"):
    print("[Ingestion] Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"[Ingestion] Indexing {len(docs)} documents into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"[Ingestion] Vector store successfully created at {persist_directory}")
    return vector_store

if __name__ == "__main__":
    create_vector_store()
