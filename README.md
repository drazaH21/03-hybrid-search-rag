# 03-Hybrid-Search-RAG

A production-grade, advanced Retrieval-Augmented Generation (RAG) pipeline designed for enterprise technical documentation. This system implements a **Hybrid Search architecture** combining sparse keyword retrieval (BM25) with dense vector embeddings, followed by a **Cross-Encoder reranking** stage for high-precision context filtering, powered by **Groq** for ultra-low-latency generation.

---

## Architecture

```
[User Query] 
     │
     ├──────────────────────┐
     ▼                      ▼
[Sparse BM25 Search]   [Dense Vector Search (ChromaDB + HuggingFace)]
     │                      │
     └──────────┬───────────┘
                ▼
     [Hybrid Candidate Pool (De-duplicated)]
                │
                ▼
     [Cross-Encoder Reranker Node (ms-marco-MiniLM-L-6-v2)]
                │
                ▼
     [Top-K Precision Chunks Context]
                │
                ▼
     [Groq LLM Generation (openai/gpt-oss-20b)] -> [Grounded Answer with Refusal Guardrail]
```

---

## Key Features

* **Hybrid Retrieval:** Blends BM25 exact-keyword matching with sentence-transformer semantic vector search to prevent missing technical terms, error codes, or domain-specific names.
* **Cross-Encoder Reranking:** Re-scores the retrieved candidate pool using a powerful `ms-marco-MiniLM-L-6-v2` cross-encoder to ensure the LLM only views the highest-precision context.
* **Grounding & Guardrails:** Strict prompt-engineering guarantees that out-of-box or unrelated queries trigger clean refusals rather than hallucinations.
* **Ultra-Low Latency:** Uses Groq LPU inference acceleration (`openai/gpt-oss-20b`) for rapid end-to-end response times.

---

## Project Structure

```text
03-hybrid-search-rag/
│
├── data/
│   └── chroma_db/             # Local Chroma vector database storage
├── src/
│   ├── __init__.py
│   ├── ingest.py              # Document chunking and embedding indexer
│   ├── retriever.py           # BM25 + Dense Hybrid search & Cross-Encoder reranker
│   └── generator.py           # Groq LLM generation client & RAG orchestrator
├── tests/
│   └── test_rag.py            # End-to-end test suite (including out-of-box grounding tests)
├── .env                       # Environment variables (Groq API key)
├── requirements.txt           # Project dependencies
└── README.md
```

---

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration & Frameworks:** LangChain, ChromaDB, HuggingFace Embeddings
* **Sparse Search:** `rank_bm25` (Okapi BM25)
* **Reranking:** Sentence-Transformers Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **LLM Engine:** Groq API (`openai/gpt-oss-20b`)
* **Data Validation:** Pydantic

---

## Quickstart Instructions

### 1. Clone & Set Up Virtual Environment
```powershell
git clone https://github.com/drazaH21/03-hybrid-search-rag.git
cd 03-hybrid-search-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run Document Ingestion
Initialize and populate the local ChromaDB vector store:
```powershell
python src/ingest.py
```

### 5. Run the End-to-End Test Suite
Execute the test suite to verify hybrid search precision, reranking, and out-of-box grounding guardrails:
```powershell
python -m tests.test_rag
```

---

## Evaluation & Testing Results

| Test Query | Retrieved Candidates | Top Reranked Context Match | Final System Behavior |
| :--- | :--- | :--- | :--- |
| **"How do Iron Mountain data pipelines run?"** | Data Engineering & Infra chunks | Data Engineering Handbook chunk | Accurately extracted daily 02:00 UTC schedule and Python/BigQuery/Looker stack. |
| **"What models are used for Groq LPU?"** | Infra Documentation chunk | Infrastructure Documentation chunk | Listed exact model identifiers (`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`). |
| **"What is the best pizza dough recipe?"** | Mixed technical chunks | None relevant | **Successfully Refused:** Stated context contained no recipe information, avoiding hallucinations. |

---

## License
Distributed under the MIT License. See `LICENSE` for more information.