import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, persist_directory="./data/chroma_db"):
        print("[Retriever] Initializing embeddings and vector store...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Fetch all stored documents for BM25 indexing
        all_data = self.vector_store.get()
        self.docs = all_data.get("documents", [])
        self.metadatas = all_data.get("metadatas", [])
        
        # Tokenize documents for BM25 sparse search
        tokenized_docs = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Initialize Cross-Encoder reranker
        print("[Retriever] Loading Cross-Encoder reranker model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def search(self, query: str, top_k: int = 2):
        print(f"\n[Retriever] Executing hybrid search for query: '{query}'")
        
        # 1. Dense Vector Search
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        dense_texts = [doc.page_content for doc in vector_results]
        
        # 2. Sparse BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = bm25_scores.argsort()[::-1][:top_k]
        sparse_texts = [self.docs[i] for i in top_bm25_idx if i < len(self.docs)]
        
        # Combine and de-duplicate candidates
        combined_candidates = list(set(dense_texts + sparse_texts))
        print(f"[Retriever] Retrieved {len(combined_candidates)} unique candidates from hybrid pool.")

        # 3. Rerank candidates using Cross-Encoder
        pairs = [[query, candidate] for candidate in combined_candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort candidates by reranker score descending
        scored_candidates = sorted(zip(combined_candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Return top_k highest precision results
        final_results = [item[0] for item in scored_candidates[:top_k]]
        print(f"[Retriever] Reranking complete. Selected top {len(final_results)} chunks for generation.")
        return final_results

if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.search("How do Iron Mountain data pipelines run?")
    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} ---\n{res}")
