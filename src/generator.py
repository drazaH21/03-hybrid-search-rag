import os
from groq import Groq
from dotenv import load_dotenv
from src.retriever import HybridRetriever

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.retriever = HybridRetriever()

    def answer_query(self, query: str) -> str:
        """
        Performs hybrid retrieval, reranking, and generation using Groq.
        """
        print(f"\n[RAG Pipeline] Processing query: '{query}'")
        
        # 1. Retrieve top reranked chunks
        relevant_chunks = self.retriever.search(query, top_k=2)
        context = "\n\n".join(relevant_chunks)
        
        # 2. Formulate prompt with retrieved context
        prompt = (
            f"You are an expert enterprise AI assistant. Answer the user's question accurately "
            f"using ONLY the provided context below. If the answer cannot be found in the context, state so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # 3. Generate response via Groq
        print("[RAG Pipeline] Generating response via Groq (openai/gpt-oss-20b)...")
        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        return answer

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "How do Iron Mountain data pipelines run?"
    ans = rag.answer_query(query)
    print(f"\n====================\nANSWER:\n{ans}\n====================")
