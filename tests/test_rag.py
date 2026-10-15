from src.generator import RAGPipeline

def test_hybrid_rag():
    print("Initializing Production Hybrid RAG Pipeline...")
    rag = RAGPipeline()
    
    test_queries = [
        "How do Iron Mountain data pipelines run?",
        "What models are used for Groq LPU inference accelerators?",
        "What architecture does Project 3 use for search?",
        "What is the best recipe for making traditional Italian wood-fired pizza dough?" # Out-of-box test
    ]

    for q in test_queries:
        print(f"\n--------------------------------------------------")
        answer = rag.answer_query(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

if __name__ == "__main__":
    test_hybrid_rag()
