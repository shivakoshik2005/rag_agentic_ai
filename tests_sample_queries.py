import json
from src.graph import build_rag_graph

TEST_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?"  # Out-of-bounds test
]

def run_tests():
    print("Initializing LangGraph execution graph for testing...\n")
    graph = build_rag_graph()

    for idx, query in enumerate(TEST_QUERIES, 1):
        print("==================================================")
        print(f"Test Query #{idx}: {query}")
        print("==================================================")
        
        initial_state = {"question": query, "context": [], "answer": "", "score": 0.0}
        result = graph.invoke(initial_state)

        payload = {
            "query": query,
            "final_answer": result["answer"],
            "retrieved_context_chunks": result["context"],
            "confidence_score": result["score"]
        }

        print(json.dumps(payload, indent=2))
        print("\n")

if __name__ == "__main__":
    run_tests()