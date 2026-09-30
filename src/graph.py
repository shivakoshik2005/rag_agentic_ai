import os
import json
from typing import List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import (
    GROQ_API_KEY,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    CHROMA_PERSIST_DIR,
)

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def get_active_groq_llm(api_key: str):
    """Attempt to instantiate a functional ChatGroq client by testing active model strings."""
    candidate_models = [LLM_MODEL] + LLM_FALLBACK_MODELS
    last_exception = None

    for model_name in candidate_models:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0,
                groq_api_key=api_key,
                max_retries=1
            )
            # Test invocation to verify model availability
            llm.invoke("Test ping")
            print(f"Successfully initialized Groq LLM with model: {model_name}")
            return llm
        except Exception as e:
            print(f"Model {model_name} failed: {e}. Trying fallback...")
            last_exception = e
            continue

    raise RuntimeError(f"Could not initialize any Groq model. Details: {last_exception}")

def build_rag_graph():
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please check your .env file or environment variables.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Instantiate LLM once during graph creation
    llm = get_active_groq_llm(api_key)

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict grounded AI assistant answering questions based strictly on the retrieved eBook context.

RULES:
1. Do NOT use outside knowledge.
2. If the context does not contain sufficient details to answer the question, state:
   "I cannot answer based on the provided document as the information is not present."
   and set confidence_score to 0.0.

Respond strictly in valid JSON format with no additional prose:
{{
  "final_answer": "your concise grounded answer here",
  "confidence_score": 0.95
}}

Retrieved Context:
{context_str}

User Question: {state['question']}"""

        res = llm.invoke(prompt)
        raw_response = res.content.strip()

        # Clean JSON markdown blocks if present
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()

        try:
            parsed = json.loads(raw_response)
            answer = parsed.get("final_answer", raw_response)
            score = float(parsed.get("confidence_score", 0.9))
        except Exception:
            answer = raw_response
            score = 0.85 if len(state["context"]) > 0 else 0.0

        return {
            "answer": answer,
            "score": score
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()