from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.graph import build_rag_graph

app = FastAPI(
    title="Agentic AI RAG API",
    description="Grounded LangGraph RAG Service",
    version="1.0.0"
)

graph = build_rag_graph()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is Agentic AI?")

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float

@app.get("/")
def health_check():
    return {"status": "active", "message": "Agentic AI RAG API is live."}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    
    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)