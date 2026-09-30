import streamlit as st
from src.graph import build_rag_graph

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI eBook RAG Assistant")
st.markdown("Ask any question regarding the **Agentic AI eBook**. Answers are strictly grounded in retrieved chunks.")

@st.cache_resource
def load_graph():
    return build_rag_graph()

try:
    graph = load_graph()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

query = st.text_input("Enter your query:", placeholder="e.g., What is the core definition of Agentic AI?")

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating grounded response..."):
            initial_state = {
                "question": query,
                "context": [],
                "answer": "",
                "score": 0.0
            }
            
            try:
                result = graph.invoke(initial_state)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("💡 Final Answer")
                    st.write(result["answer"])
                    st.metric(
                        label="Confidence Score",
                        value=f"{result['score'] * 100:.1f}%"
                    )

                with col2:
                    st.subheader("📚 Retrieved Chunks")
                    if result["context"]:
                        for idx, chunk in enumerate(result["context"], 1):
                            with st.expander(f"Chunk #{idx}"):
                                st.write(chunk)
                    else:
                        st.info("No context chunks retrieved.")

            except Exception as err:
                st.error(f"Error processing request: {err}")