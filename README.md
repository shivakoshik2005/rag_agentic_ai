Here is a complete **`README.md`** tailored for your project repository (`rag_agentic_ai`). It reflects your architecture—**LangGraph, ChromaDB, HuggingFace local embeddings, Groq LLM, FastAPI, and Streamlit**—with zero external paid API dependencies.

---

# 🤖 Agentic AI - Grounded RAG System

A **100% Free & Local-First Retrieval-Augmented Generation (RAG)** pipeline powered by **LangGraph**, **ChromaDB**, **HuggingFace Embeddings**, and **Groq Cloud API**.

This repository implements a strict grounded QA system over enterprise technical eBooks, ensuring responses are derived solely from retrieved context and eliminating hallucinations.

---

## 📌 Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Document Ingestion      │
                          │   (PyPDF + Recursive      │
                          │   Character Splitter)     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   HuggingFace Embeddings  │
                          │ (all-MiniLM-L6-v2 Local)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    Chroma Vector Store    │
                          │       (./chroma_db)       │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                       LangGraph Orchestration                         │
    │                                                                       │
    │  ┌───────────────────────┐             ┌───────────────────────────┐  │
    │  │     Retrieve Node     │ ──────────► │       Generate Node       │  │
    │  │  (Top-k Vector Search)│             │ (Groq / Structured JSON)  │  │
    │  └───────────────────────┘             └───────────────────────────┘  │
    └──────────────────────────────────┬────────────────────────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼                                   ▼
        ┌──────────────────────────┐       ┌──────────────────────────┐
        │   FastAPI REST Backend   │       │   Streamlit Interactive  │
        │      (http://.../chat)   │       │       Web Frontend       │
        └──────────────────────────┘       └──────────────────────────┘

```

---

## ✨ Features

* **Strictly Grounded QA:** Uses tailored prompt constraints and structured JSON parsing to output accurate answers with confidence scoring. Out-of-context queries return an explicit grounded fallback response.
* **Auto-Healing LLM Engine:** Built-in dynamic Groq model discovery and fallback logic to gracefully handle model deprecations or region-restricted endpoints.
* **Zero Embedding Costs:** Utilizes open-source HuggingFace `all-MiniLM-L6-v2` locally for zero API cost.
* **LangGraph State Management:** Explicit control over the RAG execution graph for full observability and state inspection.
* **Dual Interfaces:** Provides both a production-ready **FastAPI** RESTful API and a lightweight **Streamlit** Web Interface.

---

## 📁 Repository Structure

```
rag_agentic_ai/
│
├── data/                       # Document storage directory
│   └── Ebook-Agentic-AI.pdf    # Source PDF for RAG ingestion
│
├── src/                        # Core application logic
│   ├── __init__.py
│   ├── config.py               # Central environment configurations & parameters
│   ├── ingestion.py            # PDF loading, chunking, & ChromaDB vector storage
│   └── graph.py                # LangGraph state graph definition & LLM generation
│
├── chroma_db/                  # Persistent ChromaDB vector database directory
├── app.py                      # FastAPI web server endpoint
├── streamlit_app.py            # Streamlit interactive Web UI
├── tests_sample_queries.py     # Automated CLI test suite & query evaluator
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API Keys)
└── README.md                   # Project documentation

```

---

## ⚙️ Prerequisites & Setup

### 1. Requirements

* **Python 3.10+** (Tested on Python 3.13 / macOS Apple Silicon)
* A free **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/))

### 2. Environment Configuration

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here

```

### 3. Installation

Clone the repository and install the required dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/rag_agentic_ai.git
cd rag_agentic_ai

# Create virtual environment (Optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Run Document Ingestion

Place your source PDF file inside `data/Ebook-Agentic-AI.pdf` and run the ingestion pipeline:

```bash
python3 -m src.ingestion

```

*Output:* Chunks the document into overlapping segments, generates local vector embeddings, and persists them into `./chroma_db`.

---

### Step 2: Run Automated Evaluation Suite

Test the LangGraph execution flow against sample queries to verify answer grounding and confidence scoring:

```bash
python3 tests_sample_queries.py

```

---

### Step 3: Launch the FastAPI Backend Server

Start the production RESTful API:

```bash
python3 app.py

```

* The API will start on **`http://localhost:8000`**
* Interactive Swagger API Documentation is available at **`http://localhost:8000/docs`**

#### Sample API Request:

```bash
curl -X 'POST' \
  'http://localhost:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the core definition of Agentic AI?"}'

```

---

### Step 4: Launch Streamlit Web UI

Start the web interface for user interaction:

```bash
streamlit run streamlit_app.py

```

* Open your browser at **`http://localhost:8501`** to interact with the application.

---

## 🛠️ Configuration Parameters

Key settings can be modified inside `src/config.py`:

| Variable | Default Value | Description |
| --- | --- | --- |
| `LLM_MODEL` | `"llama-3.3-70b-versatile"` | Primary Groq model used for generation |
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace local embedding model |
| `CHUNK_SIZE` | `800` | Token/character size per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent text chunks |
| `CHROMA_PERSIST_DIR` | `"./chroma_db"` | Path to local persistent vector store |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more details.