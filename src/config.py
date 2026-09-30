import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Supported models on Groq
LLM_MODEL = "llama-3.1-8b-instant"
LLM_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.2-11b-vision-instruct",
    "llama-3.2-3b-preview",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150