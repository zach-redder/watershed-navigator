import chromadb
from chromadb.config import Settings
from typing import List, Dict

from src.rag.embed import get_embedding  # ✅ Use our refactored embed logic

# ChromaDB setup
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_store",  # folder for DB
    anonymized_telemetry=False
))

# Collection = like a table in Chroma
collection = chroma_client.get_or_create_collection(name="env_docs")

def embed_and_store(docs: List[Dict]):
    for i, doc in enumerate(docs):
        text = doc["text"]
        embedding = get_embedding(text)
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"source": doc["source"]}],
            ids=[f"doc-{i}"]
        )

def get_top_k_docs(query: str, k: int = 3) -> List[Dict]:
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    docs = [
        {"text": results["documents"][0][i], "source": results["metadatas"][0][i]["source"]}
        for i in range(len(results["documents"][0]))
    ]
    return docs
