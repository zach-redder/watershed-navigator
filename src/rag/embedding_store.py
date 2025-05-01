import os
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

EMBEDDING_FILE = "store/embeddings.pkl"
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    return model.encode([text])[0]

def save_embeddings(docs: List[Dict]):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(docs, f)
    print("Embeddings saved.")

def load_embeddings() -> List[Dict]:
    if not os.path.exists(EMBEDDING_FILE):
        return []
    with open(EMBEDDING_FILE, "rb") as f:
        return pickle.load(f)

def build_index(docs: List[Dict]):
    for doc in docs:
        doc["embedding"] = embed_text(doc["text"])
    save_embeddings(docs)

def search(query: str, k: int = 3, threshold: float = 0.5) -> List[Dict]:
    docs = load_embeddings()
    if not docs:
        return []

    query_vec = embed_text(query).reshape(1, -1)
    doc_vecs = np.array([doc["embedding"] for doc in docs])
    scores = cosine_similarity(query_vec, doc_vecs)[0]
    top_indices = scores.argsort()[::-1][:k]

    print(f"Top similarity score: {scores[top_indices[0]]:.3f}")

    # Check if top match is strong enough
    if scores[top_indices[0]] < threshold:
        return []  # Consider it unrelated

    return [docs[i] for i in top_indices]