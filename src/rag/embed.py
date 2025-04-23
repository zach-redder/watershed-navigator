from sentence_transformers import SentenceTransformer

# Load the embedding model once
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Good balance of speed and quality
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def get_embedding(text: str) -> list:
    """Get embedding vector for a single chunk of text."""
    return embedder.encode(text).tolist()

def get_embeddings(texts: list) -> list:
    """Batch encode multiple texts (faster)."""
    return embedder.encode(texts).tolist()
