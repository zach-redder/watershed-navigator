import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.ingest import ingest_file
from src.rag.retrieve import embed_and_store

PRELOAD_DIR = "data/preloaded"

all_docs = []

for filename in os.listdir(PRELOAD_DIR):
    filepath = os.path.join(PRELOAD_DIR, filename)
    if filepath.endswith(".pdf") or filepath.endswith(".csv"):
        print(f"🔍 Processing: {filename}")
        docs = ingest_file(filepath)
        all_docs.extend(docs)

print(f"✅ Ingested {len(all_docs)} chunks. Embedding now...")
embed_and_store(all_docs)
print("🎉 All documents embedded into ChromaDB!")