from src.rag.ingest import ingest_file
from src.rag.retrieve import embed_and_store, get_top_k_docs

docs = ingest_file("data/raw/Huron_Report.pdf")
embed_and_store(docs)

query = "What are some environmental concerns in the Huron River watershed?"
top_chunks = get_top_k_docs(query)

for i, chunk in enumerate(top_chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk["text"][:500])
