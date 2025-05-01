from pathlib import Path
from src.rag.embedding_store import build_index
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.format import chunk_text
import pandas as pd

def preload_documents(folder="data"):
    docs = []

    for file in Path(folder).glob("*"):
        if file.suffix == ".pdf":
            text = extract_text_from_pdf(str(file))
            chunks = chunk_text(text, chunk_size=1000)
        elif file.suffix == ".csv":
            df = pd.read_csv(file)
            chunks = [row.to_json() for _, row in df.iterrows()]
        else:
            continue

        docs.extend([{"text": chunk, "source": str(file)} for chunk in chunks])

    build_index(docs)
