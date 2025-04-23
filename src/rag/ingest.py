import os
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.format import chunk_text

CHUNK_SIZE = 500
OVERLAP = 50

def process_csv(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    return [row.to_json() for _, row in df.iterrows()]

def ingest_file(file_path: str) -> List[Dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        return [{"text": chunk, "source": file_path} for chunk in chunks]
    elif ext == ".csv":
        rows = process_csv(file_path)
        return [{"text": row, "source": file_path} for row in rows]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
