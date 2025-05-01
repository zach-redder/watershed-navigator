# Watershed Navigator

Watershed Navigator is a local RAG-based AI assistant for answering environmental questions using document context.

## Features
- Embeds and indexes watershed-related PDFs (e.g., EPA, SEMCOG reports)
- Retrieves relevant chunks using cosine similarity
- Generates grounded responses using a local TinyLLaMA model
- Rejects irrelevant questions (e.g., sports, pop culture)

## Technologies
- Streamlit (UI)
- SentenceTransformers (MiniLM for embedding)
- TinyLLaMA via Ollama (Hosted locally)

## How to Run

1. Place your PDFs in `data/preloaded/`
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Start the app:
```bash
streamlit run ui/app_ui.py
```
3. Ask a question for the company LimnoTech related to stormwater, TMDLs, infrastructure, etc.

## Project Files
- `src/` - ingestion, embedding, and retrieval logic along with model wrappers
- `ui/app_ui.py` - main Streamlit interface
- `store/embeddings.pkl` - vector store
- `report/watershed_navigator_report.ipynb` - technical notebook
- `README.md` - this file

## Example/Test Questions
- What are the ecological risks of unrefined liquid hydrocarbons?
- What BMPs are recommended in the Cannon River watershed?
- How does SEMCOG define effective impervious area?
- Who won the 2024 NBA championship?