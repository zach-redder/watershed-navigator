import streamlit as st
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.ingest import preload_documents
from src.rag.embedding_store import search
from src.rag.prompt import format_prompt
from src.models.llama_wrapper import ask_llama

st.set_page_config(
    page_title="Watershed Navigator",
    page_icon="🌊",
)

if not os.path.exists("store/embeddings.pkl"):
    with st.spinner("Preloading documents..."):
        preload_documents()

def format_answer(text: str, max_sentences_per_block=2) -> str:
    # Check for signs of a list
    has_numbered_list = re.search(r"^\s*\d+\.\s", text, re.MULTILINE)
    has_bulleted_list = re.search(r"^\s*[-•–]\s", text, re.MULTILINE)

    if has_numbered_list or has_bulleted_list:
        # Skip formatting if it's a list
        return text

    # Otherwise, run normal paragraph formatter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = [
        ' '.join(sentences[i:i + max_sentences_per_block])
        for i in range(0, len(sentences), max_sentences_per_block)
    ]
    return '\n\n'.join(chunks)


st.title("Watershed Navigator")
question = st.text_input("Ask your question:")

top_docs = []

if st.button("Get Answer") and question:
    top_docs = search(question, k=3)

is_relevant = bool(top_docs)

if not is_relevant:
    # Optional: Check if question contains keywords like "stormwater", "watershed", etc.
    relevant_keywords = [
        "stormwater", "watershed", "pollutant", "tmdl", "runoff", "hydrology", "modeling",
        "bacteria", "phosphorus", "green infrastructure", "riparian", "ms4", "nbs",
        "sediment", "sampling", "retention", "mitigation", "low-impact development", "swmm", "compliance"
    ]  
    if any(word in question.lower() for word in relevant_keywords):
        # Relevant, but not in docs → fall back to general LLM
        st.markdown("### Answer - General Response")
        answer = ask_llama(question)  # Skip doc context, just answer normally
        st.write(format_answer(answer))
    else:
        # Irrelevant topic → reject
        st.markdown("### No Answer")
        st.warning("That topic doesn't appear in the documents or relate to the domain of this assistant.")
else:
    # Relevant AND in docs → RAG as normal
    context = "\n\n".join([doc["text"] for doc in top_docs])
    prompt = format_prompt(context, question)
    answer = ask_llama(prompt)

    st.markdown("### Answer - RAG Response")
    st.write(format_answer(answer))

    st.markdown("### Sources")
    unique_sources = {os.path.basename(doc['source']) for doc in top_docs}
    for source in unique_sources:
        st.markdown(f"- `{source}`")