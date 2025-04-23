import streamlit as st
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.rag.ingest import ingest_file
from src.rag.retrieve import embed_and_store, get_top_k_docs
from src.rag.prompt import format_prompt
from src.models.llama_wrapper import ask_llama

st.set_page_config(page_title="Watershed Navigator")
st.title("Watershed Navigator")
st.markdown("Ask a question about environmental issues and problems you're facing.")

# File upload
uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])

# Store & embed the uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"Uploaded: {uploaded_file.name}")
    st.write("📥 Ingesting and embedding...")
    
    try:
        docs = ingest_file(file_path)
        embed_and_store(docs)
        st.success("✅ File embedded successfully!")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

# Ask a question
question = st.text_input("Ask your question:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        try:
            top_docs = get_top_k_docs(question, k=3)
            context = "\n\n".join([doc["text"] for doc in top_docs])
            prompt = format_prompt(context, question)
            answer = ask_llama(prompt)

            st.markdown("### 🤖 Answer")
            st.write(answer)

            with st.expander("📚 Sources"):
                for doc in top_docs:
                    st.markdown(f"- From `{doc['source']}`")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
