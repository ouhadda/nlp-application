import streamlit as st
from ingestion import Ingestor
from chat import RAGChat
from utils import save_uploaded_file, ensure_dirs
import os

# Initialize directories
ensure_dirs()

st.set_page_config(page_title="RAG Pedagogical Assistant", layout="wide")
st.title("NLP - RAG Application — Upload → Embed → Chat")

st.markdown(
    "Upload PDF / TXT / DOCX files, build a local FAISS vector DB, and chat with a pedagogical assistant "
    "that uses your documents (RAG) + an LLM for reasoning.\n\n"
    "Note: The pedagogical notebook you provided (for reference) is NOT ingested by this app."
)

# Initialize ingestor
ingestor = Ingestor(data_dir="data", embedding_model_name="all-MiniLM-L6-v2")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("1) Upload & Ingest files")
    uploaded = st.file_uploader(
        "Upload files (PDF, TXT, DOCX). You can upload multiple and ingest incrementally.",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded:
        target_paths = []
        for f in uploaded:
            path = save_uploaded_file(f, "uploads")
            target_paths.append(path)

        if st.button("Ingest uploaded files"):
            with st.spinner("Ingesting files..."):
                total_chunks = 0
                for p in target_paths:
                    added = ingestor.ingest_file(p)
                    total_chunks += added
                st.success(f"Ingestion finished — added {total_chunks} chunks.")
                st.rerun()

    st.markdown("---")
    st.header("2) Index status & management")
    st.write(f"Vectors in index: **{ingestor.get_num_vectors()}**")
    if ingestor.index_exists():
        st.success("FAISS index: present")
        if st.button("Rebuild index from scratch (reindex all uploads)"):
            with st.spinner("Rebuilding index..."):
                ingestor.rebuild_index_from_uploads("uploads")
            st.success("Rebuilt index.")
            st.rerun()
    else:
        st.warning("No FAISS index yet. Upload files and click 'Ingest uploaded files'.")

with col_right:
    st.header("3) Settings")
    st.write("Embedding model:", ingestor.model_name)
    top_k = st.slider("RAG: # retrieved chunks", min_value=1, max_value=8, value=3)
    use_openai = st.checkbox("Use OpenAI API (if OPENAI_API_KEY in env)", value=False)
    hf_model = st.selectbox("Local HF generator (fallback)", ["google/flan-t5-small", "google/flan-t5-base"])

st.markdown("---")
st.header("Chat with your documents")

# Chat system
chat = RAGChat(ingestor=ingestor, use_openai=use_openai, hf_model_name=hf_model)

if "history" not in st.session_state:
    st.session_state["history"] = []

# Display history
for turn in st.session_state["history"]:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['text']}")
    else:
        st.markdown(f"**Assistant:** {turn['text']}")

# Input
question = st.text_area("Ask a question about your uploaded documents or general NLP/LLM concepts:", height=140)
submit = st.button("Ask")

if submit and question.strip():
    st.session_state["history"].append({"role": "user", "text": question})
    with st.spinner("Retrieving + generating answer..."):
        answer, retrieved = chat.answer_question(question, top_k=top_k, chat_history=st.session_state["history"])
    st.session_state["history"].append({"role": "assistant", "text": answer})

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Retrieved chunks (RAG sources)"):
        for i, r in enumerate(retrieved):
            st.write(f"**{i+1}. Source:** {r.get('source','unknown')} — score: {r.get('score', None)}")
            st.write(r.get("text",""))

    st.download_button("Download conversation (JSON)", data=str(st.session_state["history"]), file_name="conversation.json")

st.sidebar.header("About")
st.sidebar.write("This app builds a local FAISS vector store and supports incremental ingestion.")
st.sidebar.write("For generation: if you have an OpenAI key, enable OpenAI in settings; otherwise a local Flan-T5 model will be used as fallback.")
