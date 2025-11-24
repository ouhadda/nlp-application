# Streamlit RAG Application (Local FAISS + LLM)

A lightweight **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, supporting dynamic ingestion of user-uploaded documents and conversational querying using either **local HuggingFace models** or **OpenAI API**.

This project is designed as an **educational yet production-ready** template for modern NLP & LLM pipelines:
chunking → embeddings → vector search → retrieval → generation.

---

## Features

### **Document Ingestion**

* Upload **PDF**, **TXT**, or **DOCX** files directly from the Streamlit sidebar.
* Files are:

  * Extracted to raw text
  * Chunked into overlapping segments
  * Converted into vector embeddings
  * Stored in a **local FAISS vector database**
* Ingestion is **incremental** — add files anytime, without restarting the app.

### **RAG Pipeline**

* Fast similarity search using **FAISS**.
* Retrieves the most relevant chunks.
* Passes retrieved context + user query to an LLM for final answer synthesis.

### **LLM Choices**

You can choose between:

1. **OpenAI API (if `OPENAI_API_KEY` is set)**
2. **Local HuggingFace model (default fallback)**

   * Uses *Flan-T5* as an example
   * Runs fully offline

### **Conversational Interface**

* Chat-like UI with message memory
* Answers are **pedagogical**, clean, and based on retrieved document knowledge
* Shows retrieved context for transparency/debugging

---

## Installation

Clone and install dependencies:

```bash
git clone https://github.com/ouhadda/nlp-application.git
cd https://github.com/ouhadda/nlp-application.git

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

---

## How to Use

### **1. Upload Documents**

In the left sidebar:

* Upload one or multiple PDF/TXT/DOCX files
* Click **"Ingest uploaded files"**

This will:

* Extract text
* Chunk it
* Embed it
* Append vectors to FAISS index

### **2. Ask Questions**

Use the chat box at the bottom:

* Ask anything related to the uploaded documents
* The system retrieves the top-k relevant chunks and generates a response

### **3. Choose LLM Backend**

In the sidebar:

* Check **Use OpenAI API** to enable OpenAI Chat Completions

  * Requires `OPENAI_API_KEY` in your environment
* Otherwise the app automatically switches to the included local model

---

## Project Structure

```
.
├── app.py     # Main UI
├── ingestion.py         # File loading, text extraction, chunking, embeddings, FAISS handling
├── rag.py               # Retrieval logic
├── chat.py              # Chat, memory, LLM calling
├── utils.py             # Helper utilities
├── faiss_store/         # Automatically created FAISS index + metadata
├── requirements.txt
└── README.md
```

---

## Configurable Parameters

You can adjust:

* Chunk size / overlap
* Embedding model
* Number of retrieved chunks
* Local model name
* FAISS index directory

All located in the script headers or constants blocks.

---

## Notes

* Local models may require several gigabytes of RAM.
* For heavy usage, consider:

  * Switching to **ChromaDB** instead of FAISS
  * Using a quantized model for faster generation
  * Adding authentication or file deletion endpoints

---

## Educational Value

This project demonstrates the full workflow of a modern RAG system:

1. Document ingestion
2. Chunking
3. Embedding generation
4. Vector-based similarity search
5. Retrieval of relevant segments
6. LLM-based answer synthesis

It is intentionally written in a **clean, modular, readable** way so you can extend it into advanced RAG functionality.

---