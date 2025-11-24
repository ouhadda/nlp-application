# RAG Streamlit App

## Setup
1. Create a Python virtual environment.
2. Install requirements:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run streamlit_app.py

## Notes
- The app stores vectors in ./data/faiss_index.faiss and metadata in ./data/metadata.json.
- Ingestion is incremental: use the upload widget, then click "Ingest uploaded files". You can add more files later — the app will add new vectors to the existing FAISS index.
- The pedagogical notebook included on the server is at: /mnt/data/NLP_LLM_Pedag.ipynb. Use the sidebar to ingest it.
- For generation: set OPENAI_API_KEY in your environment to enable OpenAI ChatCompletion. Otherwise, a local Flan-T5 model is used as fallback.
