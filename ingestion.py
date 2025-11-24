import os
import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils import read_file_text, make_sure_path_exists
from glob import glob
from tqdm import tqdm

class Ingestor:
    """
    Handles reading files, chunking text, computing embeddings and maintaining a FAISS index with metadata.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data", chunk_size: int = 800, overlap: int = 200):
        self.model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.data_dir = data_dir
        make_sure_path_exists(self.data_dir)
        self.index_path = os.path.join(self.data_dir, "faiss_index.faiss")
        self.meta_path = os.path.join(self.data_dir, "metadata.json")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []
        self._load_index()

    # -------------------------
    # Index & metadata helpers
    # -------------------------
    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print("Failed to load index/metadata:", e)
                self.index = None
                self.metadata = []
        else:
            self.index = None
            self.metadata = []

    def index_exists(self) -> bool:
        return self.index is not None

    def get_num_vectors(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Text chunking
    # -------------------------
    def _chunk_text(self, text: str) -> List[str]:
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Quick paragraph split
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for p in paras:
            if len(p) <= self.chunk_size:
                chunks.append(p)
            else:
                start = 0
                while start < len(p):
                    end = start + self.chunk_size
                    chunk = p[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = max(0, end - self.overlap)
        # Merge tiny chunks into neighbors
        merged = []
        buffer = ""
        for c in chunks:
            if not buffer:
                buffer = c
            elif len(buffer) + 1 + len(c) <= self.chunk_size:
                buffer = buffer + " " + c
            else:
                merged.append(buffer)
                buffer = c
        if buffer:
            merged.append(buffer)
        return merged

    # -------------------------
    # Ingestion: single file
    # -------------------------
    def ingest_file(self, file_path: str, source_name: str = None) -> int:
        """
        Extract text from file_path, chunk, embed, and add to FAISS index and metadata.
        Returns number of chunks added.
        """
        text = read_file_text(file_path)
        if not text or not text.strip():
            return 0
        source = source_name or os.path.basename(file_path)
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        # Compute embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = self._normalize_embeddings(embeddings)

        # Create index if needed
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

        # Add embeddings to index
        self.index.add(embeddings)

        # Append metadata (order must match added embeddings)
        start_id = len(self.metadata)
        for i, c in enumerate(chunks):
            self.metadata.append({
                "id": start_id + i,
                "source": source,
                "text": c
            })

        # persist
        self.save()
        return len(chunks)

    # -------------------------
    # Rebuild from uploads folder
    # -------------------------
    def rebuild_index_from_uploads(self, uploads_folder: str = "uploads"):
        files = []
        for ext in ("*.pdf", "*.txt", "*.docx"):
            files.extend(glob(os.path.join(uploads_folder, ext)))
        # reset
        self.index = None
        self.metadata = []
        # ingest each file
        for f in files:
            try:
                self.ingest_file(f)
            except Exception as e:
                print("Failed to ingest", f, e)
        self.save()

    # -------------------------
    # Search
    # -------------------------
    def search(self, query: str, top_k: int = 4):
        if self.index is None or len(self.metadata) == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize_embeddings(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.metadata):
                meta = dict(self.metadata[idx])
                meta["score"] = float(score)
                results.append(meta)
        return results

    # -------------------------
    # Utility
    # -------------------------
    def _normalize_embeddings(self, emb: np.ndarray):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return emb / norms
