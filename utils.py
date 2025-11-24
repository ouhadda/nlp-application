import os
from pathlib import Path
import uuid
from typing import Optional
from PyPDF2 import PdfReader
import docx
import json

ALLOWED = {".pdf", ".txt", ".docx"}

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

def make_sure_path_exists(p: str):
    os.makedirs(p, exist_ok=True)

def save_uploaded_file(uploaded_file, target_folder: str = "uploads") -> str:
    make_sure_path_exists(target_folder)
    filename = getattr(uploaded_file, "name", f"upload-{uuid.uuid4().hex}")
    safe = os.path.join(target_folder, filename)
    with open(safe, "wb") as f:
        f.write(uploaded_file.read())
    return safe

def read_file_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(pages)
    except Exception as e:
        print("PDF read error:", e)
        return ""

def read_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)
    except Exception as e:
        print("DOCX read error:", e)
        return ""

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print("TXT read error:", e)
        return ""
