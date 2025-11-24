import os
from typing import List, Tuple
from rag import Retriever

# Local HF imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import openai

class RAGChat:
    """
    Orchestrates retrieval + generation.
    If use_openai=True and OPENAI_API_KEY is found, uses OpenAI ChatCompletion.
    Otherwise, falls back to a local seq2seq HF model (flan-t5).
    """
    def __init__(self, ingestor, use_openai: bool = False, hf_model_name: str = "google/flan-t5-small"):
        self.ingestor = ingestor
        self.retriever = Retriever(ingestor)
        self.use_openai = use_openai and (os.environ.get("OPENAI_API_KEY") is not None)
        self.hf_model_name = hf_model_name

        if not self.use_openai:
            # prepare HF generator pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
            self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)

    def _build_prompt(self, question: str, retrieved: List[dict], chat_history: List[dict]) -> str:
        # concatenate retrieved contexts
        context = ""
        for i, r in enumerate(retrieved):
            src = r.get("source", "unknown")
            txt = r.get("text", "")
            context += f"[{i+1} | {src}]\n{txt}\n\n"

        # incorporate short chat history (last 6)
        history = ""
        if chat_history:
            for turn in chat_history[-6:]:
                prefix = "User:" if turn["role"] == "user" else "Assistant:"
                history += f"{prefix} {turn['text']}\n"

        prompt = (
            "You are a helpful assistant specialized in NLP and LLM concepts. "
            "Use the CONTEXT below (from the user's documents) to answer the QUESTION. "
            "If context is insufficient, use general knowledge. Explain step-by-step when appropriate.\n\n"
            f"CONTEXT:\n{context if context else 'No context available.'}\n\n"
            f"HISTORY:\n{history}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer in a clear, structured, pedagogical way. If you use context, mention the source tags like [1 | filename]."
        )
        return prompt

    def answer_question(self, question: str, top_k: int = 3, chat_history: List[dict] = None) -> Tuple[str, List[dict]]:
        if chat_history is None:
            chat_history = []

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, retrieved, chat_history)

        if self.use_openai:
            return self._generate_openai(prompt), retrieved
        else:
            return self._generate_local(prompt), retrieved

    def _generate_openai(self, prompt: str) -> str:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        # Use ChatCompletion with a stable model choice; user can change the model via env if desired.
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return text

    def _generate_local(self, prompt: str) -> str:
        out = self.generator(prompt, max_length=512, do_sample=False)
        return out[0]["generated_text"]
