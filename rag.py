from typing import List, Dict

class Retriever:
    """
    Small wrapper around Ingestor.search to perform deduping/reranking if needed.
    """
    def __init__(self, ingestor):
        self.ingestor = ingestor

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        raw = self.ingestor.search(query, top_k=top_k)
        # simple dedupe by text
        seen = set()
        out = []
        for r in raw:
            txt = r.get("text", "")
            if txt not in seen:
                out.append(r)
                seen.add(txt)
        return out
