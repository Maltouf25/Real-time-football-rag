# backend/rag_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from mistralai import Mistral


# -------------------------
# Configuration
# -------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "play_by_play_collection")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


# -------------------------
# Data model
# -------------------------

@dataclass
class RetrievedItem:
    minute: int
    tsec: int
    source: str
    text: str
    event_type: Optional[str] = None
    team: Optional[str] = None


# -------------------------
# RAG Engine
# -------------------------

class RagEngine:
    """
    Time-aware RAG engine:
      1) Embed query
      2) Retrieve from Qdrant (match_id + optional run_id + time window)
      3) Build prompt
      4) Call Mistral
    """

    def __init__(self):
        if not QDRANT_URL:
            raise ValueError("Missing env var QDRANT_URL")
        if not QDRANT_API_KEY:
            raise ValueError("Missing env var QDRANT_API_KEY")
        if not MISTRAL_API_KEY:
            raise ValueError("Missing env var MISTRAL_API_KEY")

        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.mistral = Mistral(api_key=MISTRAL_API_KEY)

    # -------------------------
    # Retrieval
    # -------------------------

    def retrieve(
        self,
        user_query: str,
        match_id: str,
        top_k: int = 6,
        minute_gte: Optional[int] = None,
        minute_lte: Optional[int] = None,
        run_id: Optional[str] = None,
        deep_think: bool = False,
    ) -> List[RetrievedItem]:

        qvec = self.embedder.encode(user_query).tolist()

        must = [
            FieldCondition(key="match_id", match=MatchValue(value=match_id)),
        ]

        if run_id:
            must.append(FieldCondition(key="run_id", match=MatchValue(value=run_id)))

        if not deep_think and (minute_gte is not None or minute_lte is not None):
            must.append(
                FieldCondition(
                    key="minute",
                    range=Range(
                        gte=minute_gte,
                        lte=minute_lte,
                    ),
                )
            )

        try:
            results = self.qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=qvec,
                limit=top_k,
                with_payload=True,
                query_filter=Filter(must=must),
            )
        except Exception as e:
            print(f"❌ Qdrant retrieval error: {e}")
            return []

        items: List[RetrievedItem] = []
        for hit in results:
            p = hit.payload or {}
            items.append(
                RetrievedItem(
                    minute=int(p.get("minute", 0)),
                    tsec=int(p.get("tsec", 0) or 0),
                    source=str(p.get("source", "comment")),
                    text=str(p.get("text", "")),
                    event_type=p.get("event_type"),
                    team=p.get("team"),
                )
            )

        # Chronological order is critical for LLM reasoning
        items.sort(key=lambda x: (x.minute, x.tsec))
        return items

    # -------------------------
    # Prompt
    # -------------------------

    def build_prompt(
        self,
        user_query: str,
        contexts: List[RetrievedItem],
        deep_think: bool = False,
    ) -> str:

        ctx_lines: List[str] = []
        for c in contexts:
            meta = []
            if c.team:
                meta.append(f"team={c.team}")
            if c.event_type:
                meta.append(f"type={c.event_type}")
            meta_s = " " + " ".join(meta) if meta else ""
            ctx_lines.append(
                f"[{c.source.upper()} {c.minute:02d}m]{meta_s} {c.text}"
            )

        system_msg = (
            "SYSTEM:\n"
            "You are a football match analyst with access to ALL match data.\n"
            if deep_think
            else
            "SYSTEM:\n"
            "You are a live football match assistant.\n"
            "Use ONLY the provided context.\n"
            "If the answer is not in the context, say you don't know.\n"
        )

        return (
            system_msg
            + "\nCONTEXT:\n"
            + "\n".join(ctx_lines)
            + "\n\nQUESTION:\n"
            + user_query
            + "\n\nANSWER:\n"
        )

    # -------------------------
    # LLM call
    # -------------------------

    def generate_answer(self, prompt: str) -> str:
        resp = self.mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a live football match assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # -------------------------
    # Public API
    # -------------------------

    def answer(
        self,
        question: str,
        match_id: str,
        current_minute: Optional[int] = None,
        minute_window: Optional[int] = 4,
        run_id: Optional[str] = None,
        deep_think: bool = False,
    ) -> Dict[str, Any]:

        if deep_think:
            minute_gte = None
            minute_lte = None
            top_k = 20
        else:
            if current_minute is None:
                return {
                    "answer": "Current match time is unknown.",
                    "contexts": [],
                }
            minute_gte = max(0, current_minute - minute_window)
            minute_lte = current_minute
            top_k = 6

        ctx = self.retrieve(
            user_query=question,
            match_id=match_id,
            top_k=top_k,
            minute_gte=minute_gte,
            minute_lte=minute_lte,
            run_id=run_id,
            deep_think=deep_think,
        )

        if not ctx:
            return {
                "answer": "I don't have enough recent context to answer this question.",
                "contexts": [],
            }

        prompt = self.build_prompt(question, ctx, deep_think=deep_think)
        answer = self.generate_answer(prompt)

        return {
            "answer": answer,
            "contexts": [c.__dict__ for c in ctx],
            "mode": "deep_think" if deep_think else "fast",
        }
