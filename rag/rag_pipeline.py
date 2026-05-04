"""
rag_pipeline.py
───────────────
Two-stage hybrid retrieval pipeline + Claude-powered generation.

Retrieval strategy (mirrors the Kaggle two-stage RAG notebook):
  Stage 1  – Candidate retrieval
              a) BM25 sparse search  (keyword / TF-IDF style)
              b) Dense semantic search via ChromaDB embeddings
  Stage 2  – Score fusion
              Reciprocal Rank Fusion (RRF) merges both ranked lists into a
              single ranked list without requiring a separate cross-encoder model.

Generation:
  Claude (claude-3-5-haiku) answers using only the fused top-k chunks
  plus a strict "cite evidence" system prompt to reduce hallucination.
"""

import logging
import math
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi
from anthropic import Anthropic

from rag.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────
DENSE_CANDIDATES = 20     # Chunks fetched from ChromaDB in stage 1
BM25_CANDIDATES  = 20     # Chunks scored by BM25 in stage 1
TOP_K_FINAL      = 6      # Chunks sent to Claude after RRF fusion
RRF_K            = 60     # RRF constant (60 is a well-established default)

SYSTEM_PROMPT = """You are a precise document analysis assistant.
Answer the user's question using ONLY information from the provided context chunks.
If the answer is not in the context, say "I could not find this in the provided documents."
Quote or closely paraphrase the source text when helpful.
Be concise and structured. Use bullet points for lists of facts."""


class RAGPipeline:
    """
    Hybrid retrieval + Claude generation pipeline.

    Parameters
    ----------
    doc_processor : DocumentProcessor
        Shared document processor / ChromaDB client.
    api_key : str
        Anthropic API key.
    model : str
        Claude model identifier (default: claude-haiku-4-5-20251001).
    """

    def __init__(
        self,
        doc_processor: DocumentProcessor,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.proc = doc_processor
        self.client = Anthropic(api_key=api_key)
        self.model = model

    # ── Public API ───────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        top_k: int = TOP_K_FINAL,
    ) -> dict:
        """
        Run end-to-end RAG: retrieve relevant chunks, then generate an answer.

        Parameters
        ----------
        question : str
            Natural-language question from the user.
        doc_id : str, optional
            Restrict retrieval to a specific document.
        top_k : int
            Number of fused chunks to feed Claude.

        Returns
        -------
        dict with keys:
            "answer"   – Claude's response string
            "sources"  – list of source chunk dicts (text, filename, page)
            "dense_count"  – number of dense hits before fusion
            "sparse_count" – number of BM25 hits before fusion
        """
        # ── Stage 1a: Dense retrieval ────────────────────────────────────
        dense_hits = self._dense_retrieve(question, doc_id, n=DENSE_CANDIDATES)

        # ── Stage 1b: BM25 sparse retrieval ─────────────────────────────
        bm25_hits = self._bm25_retrieve(question, doc_id, n=BM25_CANDIDATES)

        # ── Stage 2: RRF fusion ──────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(dense_hits, bm25_hits)
        top_chunks = fused[:top_k]

        if not top_chunks:
            return {
                "answer": "No relevant documents found. Please upload a PDF first.",
                "sources": [],
                "dense_count": 0,
                "sparse_count": 0,
            }

        # ── Generation ───────────────────────────────────────────────────
        answer = self._generate(question, top_chunks)

        sources = [
            {
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename", "unknown"),
                "page": chunk["metadata"].get("page", "?"),
                "score": chunk["score"],
            }
            for chunk in top_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "dense_count": len(dense_hits),
            "sparse_count": len(bm25_hits),
        }

    def get_context_chunks(
        self,
        question: str,
        doc_id: Optional[str] = None,
        top_k: int = TOP_K_FINAL,
    ) -> List[dict]:
        """
        Retrieve top-k chunks for a question without generating an answer.
        Used by the evaluator to compute context-based metrics.
        """
        dense_hits = self._dense_retrieve(question, doc_id, n=DENSE_CANDIDATES)
        bm25_hits  = self._bm25_retrieve(question, doc_id, n=BM25_CANDIDATES)
        fused      = self._reciprocal_rank_fusion(dense_hits, bm25_hits)
        return fused[:top_k]

    # ── Stage 1a: Dense (semantic) retrieval ─────────────────────────────────

    def _dense_retrieve(
        self,
        query: str,
        doc_id: Optional[str],
        n: int,
    ) -> List[dict]:
        """
        Query ChromaDB for the nearest `n` chunks by cosine distance.
        ChromaDB automatically embeds the query with the same model used at indexing time.
        """
        where_filter = {"doc_id": doc_id} if doc_id else None
        total = self.proc.collection.count()
        if total == 0:
            return []
        results = self.proc.collection.query(
            query_texts=[query],
            n_results=min(n, total),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": text,
                "metadata": meta,
                "score": 1.0 - dist,   # Convert cosine distance → similarity
            })
        return hits

    # ── Stage 1b: BM25 (sparse / keyword) retrieval ──────────────────────────

    def _bm25_retrieve(
        self,
        query: str,
        doc_id: Optional[str],
        n: int,
    ) -> List[dict]:
        """
        Build a BM25 index over all indexed chunks and return the top `n` matches.
        BM25 excels at keyword-heavy queries that dense retrieval can miss.

        Note: BM25 is rebuilt on each call. For large corpora, consider caching
        the index or switching to a persistent sparse store.
        """
        where_filter = {"doc_id": doc_id} if doc_id else None
        all_data = self.proc.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        docs = all_data["documents"]
        metas = all_data["metadatas"]
        if not docs:
            return []

        # Tokenise (simple whitespace split – adequate for English prose)
        tokenised = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenised)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Pair (score, index) and take top n
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return [
            {"text": docs[i], "metadata": metas[i], "score": float(s)}
            for i, s in ranked
            if s > 0  # Skip zero-score chunks (irrelevant)
        ]

    # ── Stage 2: Reciprocal Rank Fusion ──────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_hits: List[dict],
        bm25_hits: List[dict],
        k: int = RRF_K,
    ) -> List[dict]:
        """
        Merge two ranked lists using RRF:
            score(chunk) = Σ  1 / (k + rank_in_list)

        RRF is rank-based (not score-based), so it handles the incompatible
        score scales of cosine similarity and BM25 naturally.

        Parameters
        ----------
        dense_hits : list of chunk dicts sorted by dense score
        bm25_hits  : list of chunk dicts sorted by BM25 score
        k          : RRF smoothing constant

        Returns
        -------
        List of chunk dicts sorted by descending RRF score, deduplicated by text.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}

        for rank, chunk in enumerate(dense_hits, start=1):
            key = chunk["text"][:120]   # Use first 120 chars as a unique key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(bm25_hits, start=1):
            key = chunk["text"][:120]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            chunk_map[key] = chunk

        # Return chunks sorted by descending RRF score
        ranked_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        result = []
        for key in ranked_keys:
            chunk = dict(chunk_map[key])
            chunk["score"] = rrf_scores[key]
            result.append(chunk)
        return result

    # ── Generation ───────────────────────────────────────────────────────────

    def _generate(self, question: str, chunks: List[dict]) -> str:
        """
        Build a context block from the top-k chunks and call Claude to answer.

        The context block is structured as numbered citations so Claude can
        reference specific passages.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            fname = chunk["metadata"].get("filename", "doc")
            page  = chunk["metadata"].get("page", "?")
            context_parts.append(
                f"[{i}] (from {fname}, page {page}):\n{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        user_message = (
            f"Context chunks (use ONLY these to answer):\n\n"
            f"{context_block}\n\n"
            f"Question: {question}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
