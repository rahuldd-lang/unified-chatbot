"""
evaluator.py
────────────
Evaluation metrics for the RAG pipeline.

Three complementary metrics are computed, inspired by the RAGAS framework:

1. Answer Relevancy  (quantitative)
   ─────────────────
   Embedding-based cosine similarity between the user's question and the
   generated answer.  A higher score indicates that the answer stays
   "on topic" rather than drifting into tangential content.
   Formula: cos_sim(embed(question), embed(answer))

2. Faithfulness  (LLM-as-judge, qualitative → quantitative score 0-1)
   ───────────────
   Claude rates whether every factual claim in the answer is directly
   supported by the retrieved context chunks on a scale 1-5.
   The raw 1-5 score is normalised to [0, 1].
   This catches hallucination: an answer can be relevant but unfaithful.

3. Context Precision  (quantitative)
   ──────────────────
   Fraction of the retrieved chunks that are actually useful for answering
   the question.  Claude marks each chunk as relevant (1) or not (0).
   A high score means retrieval is precise (low noise).

Evaluation dataset
──────────────────
A small JSON dataset (data/eval_dataset.json) contains Q&A pairs with
expected answers.  For each item we:
  a) Run the full RAG pipeline to get the generated answer + sources.
  b) Compute all three metrics.
  c) Return a summary DataFrame for display in the Evaluation tab.
"""

import json
import logging
import re
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Compute Answer Relevancy, Faithfulness, and Context Precision for a
    RAG pipeline response.

    Parameters
    ----------
    api_key : str
        Anthropic API key (used for LLM-as-judge metrics).
    model : str
        Claude model for judge calls.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = Anthropic(api_key=api_key)
        self.model  = model
        # Lazy-loaded to avoid slowing down app startup
        self._embed_fn: Optional[DefaultEmbeddingFunction] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate_response(
        self,
        question: str,
        answer: str,
        context_chunks: List[dict],
        expected_answer: str = "",
    ) -> dict:
        """
        Compute all three metrics for a single RAG response.

        Parameters
        ----------
        question : str
            The original user question.
        answer : str
            The generated answer from Claude.
        context_chunks : list of dict
            Retrieved chunks (each dict has at least a "text" key).

        Returns
        -------
        dict with keys:
            "answer_relevancy"          – float [0, 1]
            "faithfulness"              – float [0, 1]
            "faithfulness_raw"          – int 1-5 (raw LLM score)
            "context_precision"         – float [0, 1]
            "context_recall"            – float [0, 1]  (None if no expected_answer)
            "context_recall_supported"  – int
            "context_recall_total"      – int
        """
        ar  = self.answer_relevancy(question, answer)
        fp  = self.faithfulness(answer, context_chunks)
        cp  = self.context_precision(question, context_chunks)
        cr  = (
            self.context_recall(expected_answer, context_chunks)
            if expected_answer
            else {"score": None, "supported": 0, "total": 0}
        )

        return {
            "answer_relevancy":         ar,
            "faithfulness":             fp["score"],
            "faithfulness_raw":         fp["raw"],
            "context_precision":        cp,
            "context_recall":           cr["score"],
            "context_recall_supported": cr["supported"],
            "context_recall_total":     cr["total"],
        }

    def run_eval_dataset(
        self,
        eval_path: str,
        rag_pipeline,
        doc_id: Optional[str] = None,
    ) -> list:
        """
        Run the full evaluation loop over the eval dataset JSON file.

        Parameters
        ----------
        eval_path : str
            Path to eval_dataset.json.
        rag_pipeline : RAGPipeline
            Live pipeline instance (must already have documents indexed).
        doc_id : str, optional
            Restrict retrieval to a specific document.

        Returns
        -------
        list of dicts, one per eval question, with all metric scores.
        """
        with open(eval_path) as f:
            dataset = json.load(f)

        results = []
        for item in dataset.get("questions", []):
            qid      = item.get("id", "?")
            question = item["question"]
            expected = item.get("expected_answer", "")

            try:
                # Run RAG pipeline
                rag_result = rag_pipeline.query(question, doc_id=doc_id)
                answer  = rag_result["answer"]
                chunks  = rag_result["sources"]

                # Compute metrics
                metrics = self.evaluate_response(question, answer, chunks)

                # Optional: exact-match baseline vs. expected answer
                em_score = self._token_overlap(expected, answer) if expected else None

                results.append({
                    "id":                 qid,
                    "question":           question,
                    "expected_answer":    expected,
                    "generated_answer":   answer,
                    "answer_relevancy":   round(metrics["answer_relevancy"], 3),
                    "faithfulness":       round(metrics["faithfulness"], 3),
                    "context_precision":  round(metrics["context_precision"], 3),
                    "token_overlap_f1":   round(em_score, 3) if em_score is not None else "N/A",
                    "sources_count":      len(chunks),
                })

            except Exception as e:
                logger.error("Eval item %s failed: %s", qid, e)
                results.append({"id": qid, "question": question, "error": str(e)})

        return results

    # ── Metric 1: Answer Relevancy ────────────────────────────────────────────

    def answer_relevancy(self, question: str, answer: str) -> float:
        """
        Cosine similarity between question and answer embeddings.

        Intuition: a good answer should be semantically close to the question —
        same topic, same entities.  Hallucinated or off-topic answers will drift.

        Returns float in [0, 1].
        """
        return self.answer_relevancy_score(question, answer)

    def answer_relevancy_score(self, question: str, answer: str) -> float:
        """Alias for answer_relevancy (used by some test paths)."""
        ef = self._get_embed_fn()
        q_emb = np.array(ef([question]))
        a_emb = np.array(ef([answer]))
        score = float(cosine_similarity(q_emb, a_emb)[0][0])
        # Cosine can be slightly negative; clip to [0, 1]
        return max(0.0, min(1.0, score))

    # ── Metric 2: Faithfulness (LLM-as-judge) ────────────────────────────────

    def faithfulness(self, answer: str, context_chunks: List[dict]) -> dict:
        """
        Ask Claude to rate whether the answer is grounded in the context.

        Prompt: "Rate 1-5 how well every claim in the answer is supported by
        the context (5 = fully supported, 1 = mostly hallucinated)."

        Returns dict:  {"score": float [0,1], "raw": int [1,5]}
        """
        if not context_chunks:
            return {"score": 0.0, "raw": 1}

        context_text = "\n\n".join(
            c.get("text", c) if isinstance(c, dict) else str(c)
            for c in context_chunks[:5]   # Use top 5 to stay within token limits
        )

        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Answer:\n{answer}\n\n"
            "Rate how faithfully the answer is supported by the context on a scale "
            "from 1 to 5, where:\n"
            "  5 = Every claim is directly supported by the context\n"
            "  4 = Most claims supported; minor gaps\n"
            "  3 = Some claims supported; others extrapolated\n"
            "  2 = Few claims supported; substantial hallucination\n"
            "  1 = Answer contradicts or ignores the context\n\n"
            "Respond with ONLY the integer rating (1-5)."
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = resp.content[0].text.strip()
            # Extract first integer found
            match = re.search(r"[1-5]", raw_text)
            raw_score = int(match.group()) if match else 3
            return {"score": (raw_score - 1) / 4, "raw": raw_score}  # Normalise to [0,1]
        except Exception as e:
            logger.error("Faithfulness judge failed: %s", e)
            return {"score": 0.5, "raw": 3}  # Neutral fallback

    # ── Metric 3: Context Precision ───────────────────────────────────────────

    def context_precision(self, question: str, context_chunks: List[dict]) -> float:
        """
        Fraction of retrieved chunks that are relevant to the question.

        Claude labels each chunk as relevant (1) or not (0).  The mean gives
        a precision score that reflects retrieval quality independent of the answer.

        Returns float in [0, 1].
        """
        if not context_chunks:
            return 0.0

        chunks_text = []
        for i, c in enumerate(context_chunks[:6], start=1):
            text = c.get("text", c) if isinstance(c, dict) else str(c)
            chunks_text.append(f"Chunk {i}: {text[:400]}")

        chunks_block = "\n\n".join(chunks_text)

        prompt = (
            f"Question: {question}\n\n"
            f"Retrieved chunks:\n{chunks_block}\n\n"
            f"For each chunk (1 to {len(chunks_text)}), reply with 1 if it contains "
            "information useful for answering the question, or 0 if it does not.\n"
            "Respond with ONLY a JSON array of 0s and 1s, e.g. [1, 0, 1, 1, 0, 1]."
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Extract the JSON array
            match = re.search(r"\[[\d\s,]+\]", raw)
            if match:
                flags = json.loads(match.group())
                precision = sum(flags) / len(flags) if flags else 0.0
                return float(max(0.0, min(1.0, precision)))
        except Exception as e:
            logger.error("Context precision judge failed: %s", e)

        return 0.5  # Neutral fallback

    # ── Metric 4: Context Recall ──────────────────────────────────────────────

    def context_recall(self, expected_answer: str, context_chunks: List[dict]) -> dict:
        """
        What fraction of the statements in the expected (reference) answer are
        supported by the retrieved context?

        Algorithm:
          1. Claude decomposes `expected_answer` into atomic factual statements.
          2. For each statement, Claude checks if the context contains enough
             information to support it.
          3. Recall = supported_statements / total_statements

        Intuition: high recall means the retriever surfaced all the information
        needed to answer correctly. Low recall means relevant chunks were missed.
        Complements Context Precision: precision penalises noise, recall penalises gaps.

        Requires a reference answer — only run during eval dataset scoring,
        not on live chat queries.

        Parameters
        ----------
        expected_answer : str
            Ground-truth reference answer from the eval dataset.
        context_chunks : list of dict
            Retrieved chunks (each dict has at least a "text" key).

        Returns
        -------
        dict with keys:
            "score"     – float [0, 1]
            "supported" – int   (number of supported statements)
            "total"     – int   (total number of statements decomposed)
        """
        if not expected_answer or not context_chunks:
            return {"score": 0.0, "supported": 0, "total": 0}

        context_text = "\n\n".join(
            c.get("text", c) if isinstance(c, dict) else str(c)
            for c in context_chunks[:6]
        )

        prompt = (
            f"Reference answer: {expected_answer}\n\n"
            f"Context:\n{context_text}\n\n"
            "1. Break the reference answer into individual factual statements (one per statement).\n"
            "2. For each statement, mark supported=1 if the context contains enough information "
            "to support it, or supported=0 if the context does not.\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"statements": [{"text": "<statement>", "supported": <1 or 0>}]}'
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            statements = data.get("statements", [])
            if not statements:
                return {"score": 0.0, "supported": 0, "total": 0}
            supported = sum(int(s.get("supported", 0)) for s in statements)
            total = len(statements)
            return {"score": round(supported / total, 4), "supported": supported, "total": total}
        except Exception as e:
            logger.error("context_recall failed: %s", e)
            return {"score": None, "supported": 0, "total": 0}  # None = judge failed, not a real score

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_embed_fn(self) -> DefaultEmbeddingFunction:
        """Lazy-load ChromaDB's ONNX embedding function (avoids startup cost)."""
        if self._embed_fn is None:
            self._embed_fn = DefaultEmbeddingFunction()
        return self._embed_fn

    @staticmethod
    def _token_overlap(reference: str, hypothesis: str) -> float:
        """
        Token-level F1 score between reference and hypothesis strings.
        Used as a lightweight lexical overlap baseline (like SQuAD metric).
        """
        ref_tokens  = set(reference.lower().split())
        hyp_tokens  = set(hypothesis.lower().split())
        common      = ref_tokens & hyp_tokens
        if not common:
            return 0.0
        precision = len(common) / len(hyp_tokens)
        recall    = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)
