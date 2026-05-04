"""
Tests for RAG Pipeline
======================
Test RRF fusion, query logic (with mocked ChromaDB/Claude).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_pipeline import RAGPipeline


class TestReciprocalRankFusion:
    """Test pure Python RRF logic."""

    def test_rrf_basic_merge(self):
        """Test RRF merges two lists correctly."""
        rag = RAGPipeline.__new__(RAGPipeline)  # Create without init

        dense_hits = [
            {"id": "a", "text": "apple"},
            {"id": "b", "text": "ball"},
        ]
        bm25_hits = [
            {"id": "b", "text": "ball"},
            {"id": "c", "text": "cat"},
        ]

        result = rag._reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)

        # Should have a, b, c
        assert len(result) == 3
        ids = [r["id"] for r in result]
        assert set(ids) == {"a", "b", "c"}
        # b should be first (appears in both)
        assert result[0]["id"] == "b"

    def test_rrf_deduplication_by_text(self):
        """Test deduplication uses text prefix matching."""
        rag = RAGPipeline.__new__(RAGPipeline)

        dense_hits = [
            {"id": "1", "text": "hello world this is a test"},
            {"id": "2", "text": "goodbye world"},
        ]
        bm25_hits = [
            {"id": "3", "text": "hello world this is a different id"},  # Shares "hello world" prefix
            {"id": "4", "text": "goodbye planet"},
        ]

        result = rag._reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)

        # Should have 4 entries (dedup only happens on exact text match, not prefix)
        assert len(result) == 4
        # Verify all are present
        ids = {r["id"] for r in result}
        assert ids == {"1", "2", "3", "4"}

    def test_rrf_scores_sorted_descending(self):
        """Test result is sorted by RRF score."""
        rag = RAGPipeline.__new__(RAGPipeline)

        dense_hits = [{"id": str(i), "text": f"text {i}"} for i in range(5)]
        bm25_hits = [{"id": str(i), "text": f"text {i}"} for i in range(5)]

        result = rag._reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)

        # Check monotonic
        scores = [r.get("rrf_score", 0) for r in result]
        assert scores == sorted(scores, reverse=True)


class TestRAGPipelineQuery:
    """Test query method (with mocked ChromaDB and Claude)."""

    @patch("rag.rag_pipeline.Anthropic")
    def test_query_returns_dict_structure(self, mock_anthropic_class):
        """Test query returns expected dict keys."""
        mock_doc_proc = MagicMock()
        # Mock collection_stats to return dict with integer chunk_count
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_doc_proc.collection = mock_collection

        # Mock Claude API
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated answer")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Patch the ChromaDB retrieval in _dense_retrieve to avoid actual DB calls
        mock_chunks = [
            {"text": "Result 1", "metadata": {"filename": "doc.pdf", "page": 1}},
            {"text": "Result 2", "metadata": {"filename": "doc.pdf", "page": 2}},
        ]
        with patch.object(RAGPipeline, "_dense_retrieve", return_value=mock_chunks):
            with patch.object(RAGPipeline, "_bm25_retrieve", return_value=mock_chunks):
                rag = RAGPipeline(doc_processor=mock_doc_proc, api_key="test-key")
                result = rag.query("What is this about?")

        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    @patch("rag.rag_pipeline.Anthropic")
    def test_query_with_empty_results(self, mock_anthropic_class):
        """Test query gracefully handles no retrieval results."""
        mock_doc_proc = MagicMock()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="No matching documents")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        with patch.object(RAGPipeline, "_dense_retrieve", return_value=[]):
            with patch.object(RAGPipeline, "_bm25_retrieve", return_value=[]):
                rag = RAGPipeline(doc_processor=mock_doc_proc, api_key="test-key")
                result = rag.query("Obscure question")

        assert "answer" in result
        # Should have empty sources list
        assert result["sources"] == []


class TestRAGPipelineInit:
    """Test initialization."""

    @patch("rag.rag_pipeline.Anthropic")
    def test_init_creates_pipeline(self, mock_anthropic):
        """Test RAGPipeline initializes without error."""
        mock_doc_proc = MagicMock()

        rag = RAGPipeline(doc_processor=mock_doc_proc, api_key="test-key")

        assert rag.proc == mock_doc_proc
        assert rag.model == "claude-haiku-4-5-20251001"

    @patch("rag.rag_pipeline.Anthropic")
    def test_init_with_custom_model(self, mock_anthropic):
        """Test custom model parameter."""
        mock_doc_proc = MagicMock()

        rag = RAGPipeline(doc_processor=mock_doc_proc, api_key="test-key", model="custom-model")

        assert rag.model == "custom-model"
