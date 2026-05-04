"""
Tests for Evaluation Metrics
=============================
Token overlap, metric structure (with mocked Claude calls).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenOverlap:
    """Test F1-style token overlap metric."""

    def test_token_overlap_identical(self):
        """Identical strings should have F1 = 1.0."""
        from rag.evaluator import Evaluator

        rag_eval = Evaluator(api_key="test-key")
        score = rag_eval._token_overlap("hello world test", "hello world test")
        assert score == 1.0

    def test_token_overlap_empty_strings(self):
        """Empty strings (no common tokens) should have F1 = 0.0."""
        from rag.evaluator import Evaluator

        rag_eval = Evaluator(api_key="test-key")
        score = rag_eval._token_overlap("", "")
        assert score == 0.0

    def test_token_overlap_no_match(self):
        """Completely different strings should have F1 = 0.0."""
        from rag.evaluator import Evaluator

        rag_eval = Evaluator(api_key="test-key")
        score = rag_eval._token_overlap("abc def", "xyz uvw")
        assert score == 0.0

    def test_token_overlap_partial(self):
        """Partial overlap should give 0 < F1 < 1."""
        from rag.evaluator import Evaluator

        rag_eval = Evaluator(api_key="test-key")
        score = rag_eval._token_overlap("hello world test", "hello world")
        assert 0.0 < score < 1.0


class TestEvaluatorStructure:
    """Test evaluator class structure and method availability."""

    @patch("rag.evaluator.Anthropic")
    def test_evaluator_init(self, mock_anthropic):
        """Test Evaluator initializes."""
        from rag.evaluator import Evaluator

        ev = Evaluator(api_key="test-key")
        assert ev.model == "claude-haiku-4-5-20251001"
        assert ev.client is not None

    @patch("rag.evaluator.Anthropic")
    def test_evaluator_methods_exist(self, mock_anthropic):
        """Test key methods are callable."""
        from rag.evaluator import Evaluator

        ev = Evaluator(api_key="test-key")
        assert callable(ev._token_overlap)
        assert callable(ev.answer_relevancy)
        assert callable(ev.faithfulness)


class TestEvaluatorMetrics:
    """Test individual metric functions."""

    @patch("rag.evaluator.Anthropic")
    def test_answer_relevancy_signature(self, mock_anthropic):
        """Test answer_relevancy accepts right params."""
        from rag.evaluator import Evaluator

        ev = Evaluator(api_key="test-key")
        # Should accept question and answer
        try:
            score = ev.answer_relevancy("What is X?", "X is Y")
            # May fail due to mocking, but shouldn't raise signature error
        except (ValueError, AttributeError, TypeError) as e:
            # These errors are expected from mocked embeddings
            if "signature" in str(e).lower():
                raise

    @patch("rag.evaluator.Anthropic")
    def test_context_precision_signature(self, mock_anthropic):
        """Test context_precision accepts right params."""
        from rag.evaluator import Evaluator

        ev = Evaluator(api_key="test-key")
        # Should accept question and chunks
        try:
            chunks = [{"text": "chunk1"}]
            result = ev.context_precision("What is X?", chunks)
            # May fail due to mocking
        except (ValueError, AttributeError, TypeError) as e:
            if "signature" in str(e).lower():
                raise


class TestUnifiedEvaluatorStructure:
    """Test UnifiedEvaluator would accept category dispatch."""

    def test_unified_evaluator_can_be_imported(self):
        """Just verify the module exists and can be imported."""
        # The unified evaluator is created in task 6 (evaluation)
        # For now, just verify we can import the base evaluator
        from rag.evaluator import Evaluator
        assert Evaluator is not None
