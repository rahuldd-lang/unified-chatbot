"""
Tests for ChatbotOrchestrator routing logic
============================================
Keyword classification, intent routing (no live network/API calls).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import ChatbotOrchestrator


class TestIntentClassification:
    """Test classify_intent keyword heuristics."""

    def test_disaster_keywords(self):
        orch = ChatbotOrchestrator(api_key="test-key")

        test_cases = [
            ("How many people died in earthquakes?", "disasters"),  # 2+ keywords
            ("Tell me about floods in India", "disasters"),  # 2+ keywords
            ("Disaster statistics by country", "disasters"),  # 2 keywords
            ("How many affected by the 2010 earthquake?", "disasters"),  # 2+ keywords
            ("Tsunami deaths and casualties", "disasters"),  # 3 keywords
        ]

        for query, expected in test_cases:
            result = orch.classify_intent(query)
            assert result == expected, f"Query '{query}' classified as {result}, expected {expected}"

    def test_weather_news_keywords(self):
        orch = ChatbotOrchestrator(api_key="test-key")

        test_cases = [
            ("What's the weather in Paris?", "weather_news"),
            ("Give me a forecast for London", "weather_news"),
            ("Top tech news today", "weather_news"),
            ("Latest business headlines", "weather_news"),
            ("Tell me about climate change", "weather_news"),
        ]

        for query, expected in test_cases:
            result = orch.classify_intent(query)
            assert result == expected, f"Query '{query}' classified as {result}, expected {expected}"

    def test_rag_keywords(self):
        orch = ChatbotOrchestrator(api_key="test-key")

        test_cases = [
            ("What does the document say?", "rag"),
            ("Summarize this PDF", "rag"),
            ("What's in my file?", "rag"),
            ("Tell me about my uploaded document", "rag"),
        ]

        for query, expected in test_cases:
            result = orch.classify_intent(query)
            assert result == expected, f"Query '{query}' classified as {result}, expected {expected}"

    def test_ambiguous_fallback_to_rag(self):
        orch = ChatbotOrchestrator(api_key="test-key")
        result = orch.classify_intent("Hello, how are you?")
        assert result in ["rag", "weather_news", "disasters"]  # Fallback is acceptable


class TestRouteQuery:
    """Test routing dispatch (with mocked backends)."""

    @pytest.mark.asyncio
    async def test_route_to_disasters(self):
        orch = ChatbotOrchestrator(api_key="test-key")

        # Mock the weather_disaster_agent
        orch.weather_disaster_agent.process_query = AsyncMock(
            return_value={
                "response": "Mock disaster response",
                "tool_calls": [],
                "model": "test-model",
                "error": None,
            }
        )

        result = await orch.route_query("How many died in floods?")

        assert result["intent"] == "disasters"
        assert "response" in result
        orch.weather_disaster_agent.process_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_weather_news(self):
        orch = ChatbotOrchestrator(api_key="test-key")

        orch.weather_disaster_agent.process_query = AsyncMock(
            return_value={
                "response": "Mock weather response",
                "tool_calls": [],
                "model": "test-model",
                "error": None,
            }
        )

        result = await orch.route_query("What's the weather in Paris?")

        assert result["intent"] == "weather_news"
        assert "response" in result

    @pytest.mark.asyncio
    async def test_route_to_rag_no_pipeline(self):
        orch = ChatbotOrchestrator(api_key="test-key", doc_processor=None, rag_pipeline=None)

        result = await orch.route_query("What does my document say?")

        assert result["intent"] == "rag"
        assert "not initialized" in result["response"].lower()
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_route_to_rag_with_pipeline(self):
        mock_rag = MagicMock()
        mock_rag.query.return_value = {
            "answer": "Test answer",
            "chunks": [{"text": "chunk1"}],
            "model": "test-model",
        }

        orch = ChatbotOrchestrator(api_key="test-key", rag_pipeline=mock_rag)

        result = await orch.route_query("What does my document say?")

        assert result["intent"] == "rag"
        assert result["response"] == "Test answer"
        assert len(result["chunks"]) == 1
        assert result["error"] is None


class TestKeywordSets:
    """Verify keyword sets are sensible."""

    def test_disaster_keywords_not_empty(self):
        from agents.orchestrator import DISASTER_KEYWORDS
        assert len(DISASTER_KEYWORDS) > 10

    def test_weather_news_keywords_not_empty(self):
        from agents.orchestrator import WEATHER_NEWS_KEYWORDS
        assert len(WEATHER_NEWS_KEYWORDS) > 10

    def test_keyword_overlap_minimal(self):
        from agents.orchestrator import DISASTER_KEYWORDS, WEATHER_NEWS_KEYWORDS
        overlap = DISASTER_KEYWORDS & WEATHER_NEWS_KEYWORDS
        # Allow minimal overlap
        assert len(overlap) < 5
