"""
Top-Level Router Orchestrator
==============================
Classifies user intent (via keyword heuristics) and routes queries to the appropriate backend:
- RAG pipeline (document Q&A)
- Weather/News/Disasters agent (tool-use loop)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from rag.rag_pipeline import RAGPipeline
from rag.document_processor import DocumentProcessor
from weather_agent.orchestrator import WeatherNewsDisasterOrchestrator

logger = logging.getLogger(__name__)

DISASTER_KEYWORDS = {
    "disaster", "earthquake", "flood", "tsunami", "drought", "cyclone", "storm death",
    "death toll", "casualties", "emdat", "natural disaster", "wildfire", "hurricane",
    "typhoon", "landslide", "volcano", "epidemic", "famine", "injured", "affected",
    "homeless", "damages", "reconstruction", "1900-2021", "1970-2021"
}

WEATHER_NEWS_KEYWORDS = {
    "weather", "forecast", "temperature", "rain", "snow", "wind", "cloud", "humidity",
    "news", "headline", "article", "story", "top story", "latest", "today", "breaking",
    "tech", "business", "sports", "entertainment", "health", "science", "world",
    "climate change", "heat wave", "cold snap"
}


class ChatbotOrchestrator:
    """Routes queries to RAG, Weather/News, or Disasters backend."""

    def __init__(
        self,
        api_key: str,
        gnews_api_key: str = "",
        doc_processor: DocumentProcessor | None = None,
        rag_pipeline: RAGPipeline | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.api_key = api_key
        self.gnews_api_key = gnews_api_key
        self.model = model
        self.doc_processor = doc_processor
        self.rag_pipeline = rag_pipeline
        self.weather_disaster_agent = WeatherNewsDisasterOrchestrator(
            api_key=api_key, gnews_api_key=gnews_api_key, model=model
        )

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent using keyword heuristics.
        Returns one of: 'disasters', 'weather_news', 'rag'
        """
        query_lower = query.lower()

        # Count keyword matches
        disaster_score = sum(1 for kw in DISASTER_KEYWORDS if kw in query_lower)
        weather_score = sum(1 for kw in WEATHER_NEWS_KEYWORDS if kw in query_lower)

        # Strong disaster signal
        if disaster_score >= 2:
            return "disasters"

        # Strong weather/news signal
        if weather_score >= 2:
            return "weather_news"

        # Weak signals or default to RAG (document Q&A)
        if "document" in query_lower or "pdf" in query_lower or "file" in query_lower:
            return "rag"

        if disaster_score > weather_score:
            return "disasters"
        if weather_score > disaster_score:
            return "weather_news"

        # Default fallback
        return "rag"

    async def route_query(
        self,
        query: str,
        history: list[dict] | None = None,
        active_doc_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Route query to appropriate backend and return response dict.
        """
        intent = self.classify_intent(query)

        if intent == "disasters" or intent == "weather_news":
            # Both go through the weather_disaster_agent (which has 3 MCP servers)
            result = await self.weather_disaster_agent.process_query(
                query=query, history=history or []
            )
            result["intent"] = intent
            return result

        elif intent == "rag":
            if not self.rag_pipeline:
                return {
                    "response": "RAG pipeline not initialized. Please set up document processing first.",
                    "intent": "rag",
                    "chunks": [],
                    "error": "RAG not available",
                }

            try:
                result = self.rag_pipeline.query(query=query, doc_id=active_doc_id)
                return {
                    "response": result.get("answer", ""),
                    "intent": "rag",
                    "chunks": result.get("chunks", []),
                    "model": result.get("model", self.model),
                    "tool_calls": [],
                    "error": None,
                }
            except Exception as e:
                logger.error(f"RAG query error: {e}")
                return {
                    "response": f"Error processing document query: {str(e)}",
                    "intent": "rag",
                    "chunks": [],
                    "error": str(e),
                }

        return {
            "response": "Unable to classify query intent.",
            "intent": "unknown",
            "error": "Unknown intent",
        }


def _run_in_thread(coro):
    """
    Run an async coroutine in a fresh thread with its own event loop.
    Needed for Streamlit compatibility on macOS/Linux where uvloop blocks the UI.
    """
    import streamlit as st

    def run_coro():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    ctx = st.runtime.scriptrunner.get_script_run_ctx()
    container = {"result": None, "error": None}

    def worker():
        if ctx is not None:
            st.runtime.scriptrunner.add_script_run_ctx(threading.current_thread(), ctx)
        try:
            container["result"] = run_coro()
        except Exception as e:
            container["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()

    if container["error"]:
        raise container["error"]
    return container["result"]
