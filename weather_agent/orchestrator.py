"""
Agent Orchestrator (Extended)
==============================
Connects to three MCP servers (weather + news + disasters) via stdio subprocesses,
collects their tools, and runs a Claude-powered agent loop that calls those tools on demand.

Usage (from async code):
    orchestrator = WeatherNewsDisasterOrchestrator(api_key="sk-ant-...")
    reply = await orchestrator.process_query("What's the weather in Paris?")
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SERVERS_DIR = _HERE.parent / "mcp_servers"
WEATHER_SERVER = str(_SERVERS_DIR / "weather_server.py")
NEWS_SERVER = str(_SERVERS_DIR / "news_server.py")
DISASTERS_SERVER = str(_SERVERS_DIR / "disasters_server.py")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and friendly AI assistant that specialises in:
1. **Current weather** — real-time conditions and multi-day forecasts for any city worldwide.
2. **Latest news** — top stories and topic-specific searches from GNews.io.
3. **Natural disasters** — historical statistics and trends from the EM-DAT database (1900–2021).

Guidelines:
- Always use the provided tools to fetch live data before answering.
- Present weather data clearly: include temperature (°C), description, humidity, wind, and feels-like.
- Present news as a concise bulleted list with title, summary, and URL.
- Present disaster stats with event counts, deaths, affected populations, and damages (in USD thousands).
- If asked about multiple domains (weather + news + disasters), address each with its relevant tools.
- Be friendly, concise, and factually accurate based on the tool output.
- If a location is not found, politely say so and suggest trying a different name.
- Units: temperature in °C, wind speed in km/h, precipitation in mm, damages in '000 US$.
"""

# ── Tool conversion helpers ────────────────────────────────────────────────────

def _sanitize_schema(raw) -> dict:
    """
    Return a clean JSON Schema dict that Anthropic's API accepts.

    FastMCP emits extra fields (title, $defs, additionalProperties, etc.)
    that are valid JSON Schema but cause Anthropic's tool validator to
    silently drop or reject the tool definition.  We keep only the three
    fields the API actually uses: type, properties, required.
    """
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}

    schema: dict = {"type": "object"}

    props = raw.get("properties", {})
    if props:
        # Strip per-property 'title' fields too — they're noise to the API
        schema["properties"] = {
            k: {pk: pv for pk, pv in v.items() if pk != "title"}
            if isinstance(v, dict) else v
            for k, v in props.items()
        }
    else:
        schema["properties"] = {}

    required = raw.get("required", [])
    if required:
        schema["required"] = required

    return schema


def _mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP Tool object to Anthropic API tool format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": _sanitize_schema(tool.inputSchema),
    }


# ── Main orchestrator class ────────────────────────────────────────────────────

class WeatherNewsDisasterOrchestrator:
    """
    Agent that orchestrates three MCP servers (weather + news + disasters) using Claude.

    Each call to `process_query` spawns fresh MCP server sub-processes,
    runs the full agent loop (potentially multiple tool calls), then
    terminates the servers.  This is safe for multi-user Streamlit apps.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        gnews_api_key: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.gnews_api_key = gnews_api_key

    # ── Public API ─────────────────────────────────────────────────────────────

    async def process_query(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Process a user query using the agent loop.

        Returns a dict with keys:
          - "response" (str): The final text answer.
          - "tool_calls" (list): Log of every tool call made.
          - "model" (str): Model used.
          - "error" (str | None): Any fatal error message.
        """
        if not MCP_AVAILABLE:
            return {
                "response": "MCP library is not installed. Please run: pip install mcp",
                "tool_calls": [],
                "model": self.model,
                "error": "MCP not available",
            }

        tool_call_log: list[dict] = []
        error: str | None = None
        response_text = ""

        try:
            weather_params = StdioServerParameters(
                command=sys.executable, args=[WEATHER_SERVER]
            )
            news_env = {**os.environ, "GNEWS_API_KEY": self.gnews_api_key}
            news_params = StdioServerParameters(
                command=sys.executable, args=[NEWS_SERVER], env=news_env
            )
            disasters_params = StdioServerParameters(
                command=sys.executable, args=[DISASTERS_SERVER]
            )

            async with stdio_client(weather_params) as (wr, ww):
                async with ClientSession(wr, ww) as weather_session:
                    await weather_session.initialize()

                    async with stdio_client(news_params) as (nr, nw):
                        async with ClientSession(nr, nw) as news_session:
                            await news_session.initialize()

                            async with stdio_client(disasters_params) as (dr, dw):
                                async with ClientSession(dr, dw) as disasters_session:
                                    await disasters_session.initialize()

                                    response_text, tool_call_log = await self._agent_loop(
                                        query=query,
                                        history=history or [],
                                        weather_session=weather_session,
                                        news_session=news_session,
                                        disasters_session=disasters_session,
                                    )

        except Exception as exc:
            error = str(exc)
            response_text = (
                f"An error occurred while processing your request: {error}\n\n"
                "Please check your API key and ensure the MCP servers can start."
            )

        return {
            "response": response_text,
            "tool_calls": tool_call_log,
            "model": self.model,
            "error": error,
        }

    # ── Internal agent loop ────────────────────────────────────────────────────

    async def _agent_loop(
        self,
        query: str,
        history: list[dict],
        weather_session: "ClientSession",
        news_session: "ClientSession",
        disasters_session: "ClientSession",
        max_iterations: int = 10,
    ) -> tuple[str, list[dict]]:
        """Run the Claude tool-use loop until a final text response is returned."""

        anthropic_tools: list[dict] = []
        tool_session_map: dict[str, "ClientSession"] = {}

        for label, session in [
            ("weather", weather_session),
            ("news", news_session),
            ("disasters", disasters_session),
        ]:
            caps = (
                getattr(session, "server_capabilities", None)
                or getattr(session, "_server_capabilities", None)
            )
            if caps is not None:
                if getattr(caps, "tools", None) is None:
                    continue
            try:
                tools_resp = await session.list_tools()
            except Exception:
                continue
            for t in tools_resp.tools:
                anthropic_tools.append(_mcp_tool_to_anthropic(t))
                tool_session_map[t.name] = session

        client = anthropic.Anthropic(api_key=self.api_key)

        messages: list[dict] = list(history)
        messages.append({"role": "user", "content": query})

        tool_call_log: list[dict] = []

        for _iteration in range(max_iterations):
            resp = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=anthropic_tools,
            )

            if resp.stop_reason == "end_turn":
                text_parts = [
                    block.text
                    for block in resp.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_parts), tool_call_log

            if resp.stop_reason == "max_tokens":
                text_parts = [
                    block.text
                    for block in resp.content
                    if hasattr(block, "text")
                ]
                partial = "\n".join(text_parts)
                return (
                    partial + "\n\n_(Note: the response was truncated because "
                    "it reached the maximum length. Try asking a more specific "
                    "question for a complete answer.)_"
                ), tool_call_log

            if resp.stop_reason == "tool_use":
                messages.append(
                    {"role": "assistant", "content": resp.content}
                )

                tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]

                async def _invoke(block) -> tuple[dict, dict]:
                    """Call one MCP tool and return (log_entry, tool_result)."""
                    tool_name: str = block.name
                    tool_input: dict = block.input
                    session = tool_session_map.get(tool_name)

                    log_entry = {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": None,
                        "error": None,
                    }

                    if session is None:
                        err = f"Unknown tool: {tool_name}"
                        log_entry["error"] = err
                        result_text = json.dumps({"error": err})
                        is_error = True
                    else:
                        try:
                            mcp_result = await session.call_tool(tool_name, tool_input)

                            is_error = getattr(mcp_result, "isError", False)
                            if mcp_result.content:
                                result_text = "".join(
                                    c.text
                                    for c in mcp_result.content
                                    if hasattr(c, "text")
                                )
                            else:
                                result_text = json.dumps({"result": "empty"})

                            if is_error:
                                log_entry["error"] = result_text
                            else:
                                log_entry["output"] = result_text

                        except Exception as exc:
                            is_error = True
                            result_text = json.dumps({"error": str(exc)})
                            log_entry["error"] = str(exc)

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        **({"is_error": True} if is_error else {}),
                    }
                    return log_entry, tool_result

                outcomes = await asyncio.gather(*[_invoke(b) for b in tool_use_blocks])

                for log_entry, tool_result in outcomes:
                    tool_call_log.append(log_entry)

                tool_results = [tr for _, tr in outcomes]
                messages.append({"role": "user", "content": tool_results})

            else:
                break

        return (
            "I reached the maximum number of steps without a final answer. "
            "Please try a simpler question.",
            tool_call_log,
        )

    def get_tool_list(self) -> list[str]:
        """Return a static list of known tools (for UI display)."""
        return [
            "get_current_weather",
            "get_weather_forecast",
            "get_top_news",
            "get_recent_news",
            "search_news",
            "get_news_by_topic",
            "query_disasters_by_country",
            "query_disasters_by_type",
            "query_top_deadly_disasters",
            "query_disaster_trends",
            "query_disasters_summary_stats",
        ]
