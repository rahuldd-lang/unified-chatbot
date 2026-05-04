# Unified Chatbot: RAG + Weather + Disaster Intelligence

Integrated conversational AI system combining three query modules with auto-detection and evaluation metrics.

## Features

- **Auto-Intent Detection**: Automatically routes queries to appropriate module (RAG, Weather, Disaster)
- **RAG Module**: Document-based question answering with hybrid retrieval (BM25 + semantic)
- **Weather Agent**: Real-time weather and news integration
- **Disaster Intelligence**: Natural disaster statistics from EM-DAT database
- **Evaluation Metrics**: Answer relevancy, faithfulness, context precision/recall
- **Streamlit UI**: Centered interface with chat + evaluation tabs

## Architecture

### Modules

1. **RAG Pipeline** (`rag/`)
   - Hybrid retrieval fusion (BM25 + dense embeddings with RRF)
   - ChromaDB vector storage
   - Anthropic Claude for synthesis

2. **Weather Agent** (`weather_agent/`)
   - Real-time weather queries
   - News integration via GNews API
   - Fallback handling

3. **Disaster MCP Server** (`mcp_servers/disasters_server.py`)
   - FastMCP server
   - Queries EM-DAT CSV datasets (1900-2021, 1970-2021)
   - 11,624 total disaster records
   - Query functions:
     - `query_disasters_by_country()` - Filter by country + year range
     - `query_disasters_by_type()` - Filter by disaster type
     - `query_top_deadly_disasters()` - Top N by deaths
     - `query_disaster_trends()` - Decade breakdown
     - `query_disasters_summary_stats()` - Global/country/year summaries

### Intent Routing

Dynamic pattern matching determines query intent:

```
Priority: country+type > country > type > decade > ranking > summary

- Country detection: ["india", "japan", "china", ...]
- Type detection: Earthquake, Flood, Storm, Drought, Tsunami, etc.
- Year range: "last 50 years" → 1971-2021, explicit years → parsed from text
- Decade: "decade" or "frequency" keywords
- Ranking: "top", "deadliest", "worst", "which countries"
- Summary: Default fallback
```

**Special cases:**
- Tsunami → routes to Earthquake type (EM-DAT classification)
- Cyclone/Hurricane/Typhoon → Storm type

### Evaluation Metrics

Located in `rag/evaluator.py`:

- **answer_relevancy**: Cosine similarity between question + answer embeddings [0-1]
- **faithfulness**: LLM-as-judge scoring (1-5 scale)
- **context_precision**: Fraction of retrieved context relevant to question
- **context_recall**: Fraction of expected answer covered by context
- **token_overlap_f1**: F1 score of token overlap between answer + context

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- streamlit >= 1.28
- anthropic >= 0.7
- pandas >= 2.0
- chromadb >= 0.4
- mcp >= 0.2
- sentence-transformers >= 2.2
- rank-bm25 >= 0.2

### Environment Variables

Create `.env`:
```
GNEWS_API_KEY=your_key_here
```

**Note:** `ANTHROPIC_API_KEY` is prompted at runtime for security.

### Data Setup

Place disaster CSV files under `DISASTERS/`:
```
chatbot/
├── DISASTERS/
│   ├── 1900_2021_DISASTERS.xlsx - emdat data.csv    (11.6k records, full history)
│   └── 1970-2021_DISASTERS.xlsx - emdat data.csv    (overlapping subset)
```

**Data columns:** Year, Country, ISO, Disaster Type, Event Name, Total Deaths, Total Affected, Total Damages ('000 US$), No Homeless, Location, Region, Continent

## Quick Start

```bash
cd chatbot/
streamlit run app.py
```

1. Enter Anthropic API key when prompted
2. Ask questions in Chat tab (auto-routed to RAG/Weather/Disaster)
3. Visit Evaluation tab → "Run Evaluation" for metrics on 12 disaster queries

## Usage

### Run Streamlit App

```bash
cd chatbot/
streamlit run app.py
```

Opens at http://localhost:8502
- **Chat tab**: Ask questions → auto-routed to appropriate module
- **Evaluation tab**: "Run Evaluation" button tests 12 disaster queries

### Test Queries with Real API

Run all 12 evaluation queries (requires `ANTHROPIC_API_KEY` env var):

```bash
cd chatbot/
export ANTHROPIC_API_KEY=sk-ant-...
python3 test_all_queries.py
```

Output shows routing intent + Claude response for each query.

### Run Unit Tests

```bash
python3 run_tests.py
```

All 51 tests pass:
- Disaster server (24 tests)
- Evaluator (9 tests)
- Orchestrator routing (9 tests)
- RAG pipeline (9 tests)

## Evaluation Dataset

`evaluation/eval_dataset.json` contains 12 disaster queries:

| # | Query | Category | Expected |
|---|-------|----------|----------|
| Q0 | Top 5 deadliest disasters | disaster_stats | 1931 China Flood (3.7M), etc. |
| Q1 | Japan earthquakes 1970-2021 | disaster_query | 6 events, 25k deaths |
| Q2 | Floods last 50 years by country | disaster_trends | Aggregate 2,810 events, 178k deaths (no per-country breakdown) |
| Q3 | Avg tropical cyclone damage | disaster_economics | $379M/event |
| Q4 | Decade with most events | disaster_trends | 2000s (1,099 events) |
| Q5 | 2004 Indian Ocean tsunami affected | disaster_event | ~1.7M people |
| Q6 | Disaster types → economic damage | disaster_economics | Storms > Earthquakes > Floods |
| Q7 | Highest mortality rate by type | disaster_stats | Drought (15.6k/event) |
| Q8 | Frequency change 1900-2021 | disaster_trends | Increased (9→1,099 events) |
| Q9 | Total EM-DAT records | dataset_meta | 11,624 records |
| Q10 | Vulnerable regions | disaster_geography | Asia-Pacific |
| Q11 | Homeless since 1970 | disaster_impact | 102M+ |

### Results Summary

✅ **Working (5/12):** Q0, Q1, Q4, Q8, Q9
⚠️ **Limited (7/12):** Q2, Q3, Q5, Q6, Q7, Q10, Q11

**Limitations:**
- Damage costs not in synthesis for Q3, Q6, Q7
- Homeless field not passed to Claude for Q11
- Country-level type breakdowns not available for Q2, Q6
- 2004 tsunami only returns India, not full 3-country aggregate
- Type-level mortality not computed (only global aggregate)

## Code Structure

```
chatbot/
├── app.py                          # Main Streamlit app
├── mcp_servers/
│   └── disasters_server.py         # FastMCP disaster queries
├── rag/
│   ├── rag_pipeline.py             # Hybrid retrieval + synthesis
│   ├── evaluator.py                # Metrics (relevancy, faithfulness, etc.)
│   └── __init__.py
├── weather_agent/
│   └── weather_news_agent.py       # Weather + news queries
├── agents/                         # Agent implementations
├── evaluation/
│   └── eval_dataset.json           # 12 test questions
├── tests/                          # 51 unit tests
├── DISASTERS/                      # EM-DAT CSV data
├── chroma_db/                      # Vector store (auto-created)
├── requirements.txt
└── README.md
```

## Intent Detection Examples

| Question | Detected Intent | Year Range | Routing |
|----------|-----------------|------------|---------|
| "Top 5 deadliest" | ranking | 1900-2021 | query_top_deadly_disasters() |
| "Japan earthquakes 1970-2021" | country_type:Japan:Earthquake | 1970-2021 | country + type filter |
| "Floods last 50 years" | type:Flood | 1971-2021 | query_disasters_by_type() |
| "Tropical cyclone damage" | type:Storm | 1900-2021 | query_disasters_by_type() |
| "Which decade most events" | decade | 1900-2021 | query_disaster_trends() |
| "2004 tsunami" | country_type:India:Earthquake | 2004-2004 | year narrowed to 2004 |

## API Integration

- **Anthropic Claude**: Answer synthesis, evaluation metrics, intent clarification
- **GNews**: News results for weather queries
- **ChromaDB**: Vector storage for RAG documents
- **FastMCP**: Natural disaster query protocol

## Performance

- **Query latency**: ~2-5s (routing + synthesis)
- **Evaluation latency**: ~30-60s for 12 queries
- **Storage**: 11,624 disaster records, ~5MB CSV
- **Tests**: 51 unit tests in ~5s

## Troubleshooting

### Streamlit App Won't Start
```
Error: No module named 'streamlit'
→ pip install -r requirements.txt
```

### CSV Files Not Found
```
Error: FileNotFoundError: DISASTERS/...csv
→ Check DISASTERS/ folder exists under chatbot/
→ Run: ls DISASTERS/ | grep "emdat data.csv"
```

### API Key Errors
```
Error: "Could not resolve authentication method"
→ Paste valid sk-ant-... key when prompted in Streamlit
→ For test_all_queries.py: export ANTHROPIC_API_KEY=sk-ant-...
```

### Tests Failing
```
pytest: ERROR at setup
→ Install test dependencies: pip install pytest pytest-asyncio
→ Run from chatbot/ directory: python3 run_tests.py
```

### Slow Evaluation
```
Evaluation takes > 60s
→ Normal (calls Claude 12 times)
→ Network latency or API rate limiting possible
→ Check: echo $ANTHROPIC_API_KEY (should be set)
```

## Known Issues

1. **Damage data**: Not synthesized for economic questions (Q3, Q6, Q7)
2. **Country breakdowns**: Type queries don't return per-country stats (Q2)
3. **Aggregate queries**: Global summaries only, not per-type (Q7)
4. **Homeless tracking**: "No Homeless" column not included in context (Q11)
5. **Multi-country events**: 2004 tsunami routes to single country instead of aggregate (Q5)

## Future Improvements

- [ ] Add per-country type breakdowns
- [ ] Include damage costs in synthesis
- [ ] Implement multi-country event aggregation
- [ ] Add homeless field to disaster summaries
- [ ] Type-level mortality rate calculations
- [ ] Document upload for RAG module
- [ ] Real-time weather integration
- [ ] Cache evaluation results

## Testing

Run full test suite:
```bash
python3 run_tests.py
```

Test individual module:
```bash
python3 -m pytest tests/test_disaster_server.py -v
```

Evaluate all 12 queries:
```bash
python3 test_all_queries.py
```

## License

MIT

## Author

Rahul Dubey (rahul.d82@gmail.com)
