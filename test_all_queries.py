#!/usr/bin/env python3
"""Test all 12 evaluation queries with real API."""

import sys
import json
import os
from pathlib import Path
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent))

from mcp_servers import disasters_server as ds

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable not set")
    print("Usage: export ANTHROPIC_API_KEY=sk-ant-... && python3 test_all_queries.py")
    sys.exit(1)

MODEL = "claude-opus-4-7"

# Load evaluation dataset
with open("evaluation/eval_dataset.json") as f:
    eval_data = json.load(f)

questions = eval_data["questions"]

print("=" * 100)
print("TESTING ALL 12 EVALUATION QUERIES")
print("=" * 100)

client = Anthropic(api_key=API_KEY)
results = []

for i, q in enumerate(questions):
    qid = q['id']
    question = q['question']
    expected = q['expected_answer']

    print(f"\n{'='*100}")
    print(f"Q{i}: {question}")
    print(f"{'='*100}")

    # Extract intent + route query
    question_lower = question.lower()

    disaster_types = {
        "earthquake": "Earthquake", "earthquakes": "Earthquake",
        "flood": "Flood", "floods": "Flood",
        "tsunami": "Tsunami", "tsunamis": "Tsunami",
        "drought": "Drought", "droughts": "Drought",
        "cyclone": "Storm", "cyclones": "Storm", "hurricane": "Storm", "typhoon": "Storm",
        "storm": "Storm", "storms": "Storm",
        "wildfire": "Wildfire", "wildfires": "Wildfire",
        "landslide": "Landslide", "landslides": "Landslide",
        "volcano": "Volcanic activity", "volcanic": "Volcanic activity",
        "epidemic": "Epidemic", "epidemics": "Epidemic",
        "extreme temperature": "Extreme temperature",
        "insect": "Insect infestation"
    }

    countries = ["india", "japan", "china", "usa", "united states", "bangladesh", "indonesia",
                "pakistan", "philippines", "thailand", "vietnam", "korea", "turkey", "iran",
                "mexico", "argentina", "brazil", "peru", "chile", "nepal", "afghanistan"]

    # Year extraction
    import re
    year_start, year_end = 1900, 2021

    if "last 50" in question_lower or "past 50" in question_lower:
        year_start = 1971
    elif "last 100" in question_lower or "recorded history" in question_lower:
        year_start = 1900
    else:
        year_match = re.findall(r'\b([1-2]\d{3})\b', question_lower)
        if year_match:
            years = sorted([int(y) for y in year_match])
            if len(years) == 2:
                year_start = years[0]
                year_end = years[1]
            elif len(years) >= 1:
                year = years[0]
                if year > 1990:
                    year_start = year
                    year_end = year
                else:
                    year_start = year
                    year_end = years[-1] if len(years) > 1 else 2021

    # Country extraction
    country_found = None
    for country in countries:
        if country in question_lower:
            country_found = country.title()
            break

    # Type extraction
    type_found = None
    for dtype_key, dtype_val in disaster_types.items():
        if dtype_key in question_lower:
            type_found = dtype_val
            break

    # Tsunami special case
    if "tsunami" in question_lower:
        type_found = "Earthquake"

    # Intent determination
    if "decade" in question_lower or "frequency" in question_lower:
        intent = "decade"
    elif country_found and type_found:
        intent = f"country_type:{country_found}:{type_found}"
    elif country_found:
        intent = f"country:{country_found}"
    elif type_found:
        intent = f"type:{type_found}"
    elif "which countr" in question_lower or "worst" in question_lower or "deadli" in question_lower or "top" in question_lower or "region" in question_lower:
        intent = "ranking"
    else:
        intent = "summary"

    print(f"Intent: {intent}, Years: {year_start}-{year_end}")

    # Execute query
    try:
        if intent.startswith("country_type:"):
            parts = intent.split(":")
            country_param = parts[1].strip()
            type_param = parts[2].strip()
            country_data = ds.query_disasters_by_country(country_param, start_year=year_start, end_year=year_end)
            country_dict = json.loads(country_data)
            country_events = country_dict.get("events", [])
            filtered_events = [e for e in country_events if type_param.lower() in str(e.get("Disaster Type", "")).lower()]
            events = filtered_events
            stats = {
                "total_events": len(filtered_events),
                "total_deaths": sum(e.get("Total Deaths", 0) for e in filtered_events),
                "total_affected": sum(e.get("Total Affected", 0) for e in filtered_events)
            }
        elif intent.startswith("country:"):
            country_param = intent.split(":")[-1].strip()
            result = ds.query_disasters_by_country(country_param, start_year=year_start, end_year=year_end)
            data_dict = json.loads(result)
            events = data_dict.get("events", [])
            stats = data_dict.get("stats", {})
        elif intent.startswith("type:"):
            type_param = intent.split(":")[-1].strip()
            result = ds.query_disasters_by_type(type_param, start_year=year_start, end_year=year_end)
            data_dict = json.loads(result)
            events = None
            stats = data_dict.get("summary", {})
        elif intent == "decade":
            result = ds.query_disaster_trends()
            data_dict = json.loads(result)
            events = None
            stats = data_dict.get("decade_breakdown", [])
        elif intent == "ranking":
            result = ds.query_top_deadly_disasters(n=20, start_year=year_start, end_year=year_end)
            data_dict = json.loads(result)
            events = data_dict.get("events", [])
            stats = {}
        else:
            result = ds.query_disasters_summary_stats()
            data_dict = json.loads(result)
            events = None
            stats = data_dict

        # Build context summary
        if isinstance(stats, list):  # decade format
            summary_text = "Disaster Trends by Decade:\n"
            for decade in stats:
                summary_text += f"{decade['Decade']}: {decade['event_count']} events, {int(decade['Total Deaths'])} deaths\n"
        else:
            summary_text = f"Disaster Query Results:\n"
            if stats:
                summary_text += f"Total events: {stats.get('total_events', 0)}\n"
                summary_text += f"Total deaths: {stats.get('total_deaths', 0)}\n"
                summary_text += f"Total affected: {stats.get('total_affected', 0)}\n\n"
            elif events:
                total_deaths = sum(e.get('Total Deaths', 0) for e in events)
                total_affected = sum(e.get('Total Affected', 0) for e in events)
                summary_text += f"Total events: {len(events)}\n"
                summary_text += f"Total deaths: {int(total_deaths)}\n"
                summary_text += f"Total affected: {int(total_affected)}\n\n"

            if events:
                summary_text += "Top Events:\n"
                for i, evt in enumerate(events[:5], 1):
                    evt_name = evt.get('Event Name')
                    evt_name = evt_name if (evt_name and str(evt_name).lower() != 'nan') else "Unnamed"
                    location = evt.get('Location', 'Unknown')
                    location = location if (location and str(location).lower() != 'nan') else "Unknown"
                    summary_text += f"{i}. {evt.get('Year')} - {evt.get('Disaster Type')}: {evt_name} ({int(evt.get('Total Deaths', 0))} deaths) at {location}\n"

        # Get Claude response
        synthesis_prompt = f"""Based on this disaster data, answer the question accurately:

{summary_text}

Question: {question}

Answer factually using the data provided."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )

        answer = response.content[0].text if response.content else "No response"

        print(f"\nData Summary:\n{summary_text}")
        print(f"\nClaude Response:\n{answer[:300]}...")

        results.append({
            "question": question,
            "answer": answer,
            "expected": expected,
            "intent": intent
        })

    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "question": question,
            "answer": f"Error: {e}",
            "expected": expected,
            "intent": intent
        })

# Print summary
print("\n" + "=" * 100)
print("SUMMARY OF ALL 12 RESPONSES")
print("=" * 100)

for i, r in enumerate(results):
    print(f"\nQ{i}: {r['question'][:70]}...")
    print(f"Intent: {r['intent']}")
    print(f"Response: {r['answer'][:150]}...")
    print()
