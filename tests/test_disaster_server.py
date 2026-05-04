"""
Unit tests for Disaster MCP Server
==================================
Tests pure CSV loading, tool logic, JSON serialization. No network calls.
"""

import json
import sys
from pathlib import Path

import pytest
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers import disasters_server as ds


class TestDisasterDataLoading:
    """Test CSV loading and DataFrame construction."""

    def test_load_data_returns_dataframe(self):
        df = ds._load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_has_required_columns(self):
        df = ds._load_data()
        required_cols = [
            "Year",
            "Country",
            "ISO",
            "Disaster Type",
            "Total Deaths",
            "Total Affected",
            "Total Damages ('000 US$)",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_numeric_columns_are_numeric(self):
        df = ds._load_data()
        for col in ["Year", "Total Deaths", "Total Affected"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"

    def test_load_data_no_nan_in_aggregates(self):
        df = ds._load_data()
        # After fillna(0) in loader, numeric columns should not have NaN
        assert df["Total Deaths"].isna().sum() == 0
        assert df["Total Affected"].isna().sum() == 0


class TestQueryDisastersByCountry:
    """Test query_disasters_by_country tool."""

    def test_query_known_country(self):
        result_json = ds.query_disasters_by_country("India", start_year=1970, end_year=2021)
        result = json.loads(result_json)

        assert "stats" in result
        assert "events" in result
        assert result["stats"]["total_events"] > 0
        assert result["stats"]["total_deaths"] >= 0

    def test_query_iso_code(self):
        result_json = ds.query_disasters_by_country("IND", start_year=1970, end_year=2021)
        result = json.loads(result_json)
        assert result["stats"]["total_events"] > 0

    def test_query_unknown_country_returns_empty(self):
        result_json = ds.query_disasters_by_country("ZZZCountryNotExist", start_year=1970, end_year=2021)
        result = json.loads(result_json)
        assert result["stats"]["total_events"] == 0
        assert result["stats"]["total_deaths"] == 0

    def test_query_year_range(self):
        # All events should be in range
        result_json = ds.query_disasters_by_country("China", start_year=2000, end_year=2010)
        result = json.loads(result_json)
        for event in result["events"]:
            assert 2000 <= event["Year"] <= 2010


class TestQueryDisastersByType:
    """Test query_disasters_by_type tool."""

    def test_query_flood(self):
        result_json = ds.query_disasters_by_type("Flood", start_year=1970, end_year=2021)
        result = json.loads(result_json)

        assert "summary" in result
        assert "yearly_breakdown" in result
        assert result["summary"]["total_events"] > 0

    def test_query_earthquake(self):
        result_json = ds.query_disasters_by_type("Earthquake", start_year=1970, end_year=2021)
        result = json.loads(result_json)
        assert result["summary"]["total_events"] > 0

    def test_query_unknown_type_returns_empty(self):
        result_json = ds.query_disasters_by_type("NonexistentDisaster", start_year=1970, end_year=2021)
        result = json.loads(result_json)
        assert result["summary"]["total_events"] == 0

    def test_yearly_breakdown_structure(self):
        result_json = ds.query_disasters_by_type("Flood")
        result = json.loads(result_json)
        if result["yearly_breakdown"]:
            first_entry = result["yearly_breakdown"][0]
            assert "Year" in first_entry
            assert "event_count" in first_entry
            assert "Total Deaths" in first_entry


class TestQueryTopDeadly:
    """Test query_top_deadly_disasters tool."""

    def test_returns_n_results(self):
        result_json = ds.query_top_deadly_disasters(n=5)
        result = json.loads(result_json)
        assert len(result["events"]) == min(5, result["n"])

    def test_sorted_by_deaths_descending(self):
        result_json = ds.query_top_deadly_disasters(n=10)
        result = json.loads(result_json)
        events = result["events"]
        if len(events) > 1:
            deaths_list = [e["Total Deaths"] for e in events]
            assert deaths_list == sorted(deaths_list, reverse=True)

    def test_continent_filter(self):
        result_json = ds.query_top_deadly_disasters(n=10, continent="Asia")
        result = json.loads(result_json)
        assert result["continent_filter"] == "Asia"

    def test_clamped_n_value(self):
        # Request more than max (50)
        result_json = ds.query_top_deadly_disasters(n=100)
        result = json.loads(result_json)
        assert result["n"] <= 50


class TestQueryTrends:
    """Test query_disaster_trends tool."""

    def test_decade_keys_format(self):
        result_json = ds.query_disaster_trends()
        result = json.loads(result_json)
        breakdown = result["decade_breakdown"]
        if breakdown:
            first_decade = breakdown[0]["Decade"]
            assert first_decade.endswith("s"), f"Decade format wrong: {first_decade}"

    def test_flood_trends(self):
        result_json = ds.query_disaster_trends(disaster_type="Flood")
        result = json.loads(result_json)
        assert result["filters"]["disaster_type"] == "Flood"
        assert len(result["decade_breakdown"]) > 0

    def test_continent_filter(self):
        result_json = ds.query_disaster_trends(continent="Africa")
        result = json.loads(result_json)
        assert result["filters"]["continent"] == "Africa"


class TestQuerySummaryStats:
    """Test query_disasters_summary_stats tool."""

    def test_global_stats(self):
        result_json = ds.query_disasters_summary_stats()
        result = json.loads(result_json)

        assert result["scope"] == "global"
        assert "total_events" in result
        assert "total_deaths" in result
        assert "total_affected" in result
        assert result["total_events"] > 0

    def test_country_stats(self):
        result_json = ds.query_disasters_summary_stats(country="Bangladesh")
        result = json.loads(result_json)

        assert "Bangladesh" in result["scope"]
        assert result["total_events"] > 0

    def test_year_stats(self):
        result_json = ds.query_disasters_summary_stats(year=2010)
        result = json.loads(result_json)

        assert "2010" in result["scope"]

    def test_stats_are_nonnegative(self):
        result_json = ds.query_disasters_summary_stats(country="India")
        result = json.loads(result_json)

        assert result["total_events"] >= 0
        assert result["total_deaths"] >= 0
        assert result["total_affected"] >= 0
        assert result["total_damages_usd_thousands"] >= 0


class TestJSONSerialization:
    """Test all responses are valid JSON."""

    def test_all_responses_valid_json(self):
        responses = [
            ds.query_disasters_by_country("China"),
            ds.query_disasters_by_type("Flood"),
            ds.query_top_deadly_disasters(n=5),
            ds.query_disaster_trends(),
            ds.query_disasters_summary_stats(),
        ]

        for resp in responses:
            assert isinstance(resp, str)
            # Should not raise
            parsed = json.loads(resp)
            assert isinstance(parsed, dict)
