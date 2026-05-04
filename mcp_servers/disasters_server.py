"""
Natural Disaster Query MCP Server
==================================
FastMCP server that queries EM-DAT disaster CSV files via Pandas.

Two datasets:
  - 1900_2021_DISASTERS.xlsx - emdat data.csv  (1900–2021, ~16k rows)
  - 1970-2021_DISASTERS.xlsx - emdat data.csv  (1970–2021, ~14k rows)

Exposes 5 tools for disaster statistics queries (deaths, damages, trends, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Disaster Data Server",
    instructions="Query natural disaster statistics from EM-DAT datasets (1900–2021). "
    "Tools available: query by country/type/trends, top deadly events, and summary statistics."
)

_CSV_1900 = Path(__file__).parent.parent / "DISASTERS" / "1900_2021_DISASTERS.xlsx - emdat data.csv"
_CSV_1970 = Path(__file__).parent.parent / "DISASTERS" / "1970-2021_DISASTERS.xlsx - emdat data.csv"

_DF: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    """Load and combine both CSV files. Call once at module import."""
    global _DF
    if _DF is not None:
        return _DF

    try:
        df_1900 = pd.read_csv(_CSV_1900)
        df_1970 = pd.read_csv(_CSV_1970)

        df_1900.columns = df_1900.columns.str.strip()
        df_1970.columns = df_1970.columns.str.strip()

        for col in ["Total Deaths", "No Injured", "No Affected", "No Homeless", "Total Affected",
                     "Total Damages ('000 US$)", "Insured Damages ('000 US$)"]:
            if col in df_1900.columns:
                df_1900[col] = pd.to_numeric(df_1900[col], errors="coerce").fillna(0)
            if col in df_1970.columns:
                df_1970[col] = pd.to_numeric(df_1970[col], errors="coerce").fillna(0)

        df_combined = pd.concat([df_1900, df_1970], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["Year", "Disaster Type", "Country", "Event Name"], keep="first"
        )
        df_combined = df_combined.sort_values("Year").reset_index(drop=True)

        _DF = df_combined
        logger.info(f"Loaded {len(_DF)} disaster records from {_CSV_1900.name} and {_CSV_1970.name}")
        return _DF

    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise


@mcp.tool()
def query_disasters_by_country(
    country: str, start_year: int = 1900, end_year: int = 2021
) -> str:
    """
    Query disaster events for a specific country within a year range.

    Args:
        country: Country name or ISO code (e.g., 'India', 'IND', 'Bangladesh', 'BGD')
        start_year: Earliest year to include (default 1900)
        end_year: Latest year to include (default 2021)

    Returns:
        JSON with events list and aggregate statistics (deaths, affected, damages)
    """
    df = _load_data()

    mask = (
        ((df["Country"].str.lower() == country.lower()) |
         (df["ISO"].str.lower() == country.lower())) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask].sort_values("Total Deaths", ascending=False)

    stats = {
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
        "total_damages_usd_thousands": int(result_df["Total Damages ('000 US$)"].sum()),
    }

    events = result_df[["Year", "Disaster Type", "Event Name", "Total Deaths", "No Affected", "Location"]].head(50).to_dict(orient="records")

    return json.dumps({"country": country, "year_range": [start_year, end_year], "stats": stats, "events": events}, default=str)


@mcp.tool()
def query_disasters_by_type(
    disaster_type: str, start_year: int = 1900, end_year: int = 2021
) -> str:
    """
    Query aggregate statistics for a specific disaster type.

    Args:
        disaster_type: Type such as 'Flood', 'Earthquake', 'Drought', 'Storm', 'Wildfire', 'Tsunami'
        start_year: Earliest year (default 1900)
        end_year: Latest year (default 2021)

    Returns:
        JSON with event count, total deaths, affected, damages grouped by year
    """
    df = _load_data()

    mask = (
        (df["Disaster Type"].str.lower() == disaster_type.lower()) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask]

    yearly_stats = result_df.groupby("Year").agg({
        "Event Name": "count",
        "Total Deaths": "sum",
        "Total Affected": "sum",
        "Total Damages ('000 US$)": "sum"
    }).rename(columns={"Event Name": "event_count"}).reset_index().to_dict(orient="records")

    summary = {
        "disaster_type": disaster_type,
        "year_range": [start_year, end_year],
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
    }

    return json.dumps({"summary": summary, "yearly_breakdown": yearly_stats}, default=str)


@mcp.tool()
def query_top_deadly_disasters(
    n: int = 10, start_year: int = 1900, end_year: int = 2021, continent: str = ""
) -> str:
    """
    Query the N deadliest disaster events.

    Args:
        n: Number of results to return (1–50, default 10)
        start_year: Earliest year (default 1900)
        end_year: Latest year (default 2021)
        continent: Optional filter (e.g., 'Asia', 'Africa', 'Americas', 'Europe', 'Oceania'). Empty = worldwide.

    Returns:
        JSON list of top N events ranked by Total Deaths (descending)
    """
    df = _load_data()
    n = max(1, min(n, 50))

    mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
    if continent:
        mask = mask & (df["Continent"].str.lower() == continent.lower())

    result_df = df[mask].nlargest(n, "Total Deaths")

    events = result_df[["Year", "Disaster Type", "Event Name", "Country", "Total Deaths", "Total Affected", "Location"]].to_dict(orient="records")

    return json.dumps({"n": n, "continent_filter": continent, "events": events}, default=str)


@mcp.tool()
def query_disaster_trends(disaster_type: str = "", continent: str = "") -> str:
    """
    Query decade-by-decade trends of disaster frequency and impact.

    Args:
        disaster_type: Optional filter (e.g., 'Flood'). Empty = all types.
        continent: Optional filter. Empty = worldwide.

    Returns:
        JSON with decade-grouped counts, deaths, and affected
    """
    df = _load_data()

    mask = pd.Series([True] * len(df))
    if disaster_type:
        mask = mask & (df["Disaster Type"].str.lower() == disaster_type.lower())
    if continent:
        mask = mask & (df["Continent"].str.lower() == continent.lower())

    result_df = df[mask].copy()
    result_df["Decade"] = (result_df["Year"] // 10 * 10).astype(str) + "s"

    decade_stats = result_df.groupby("Decade").agg({
        "Event Name": "count",
        "Total Deaths": "sum",
        "Total Affected": "sum"
    }).rename(columns={"Event Name": "event_count"}).reset_index().to_dict(orient="records")

    return json.dumps({
        "filters": {"disaster_type": disaster_type or "all", "continent": continent or "worldwide"},
        "decade_breakdown": decade_stats
    }, default=str)


@mcp.tool()
def query_disasters_summary_stats(country: str = "", year: int = 0) -> str:
    """
    Query global or scoped summary statistics.

    Args:
        country: Optional country filter (empty = global). Can be name or ISO code.
        year: Optional single year filter (0 = all years).

    Returns:
        JSON with total events, deaths, affected, damages
    """
    df = _load_data()

    mask = pd.Series([True] * len(df))
    if country:
        mask = mask & ((df["Country"].str.lower() == country.lower()) |
                       (df["ISO"].str.lower() == country.lower()))
    if year > 0:
        mask = mask & (df["Year"] == year)

    result_df = df[mask]

    summary = {
        "scope": f"country={country}" if country else ("year=" + str(year) if year > 0 else "global"),
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
        "total_damages_usd_thousands": int(result_df["Total Damages ('000 US$)"].sum()),
        "insured_damages_usd_thousands": int(result_df["Insured Damages ('000 US$)"].sum()),
    }

    return json.dumps(summary, default=str)


@mcp.tool()
def query_disasters_by_location(location: str, start_year: int = 1900, end_year: int = 2021) -> str:
    """Query disasters for specific location (region within country)."""
    df = _load_data()
    mask = (
        (df["Location"].str.lower().str.contains(location.lower(), na=False)) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask].sort_values("Total Deaths", ascending=False)
    stats = {
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
    }
    events = result_df[["Year", "Country", "Location", "Disaster Type", "Total Deaths"]].head(50).to_dict(orient="records")
    return json.dumps({"location": location, "year_range": [start_year, end_year], "stats": stats, "events": events}, default=str)


@mcp.tool()
def query_disasters_by_region(region: str, start_year: int = 1900, end_year: int = 2021) -> str:
    """Query disasters for geographic region (e.g., 'Asia', 'Africa', 'South America')."""
    df = _load_data()
    mask = (
        (df["Region"].str.lower() == region.lower()) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask].sort_values("Total Deaths", ascending=False)
    stats = {
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
    }
    events = result_df[["Year", "Country", "Region", "Disaster Type", "Total Deaths"]].head(50).to_dict(orient="records")
    return json.dumps({"region": region, "year_range": [start_year, end_year], "stats": stats, "events": events}, default=str)


@mcp.tool()
def query_disasters_by_continent(continent: str, start_year: int = 1900, end_year: int = 2021) -> str:
    """Query disasters for continent (e.g., 'Asia', 'Africa', 'Americas', 'Europe', 'Oceania')."""
    df = _load_data()
    mask = (
        (df["Continent"].str.lower() == continent.lower()) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask].sort_values("Total Deaths", ascending=False)
    stats = {
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
    }
    events = result_df[["Year", "Country", "Continent", "Disaster Type", "Total Deaths"]].head(50).to_dict(orient="records")
    return json.dumps({"continent": continent, "year_range": [start_year, end_year], "stats": stats, "events": events}, default=str)


@mcp.tool()
def query_disasters_by_iso(iso_code: str, start_year: int = 1900, end_year: int = 2021) -> str:
    """Query disasters by ISO 3-letter country code (e.g., 'IND' for India, 'JPN' for Japan)."""
    df = _load_data()
    mask = (
        (df["ISO"].str.upper() == iso_code.upper()) &
        (df["Year"] >= start_year) & (df["Year"] <= end_year)
    )
    result_df = df[mask].sort_values("Total Deaths", ascending=False)
    stats = {
        "total_events": len(result_df),
        "total_deaths": int(result_df["Total Deaths"].sum()),
        "total_affected": int(result_df["Total Affected"].sum()),
    }
    events = result_df[["Year", "Country", "ISO", "Disaster Type", "Total Deaths"]].head(50).to_dict(orient="records")
    return json.dumps({"iso_code": iso_code, "year_range": [start_year, end_year], "stats": stats, "events": events}, default=str)


if __name__ == "__main__":
    _load_data()
    mcp.run()
