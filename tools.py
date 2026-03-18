"""
Tools for EcoHome Energy Advisor Agent
"""
import math
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://openai.vocareum.com/v1"


def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_base="https://openai.vocareum.com/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)
    
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
    """
    rng = random.Random(hash(location) & 0xFFFFFFFF)
    conditions = ["sunny", "partly_cloudy", "cloudy", "rainy"]
    weights = [0.55, 0.20, 0.15, 0.10]

    loc = location.lower()
    if any(x in loc for x in ["miami", "phoenix", "los angeles", "san diego", "houston"]):
        base_temp = 24
    elif any(x in loc for x in ["chicago", "boston", "new york", "minneapolis"]):
        base_temp = 12
    elif any(x in loc for x in ["seattle", "portland", "san francisco", "denver"]):
        base_temp = 15
    else:
        base_temp = 18

    peak_irradiance = {"sunny": 850, "partly_cloudy": 520, "cloudy": 180, "rainy": 55}

    forecast_days = []
    for day_offset in range(days):
        day_date = datetime.now() + timedelta(days=day_offset)
        condition = rng.choices(conditions, weights=weights)[0]
        daily_high = base_temp + rng.uniform(-3, 6)

        hourly = []
        for hour in range(24):
            # diurnal cycle: coldest ~6am, warmest ~3pm
            if 6 <= hour <= 15:
                t_offset = (hour - 6) / 9 * 6
            elif hour > 15:
                t_offset = max(0.0, (1 - (hour - 15) / 8) * 6)
            else:
                t_offset = 0.0
            temp = daily_high - 4 + t_offset + rng.uniform(-0.5, 0.5)

            if 6 <= hour <= 19:
                irradiance = peak_irradiance[condition] * math.exp(-0.5 * ((hour - 13) / 3.2) ** 2)
                irradiance = max(0.0, irradiance + rng.uniform(-25, 25))
            else:
                irradiance = 0.0

            hourly.append({
                "hour": hour,
                "temperature_c": round(temp, 1),
                "condition": condition,
                "solar_irradiance": round(irradiance, 1),
                "humidity": rng.randint(30, 75),
                "wind_speed": round(rng.uniform(3, 22), 1)
            })

        solar_hours = [h["hour"] for h in hourly if h["solar_irradiance"] > 150]
        forecast_days.append({
            "date": day_date.strftime("%Y-%m-%d"),
            "condition": condition,
            "max_temp_c": round(daily_high + 2, 1),
            "min_temp_c": round(daily_high - 4, 1),
            "peak_solar_start": solar_hours[0] if solar_hours else 10,
            "peak_solar_end": solar_hours[-1] if solar_hours else 16,
            "hourly": hourly
        })

    return {
        "location": location,
        "forecast_days": days,
        "current": {
            "temperature_c": round(base_temp + rng.uniform(-2, 4), 1),
            "condition": forecast_days[0]["condition"],
            "humidity": rng.randint(40, 70),
            "wind_speed": round(rng.uniform(5, 15), 1)
        },
        "daily_forecast": forecast_days
    }

@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # PG&E-style TOU-E schedule
    tou = {
        "off_peak": {"hours": list(range(0, 7)) + [23], "rate": 0.08},
        "mid_peak": {"hours": list(range(7, 16)) + [21, 22], "rate": 0.15},
        "peak":     {"hours": list(range(16, 21)), "rate": 0.28},
    }

    hourly_rates = []
    for hour in range(24):
        for period, cfg in tou.items():
            if hour in cfg["hours"]:
                hourly_rates.append({
                    "hour": hour,
                    "rate": cfg["rate"],
                    "period": period,
                    "demand_charge": round(cfg["rate"] * 0.10, 4) if period == "peak" else 0.0
                })
                break

    daily_avg = round(sum(h["rate"] for h in hourly_rates) / 24, 4)

    return {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates,
        "peak_hours": list(range(16, 21)),
        "off_peak_hours": list(range(0, 7)) + [23],
        "mid_peak_hours": list(range(7, 16)) + [21, 22],
        "cheapest_window_start": 0,
        "cheapest_window_end": 6,
        "daily_average_rate": daily_avg
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            import glob
            for doc_path in sorted(glob.glob("data/documents/*.txt")):
                loader = TextLoader(doc_path)
                docs = loader.load()
                documents.extend(docs)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = get_embeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
