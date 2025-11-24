"""Test script to verify OpenWeather and Groq API connectivity"""
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load .env file
load_dotenv()

# API Keys
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", os.getenv("OPENWEATHER_API_KEY"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# City coordinates for testing
CITY_COORDS = {
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
}

print("=" * 60)
print("API CONNECTIVITY TEST")
print("=" * 60)

# Test WeatherAPI (FreeWeather API)
print("\n1. Testing WeatherAPI (FreeWeather)...")
if not WEATHERAPI_KEY:
    print("   [X] WEATHERAPI_KEY not found in .env file")
else:
    print(f"   [OK] API Key found: {WEATHERAPI_KEY[:10]}...")
    coords = CITY_COORDS["Jaipur"]
    location = f"{coords['lat']},{coords['lon']}"
    params = {
        "key": WEATHERAPI_KEY,
        "q": location,
        "aqi": "no",
    }
    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        location_data = data.get("location", {})
        condition = current.get("condition", {})
        print(f"   [SUCCESS] WeatherAPI working!")
        print(f"   City: {location_data.get('name', 'N/A')}")
        print(f"   Temperature: {current.get('temp_c', 'N/A')}C")
        print(f"   Condition: {condition.get('text', 'N/A')}")
        print(f"   Humidity: {current.get('humidity', 'N/A')}%")
        print(f"   Wind Speed: {current.get('wind_kph', 'N/A')} km/h")
    except requests.exceptions.RequestException as e:
        print(f"   [FAILED] WeatherAPI failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")

# Test Groq API
print("\n2. Testing Groq API...")
if not GROQ_API_KEY:
    print("   [X] GROQ_API_KEY not found in .env file")
else:
    print(f"   [OK] API Key found: {GROQ_API_KEY[:10]}...")
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": "Say 'API test successful' if you can read this.",
            }
        ],
        "max_tokens": 20,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        print(f"   [SUCCESS] Groq API working!")
        print(f"   Response: {content}")
    except requests.exceptions.RequestException as e:
        print(f"   [FAILED] Groq API failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

