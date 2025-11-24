import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from config import CITIES, START_DATE, END_DATE, OPENWEATHER_API_KEY

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def get_historical_weather(city_name, lat, lon, date):
    """
    Fetch historical weather data for a specific date using OpenWeatherMap
    Note: For free API, we'll use daily aggregated data
    """
    timestamp = int(datetime.strptime(date, '%Y-%m-%d').timestamp())
    
    # For demonstration, using current weather API
    # You'll need paid plan for full historical data
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': city_name,
                'date': date,
                'temp_avg': data['main']['temp'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather_condition': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed']
            }
        else:
            print(f"Error fetching data for {city_name} on {date}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception for {city_name} on {date}: {str(e)}")
        return None

def generate_synthetic_weather_data():
    """
    Generate synthetic weather data for demonstration
    In real project, you'd use actual API or purchase historical data
    """
    import numpy as np
    
    print("Generating synthetic weather data for demonstration...")
    
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    weather_data = []
    
    for city_name, coords in CITIES.items():
        print(f"Generating data for {city_name}...")
        
        for date in date_range:
            # Seasonal temperature patterns for Rajasthan
            month = date.month
            base_temp = 25 + 10 * np.sin((month - 3) * np.pi / 6)  # Peaks in summer
            
            weather_data.append({
                'city': city_name,
                'date': date.strftime('%Y-%m-%d'),
                'temp_avg': base_temp + np.random.randn() * 3,
                'temp_min': base_temp - 5 + np.random.randn() * 2,
                'temp_max': base_temp + 8 + np.random.randn() * 2,
                'humidity': 40 + 30 * np.sin((month - 8) * np.pi / 6) + np.random.randn() * 5,
                'pressure': 1010 + np.random.randn() * 5,
                'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.7, 0.25, 0.05]),
                'wind_speed': 5 + np.random.randn() * 2
            })
    
    df = pd.DataFrame(weather_data)
    output_path = PROJECT_ROOT / 'data' / 'raw' / 'weather_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Weather data saved to {output_path} ({len(df)} records)")
    return df

if __name__ == "__main__":
    # For now, generate synthetic data
    # Later, you can integrate real API calls
    df = generate_synthetic_weather_data()
    print(f"\nSample data:\n{df.head()}")
