import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config import CITIES, START_DATE, END_DATE

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def generate_tourism_footfall():
    """
    Generate synthetic tourism footfall data with realistic patterns:
    - Peak season: October to March (winter)
    - Low season: April to June (summer)
    - Monsoon: July to September (moderate)
    - Festival spikes
    - Weekend effects
    - Year-over-year growth
    - City-specific popularity
    """
    
    print("Generating realistic tourism footfall data...")
    
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    tourism_data = []
    
    # City-specific base popularity (relative tourists per day)
    city_popularity = {
        'Jaipur': 3000,      # Most popular - Pink City
        'Udaipur': 2500,     # Very popular - City of Lakes
        'Jodhpur': 2000,     # Popular - Blue City
        'Jaisalmer': 1800,   # Popular - Golden City
        'Pushkar': 1200      # Smaller but famous for fair
    }
    
    # Load events data to create spikes
    events_path = PROJECT_ROOT / 'data' / 'raw' / 'festivals_events.csv'
    events_df = pd.read_csv(events_path)
    
    for city_name in CITIES.keys():
        print(f"Generating footfall for {city_name}...")
        base_footfall = city_popularity[city_name]
        
        for idx, date in enumerate(date_range):
            month = date.month
            year = date.year
            day_of_week = date.dayofweek  # Monday=0, Sunday=6
            
            # 1. Seasonal pattern (Rajasthan peak: Oct-Mar)
            if month in [10, 11, 12, 1, 2, 3]:  # Peak winter season
                seasonal_factor = 1.5
            elif month in [7, 8, 9]:  # Monsoon - moderate
                seasonal_factor = 0.9
            else:  # Summer (Apr-Jun) - low season
                seasonal_factor = 0.5
            
            # 2. Year-over-year growth (5% annual growth)
            years_since_start = (year - 2018) + (date.dayofyear / 365)
            growth_factor = 1 + (0.05 * years_since_start)
            
            # 3. Weekend effect (20% more tourists on weekends)
            weekend_factor = 1.2 if day_of_week >= 5 else 1.0
            
            # 4. Festival/Event spikes
            festival_factor = 1.0
            city_events = events_df[
                (events_df['city'] == city_name) | (events_df['city'] == 'All')
            ]
            
            for _, event in city_events.iterrows():
                if event['month'] == month:
                    # Create spike around festival dates (approximate)
                    if 10 <= date.day <= (10 + event['duration_days'] + 5):
                        if city_name == 'Pushkar' and event['event_name'] == 'Pushkar Camel Fair':
                            festival_factor = 4.0  # Huge spike
                        else:
                            festival_factor = 1.8  # Moderate spike
            
            # 5. Random daily variation
            random_factor = np.random.uniform(0.85, 1.15)
            
            # Calculate final footfall
            footfall = (base_footfall * 
                       seasonal_factor * 
                       growth_factor * 
                       weekend_factor * 
                       festival_factor * 
                       random_factor)
            
            # Add some noise
            footfall = max(int(footfall + np.random.normal(0, base_footfall * 0.05)), 100)
            
            tourism_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'city': city_name,
                'tourist_footfall': footfall,
                'year': year,
                'month': month,
                'day': date.day,
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'season': 'Winter' if month in [10,11,12,1,2,3] else 
                         ('Monsoon' if month in [7,8,9] else 'Summer')
            })
    
    df = pd.DataFrame(tourism_data)
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    output_path = PROJECT_ROOT / 'data' / 'raw' / 'tourism_footfall.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Tourism footfall data saved to {output_path} ({len(df)} records)")
    print(f"\nSummary Statistics by City:")
    print(df.groupby('city')['tourist_footfall'].describe().round(0))
    
    return df

if __name__ == "__main__":
    df = generate_tourism_footfall()
    
    # Show sample data
    print(f"\n Sample records:\n{df.head(10)}")
    
    # Show seasonal patterns
    print(f"\n Average footfall by season:")
    print(df.groupby('season')['tourist_footfall'].mean().round(0))
