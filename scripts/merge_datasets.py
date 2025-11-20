import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def merge_all_datasets():
    """
    Merge tourism, weather, and events data into a single dataset
    """
    
    print("Loading datasets...")
    
    # Load all datasets
    tourism_path = PROJECT_ROOT / 'data' / 'raw' / 'tourism_footfall.csv'
    weather_path = PROJECT_ROOT / 'data' / 'raw' / 'weather_data.csv'
    events_path = PROJECT_ROOT / 'data' / 'raw' / 'festivals_events.csv'
    
    tourism_df = pd.read_csv(tourism_path)
    weather_df = pd.read_csv(weather_path)
    events_df = pd.read_csv(events_path)
    
    print(f"✓ Tourism data: {len(tourism_df)} records")
    print(f"✓ Weather data: {len(weather_df)} records")
    print(f"✓ Events data: {len(events_df)} events")
    
    # Merge tourism and weather data
    merged_df = pd.merge(
        tourism_df,
        weather_df,
        on=['date', 'city'],
        how='left'
    )
    
    print(f"\n✓ After merging with weather: {len(merged_df)} records")
    
    # Add event indicators
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['is_festival'] = 0
    merged_df['festival_name'] = ''
    
    # Mark festival days
    for _, event in events_df.iterrows():
        event_month = event['month']
        event_duration = event['duration_days']
        
        # Approximate festival dates (10th-20th of the month for simplicity)
        for city in merged_df['city'].unique():
            if event['city'] == city or event['city'] == 'All':
                mask = (
                    (merged_df['city'] == city) &
                    (merged_df['month'] == event_month) &
                    (merged_df['day'] >= 10) &
                    (merged_df['day'] <= 10 + event_duration)
                )
                merged_df.loc[mask, 'is_festival'] = 1
                merged_df.loc[mask, 'festival_name'] = event['event_name']
    
    # Add lag features (previous days' footfall)
    merged_df = merged_df.sort_values(['city', 'date'])
    
    for lag in [1, 7, 30]:
        merged_df[f'footfall_lag_{lag}'] = merged_df.groupby('city')['tourist_footfall'].shift(lag)
    
    # Add rolling averages
    merged_df['footfall_rolling_7'] = merged_df.groupby('city')['tourist_footfall'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    merged_df['footfall_rolling_30'] = merged_df.groupby('city')['tourist_footfall'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    
    # Save merged dataset
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'merged_dataset.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Final merged dataset saved: {len(merged_df)} records")
    print(f"✓ Total features: {len(merged_df.columns)}")
    print(f"\nFeatures: {list(merged_df.columns)}")
    
    # Check for missing values
    print(f"\nMissing values summary:")
    missing = merged_df.isnull().sum()
    print(missing[missing > 0])
    
    # Show sample
    print(f"\nSample merged data:")
    print(merged_df.head())
    
    return merged_df

if __name__ == "__main__":
    df = merge_all_datasets()
