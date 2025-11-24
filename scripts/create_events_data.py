import pandas as pd
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def create_events_dataset():
    """
    Create festival and events dataset for Rajasthan
    Based on major festivals and tourist events
    """
    
    events = [
        # Annual recurring festivals
        {'event_name': 'Jaipur Literature Festival', 'city': 'Jaipur', 'month': 1, 'duration_days': 5, 'category': 'Cultural'},
        {'event_name': 'Desert Festival', 'city': 'Jaisalmer', 'month': 2, 'duration_days': 3, 'category': 'Cultural'},
        {'event_name': 'Holi', 'city': 'All', 'month': 3, 'duration_days': 2, 'category': 'Religious'},
        {'event_name': 'Gangaur Festival', 'city': 'Jaipur', 'month': 3, 'duration_days': 2, 'category': 'Religious'},
        {'event_name': 'Mewar Festival', 'city': 'Udaipur', 'month': 3, 'duration_days': 3, 'category': 'Cultural'},
        {'event_name': 'Summer Festival', 'city': 'Mount Abu', 'month': 5, 'duration_days': 3, 'category': 'Cultural'},
        {'event_name': 'Teej Festival', 'city': 'Jaipur', 'month': 7, 'duration_days': 2, 'category': 'Religious'},
        {'event_name': 'Raksha Bandhan', 'city': 'All', 'month': 8, 'duration_days': 1, 'category': 'Religious'},
        {'event_name': 'Janmashtami', 'city': 'All', 'month': 8, 'duration_days': 2, 'category': 'Religious'},
        {'event_name': 'Dussehra', 'city': 'All', 'month': 10, 'duration_days': 3, 'category': 'Religious'},
        {'event_name': 'Pushkar Camel Fair', 'city': 'Pushkar', 'month': 11, 'duration_days': 7, 'category': 'Cultural'},
        {'event_name': 'Diwali', 'city': 'All', 'month': 10, 'duration_days': 3, 'category': 'Religious'},
        {'event_name': 'Marwar Festival', 'city': 'Jodhpur', 'month': 10, 'duration_days': 2, 'category': 'Cultural'},
    ]
    
    df = pd.DataFrame(events)
    output_path = PROJECT_ROOT / 'data' / 'raw' / 'festivals_events.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Events data saved to {output_path} ({len(df)} events)")
    return df

if __name__ == "__main__":
    df = create_events_dataset()
    print(f"\nFestivals and Events:\n{df}")
