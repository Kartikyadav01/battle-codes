import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def verify_dataset():
    """
    Quick verification and visualization of the merged dataset
    """
    
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'merged_dataset.csv'
    df = pd.read_csv(data_path)
    
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"   - Total records: {df.shape[0]:,}")
    print(f"   - Total features: {df.shape[1]}")
    
    print(f"\n2. Date Range:")
    print(f"   - Start: {df['date'].min()}")
    print(f"   - End: {df['date'].max()}")
    
    print(f"\n3. Cities Covered:")
    for city in df['city'].unique():
        count = len(df[df['city'] == city])
        avg_footfall = df[df['city'] == city]['tourist_footfall'].mean()
        print(f"   - {city}: {count} days, Avg footfall: {avg_footfall:.0f}")
    
    print(f"\n4. Footfall Statistics:")
    print(df['tourist_footfall'].describe())
    
    print(f"\n5. Festival Days: {df['is_festival'].sum()} days")
    
    print(f"\n6. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values!")
    else:
        print(missing[missing > 0])
    
    print(f"\n7. Sample Records:")
    print(df[['date', 'city', 'tourist_footfall', 'temp_avg', 'is_festival']].head(10))
    
    print("\n✅ Dataset verification complete!")
    print("="*60)

if __name__ == "__main__":
    verify_dataset()
