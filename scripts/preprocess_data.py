import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def preprocess_data():
    """
    Complete preprocessing pipeline:
    - Handle missing values
    - Feature engineering
    - Encoding categorical variables
    - Train/test split (chronological)
    - Feature scaling
    """
    
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load merged dataset
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'merged_dataset.csv'
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n1. Initial dataset shape: {df.shape}")
    
    # -----------------------------------------------------------
    # STEP 1: Handle Missing Values
    # -----------------------------------------------------------
    print("\n2. Handling missing values...")
    
    # For lag features, fill initial missing values with median
    lag_columns = ['footfall_lag_1', 'footfall_lag_7', 'footfall_lag_30']
    for col in lag_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill festival_name missing values with 'No Festival'
    df['festival_name'].fillna('No Festival', inplace=True)
    
    print(f"   ✓ Missing values handled")
    print(f"   Remaining missing: {df.isnull().sum().sum()}")
    
    # -----------------------------------------------------------
    # STEP 2: Feature Engineering
    # -----------------------------------------------------------
    print("\n3. Feature engineering...")
    
    # Cyclical encoding for month (sine/cosine transformation)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Temperature range
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    # Weather condition encoding (one-hot) - FIXED
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
    df = pd.concat([df, weather_dummies], axis=1)
    df.drop('weather_condition', axis=1, inplace=True)  # REMOVE ORIGINAL COLUMN
    
    # City encoding (label encoding)
    le_city = LabelEncoder()
    df['city_encoded'] = le_city.fit_transform(df['city'])
    
    # Season encoding (one-hot) - FIXED
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    df.drop('season', axis=1, inplace=True)  # REMOVE ORIGINAL COLUMN
    
    print(f"   ✓ New features created")
    print(f"   Current shape: {df.shape}")
    
    # -----------------------------------------------------------
    # STEP 3: Select Features for Modeling
    # -----------------------------------------------------------
    print("\n4. Selecting features for modeling...")
    
    feature_columns = [
        # Time features
        'year', 'month', 'day', 'day_of_week', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        
        # Weather features
        'temp_avg', 'temp_min', 'temp_max', 'temp_range',
        'humidity', 'pressure', 'wind_speed',
        
        # Event features
        'is_festival',
        
        # Lag features
        'footfall_lag_1', 'footfall_lag_7', 'footfall_lag_30',
        'footfall_rolling_7', 'footfall_rolling_30',
        
        # Encoded features
        'city_encoded'
    ]
    
    # Add weather dummies
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    feature_columns.extend(weather_cols)
    
    # Add season dummies
    season_cols = [col for col in df.columns if col.startswith('season_')]
    feature_columns.extend(season_cols)
    
    target_column = 'tourist_footfall'
    
    # Verify all feature columns exist and are numeric
    print(f"   ✓ Total features selected: {len(feature_columns)}")
    
    # Check data types
    for col in feature_columns:
        if col not in df.columns:
            print(f"   ⚠ Warning: {col} not found in dataframe")
        elif df[col].dtype == 'object':
            print(f"   ⚠ Warning: {col} is not numeric (dtype: {df[col].dtype})")
    
    # -----------------------------------------------------------
    # STEP 4: Train/Test Split (Chronological)
    # -----------------------------------------------------------
    print("\n5. Creating train/test split (80-20, chronological)...")
    
    # Sort by date
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    # Split by date (80% train, 20% test)
    split_date = df['date'].quantile(0.8)
    
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    print(f"   Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"   Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"   Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Test size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Separate features and target
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].copy()
    
    # -----------------------------------------------------------
    # STEP 5: Feature Scaling
    # -----------------------------------------------------------
    print("\n6. Scaling features...")
    
    # Ensure all features are numeric
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    print(f"   ✓ Features scaled using StandardScaler")
    
    # -----------------------------------------------------------
    # STEP 6: Save Preprocessed Data
    # -----------------------------------------------------------
    print("\n7. Saving preprocessed data...")
    
    # Create directories if not exists
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    models_dir = PROJECT_ROOT / 'models'
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    X_train_scaled.to_csv(processed_dir / 'X_train.csv', index=False)
    X_test_scaled.to_csv(processed_dir / 'X_test.csv', index=False)
    y_train.to_csv(processed_dir / 'y_train.csv', index=False)
    y_test.to_csv(processed_dir / 'y_test.csv', index=False)
    
    # Save metadata
    train_df[['date', 'city', 'tourist_footfall']].to_csv(processed_dir / 'train_metadata.csv', index=False)
    test_df[['date', 'city', 'tourist_footfall']].to_csv(processed_dir / 'test_metadata.csv', index=False)
    
    # Save scaler and encoders
    joblib.dump(scaler, models_dir / 'scaler.pkl')
    joblib.dump(le_city, models_dir / 'city_encoder.pkl')
    joblib.dump(feature_columns, models_dir / 'feature_columns.pkl')
    
    print(f"   ✓ Training data: X_train.csv, y_train.csv")
    print(f"   ✓ Testing data: X_test.csv, y_test.csv")
    print(f"   ✓ Scaler saved: models/scaler.pkl")
    print(f"   ✓ Feature list saved: models/feature_columns.pkl")
    
    # -----------------------------------------------------------
    # Summary Statistics
    # -----------------------------------------------------------
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"\nFeature Matrix Shape:")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test: {X_test_scaled.shape}")
    print(f"\nTarget Statistics:")
    print(f"  Train - Mean: {y_train.mean():.0f}, Std: {y_train.std():.0f}")
    print(f"  Test - Mean: {y_test.mean():.0f}, Std: {y_test.std():.0f}")
    print(f"\nTotal Features: {len(feature_columns)}")
    print(f"\n✅ Data preprocessing complete!")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Display feature names
    print(f"\nFeature Names ({len(X_train.columns)}):")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i}. {col}")
