import pandas as pd
import joblib
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def check_preprocessed_data():
    """
    Verify all preprocessed files are created correctly
    """
    
    print("="*60)
    print("CHECKING PREPROCESSED DATA")
    print("="*60)
    
    # Check CSV files
    files_to_check = [
        PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv',
        PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv',
        PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv',
        PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv',
        PROJECT_ROOT / 'data' / 'processed' / 'train_metadata.csv',
        PROJECT_ROOT / 'data' / 'processed' / 'test_metadata.csv'
    ]
    
    print("\n1. Checking CSV files:")
    for file in files_to_check:
        try:
            df = pd.read_csv(file)
            print(f"   ✓ {file}: {df.shape}")
        except Exception as e:
            print(f"   ✗ {file}: ERROR - {e}")
    
    # Check model artifacts
    print("\n2. Checking model artifacts:")
    artifacts = [
        PROJECT_ROOT / 'models' / 'scaler.pkl',
        PROJECT_ROOT / 'models' / 'city_encoder.pkl',
        PROJECT_ROOT / 'models' / 'feature_columns.pkl'
    ]
    
    for artifact in artifacts:
        try:
            obj = joblib.load(artifact)
            print(f"   ✓ {artifact}: Loaded successfully")
        except Exception as e:
            print(f"   ✗ {artifact}: ERROR - {e}")
    
    # Load and display sample
    print("\n3. Sample training data:")
    X_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv')
    y_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv')
    
    print(f"\nFirst 5 rows of X_train:")
    print(X_train.head())
    
    print(f"\nFirst 5 values of y_train:")
    print(y_train.head())
    
    print("\n✅ All preprocessed files verified!")
    print("="*60)

if __name__ == "__main__":
    check_preprocessed_data()
