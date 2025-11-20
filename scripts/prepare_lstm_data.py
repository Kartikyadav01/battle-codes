import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def create_sequences(X, y, time_steps=7):
    """
    Create sequences for LSTM input
    Each sequence contains 'time_steps' days of data to predict the next day
    """
    Xs, ys = [], []
    
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    
    return np.array(Xs), np.array(ys)

def prepare_lstm_data():
    """
    Prepare sequential data for LSTM model
    """
    
    print("="*60)
    print("LSTM DATA PREPARATION")
    print("="*60)
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv').values
    X_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv').values
    y_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv').values.ravel()
    
    print(f"   Original X_train shape: {X_train.shape}")
    print(f"   Original X_test shape: {X_test.shape}")
    
    # Create sequences
    TIME_STEPS = 7  # Use 7 days of history to predict next day
    
    print(f"\n2. Creating sequences (time_steps={TIME_STEPS})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)
    
    print(f"   X_train_seq shape: {X_train_seq.shape}")
    print(f"   X_test_seq shape: {X_test_seq.shape}")
    print(f"   y_train_seq shape: {y_train_seq.shape}")
    print(f"   y_test_seq shape: {y_test_seq.shape}")
    
    # Save sequences
    print("\n3. Saving sequential data...")
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / 'X_train_lstm.npy', X_train_seq)
    np.save(processed_dir / 'X_test_lstm.npy', X_test_seq)
    np.save(processed_dir / 'y_train_lstm.npy', y_train_seq)
    np.save(processed_dir / 'y_test_lstm.npy', y_test_seq)
    
    print(f"   ✓ Files saved in {processed_dir}")
    
    print("\n✅ LSTM data preparation complete!")
    print("="*60)
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq

if __name__ == "__main__":
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_lstm_data()
