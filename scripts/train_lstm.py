import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import joblib
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"{'='*60}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def build_lstm_model(input_shape):
    """
    Build LSTM neural network architecture
    """
    model = keras.Sequential([
        # First LSTM layer with return sequences
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Second LSTM layer with return sequences
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        
        # Third LSTM layer
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        
        # Output layer
        layers.Dense(1)
    ])
    
    return model

def train_lstm():
    """
    Train LSTM model for tourist footfall prediction
    """
    
    print("="*60)
    print("LSTM MODEL TRAINING")
    print("="*60)
    
    # Load sequential data
    print("\n1. Loading sequential data...")
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    X_train_seq = np.load(processed_dir / 'X_train_lstm.npy')
    X_test_seq = np.load(processed_dir / 'X_test_lstm.npy')
    y_train_seq = np.load(processed_dir / 'y_train_lstm.npy')
    y_test_seq = np.load(processed_dir / 'y_test_lstm.npy')
    
    print(f"   X_train shape: {X_train_seq.shape}")
    print(f"   X_test shape: {X_test_seq.shape}")
    print(f"   y_train shape: {y_train_seq.shape}")
    print(f"   y_test shape: {y_test_seq.shape}")
    
    # Build model
    print("\n2. Building LSTM architecture...")
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(input_shape)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\n3. Model Architecture:")
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    print("\n4. Training LSTM model...")
    start_time = time.time()
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n   ✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("\n5. Making predictions...")
    y_train_pred = model.predict(X_train_seq).flatten()
    y_test_pred = model.predict(X_test_seq).flatten()
    
    # Evaluate on training set
    print("\n6. Training Set Performance:")
    train_metrics = evaluate_model(y_train_seq, y_train_pred, "LSTM - TRAINING")
    
    # Evaluate on test set
    print("\n7. Test Set Performance:")
    test_metrics = evaluate_model(y_test_seq, y_test_pred, "LSTM - TEST")
    
    # Save model
    print("\n8. Saving model and results...")
    models_dir = PROJECT_ROOT / 'models'
    results_dir = PROJECT_ROOT / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(models_dir / 'lstm_model.h5')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test_seq,
        'y_pred': y_test_pred,
        'error': y_test_seq - y_test_pred
    })
    predictions_df.to_csv(results_dir / 'lstm_predictions.csv', index=False)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(results_dir / 'lstm_training_history.csv', index=False)
    
    print(f"   ✓ Model saved: {models_dir / 'lstm_model.h5'}")
    print(f"   ✓ Predictions saved: {results_dir / 'lstm_predictions.csv'}")
    print(f"   ✓ Training history saved: {results_dir / 'lstm_training_history.csv'}")
    
    print("\n✅ LSTM training complete!")
    print("="*60)
    
    return model, test_metrics, history

if __name__ == "__main__":
    model, metrics, history = train_lstm()
