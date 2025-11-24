import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time
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

def train_random_forest():
    """
    Train Random Forest model for tourist footfall prediction
    """
    
    print("="*60)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv')
    X_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv')
    y_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv').values.ravel()
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Initialize Random Forest model
    print("\n2. Initializing Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,        # Number of trees
        max_depth=20,            # Maximum depth of trees
        min_samples_split=10,    # Minimum samples to split a node
        min_samples_leaf=4,      # Minimum samples in leaf node
        max_features='sqrt',     # Number of features for best split
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        verbose=1
    )
    
    # Train the model
    print("\n3. Training Random Forest...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate on training set
    print("\n5. Training Set Performance:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Random Forest - TRAINING")
    
    # Evaluate on test set
    print("\n6. Test Set Performance:")
    test_metrics = evaluate_model(y_test, y_test_pred, "Random Forest - TEST")
    
    # Feature importance
    print("\n7. Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    print("\n8. Saving model...")
    models_dir = PROJECT_ROOT / 'models'
    results_dir = PROJECT_ROOT / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_dir / 'random_forest_model.pkl')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'error': y_test - y_test_pred
    })
    predictions_df.to_csv(results_dir / 'random_forest_predictions.csv', index=False)
    
    # Save feature importance
    feature_importance.to_csv(results_dir / 'random_forest_feature_importance.csv', index=False)
    
    print(f"   ✓ Model saved: {models_dir / 'random_forest_model.pkl'}")
    print(f"   ✓ Predictions saved: {results_dir / 'random_forest_predictions.csv'}")
    print(f"   ✓ Feature importance saved: {results_dir / 'random_forest_feature_importance.csv'}")
    
    print("\n✅ Random Forest training complete!")
    print("="*60)
    
    return model, test_metrics

if __name__ == "__main__":
    model, metrics = train_random_forest()
