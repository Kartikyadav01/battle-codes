import pandas as pd
import numpy as np
import xgboost as xgb
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

def train_xgboost():
    """
    Train XGBoost model for tourist footfall prediction
    """
    
    print("="*60)
    print("XGBOOST MODEL TRAINING")
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
    
    # Initialize XGBoost model
    print("\n2. Initializing XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=300,           # Number of boosting rounds
        learning_rate=0.05,         # Step size shrinkage
        max_depth=8,                # Maximum tree depth
        min_child_weight=3,         # Minimum sum of instance weight
        subsample=0.8,              # Subsample ratio of training data
        colsample_bytree=0.8,       # Subsample ratio of columns
        gamma=0.1,                  # Minimum loss reduction
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    # Train the model
    print("\n3. Training XGBoost...")
    start_time = time.time()
    
    # Use evaluation set for early stopping
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50  # Print progress every 50 iterations
    )
    
    training_time = time.time() - start_time
    print(f"   ✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate on training set
    print("\n5. Training Set Performance:")
    train_metrics = evaluate_model(y_train, y_train_pred, "XGBoost - TRAINING")
    
    # Evaluate on test set
    print("\n6. Test Set Performance:")
    test_metrics = evaluate_model(y_test, y_test_pred, "XGBoost - TEST")
    
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
    
    joblib.dump(model, models_dir / 'xgboost_model.pkl')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'error': y_test - y_test_pred
    })
    predictions_df.to_csv(results_dir / 'xgboost_predictions.csv', index=False)
    
    # Save feature importance
    feature_importance.to_csv(results_dir / 'xgboost_feature_importance.csv', index=False)
    
    print(f"   ✓ Model saved: {models_dir / 'xgboost_model.pkl'}")
    print(f"   ✓ Predictions saved: {results_dir / 'xgboost_predictions.csv'}")
    print(f"   ✓ Feature importance saved: {results_dir / 'xgboost_feature_importance.csv'}")
    
    print("\n✅ XGBoost training complete!")
    print("="*60)
    
    return model, test_metrics

if __name__ == "__main__":
    model, metrics = train_xgboost()
