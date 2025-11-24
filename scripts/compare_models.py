import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def load_predictions():
    """Load predictions from all models"""
    
    results_dir = PROJECT_ROOT / 'results'
    rf_pred = pd.read_csv(results_dir / 'random_forest_predictions.csv')
    xgb_pred = pd.read_csv(results_dir / 'xgboost_predictions.csv')
    lstm_pred = pd.read_csv(results_dir / 'lstm_predictions.csv')
    
    return rf_pred, xgb_pred, lstm_pred

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def create_comparison_visualizations():
    """Create comprehensive comparison visualizations"""
    
    print("="*60)
    print("MODEL COMPARISON & VISUALIZATION")
    print("="*60)
    
    # Load predictions
    print("\n1. Loading predictions...")
    rf_pred, xgb_pred, lstm_pred = load_predictions()
    
    # Calculate metrics
    print("\n2. Calculating metrics...")
    rf_metrics = calculate_metrics(rf_pred['y_true'], rf_pred['y_pred'])
    xgb_metrics = calculate_metrics(xgb_pred['y_true'], xgb_pred['y_pred'])
    lstm_metrics = calculate_metrics(lstm_pred['y_true'], lstm_pred['y_pred'])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LSTM'],
        'RMSE': [rf_metrics['RMSE'], xgb_metrics['RMSE'], lstm_metrics['RMSE']],
        'MAE': [rf_metrics['MAE'], xgb_metrics['MAE'], lstm_metrics['MAE']],
        'R²': [rf_metrics['R2'], xgb_metrics['R2'], lstm_metrics['R2']]
    })
    
    print("\n3. Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tourist Footfall Prediction - Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    models = comparison_df['Model']
    rmse_values = comparison_df['RMSE']
    bars1 = ax1.bar(models, rmse_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim(0, max(rmse_values) * 1.2)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. MAE Comparison
    ax2 = axes[0, 1]
    mae_values = comparison_df['MAE']
    bars2 = ax2.bar(models, mae_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_title('MAE Comparison (Lower is Better)', fontweight='bold')
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, max(mae_values) * 1.2)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. R² Comparison
    ax3 = axes[0, 2]
    r2_values = comparison_df['R²']
    bars3 = ax3.bar(models, r2_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax3.set_title('R² Score Comparison (Higher is Better)', fontweight='bold')
    ax3.set_ylabel('R² Score')
    ax3.set_ylim(0.85, 1.0)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Actual vs Predicted - Random Forest
    ax4 = axes[1, 0]
    sample_size = min(500, len(rf_pred))
    ax4.scatter(rf_pred['y_true'][:sample_size], rf_pred['y_pred'][:sample_size], 
                alpha=0.5, s=20, color='#2ecc71')
    ax4.plot([rf_pred['y_true'].min(), rf_pred['y_true'].max()],
             [rf_pred['y_true'].min(), rf_pred['y_true'].max()], 
             'r--', lw=2)
    ax4.set_title('Random Forest: Actual vs Predicted', fontweight='bold')
    ax4.set_xlabel('Actual Footfall')
    ax4.set_ylabel('Predicted Footfall')
    
    # 5. Actual vs Predicted - XGBoost
    ax5 = axes[1, 1]
    ax5.scatter(xgb_pred['y_true'][:sample_size], xgb_pred['y_pred'][:sample_size], 
                alpha=0.5, s=20, color='#3498db')
    ax5.plot([xgb_pred['y_true'].min(), xgb_pred['y_true'].max()],
             [xgb_pred['y_true'].min(), xgb_pred['y_true'].max()], 
             'r--', lw=2)
    ax5.set_title('XGBoost: Actual vs Predicted', fontweight='bold')
    ax5.set_xlabel('Actual Footfall')
    ax5.set_ylabel('Predicted Footfall')
    
    # 6. Actual vs Predicted - LSTM
    ax6 = axes[1, 2]
    ax6.scatter(lstm_pred['y_true'][:sample_size], lstm_pred['y_pred'][:sample_size], 
                alpha=0.5, s=20, color='#e74c3c')
    ax6.plot([lstm_pred['y_true'].min(), lstm_pred['y_true'].max()],
             [lstm_pred['y_true'].min(), lstm_pred['y_true'].max()], 
             'r--', lw=2)
    ax6.set_title('LSTM: Actual vs Predicted', fontweight='bold')
    ax6.set_xlabel('Actual Footfall')
    ax6.set_ylabel('Predicted Footfall')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {results_dir / 'model_comparison_comprehensive.png'}")
    
    # Create error distribution plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')
    
    axes2[0].hist(rf_pred['error'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes2[0].set_title('Random Forest Errors')
    axes2[0].set_xlabel('Error (Actual - Predicted)')
    axes2[0].set_ylabel('Frequency')
    axes2[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    axes2[1].hist(xgb_pred['error'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes2[1].set_title('XGBoost Errors')
    axes2[1].set_xlabel('Error (Actual - Predicted)')
    axes2[1].set_ylabel('Frequency')
    axes2[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    axes2[2].hist(lstm_pred['error'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes2[2].set_title('LSTM Errors')
    axes2[2].set_xlabel('Error (Actual - Predicted)')
    axes2[2].set_ylabel('Frequency')
    axes2[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {results_dir / 'error_distribution.png'}")
    
    # Best model summary
    print("\n5. Best Model:")
    best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
    print(f"   🏆 {best_model['Model']} achieves the best performance!")
    print(f"   - RMSE: {best_model['RMSE']:.2f}")
    print(f"   - MAE: {best_model['MAE']:.2f}")
    print(f"   - R²: {best_model['R²']:.4f}")
    
    print("\n✅ Model comparison complete!")
    print("="*60)

if __name__ == "__main__":
    create_comparison_visualizations()
