"""
Credit Risk Model Training Module
Trains and evaluates Logistic Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MODELS_PATH = "models/"
PROCESSED_DATA_PATH = "data/processed/"
RANDOM_STATE = 42


def load_processed_data() -> dict:
    """
    Load preprocessed train/val/test splits.
    
    Returns:
        Dictionary with data splits
    """
    splits = {}
    for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        filepath = os.path.join(PROCESSED_DATA_PATH, f"{name}.csv")
        splits[name] = pd.read_csv(filepath)
        if 'y_' in name:
            splits[name] = splits[name].squeeze()
    
    # Load feature names
    with open(os.path.join(PROCESSED_DATA_PATH, "feature_names.txt"), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    print(f"✅ Loaded processed data")
    print(f"   Train: {len(splits['X_train']):,} | Val: {len(splits['X_val']):,} | Test: {len(splits['X_test']):,}")
    
    return splits, feature_names


def calculate_class_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost based on class imbalance.
    
    Args:
        y_train: Training target values
        
    Returns:
        scale_pos_weight value
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    print(f"   Class distribution: {n_neg:,} paid (0) vs {n_pos:,} default (1)")
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")
    
    return scale_pos_weight


def train_logistic_regression(X_train, y_train, class_weight='balanced') -> LogisticRegression:
    """
    Train Logistic Regression baseline model.
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weight: How to handle class imbalance
        
    Returns:
        Trained model
    """
    print("\n🔵 Training Logistic Regression (Baseline)...")
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Logistic Regression trained")
    
    return model


def train_random_forest(X_train, y_train, class_weight='balanced') -> RandomForestClassifier:
    """
    Train Random Forest comparison model.
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weight: How to handle class imbalance
        
    Returns:
        Trained model
    """
    print("\n🌲 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Random Forest trained")
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val, 
                  scale_pos_weight: float = None,
                  tune: bool = False) -> XGBClassifier:
    """
    Train XGBoost primary model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scale_pos_weight: Weight for positive class
        tune: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    print("\n🚀 Training XGBoost (Primary Model)...")
    
    if scale_pos_weight is None:
        scale_pos_weight = calculate_class_weight(y_train)
    
    if tune:
        print("   🔧 Performing hyperparameter tuning...")
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3, 5]
        }
        
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='auc'
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"   Best params: {grid_search.best_params_}")
        
    else:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='auc',
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    print("   ✅ XGBoost trained")
    
    return model


def evaluate_model(model, X, y, model_name: str) -> dict:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        model_name: Name for display
        
    Returns:
        Dictionary with metrics
    """
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'avg_precision': average_precision_score(y, y_pred_proba),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    return metrics


def print_evaluation_results(metrics: dict, dataset_name: str = "Test"):
    """
    Print formatted evaluation results.
    
    Args:
        metrics: Dictionary with metrics
        dataset_name: Name of dataset being evaluated
    """
    print(f"\n📊 {metrics['model']} - {dataset_name} Set Results:")
    print(f"   ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"   Avg Precision:  {metrics['avg_precision']:.4f}")
    print(f"   Precision:      {metrics['precision']:.4f}")
    print(f"   Recall:         {metrics['recall']:.4f}")
    print(f"   F1 Score:       {metrics['f1']:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"   FN={cm[1,0]:,}  TP={cm[1,1]:,}")


def compare_models(models_metrics: list) -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Args:
        models_metrics: List of metric dictionaries
        
    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame([
        {
            'Model': m['model'],
            'ROC-AUC': m['roc_auc'],
            'Avg Precision': m['avg_precision'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1': m['f1']
        }
        for m in models_metrics
    ])
    
    comparison = comparison.sort_values('ROC-AUC', ascending=False)
    
    return comparison


def save_model(model, model_name: str, feature_names: list = None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for the saved file
        feature_names: List of feature names
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    filepath = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    
    if feature_names:
        with open(os.path.join(MODELS_PATH, "feature_names.txt"), 'w') as f:
            f.write('\n'.join(feature_names))
    
    print(f"   💾 Saved {model_name} to {filepath}")


def load_model(model_name: str):
    """
    Load trained model from disk.
    
    Args:
        model_name: Name of the saved model
        
    Returns:
        Loaded model
    """
    filepath = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    return joblib.load(filepath)


def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for all models.
    
    Args:
        models_dict: Dictionary of {name: model}
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Credit Risk Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📈 Saved ROC curves to {save_path}")
    
    plt.close()


def run_training_pipeline(tune_xgboost: bool = False) -> dict:
    """
    Run the complete model training pipeline.
    
    Args:
        tune_xgboost: Whether to tune XGBoost hyperparameters
        
    Returns:
        Dictionary with trained models and results
    """
    print("=" * 50)
    print("🚀 CREDIT RISK MODEL TRAINING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📁 Step 1: Loading processed data...")
    splits, feature_names = load_processed_data()
    
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    # Step 2: Calculate class weight
    print("\n⚖️ Step 2: Calculating class weights...")
    scale_pos_weight = calculate_class_weight(y_train)
    
    # Step 3: Train models
    print("\n🎯 Step 3: Training models...")
    
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, 
                               scale_pos_weight, tune=tune_xgboost)
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    # Step 4: Evaluate on validation set
    print("\n📊 Step 4: Evaluating on validation set...")
    val_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        val_metrics.append(metrics)
        print_evaluation_results(metrics, "Validation")
    
    # Step 5: Evaluate on test set
    print("\n📊 Step 5: Evaluating on test set...")
    test_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        test_metrics.append(metrics)
        print_evaluation_results(metrics, "Test")
    
    # Step 6: Compare models
    print("\n📋 Step 6: Model Comparison (Test Set):")
    comparison = compare_models(test_metrics)
    print(comparison.to_string(index=False))
    
    # Step 7: Save best model (XGBoost)
    print("\n💾 Step 7: Saving models...")
    for name, model in models.items():
        save_model(model, name.lower().replace(' ', '_'), feature_names)
    
    # Step 8: Plot ROC curves
    print("\n📈 Step 8: Generating ROC curves...")
    plot_roc_curves(models, X_test, y_test, 
                    os.path.join(MODELS_PATH, "roc_curves.png"))
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Model: {comparison.iloc[0]['Model']}")
    print(f"   Best ROC-AUC: {comparison.iloc[0]['ROC-AUC']:.4f}")
    print("=" * 50)
    
    return {
        'models': models,
        'test_metrics': test_metrics,
        'comparison': comparison,
        'feature_names': feature_names,
        'best_model': xgb_model
    }


if __name__ == "__main__":
    result = run_training_pipeline(tune_xgboost=False)
