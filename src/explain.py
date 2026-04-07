"""
Credit Risk Explainability Module
Provides SHAP-based global and local explanations for model predictions.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

# Constants
MODELS_PATH = "models/"
PROCESSED_DATA_PATH = "data/processed/"


def load_model_and_data():
    """
    Load the trained XGBoost model and test data.
    
    Returns:
        Tuple of (model, X_test, y_test, feature_names)
    """
    # Load model
    model = joblib.load(os.path.join(MODELS_PATH, "xgboost.pkl"))
    
    # Load test data
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv")).squeeze()
    
    # Load feature names
    with open(os.path.join(MODELS_PATH, "feature_names.txt"), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    return model, X_test, y_test, feature_names


def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for given data.
    
    Args:
        model: Trained model
        X: Features DataFrame
        
    Returns:
        Tuple of (explainer, shap_values)
    """
    print("🔍 Computing SHAP values...")
    
    # Create SHAP explainer for tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print(f"   ✅ Computed SHAP values for {len(X):,} samples")
    
    return explainer, shap_values


def plot_global_importance(shap_values, X: pd.DataFrame, 
                           save_path: str = None,
                           max_display: int = 15):
    """
    Plot global feature importance using SHAP summary.
    
    Args:
        shap_values: Computed SHAP values
        X: Features DataFrame
        save_path: Path to save figure
        max_display: Maximum features to display
    """
    print("\n📊 Generating Global Feature Importance...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved to {save_path}")
    
    plt.close()


def plot_bar_importance(shap_values, X: pd.DataFrame,
                        save_path: str = None,
                        max_display: int = 15):
    """
    Plot bar chart of mean absolute SHAP values.
    
    Args:
        shap_values: Computed SHAP values
        X: Features DataFrame
        save_path: Path to save figure
        max_display: Maximum features to display
    """
    print("\n📊 Generating Feature Importance Bar Chart...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", 
                      max_display=max_display, show=False)
    plt.title("Mean |SHAP Value| (Feature Importance)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved to {save_path}")
    
    plt.close()


def explain_single_prediction(model, explainer, X: pd.DataFrame, 
                              idx: int, feature_names: list,
                              save_path: str = None) -> dict:
    """
    Explain a single prediction with local SHAP values.
    
    Args:
        model: Trained model
        explainer: SHAP explainer
        X: Features DataFrame
        idx: Index of sample to explain
        feature_names: List of feature names
        save_path: Path to save waterfall plot
        
    Returns:
        Dictionary with explanation details
    """
    # Get single sample
    sample = X.iloc[idx:idx+1]
    
    # Prediction
    pred_proba = model.predict_proba(sample)[0][1]
    pred_class = model.predict(sample)[0]
    
    # SHAP values for this sample
    shap_values = explainer.shap_values(sample)
    
    # Get top contributing features
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0],
        'feature_value': sample.values[0]
    })
    feature_contributions['abs_shap'] = abs(feature_contributions['shap_value'])
    feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
    
    # Determine risk category
    if pred_proba < 0.3:
        risk_level = "🟢 LOW RISK"
    elif pred_proba < 0.6:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🔴 HIGH RISK"
    
    explanation = {
        'sample_idx': idx,
        'default_probability': pred_proba,
        'predicted_class': pred_class,
        'risk_level': risk_level,
        'base_value': explainer.expected_value,
        'top_factors': feature_contributions.head(5).to_dict('records')
    }
    
    # Generate waterfall plot
    if save_path:
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=sample.values[0],
                feature_names=feature_names
            ),
            max_display=10,
            show=False
        )
        plt.title(f"Prediction Explanation - {risk_level}", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved waterfall plot to {save_path}")
    
    return explanation


def format_explanation(explanation: dict) -> str:
    """
    Format explanation as a readable text summary.
    
    Args:
        explanation: Dictionary from explain_single_prediction
        
    Returns:
        Formatted string explanation
    """
    output = []
    output.append("=" * 50)
    output.append("📋 LOAN DEFAULT PREDICTION EXPLANATION")
    output.append("=" * 50)
    output.append(f"\n{explanation['risk_level']}")
    output.append(f"Default Probability: {explanation['default_probability']:.1%}")
    output.append(f"Prediction: {'Default' if explanation['predicted_class'] == 1 else 'Fully Paid'}")
    
    output.append("\n📊 Top Contributing Factors:")
    output.append("-" * 40)
    
    for i, factor in enumerate(explanation['top_factors'], 1):
        direction = "↑ increases" if factor['shap_value'] > 0 else "↓ decreases"
        output.append(
            f"{i}. {factor['feature']}: {factor['feature_value']:.2f}"
            f"\n   Impact: {direction} default risk by {abs(factor['shap_value']):.3f}"
        )
    
    output.append("=" * 50)
    
    return "\n".join(output)


def get_risk_factors_summary(explanation: dict) -> str:
    """
    Generate a brief summary of risk factors for dashboard display.
    
    Args:
        explanation: Dictionary from explain_single_prediction
        
    Returns:
        Brief summary string
    """
    positive_factors = [f for f in explanation['top_factors'] if f['shap_value'] > 0]
    negative_factors = [f for f in explanation['top_factors'] if f['shap_value'] < 0]
    
    summary = []
    
    if positive_factors:
        summary.append("⚠️ Risk Factors:")
        for f in positive_factors[:3]:
            summary.append(f"  • High {f['feature'].replace('_', ' ')}")
    
    if negative_factors:
        summary.append("\n✅ Positive Factors:")
        for f in negative_factors[:3]:
            summary.append(f"  • Good {f['feature'].replace('_', ' ')}")
    
    return "\n".join(summary)


def run_explanation_pipeline(sample_idx: int = 0) -> dict:
    """
    Run the complete explanation pipeline.
    
    Args:
        sample_idx: Index of sample to explain
        
    Returns:
        Dictionary with explanation results
    """
    print("=" * 50)
    print("🔍 CREDIT RISK EXPLANATION PIPELINE")
    print("=" * 50)
    
    # Step 1: Load model and data
    print("\n📁 Step 1: Loading model and data...")
    model, X_test, y_test, feature_names = load_model_and_data()
    
    # Step 2: Compute SHAP values
    print("\n🧮 Step 2: Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_test)
    
    # Step 3: Generate global importance plots
    print("\n📊 Step 3: Generating global importance plots...")
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    plot_global_importance(
        shap_values, X_test,
        save_path=os.path.join(MODELS_PATH, "shap_summary.png")
    )
    
    plot_bar_importance(
        shap_values, X_test,
        save_path=os.path.join(MODELS_PATH, "shap_bar.png")
    )
    
    # Step 4: Explain single prediction
    print("\n🎯 Step 4: Explaining single prediction...")
    explanation = explain_single_prediction(
        model, explainer, X_test, sample_idx, feature_names,
        save_path=os.path.join(MODELS_PATH, f"waterfall_sample_{sample_idx}.png")
    )
    
    # Print formatted explanation
    print(format_explanation(explanation))
    
    print("\n" + "=" * 50)
    print("✅ EXPLANATION PIPELINE COMPLETE!")
    print("=" * 50)
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'explanation': explanation,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    # Run explanation for first test sample
    result = run_explanation_pipeline(sample_idx=0)
    
    # Also explain a few more samples
    for idx in [10, 50, 100]:
        model, X_test, y_test, feature_names = load_model_and_data()
        explainer, _ = compute_shap_values(model, X_test)
        explanation = explain_single_prediction(
            model, explainer, X_test, idx, feature_names,
            save_path=os.path.join(MODELS_PATH, f"waterfall_sample_{idx}.png")
        )
        print(f"\n--- Sample {idx} ---")
        print(format_explanation(explanation))
