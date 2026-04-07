"""
Credit Risk Preprocessing Module
Handles data loading, cleaning, and preparation for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Constants
RAW_DATA_PATH = "data/raw/loan_data.csv"
PROCESSED_DATA_PATH = "data/processed/"
RANDOM_STATE = 42

# Target column
TARGET_COL = "not_fully_paid"

# Feature columns after preprocessing
NUMERICAL_COLS = [
    'credit_policy', 'int_rate', 'installment', 'log_annual_inc', 
    'dti', 'fico', 'days_with_cr_line', 'revol_bal', 'revol_util',
    'inq_last_6mths', 'delinq_2yrs', 'pub_rec'
]

CATEGORICAL_COLS = ['purpose']


def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw loan data from CSV.
    
    Args:
        filepath: Path to the raw CSV file
        
    Returns:
        DataFrame with raw loan data
    """
    df = pd.read_csv(filepath)
    
    # Rename columns: replace '.' with '_' for Python compatibility
    df.columns = [col.replace('.', '_') for col in df.columns]
    
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"   Target distribution: {df[TARGET_COL].value_counts().to_dict()}")
    
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze data quality and return summary statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'target_balance': df[TARGET_COL].value_counts().to_dict(),
        'class_imbalance_ratio': df[TARGET_COL].value_counts()[0] / df[TARGET_COL].value_counts()[1]
    }
    
    print("\n📊 Data Quality Report:")
    print(f"   Total rows: {quality_report['total_rows']:,}")
    print(f"   Missing values: {sum(df.isnull().sum())}")
    print(f"   Class imbalance ratio: {quality_report['class_imbalance_ratio']:.2f}:1 (Paid:Default)")
    
    return quality_report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Numerical: Fill with median
    - Categorical: Fill with mode
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        print("✅ No missing values found")
        return df
    
    for col in missing_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   Filled {col} with median: {df[col].median():.2f}")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   Filled {col} with mode: {df[col].mode()[0]}")
    
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict = None) -> tuple:
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df: Input DataFrame
        fit: Whether to fit new encoders or use existing
        encoders: Pre-fitted encoders (if fit=False)
        
    Returns:
        Tuple of (encoded DataFrame, encoders dict)
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    # One-hot encode 'purpose' column
    if 'purpose' in df.columns:
        purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose', drop_first=True)
        df = pd.concat([df.drop('purpose', axis=1), purpose_dummies], axis=1)
        print(f"✅ One-hot encoded 'purpose' → {len(purpose_dummies.columns)} new columns")
    
    return df, encoders


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features for credit risk prediction.
    
    Features created:
    - income_to_installment_ratio: Affordability indicator
    - fico_category: Risk bands based on FICO score
    - dti_category: Risk bands based on DTI
    - credit_age_years: Credit history in years
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # 1. Income to Installment Ratio (affordability)
    # Note: log_annual_inc is already log-transformed, so we exponentiate
    df['annual_income'] = np.exp(df['log_annual_inc'])
    df['monthly_income'] = df['annual_income'] / 12
    df['income_to_installment_ratio'] = df['monthly_income'] / df['installment']
    
    # 2. FICO Category (Risk Bands)
    df['fico_category'] = pd.cut(
        df['fico'],
        bins=[0, 579, 669, 739, 799, 850],
        labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
    
    # 3. DTI Category (Risk Bands)
    df['dti_category'] = pd.cut(
        df['dti'],
        bins=[-1, 10, 20, 30, 100],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # 4. Credit Age in Years
    df['credit_age_years'] = df['days_with_cr_line'] / 365.25
    
    # 5. Delinquency Risk Score (weighted)
    df['delinq_risk_score'] = (
        df['delinq_2yrs'] * 2 + 
        df['pub_rec'] * 3 + 
        df['inq_last_6mths'] * 1
    )
    
    print(f"✅ Created 5 engineered features")
    
    # Drop intermediate columns
    df.drop(['annual_income', 'monthly_income'], axis=1, inplace=True)
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare final feature matrix for modeling.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Tuple of (X features DataFrame, y target Series, feature_names list)
    """
    # Separate target
    y = df[TARGET_COL].copy()
    X = df.drop(TARGET_COL, axis=1).copy()
    
    # Convert categorical columns to numeric for modeling
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Prepared {len(feature_names)} features for modeling")
    
    return X, y, feature_names


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.15, 
               val_size: float = 0.15) -> dict:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        val_size: Proportion for validation set
        
    Returns:
        Dictionary with train/val/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return splits


def run_preprocessing_pipeline(save: bool = True) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        save: Whether to save processed data
        
    Returns:
        Dictionary with processed data splits
    """
    print("=" * 50)
    print("🚀 CREDIT RISK PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📁 Step 1: Loading data...")
    df = load_data()
    
    # Step 2: Check data quality
    print("\n🔍 Step 2: Checking data quality...")
    quality_report = check_data_quality(df)
    
    # Step 3: Handle missing values
    print("\n🔧 Step 3: Handling missing values...")
    df = handle_missing_values(df)
    
    # Step 4: Encode categoricals
    print("\n🏷️ Step 4: Encoding categorical variables...")
    df, encoders = encode_categoricals(df)
    
    # Step 5: Feature engineering
    print("\n⚙️ Step 5: Engineering features...")
    df = create_features(df)
    
    # Step 6: Prepare features
    print("\n📦 Step 6: Preparing feature matrix...")
    X, y, feature_names = prepare_features(df)
    
    # Step 7: Split data
    print("\n✂️ Step 7: Splitting data...")
    splits = split_data(X, y)
    
    # Save processed data
    if save:
        print("\n💾 Saving processed data...")
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # Save splits
        for name, data in splits.items():
            filepath = os.path.join(PROCESSED_DATA_PATH, f"{name}.csv")
            data.to_csv(filepath, index=False)
        
        # Save feature names
        with open(os.path.join(PROCESSED_DATA_PATH, "feature_names.txt"), 'w') as f:
            f.write('\n'.join(feature_names))
        
        # Save quality report
        joblib.dump(quality_report, os.path.join(PROCESSED_DATA_PATH, "quality_report.pkl"))
        
        print(f"   ✅ Saved to {PROCESSED_DATA_PATH}")
    
    print("\n" + "=" * 50)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 50)
    
    return {
        'splits': splits,
        'feature_names': feature_names,
        'quality_report': quality_report,
        'class_weights': {
            0: 1.0,
            1: quality_report['class_imbalance_ratio']
        }
    }


if __name__ == "__main__":
    # Run preprocessing pipeline
    result = run_preprocessing_pipeline(save=True)
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"   Features: {len(result['feature_names'])}")
    print(f"   Class weights for XGBoost: {result['class_weights']}")
