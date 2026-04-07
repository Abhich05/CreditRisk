"""
Credit Risk Prediction Dashboard
Streamlit app for loan default prediction with explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODELS_PATH = "models/"

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2980b9, #1f618d);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    model_path = os.path.join(MODELS_PATH, "xgboost.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_feature_names():
    """Load feature names."""
    filepath = os.path.join(MODELS_PATH, "feature_names.txt")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read().strip().split('\n')
    return None


def create_probability_gauge(probability):
    """Create a gauge chart for default probability."""
    # Determine color based on probability
    if probability < 0.3:
        color = "#27ae60"  # Green
    elif probability < 0.6:
        color = "#f39c12"  # Orange
    else:
        color = "#e74c3c"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': "#d5f5e3"},
                {'range': [30, 60], 'color': "#fdebd0"},
                {'range': [60, 100], 'color': "#fadbd8"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        },
        title={'text': "Default Probability", 'font': {'size': 20}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def get_risk_badge(probability):
    """Return risk level badge HTML."""
    if probability < 0.3:
        return '<div class="risk-low">🟢 LOW RISK</div>'
    elif probability < 0.6:
        return '<div class="risk-medium">🟡 MEDIUM RISK</div>'
    else:
        return '<div class="risk-high">🔴 HIGH RISK</div>'


def prepare_input_features(input_data, feature_names):
    """Prepare input features for prediction."""
    # Create DataFrame with all required features
    features = pd.DataFrame(columns=feature_names)
    
    # Map input data to features
    features.loc[0] = 0  # Initialize with zeros
    
    # Direct mappings
    features.loc[0, 'credit_policy'] = input_data['credit_policy']
    features.loc[0, 'int_rate'] = input_data['int_rate']
    features.loc[0, 'installment'] = input_data['installment']
    features.loc[0, 'log_annual_inc'] = np.log(input_data['annual_income'])
    features.loc[0, 'dti'] = input_data['dti']
    features.loc[0, 'fico'] = input_data['fico']
    features.loc[0, 'days_with_cr_line'] = input_data['credit_history_years'] * 365.25
    features.loc[0, 'revol_bal'] = input_data['revol_bal']
    features.loc[0, 'revol_util'] = input_data['revol_util']
    features.loc[0, 'inq_last_6mths'] = input_data['inq_last_6mths']
    features.loc[0, 'delinq_2yrs'] = input_data['delinq_2yrs']
    features.loc[0, 'pub_rec'] = input_data['pub_rec']
    
    # Engineered features
    monthly_income = input_data['annual_income'] / 12
    features.loc[0, 'income_to_installment_ratio'] = monthly_income / input_data['installment']
    features.loc[0, 'credit_age_years'] = input_data['credit_history_years']
    features.loc[0, 'delinq_risk_score'] = (
        input_data['delinq_2yrs'] * 2 + 
        input_data['pub_rec'] * 3 + 
        input_data['inq_last_6mths'] * 1
    )
    
    # FICO category (encoded as numeric)
    fico = input_data['fico']
    if fico < 580:
        features.loc[0, 'fico_category'] = 0
    elif fico < 670:
        features.loc[0, 'fico_category'] = 1
    elif fico < 740:
        features.loc[0, 'fico_category'] = 2
    elif fico < 800:
        features.loc[0, 'fico_category'] = 3
    else:
        features.loc[0, 'fico_category'] = 4
    
    # DTI category
    dti = input_data['dti']
    if dti < 10:
        features.loc[0, 'dti_category'] = 0
    elif dti < 20:
        features.loc[0, 'dti_category'] = 1
    elif dti < 30:
        features.loc[0, 'dti_category'] = 2
    else:
        features.loc[0, 'dti_category'] = 3
    
    # Purpose one-hot encoding
    purpose = input_data['purpose']
    purpose_cols = [col for col in feature_names if col.startswith('purpose_')]
    for col in purpose_cols:
        purpose_name = col.replace('purpose_', '')
        features.loc[0, col] = 1 if purpose == purpose_name else 0
    
    return features


def explain_prediction(model, features, feature_names):
    """Generate SHAP explanation for prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Create waterfall data
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0],
        'Feature Value': features.values[0]
    })
    contributions['Abs SHAP'] = abs(contributions['SHAP Value'])
    contributions = contributions.sort_values('Abs SHAP', ascending=False)
    
    return contributions, explainer.expected_value


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model = load_model()
    feature_names = load_feature_names()
    
    if model is None:
        st.error("⚠️ Model not found! Please run the training pipeline first.")
        st.code("cd credit_risk\npython -m src.preprocessing\npython -m src.model", language="bash")
        return
    
    # Sidebar - Input Form
    st.sidebar.header("📝 Applicant Information")
    
    # Personal Information
    st.sidebar.subheader("💰 Financial Details")
    
    annual_income = st.sidebar.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=500000,
        value=60000,
        step=5000,
        help="Applicant's annual income"
    )
    
    installment = st.sidebar.number_input(
        "Monthly Loan Payment ($)",
        min_value=50,
        max_value=2000,
        value=300,
        step=25,
        help="Monthly installment amount"
    )
    
    int_rate = st.sidebar.slider(
        "Interest Rate (%)",
        min_value=5.0,
        max_value=25.0,
        value=12.0,
        step=0.5,
        help="Loan interest rate"
    ) / 100  # Convert to decimal
    
    dti = st.sidebar.slider(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
        help="Monthly debt payments / monthly income"
    )
    
    # Credit History
    st.sidebar.subheader("📊 Credit History")
    
    fico = st.sidebar.slider(
        "FICO Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        step=5,
        help="Credit score (300-850)"
    )
    
    credit_history_years = st.sidebar.slider(
        "Credit History (Years)",
        min_value=0.5,
        max_value=30.0,
        value=10.0,
        step=0.5,
        help="Length of credit history"
    )
    
    revol_bal = st.sidebar.number_input(
        "Revolving Balance ($)",
        min_value=0,
        max_value=100000,
        value=5000,
        step=500,
        help="Total credit card balance"
    )
    
    revol_util = st.sidebar.slider(
        "Credit Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="% of credit limit being used"
    )
    
    # Risk Indicators
    st.sidebar.subheader("⚠️ Risk Indicators")
    
    inq_last_6mths = st.sidebar.number_input(
        "Credit Inquiries (Last 6 Months)",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of credit inquiries"
    )
    
    delinq_2yrs = st.sidebar.number_input(
        "Delinquencies (Last 2 Years)",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of 30+ day delinquencies"
    )
    
    pub_rec = st.sidebar.number_input(
        "Public Records",
        min_value=0,
        max_value=5,
        value=0,
        help="Bankruptcies, liens, judgments"
    )
    
    # Loan Details
    st.sidebar.subheader("📋 Loan Details")
    
    credit_policy = st.sidebar.selectbox(
        "Meets Credit Policy",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Does applicant meet underwriting criteria?"
    )
    
    purpose = st.sidebar.selectbox(
        "Loan Purpose",
        options=[
            'debt_consolidation', 'credit_card', 'home_improvement',
            'small_business', 'major_purchase', 'all_other', 'educational'
        ],
        help="Purpose of the loan"
    )
    
    # Predict Button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Default Risk", use_container_width=True)
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    if predict_button:
        # Prepare input
        input_data = {
            'credit_policy': credit_policy,
            'purpose': purpose,
            'int_rate': int_rate,
            'installment': installment,
            'annual_income': annual_income,
            'dti': dti,
            'fico': fico,
            'credit_history_years': credit_history_years,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'inq_last_6mths': inq_last_6mths,
            'delinq_2yrs': delinq_2yrs,
            'pub_rec': pub_rec
        }
        
        try:
            # Prepare features
            features = prepare_input_features(input_data, feature_names)
            
            # Make prediction
            probability = model.predict_proba(features)[0][1]
            prediction = model.predict(features)[0]
            
            # Display results
            with col1:
                st.subheader("🎯 Prediction Result")
                st.markdown(get_risk_badge(probability), unsafe_allow_html=True)
                st.plotly_chart(create_probability_gauge(probability), use_container_width=True)
                
                # Key metrics
                st.subheader("📊 Key Metrics")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("FICO Score", f"{fico}", 
                              delta="Good" if fico >= 700 else "Fair" if fico >= 650 else "Poor")
                with m2:
                    st.metric("DTI Ratio", f"{dti}%",
                              delta="Low" if dti < 15 else "Medium" if dti < 25 else "High")
                with m3:
                    st.metric("Income/Payment", f"{(annual_income/12)/installment:.1f}x",
                              delta="Good" if (annual_income/12)/installment > 5 else "Fair")
            
            with col2:
                st.subheader("🔍 Explanation (SHAP Analysis)")
                
                # Get SHAP explanation
                contributions, base_value = explain_prediction(model, features, feature_names)
                
                # Top factors
                st.markdown("**Top Contributing Factors:**")
                
                top_factors = contributions.head(8)
                
                for _, row in top_factors.iterrows():
                    feature = row['Feature'].replace('_', ' ').title()
                    shap_val = row['SHAP Value']
                    
                    if shap_val > 0:
                        st.markdown(f"⬆️ **{feature}**: Increases risk by {abs(shap_val):.3f}")
                    else:
                        st.markdown(f"⬇️ **{feature}**: Decreases risk by {abs(shap_val):.3f}")
                
                # Bar chart of contributions
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#e74c3c' if x > 0 else '#27ae60' for x in top_factors['SHAP Value']]
                bars = ax.barh(
                    top_factors['Feature'].str.replace('_', ' ').str.title(),
                    top_factors['SHAP Value'],
                    color=colors
                )
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Feature Contributions to Prediction')
                ax.axvline(x=0, color='black', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure the model is trained with all required features.")
    
    else:
        # Welcome message when no prediction yet
        with col1:
            st.info("👈 Fill in the applicant information in the sidebar and click **Predict Default Risk** to get started.")
            
            st.subheader("ℹ️ About This System")
            st.markdown("""
            This **Credit Risk Prediction System** uses machine learning to:
            
            1. **Predict** the probability of loan default
            2. **Classify** applicants into risk categories (Low/Medium/High)
            3. **Explain** the key factors driving each prediction
            
            **Model Details:**
            - Algorithm: XGBoost Classifier
            - Trained on: Lending Club Loan Data
            - Explainability: SHAP (SHapley Additive exPlanations)
            """)
        
        with col2:
            st.subheader("📈 Risk Categories")
            st.markdown("""
            | Category | Probability Range | Description |
            |----------|------------------|-------------|
            | 🟢 Low | < 30% | Low default risk |
            | 🟡 Medium | 30% - 60% | Moderate risk |
            | 🔴 High | > 60% | High default risk |
            """)
            
            st.subheader("🎯 Key Features")
            st.markdown("""
            - **FICO Score**: Credit score (most important)
            - **DTI Ratio**: Debt burden indicator
            - **Delinquencies**: Past payment behavior
            - **Credit Utilization**: How much credit is used
            - **Income**: Ability to repay
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🏦 Credit Risk Prediction System | Built with Streamlit & XGBoost | "
        "For educational purposes only"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
