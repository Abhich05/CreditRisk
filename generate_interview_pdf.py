"""
Generate comprehensive interview Q&A PDF for Credit Risk Prediction Project
"""

from fpdf import FPDF


class InterviewPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Credit Risk Prediction - Interview Q&A", align="C")
            self.ln(3)
            self.line(10, 13, 200, 13)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_title(self, title):
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(31, 78, 121)
        self.multi_cell(0, 12, title, align="C")
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def add_subtitle(self, subtitle):
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 8, subtitle, align="C")
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def add_section(self, title):
        self.ln(5)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(31, 78, 121)
        self.cell(0, 10, title)
        self.ln(12)
        self.set_text_color(0, 0, 0)

    def add_subsection(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(41, 128, 185)
        self.cell(0, 9, title)
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def add_question(self, question, answer):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(192, 57, 43)
        q_lines = self.multi_cell(0, 7, f"Q: {question}", output=True)
        self.ln(3)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, answer)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def add_code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        lines = code.split("\n")
        for line in lines:
            self.cell(0, 5, f"  {line}", fill=True)
            self.ln(5)
        self.ln(3)


pdf = InterviewPDF()
pdf.set_auto_page_break(auto=True, margin=20)

# ==================== COVER PAGE ====================
pdf.add_page()
pdf.ln(40)
pdf.add_title("Credit Risk Prediction System")
pdf.ln(5)
pdf.add_subtitle("Comprehensive Interview Preparation Guide")
pdf.ln(10)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(
    0,
    7,
    (
        "Complete Q&A covering all aspects of the project:\n"
        "Project Overview, Data Preprocessing, Feature Engineering,\n"
        "Machine Learning Models, Model Evaluation, Explainable AI (SHAP),\n"
        "Streamlit Dashboard, Deployment, and Industry Best Practices"
    ),
    align="C",
)
pdf.ln(15)
pdf.set_font("Helvetica", "I", 10)
pdf.multi_cell(
    0,
    7,
    (
        "Technologies: Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit\n"
        "Dataset: Lending Club Loan Data (~9,600 samples, 14+ features)\n"
        "Target: not_fully_paid (Loan Default Prediction)"
    ),
    align="C",
)
pdf.ln(20)
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "50+ Interview Questions with Detailed Answers", align="C")

# ==================== TABLE OF CONTENTS ====================
pdf.add_page()
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(31, 78, 121)
pdf.cell(0, 12, "Table of Contents")
pdf.ln(15)
pdf.set_text_color(0, 0, 0)

toc = [
    ("Section 1", "Project Overview & Basics", 5),
    ("Section 2", "Dataset & Data Understanding", 8),
    ("Section 3", "Data Preprocessing & Cleaning", 11),
    ("Section 4", "Feature Engineering", 14),
    ("Section 5", "Machine Learning Models", 17),
    ("Section 6", "Model Evaluation & Metrics", 20),
    ("Section 7", "Class Imbalance Handling", 23),
    ("Section 8", "Explainable AI (SHAP)", 26),
    ("Section 9", "Streamlit Dashboard & Deployment", 29),
    ("Section 10", "Industry & Regulatory Context", 32),
    ("Section 11", "Advanced & Behavioral Questions", 35),
]

for sec, title, _ in toc:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(35, 8, sec)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, title)
    pdf.ln(8)

# ==================== SECTION 1: PROJECT OVERVIEW ====================
pdf.add_page()
pdf.add_section("Section 1: Project Overview & Basics")

pdf.add_question(
    "What is this project about?",
    "This is a Credit Risk Prediction System that predicts the probability of a loan applicant defaulting on their loan. "
    "It uses historical financial and behavioral data from Lending Club to build machine learning models. "
    "The project includes data preprocessing, feature engineering, training 3 ML models (Logistic Regression, Random Forest, XGBoost), "
    "explainable AI using SHAP values, and an interactive Streamlit dashboard for real-time predictions with explanations.",
)

pdf.add_question(
    "What is the business problem being solved?",
    "Banks and lending institutions need to assess whether a loan applicant will repay their loan or default. "
    "Manual assessment is slow and error-prone. This project automates credit risk assessment by building a predictive model "
    "that can quickly evaluate loan applications, classify applicants into risk categories (Low/Medium/High), "
    "and provide explainable decisions for regulatory compliance (Basel III, GDPR).",
)

pdf.add_question(
    "Why is credit risk prediction important in the financial industry?",
    "Credit risk is the risk of financial loss when a borrower fails to repay a loan. "
    "Accurate prediction helps: (1) Reduce loan defaults and financial losses, (2) Automate loan approval processes, "
    "(3) Ensure regulatory compliance with explainable decisions, (4) Set appropriate interest rates based on risk, "
    "(5) Improve portfolio management and capital allocation.",
)

pdf.add_question(
    "What is the project structure?",
    "The project follows a modular structure:\n"
    "- data/raw/: Original loan_data.csv dataset\n"
    "- data/processed/: Cleaned train/val/test splits\n"
    "- models/: Saved model artifacts (.pkl files), feature names, ROC curves\n"
    "- notebooks/: EDA and experiments\n"
    "- src/preprocessing.py: Data cleaning, feature engineering, train/val/test splitting\n"
    "- src/model.py: Model training (LR, RF, XGBoost), evaluation, and saving\n"
    "- src/explain.py: SHAP-based global and local explanations\n"
    "- app.py: Streamlit interactive dashboard\n"
    "- requirements.txt: All Python dependencies",
)

pdf.add_question(
    "What are the key technologies used?",
    "Python 3.10+, pandas/numpy (data manipulation), scikit-learn (preprocessing, LR, RF), "
    "XGBoost (primary model), SHAP (explainability), Streamlit (dashboard), "
    "Plotly/matplotlib/seaborn (visualizations), joblib (model serialization).",
)

pdf.add_question(
    "What is the end-to-end pipeline of this project?",
    "Step 1: Load raw CSV data -> Step 2: Check data quality -> Step 3: Handle missing values (median/mode) -> "
    "Step 4: Encode categorical variables (one-hot encoding) -> Step 5: Feature engineering (create 5 new features) -> "
    "Step 6: Split into train/val/test (stratified) -> Step 7: Train 3 models -> Step 8: Evaluate on validation and test sets -> "
    "Step 9: Save best model (XGBoost) -> Step 10: Generate SHAP explanations -> Step 11: Deploy via Streamlit dashboard",
)

# ==================== SECTION 2: DATASET ====================
pdf.add_page()
pdf.add_section("Section 2: Dataset & Data Understanding")

pdf.add_question(
    "What dataset is used in this project?",
    "The Lending Club Loan Data is used. It contains approximately 9,600 loan records with "
    '14 original features plus 5 engineered features. The target variable is "not_fully_paid" '
    "where 0 means the loan was fully paid and 1 means the loan defaulted.",
)

pdf.add_question(
    "What is the target variable and what does it represent?",
    'The target variable is "not_fully_paid". It is a binary classification target:\n'
    "- 0 = Loan was fully paid (negative class)\n"
    "- 1 = Loan was not fully paid/defaulted (positive class)\n"
    "This is a classification problem, specifically binary classification.",
)

pdf.add_question(
    "What are the key features in the dataset? Explain each.",
    "1. credit_policy: Whether the customer meets credit underwriting criteria (0/1)\n"
    "2. int_rate: Interest rate of the loan (decimal)\n"
    "3. installment: Monthly loan payment amount\n"
    "4. log_annual_inc: Log-transformed annual income\n"
    "5. dti: Debt-to-income ratio (monthly debt payments / monthly income)\n"
    "6. fico: FICO credit score (300-850)\n"
    "7. days_with_cr_line: Number of days with a credit line\n"
    "8. revol_bal: Total revolving credit card balance\n"
    "9. revol_util: Credit utilization percentage\n"
    "10. inq_last_6mths: Number of credit inquiries in last 6 months\n"
    "11. delinq_2yrs: Number of 30+ day delinquencies in last 2 years\n"
    "12. pub_rec: Number of public records (bankruptcies, liens, judgments)\n"
    "13. purpose: Loan purpose (categorical: debt_consolidation, credit_card, etc.)",
)

pdf.add_question(
    "What is the class distribution in the dataset? Is it imbalanced?",
    "Yes, the dataset is imbalanced. The majority of loans are fully paid (class 0) while fewer loans default (class 1). "
    "The class imbalance ratio is approximately X:1 (Paid:Default). This imbalance is handled using:\n"
    '- class_weight="balanced" in Logistic Regression and Random Forest\n'
    "- scale_pos_weight parameter in XGBoost (calculated as n_neg/n_pos)",
)

pdf.add_question(
    "What is FICO score and why is it important?",
    "FICO (Fair Isaac Corporation) score is a credit score ranging from 300 to 850 that represents a person's creditworthiness. "
    "It is the most important predictor in credit risk modeling because:\n"
    "- Higher FICO = lower default risk\n"
    "- It summarizes credit history, payment behavior, credit utilization, and credit mix\n"
    "- In this project, FICO consistently ranks as the top feature in SHAP importance\n"
    "- Categories: Very Poor (<580), Fair (580-669), Good (670-739), Very Good (740-799), Excellent (800+)",
)

pdf.add_question(
    "What is DTI ratio and what does it indicate?",
    "DTI (Debt-to-Income) ratio is the percentage of a person's monthly gross income that goes toward paying debts. "
    "Formula: DTI = (Monthly Debt Payments / Monthly Gross Income) x 100.\n"
    "- DTI < 10%: Low risk\n"
    "- DTI 10-20%: Medium risk\n"
    "- DTI 20-30%: High risk\n"
    "- DTI > 30%: Very high risk\n"
    "Higher DTI indicates the borrower has less capacity to take on additional debt.",
)

pdf.add_question(
    "What is credit utilization (revol_util) and why does it matter?",
    "Credit utilization is the percentage of available credit that a borrower is currently using. "
    "Formula: (Total Credit Card Balance / Total Credit Limit) x 100.\n"
    "It matters because:\n"
    "- High utilization (>30%) signals financial stress\n"
    "- It is a major component of FICO score calculation (30% weight)\n"
    "- Borrowers with high utilization are more likely to default",
)

# ==================== SECTION 3: PREPROCESSING ====================
pdf.add_page()
pdf.add_section("Section 3: Data Preprocessing & Cleaning")

pdf.add_question(
    "What are the steps in your preprocessing pipeline?",
    'Step 1: Load data from CSV and rename columns (replace "." with "_")\n'
    "Step 2: Check data quality - missing values, dtypes, class balance\n"
    "Step 3: Handle missing values - numerical columns filled with median, categorical with mode\n"
    'Step 4: Encode categorical variables - one-hot encoding for "purpose" column\n'
    "Step 5: Feature engineering - create 5 new features\n"
    "Step 6: Prepare feature matrix - separate X and y, convert categorical codes\n"
    "Step 7: Split data into train/val/test sets using stratified sampling",
)

pdf.add_question(
    "How do you handle missing values and why?",
    "Numerical columns are filled with the median, and categorical columns with the mode.\n"
    "Why median over mean? The median is robust to outliers. In financial data, features like income, "
    "balance, and interest rates often have extreme values. Using the median prevents these outliers "
    "from skewing the imputation value. For categorical columns, the mode (most frequent value) is the "
    "most reasonable default.",
)

pdf.add_question(
    'Why do you use one-hot encoding instead of label encoding for the "purpose" column?',
    'One-hot encoding is used because "purpose" is a nominal categorical variable with no inherent ordering. '
    'Values like "debt_consolidation", "credit_card", "home_improvement" etc. have no numerical relationship. '
    "Label encoding would assign arbitrary numbers (0, 1, 2...) implying an order that doesn't exist, "
    "which could mislead the model. One-hot encoding creates separate binary columns for each category, "
    "preserving the nominal nature of the data.",
)

pdf.add_question(
    "What is stratified sampling and why is it used for train/val/test split?",
    "Stratified sampling ensures that the class distribution (ratio of paid to defaulted loans) is preserved "
    "in each split (train, validation, test). This is critical for imbalanced datasets because:\n"
    "- Without stratification, a split might have very few or no default cases\n"
    "- The model would not learn to predict the minority class\n"
    "- Evaluation metrics would be unreliable\n"
    "In this project, train_test_split is called twice with stratify=y to maintain class proportions.",
)

pdf.add_question(
    "What is the data split ratio and why?",
    "The data is split into: Train (~70%), Validation (~15%), Test (~15%).\n"
    "- Train set: Used to fit the model parameters\n"
    "- Validation set: Used for hyperparameter tuning and early stopping\n"
    "- Test set: Used for final unbiased evaluation\n"
    "The validation set is essential for XGBoost's early_stopping_rounds parameter and for comparing models "
    "before final test evaluation. With ~9,600 samples, this gives ~6,720 train, ~1,440 val, ~1,440 test.",
)

pdf.add_question(
    "What is data leakage and how do you prevent it in this project?",
    "Data leakage occurs when information from outside the training dataset (or future information) "
    "is used to make predictions, leading to overly optimistic performance that doesn't generalize.\n"
    "Prevention in this project:\n"
    "- Removed post-loan features that wouldn't be available at application time\n"
    "- Only use features available at the time of loan application\n"
    "- Proper train/val/test split prevents test data from influencing training\n"
    "- Feature engineering uses only pre-loan data\n"
    "- Scaling/encoding fitted only on training data (not on test)",
)

pdf.add_question(
    "Why is log_annual_inc used instead of raw annual income?",
    "Income data is typically right-skewed (a few people earn very high amounts). Log transformation:\n"
    "- Reduces skewness and makes the distribution more normal\n"
    "- Reduces the impact of extreme outliers\n"
    "- Makes the relationship with the target more linear\n"
    "- Many ML models (especially linear models) perform better with normally distributed features\n"
    "In the dashboard, we reverse this with np.exp() when the user inputs raw income.",
)

pdf.add_question(
    "How is the quality report generated?",
    "The check_data_quality() function computes:\n"
    "- Total rows and columns\n"
    "- Missing values count per column\n"
    "- Missing percentage per column\n"
    "- Data types of each column\n"
    "- Target class distribution (value counts)\n"
    "- Class imbalance ratio (n_class_0 / n_class_1)\n"
    "This report helps identify data issues before modeling.",
)

# ==================== SECTION 4: FEATURE ENGINEERING ====================
pdf.add_page()
pdf.add_section("Section 4: Feature Engineering")

pdf.add_question(
    "What is feature engineering and why is it important?",
    "Feature engineering is the process of creating new features from existing data to improve model performance. "
    "It is often the most impactful step in ML because:\n"
    "- Good features can make simple models outperform complex ones\n"
    "- They capture domain knowledge that raw data may not express directly\n"
    "- They help models learn patterns more efficiently\n"
    "In this project, 5 engineered features are created based on credit risk domain knowledge.",
)

pdf.add_question(
    "What is income_to_installment_ratio and why did you create it?",
    "This ratio measures loan affordability: (Monthly Income / Monthly Installment).\n"
    "Formula: (annual_income / 12) / installment\n"
    "Why it matters:\n"
    "- A high ratio means the borrower can easily afford the payment\n"
    "- A low ratio (<2) indicates financial strain\n"
    "- It combines two features into a more meaningful signal\n"
    '- It captures the concept of "ability to repay" better than either feature alone',
)

pdf.add_question(
    "What is fico_category and how is it created?",
    "fico_category bins the continuous FICO score into risk bands:\n"
    "- Very Poor: 0-579\n"
    "- Fair: 580-669\n"
    "- Good: 670-739\n"
    "- Very Good: 740-799\n"
    "- Excellent: 800-850\n"
    "This captures non-linear relationships - the difference between 350 and 400 may matter more "
    "than between 750 and 800. It also aligns with industry-standard credit score categories.",
)

pdf.add_question(
    "What is dti_category and why is it useful?",
    "dti_category bins the DTI ratio into risk levels:\n"
    "- Low: 0-10%\n"
    "- Medium: 10-20%\n"
    "- High: 20-30%\n"
    "- Very High: 30-100%\n"
    "This captures threshold effects - a DTI of 35% is qualitatively different from 25%, "
    "even though the numerical difference is 10. Many lenders have explicit DTI cutoffs, "
    "so this feature helps the model learn those business rules.",
)

pdf.add_question(
    "What is credit_age_years?",
    "credit_age_years converts days_with_cr_line from days to years by dividing by 365.25.\n"
    "This is a simple unit conversion that makes the feature more interpretable. "
    "A longer credit history generally means lower risk because there is more data to assess behavior.",
)

pdf.add_question(
    "What is delinq_risk_score and how is it calculated?",
    "delinq_risk_score is a weighted composite score of negative credit events:\n"
    "Formula: (delinq_2yrs * 2) + (pub_rec * 3) + (inq_last_6mths * 1)\n"
    "Weights reflect severity:\n"
    "- Public records (bankruptcies) are most serious (weight 3)\n"
    "- Delinquencies are moderately serious (weight 2)\n"
    "- Credit inquiries are least serious (weight 1)\n"
    "This combines three related risk indicators into a single, more powerful feature.",
)

pdf.add_question(
    "How do you decide which features to engineer?",
    "Feature engineering should be driven by domain knowledge, not random experimentation:\n"
    "1. Understand the business problem (credit risk assessment)\n"
    "2. Research what factors lenders actually consider (FICO, DTI, payment history)\n"
    "3. Create features that capture known risk indicators\n"
    "4. Combine related features into ratios or composite scores\n"
    "5. Bin continuous variables at meaningful thresholds\n"
    "6. Validate that engineered features improve model performance",
)

# ==================== SECTION 5: ML MODELS ====================
pdf.add_page()
pdf.add_section("Section 5: Machine Learning Models")

pdf.add_question(
    "Why did you choose these 3 specific models?",
    "The three models represent a progression of complexity:\n"
    "1. Logistic Regression: Simple baseline, interpretable, fast. Good for establishing a minimum performance bar.\n"
    "2. Random Forest: Ensemble method, handles non-linear relationships, robust to outliers. Good comparison model.\n"
    "3. XGBoost: State-of-the-art gradient boosting, best performance on tabular data, supports class weights and early stopping. Primary model.\n"
    "This progression allows us to demonstrate understanding of model trade-offs.",
)

pdf.add_question(
    "Explain how Logistic Regression works.",
    "Logistic Regression is a linear classification algorithm that models the probability of the positive class:\n"
    "1. Computes a linear combination: z = w1*x1 + w2*x2 + ... + b\n"
    "2. Applies sigmoid function: P(y=1) = 1 / (1 + e^(-z))\n"
    "3. If P >= 0.5, predict class 1; otherwise class 0\n"
    "It optimizes the log-loss (cross-entropy) function using gradient descent.\n"
    'In this project, we use class_weight="balanced" to handle class imbalance and solver="lbfgs" for optimization.',
)

pdf.add_question(
    "Explain how Random Forest works.",
    "Random Forest is an ensemble of decision trees using bagging (Bootstrap Aggregating):\n"
    "1. Create multiple bootstrap samples from the training data\n"
    "2. Train a decision tree on each sample\n"
    "3. At each split, consider only a random subset of features\n"
    "4. Final prediction: majority vote (classification) or average (regression)\n"
    "Key parameters: n_estimators (number of trees), max_depth (tree depth), class_weight (imbalance handling).\n"
    "Random forests reduce overfitting compared to single trees through averaging.",
)

pdf.add_question(
    "Explain how XGBoost works.",
    "XGBoost (Extreme Gradient Boosting) is a gradient boosting algorithm that builds trees sequentially:\n"
    "1. Start with a simple model (e.g., predict the mean)\n"
    "2. For each iteration, train a new tree on the residuals (errors) of the current model\n"
    "3. Add the new tree's predictions (scaled by learning_rate) to the ensemble\n"
    "4. Repeat for n_estimators iterations\n"
    "Key features:\n"
    "- Regularization (L1/L2) prevents overfitting\n"
    "- Handles missing values natively\n"
    "- Supports parallel tree construction\n"
    "- Early stopping prevents overfitting on validation set\n"
    "- scale_pos_weight handles class imbalance",
)

pdf.add_question(
    "What is the difference between bagging (Random Forest) and boosting (XGBoost)?",
    "Bagging (Random Forest):\n"
    "- Trees are built independently and in parallel\n"
    "- Each tree sees a random bootstrap sample\n"
    "- Reduces variance (overfitting)\n"
    "- Trees are typically deep and unpruned\n"
    "\nBoosting (XGBoost):\n"
    "- Trees are built sequentially, each correcting previous errors\n"
    "- Each tree focuses on samples the ensemble got wrong\n"
    "- Reduces both bias and variance\n"
    "- Trees are typically shallow (weak learners)\n"
    "- More prone to overfitting if not regularized",
)

pdf.add_question(
    "What hyperparameters did you use for XGBoost and why?",
    "Default configuration:\n"
    "- n_estimators=200: Number of trees (enough to learn patterns without overfitting)\n"
    "- max_depth=5: Limits tree depth to prevent overfitting\n"
    "- learning_rate=0.1: Step size for each tree (trade-off between speed and accuracy)\n"
    "- scale_pos_weight: Calculated from class distribution (n_neg/n_pos)\n"
    '- eval_metric="auc": Optimizes for ROC-AUC (good for imbalanced data)\n'
    "- early_stopping_rounds=20: Stops if validation AUC doesn't improve for 20 rounds\n"
    "\nTuning grid: max_depth [3,5,7], learning_rate [0.01,0.1,0.2], n_estimators [100,200], min_child_weight [1,3,5]",
)

pdf.add_question(
    "What is early stopping and how does it work?",
    "Early stopping monitors model performance on a validation set during training and stops when performance "
    "stops improving:\n"
    "1. After each iteration (tree), evaluate on validation set\n"
    "2. Track the best validation score\n"
    "3. If score doesn't improve for N rounds (early_stopping_rounds=20), stop training\n"
    "4. Restore the model from the best iteration\n"
    "Benefits: Prevents overfitting, saves training time, automatically finds optimal number of trees.",
)

pdf.add_question(
    "What is the learning rate in XGBoost and how does it affect training?",
    "The learning rate (eta) controls the contribution of each tree to the ensemble:\n"
    "- High learning rate (0.2-0.3): Faster training but may overshoot optimal solution\n"
    "- Low learning rate (0.01-0.1): Slower but more precise, needs more trees\n"
    "- Typical practice: Use low learning rate with many trees (e.g., 0.01 with 1000+ trees)\n"
    "In this project, 0.1 with 200 trees provides a good balance between training time and performance.",
)

pdf.add_question(
    "What is max_depth and how does it affect the model?",
    "max_depth controls the maximum depth of each decision tree:\n"
    "- Low depth (3-5): Simpler trees, less overfitting, may underfit\n"
    "- High depth (7-10): Complex trees, captures more patterns, may overfit\n"
    "- In XGBoost, shallow trees (3-6) are typical because boosting combines many weak learners\n"
    "- The optimal depth depends on data complexity and is determined via validation/tuning",
)

# ==================== SECTION 6: MODEL EVALUATION ====================
pdf.add_page()
pdf.add_section("Section 6: Model Evaluation & Metrics")

pdf.add_question(
    "What evaluation metrics did you use and why?",
    "1. ROC-AUC: Measures the model's ability to rank positive cases higher than negative ones. "
    "Robust to class imbalance. Primary metric for model comparison.\n"
    "2. Average Precision (AP): Summary of precision-recall curve. Better than ROC-AUC for imbalanced data.\n"
    "3. Precision: Of all predicted defaults, how many actually defaulted? Important for minimizing false alarms.\n"
    "4. Recall: Of all actual defaults, how many did we catch? Critical for risk management.\n"
    "5. F1 Score: Harmonic mean of precision and recall. Balances both concerns.\n"
    "6. Confusion Matrix: Shows TN, FP, FN, TP for detailed error analysis.",
)

pdf.add_question(
    "Explain ROC-AUC in simple terms.",
    "ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures how well the model distinguishes "
    "between the two classes across all possible thresholds.\n"
    "- AUC = 1.0: Perfect classifier\n"
    "- AUC = 0.5: Random guessing\n"
    "- AUC = 0.72: Our XGBoost model (72% chance of ranking a random default higher than a random paid loan)\n"
    "It plots True Positive Rate (Recall) vs False Positive Rate at different classification thresholds. "
    "The area under this curve is the AUC score.",
)

pdf.add_question(
    "What is the precision-recall trade-off?",
    "Precision and recall are inversely related:\n"
    "- Lower threshold -> more predictions as positive -> higher recall, lower precision\n"
    "- Higher threshold -> fewer predictions as positive -> lower recall, higher precision\n"
    "In credit risk:\n"
    "- High recall is important: Don't miss actual defaults (FN is costly)\n"
    "- High precision is important: Don't reject good applicants (FP loses business)\n"
    "The trade-off is managed by adjusting the classification threshold based on business needs.",
)

pdf.add_question(
    "Why is accuracy not a good metric for this problem?",
    'With imbalanced data (e.g., 84% paid, 16% defaulted), a model that always predicts "paid" would have '
    "84% accuracy but zero usefulness. It would catch no defaults at all.\n"
    "Accuracy treats all errors equally, but in credit risk:\n"
    "- False Negative (missed default) is very costly (financial loss)\n"
    "- False Positive (rejected good applicant) costs lost business\n"
    "ROC-AUC, Precision, Recall, and F1 are much more informative for imbalanced classification.",
)

pdf.add_question(
    "What does the confusion matrix tell you?",
    "The confusion matrix shows:\n"
    "- True Negatives (TN): Correctly predicted paid loans\n"
    "- False Positives (FP): Paid loans incorrectly predicted as defaults\n"
    "- False Negatives (FN): Defaults incorrectly predicted as paid (most costly error)\n"
    "- True Positives (TP): Correctly predicted defaults\n"
    "In credit risk, minimizing FN is critical because missing a default costs the lender the full loan amount.",
)

pdf.add_question(
    "What were the model performance results?",
    "Test Set Results (approximate):\n"
    "- Logistic Regression: ROC-AUC ~0.67, Precision ~0.25, Recall ~0.60\n"
    "- Random Forest: ROC-AUC ~0.70, Precision ~0.28, Recall ~0.55\n"
    "- XGBoost (Best): ROC-AUC ~0.72, Precision ~0.30, Recall ~0.58\n"
    "\nXGBoost performs best because it handles non-linear relationships and feature interactions better. "
    "The moderate precision reflects the inherent difficulty of predicting defaults - many factors are "
    "unpredictable from application data alone.",
)

pdf.add_question(
    "How would you improve model performance?",
    "1. More data: Larger dataset with more diverse loan types\n"
    "2. Better features: Employment history, payment history trends, macroeconomic indicators\n"
    "3. Hyperparameter tuning: More extensive grid/random search or Bayesian optimization\n"
    "4. Ensemble methods: Stack multiple models or use voting ensemble\n"
    "5. SMOTE/ADASYN: Synthetic oversampling of minority class\n"
    "6. Threshold optimization: Adjust classification threshold for business needs\n"
    "7. Feature selection: Remove noisy features that add variance\n"
    "8. Cross-validation: More robust evaluation with k-fold CV",
)

pdf.add_question(
    "What is cross-validation and did you use it?",
    "Cross-validation splits data into k folds, trains on k-1 folds and validates on the remaining fold, "
    "repeating k times. It provides more robust performance estimates than a single split.\n"
    "In this project:\n"
    "- GridSearchCV uses 3-fold CV for hyperparameter tuning\n"
    "- The main pipeline uses a fixed train/val/test split for simplicity\n"
    "- For production, 5-fold or 10-fold stratified CV would be recommended",
)

# ==================== SECTION 7: CLASS IMBALANCE ====================
pdf.add_page()
pdf.add_section("Section 7: Class Imbalance Handling")

pdf.add_question(
    "What is class imbalance and why is it a problem?",
    "Class imbalance occurs when one class significantly outnumbers the other. In this project, "
    "fully paid loans (class 0) far outnumber defaulted loans (class 1).\n"
    "Problems caused:\n"
    "- Model learns to predict majority class always\n"
    "- Poor recall on minority class (the class we care about most)\n"
    "- Misleading accuracy (high accuracy but useless model)\n"
    "- Decision boundary biased toward majority class",
)

pdf.add_question(
    'How does class_weight="balanced" work in scikit-learn?',
    'class_weight="balanced" automatically adjusts weights inversely proportional to class frequencies:\n'
    "weight_class_i = n_samples / (n_classes * n_samples_class_i)\n"
    "This means the minority class gets a higher weight, so misclassifying a default costs more in the "
    "loss function. The model pays more attention to the minority class during training.\n"
    "Used in: Logistic Regression and Random Forest in this project.",
)

pdf.add_question(
    "What is scale_pos_weight in XGBoost?",
    "scale_pos_weight is XGBoost's parameter for handling class imbalance:\n"
    "scale_pos_weight = n_negative_samples / n_positive_samples\n"
    "It multiplies the gradient of the positive class by this factor, effectively telling the model "
    "that each positive (default) sample is worth this many negative samples.\n"
    "For example, if ratio is 4:1, scale_pos_weight = 4, meaning each default counts as 4 paid loans "
    "in the loss function.",
)

pdf.add_question(
    "What other techniques exist for handling class imbalance?",
    "1. SMOTE (Synthetic Minority Over-sampling): Creates synthetic minority samples by interpolating between neighbors\n"
    "2. Random Oversampling: Duplicates minority class samples\n"
    "3. Random Undersampling: Removes majority class samples\n"
    "4. ADASYN: Adaptive synthetic sampling (focuses on hard-to-learn samples)\n"
    "5. Ensemble methods: BalancedBaggingClassifier, EasyEnsemble\n"
    "6. Cost-sensitive learning: Assign different misclassification costs\n"
    "7. Threshold moving: Adjust prediction threshold after training\n"
    "\nIn this project, we use class weights (simpler, less prone to overfitting than resampling).",
)

pdf.add_question(
    "Why did you choose class weights over SMOTE?",
    "Class weights are preferred because:\n"
    "1. No data duplication: SMOTE creates synthetic samples that may not reflect reality\n"
    "2. Simpler pipeline: No need for additional preprocessing steps\n"
    "3. Less overfitting: SMOTE can cause overfitting on synthetic patterns\n"
    "4. Native support: XGBoost and scikit-learn have built-in support\n"
    "5. Faster training: No data generation overhead\n"
    "\nSMOTE could be explored as a future improvement, especially combined with undersampling.",
)

# ==================== SECTION 8: SHAP ====================
pdf.add_page()
pdf.add_section("Section 8: Explainable AI (SHAP)")

pdf.add_question(
    "What is SHAP and why is it important?",
    "SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain ML model predictions. "
    "It is based on Shapley values from cooperative game theory.\n"
    "Importance:\n"
    "- Provides both global (overall feature importance) and local (individual prediction) explanations\n"
    "- Mathematically sound and consistent\n"
    "- Required for regulatory compliance (GDPR right to explanation, Basel III)\n"
    "- Builds trust with stakeholders and loan applicants\n"
    "- Helps debug models and identify data issues",
)

pdf.add_question(
    "Explain SHAP values in simple terms.",
    'SHAP values answer: "How much did each feature contribute to this specific prediction?"\n'
    "Think of it like splitting a restaurant bill:\n"
    '- The "base value" is the average prediction across all samples\n'
    "- Each feature adds or subtracts from this base\n"
    "- The sum of base + all SHAP values = the actual prediction\n"
    "Positive SHAP value: Feature pushes prediction toward default\n"
    "Negative SHAP value: Feature pushes prediction toward paid\n"
    "Magnitude: How strong the influence is",
)

pdf.add_question(
    "What is the difference between global and local explanations?",
    "Global Explanation: How features affect the model overall across all predictions.\n"
    "- SHAP summary plot: Shows which features matter most on average\n"
    "- SHAP bar plot: Mean absolute SHAP values ranked\n"
    "- Helps understand model behavior and validate it matches domain knowledge\n"
    "\nLocal Explanation: How features affect a single prediction.\n"
    "- Waterfall plot: Shows contribution of each feature to one prediction\n"
    "- Force plot: Visual representation of feature pushes\n"
    "- Helps explain individual decisions to applicants/regulators",
)

pdf.add_question(
    "What is a SHAP summary plot?",
    "The SHAP summary plot shows:\n"
    "- Y-axis: Features ranked by importance (most important at top)\n"
    "- X-axis: SHAP value (impact on prediction)\n"
    "- Color: Feature value (red=high, blue=low)\n"
    "- Each dot is one sample\n"
    "\nIt reveals:\n"
    "- Which features are most important overall\n"
    "- Whether high/low values increase or decrease risk\n"
    "- The direction and magnitude of feature effects\n"
    "- For example: High FICO (red dots) cluster on the left (negative SHAP) = reduces default risk",
)

pdf.add_question(
    "What is a SHAP waterfall plot?",
    "The waterfall plot explains a single prediction:\n"
    "- Starts at the base value (expected model output)\n"
    "- Each feature adds or subtracts from the base\n"
    "- Red bars: Features that increase prediction (toward default)\n"
    "- Blue bars: Features that decrease prediction (toward paid)\n"
    "- Ends at the final prediction value\n"
    "\nIt shows exactly why a specific applicant was classified as high/medium/low risk, "
    "listing the top contributing factors in order of impact.",
)

pdf.add_question(
    "What is TreeExplainer and why use it?",
    "TreeExplainer is a SHAP explainer optimized specifically for tree-based models (Random Forest, XGBoost, etc.).\n"
    "Advantages over the general KernelExplainer:\n"
    "- Much faster: O(T*L*D) vs O(2^M) where T=trees, L=leaves, D=depth, M=features\n"
    "- Exact SHAP values: No approximation needed\n"
    "- Handles tree structure natively\n"
    "- Supports tree-specific optimizations like tree_path_dependent method\n"
    "\nFor non-tree models, you would use LinearExplainer or KernelExplainer.",
)

pdf.add_question(
    "What are the top features according to SHAP in this project?",
    "Based on the project documentation:\n"
    "1. FICO Score: Strongest predictor - higher score = lower default risk\n"
    "2. DTI Ratio: Higher debt burden = higher risk\n"
    "3. Interest Rate: Higher rates (riskier borrowers) = higher risk\n"
    "4. Delinquency History: Past delinquencies = higher risk\n"
    "5. Credit Utilization: Higher utilization = higher risk\n"
    "6. Income-to-Installment Ratio: Lower ratio = higher risk\n"
    "7. Credit History Length: Shorter history = higher risk\n"
    "8. Public Records: More bankruptcies = higher risk",
)

pdf.add_question(
    "How does SHAP help with regulatory compliance?",
    "Regulations like GDPR (Article 22) and Basel III require:\n"
    "- Right to explanation: Applicants can ask why their loan was denied\n"
    "- Transparency: Lenders must explain automated decisions\n"
    "- Fairness: Models must not discriminate unfairly\n"
    "\nSHAP provides:\n"
    "- Individual explanations for each decision\n"
    '- Quantified feature contributions (not just "the model said so")\n'
    "- Audit trail showing which factors drove each decision\n"
    "- Ability to detect biased features (e.g., if a proxy for race is influential)\n"
    "This makes SHAP essential for production credit risk systems.",
)

# ==================== SECTION 9: STREAMLIT ====================
pdf.add_page()
pdf.add_section("Section 9: Streamlit Dashboard & Deployment")

pdf.add_question(
    "What does the Streamlit dashboard do?",
    "The Streamlit app provides an interactive interface for:\n"
    "1. Input: Users enter applicant information (income, FICO, DTI, etc.) via sidebar form\n"
    "2. Prediction: Model predicts default probability in real-time\n"
    "3. Risk Classification: Applicant classified as Low/Medium/High risk\n"
    "4. Explanation: SHAP analysis shows top factors driving the prediction\n"
    "5. Visualization: Gauge chart for probability, bar chart for feature contributions\n"
    "6. Key Metrics: Displays FICO score, DTI ratio, income-to-payment ratio with assessments",
)

pdf.add_question(
    "How does the dashboard handle user input and make predictions?",
    "1. User fills form fields in the sidebar (income, FICO, DTI, etc.)\n"
    '2. On clicking "Predict Default Risk", the input is collected into a dictionary\n'
    "3. prepare_input_features() transforms raw input into model-ready features:\n"
    "   - Converts annual income to log scale\n"
    "   - Converts credit history years to days\n"
    "   - Creates engineered features (ratios, categories, composite scores)\n"
    "   - One-hot encodes loan purpose\n"
    "4. model.predict_proba() returns default probability\n"
    "5. SHAP explainer generates feature contributions\n"
    "6. Results displayed with visualizations",
)

pdf.add_question(
    "What is @st.cache_resource and why is it used?",
    "@st.cache_resource caches resources that are expensive to create, like ML models.\n"
    "Without caching, the model would be loaded from disk on every user interaction, "
    "making the app extremely slow.\n"
    "\nIn this project:\n"
    "- load_model() is cached: Model loaded once, reused for all predictions\n"
    "- load_feature_names() is cached: Feature names loaded once\n"
    "\nDifference from @st.cache_data: cache_resource is for non-hashable objects (models, connections), "
    "while cache_data is for data (DataFrames, arrays).",
)

pdf.add_question(
    "How are risk categories determined?",
    "Risk categories are based on default probability thresholds:\n"
    "- Low Risk (Green): Probability < 30%\n"
    "- Medium Risk (Yellow): Probability 30% - 60%\n"
    "- High Risk (Red): Probability > 60%\n"
    "\nThese thresholds are configurable and should be set based on:\n"
    "- Business risk tolerance\n"
    "- Regulatory requirements\n"
    "- Historical default rates\n"
    "- Cost of false positives vs false negatives",
)

pdf.add_question(
    "What visualizations does the dashboard provide?",
    "1. Probability Gauge (Plotly): Semi-circular gauge showing default probability with color coding\n"
    "2. Feature Contribution Bar Chart (matplotlib): Horizontal bars showing SHAP values for top 8 features\n"
    "   - Red bars: Features increasing risk\n"
    "   - Green bars: Features decreasing risk\n"
    "3. Risk Badge: Color-coded HTML badge (Low/Medium/High)\n"
    "4. Key Metrics: Three-column layout showing FICO, DTI, and income-to-payment ratio with delta indicators",
)

pdf.add_question(
    "How would you deploy this application in production?",
    "1. Containerize with Docker: Create Dockerfile with all dependencies\n"
    "2. Cloud hosting: Deploy on AWS, GCP, Azure, or Streamlit Cloud\n"
    "3. API layer: Wrap model in FastAPI/Flask for programmatic access\n"
    "4. Database: Store predictions and applicant data for auditing\n"
    "5. Monitoring: Track prediction drift, model performance, and data quality\n"
    "6. CI/CD: Automated testing and deployment pipeline\n"
    "7. Security: Authentication, rate limiting, input validation\n"
    "8. Model versioning: Track model versions and enable rollback",
)

pdf.add_question(
    "How does the app handle feature engineering for user input?",
    "The prepare_input_features() function in app.py replicates the preprocessing pipeline:\n"
    "1. log_annual_inc = np.log(annual_income) - reverses the log transform for user input\n"
    "2. days_with_cr_line = credit_history_years * 365.25 - converts years to days\n"
    "3. income_to_installment_ratio = monthly_income / installment\n"
    "4. fico_category: Binned based on FICO score ranges\n"
    "5. dti_category: Binned based on DTI ranges\n"
    "6. delinq_risk_score: Weighted composite of delinq, pub_rec, inq\n"
    "7. purpose_*: One-hot encoded based on selected purpose\n"
    "\nThis ensures the input format matches what the model was trained on.",
)

# ==================== SECTION 10: INDUSTRY ====================
pdf.add_page()
pdf.add_section("Section 10: Industry & Regulatory Context")

pdf.add_question(
    "What is Basel III and how does it relate to this project?",
    "Basel III is an international regulatory framework for banks that requires:\n"
    "- Adequate capital reserves based on credit risk\n"
    "- Robust risk assessment models\n"
    "- Stress testing and scenario analysis\n"
    "- Model validation and documentation\n"
    "\nThis project relates because:\n"
    "- Banks must quantify credit risk for capital allocation\n"
    "- ML models can improve risk assessment accuracy\n"
    "- Explainability (SHAP) is required for model validation\n"
    "- Regulatory approval requires transparent, auditable models",
)

pdf.add_question(
    'What is GDPR\'s "right to explanation"?',
    "GDPR (General Data Protection Regulation) Article 22 gives EU citizens the right to:\n"
    "- Not be subject to solely automated decisions with legal/significant effects\n"
    "- Obtain human intervention\n"
    "- Express their point of view\n"
    "- Contest the decision\n"
    "- Get an explanation of the decision\n"
    "\nFor credit risk: If a loan is denied by an ML model, the applicant has the right to know why. "
    "SHAP provides these explanations by showing which factors contributed to the decision.",
)

pdf.add_question(
    "What is data leakage in the context of credit risk?",
    "Data leakage in credit risk means using information that wouldn't be available at the time of application:\n"
    "Examples of leakage:\n"
    "- Post-loan features (e.g., payment history after loan origination)\n"
    "- Future information (e.g., whether the loan was eventually paid)\n"
    "- Features derived from the target variable\n"
    "- Features that are consequences, not causes, of default\n"
    "\nPrevention: Only use features available at application time (FICO, income, DTI, etc.). "
    "This project explicitly removes post-loan features.",
)

pdf.add_question(
    "What are the business costs of False Positives vs False Negatives?",
    "False Positive (predict default, but loan would be paid):\n"
    "- Cost: Lost revenue from a good customer\n"
    "- Impact: Customer dissatisfaction, reduced market share\n"
    "- Mitigation: Can be partially recovered through manual review\n"
    "\nFalse Negative (predict paid, but loan defaults):\n"
    "- Cost: Full loan amount lost (principal + interest)\n"
    "- Impact: Direct financial loss, increased capital reserves needed\n"
    "- Mitigation: Very difficult to recover after default\n"
    "\nIn credit risk, False Negatives are typically more costly, so we prioritize recall.",
)

pdf.add_question(
    "How would you monitor this model in production?",
    "1. Prediction Drift: Monitor if prediction distributions change over time\n"
    "2. Data Drift: Monitor if input feature distributions change (PSI, KS test)\n"
    "3. Performance Monitoring: Track precision, recall, AUC on labeled outcomes\n"
    "4. Business Metrics: Default rate, approval rate, loss rate\n"
    "5. Feature Monitoring: Check for missing values, outliers, range violations\n"
    "6. Model Decay: Retrain periodically as new data becomes available\n"
    "7. A/B Testing: Compare new model versions against current production model\n"
    "8. Alerting: Set thresholds for automatic alerts on anomalies",
)

pdf.add_question(
    "What ethical considerations are important in credit risk ML?",
    "1. Fairness: Model must not discriminate based on protected attributes (race, gender, age)\n"
    "2. Transparency: Decisions must be explainable to applicants and regulators\n"
    "3. Bias Detection: Check for proxy discrimination (e.g., zip code as proxy for race)\n"
    "4. Accessibility: Ensure model doesn't systematically exclude certain groups\n"
    "5. Privacy: Protect applicant data and comply with data protection laws\n"
    "6. Human Oversight: Allow human review of automated decisions\n"
    "7. Regular Auditing: Periodic fairness and performance audits\n"
    "8. Documentation: Maintain model cards and decision logs",
)

# ==================== SECTION 11: ADVANCED ====================
pdf.add_page()
pdf.add_section("Section 11: Advanced & Behavioral Questions")

pdf.add_question(
    "If you had more time, what would you improve in this project?",
    "1. Data: Use larger dataset (full Lending Club dataset has millions of records)\n"
    "2. Features: Add employment history, bank account data, macroeconomic indicators\n"
    "3. Models: Try LightGBM, CatBoost, neural networks, or stacking ensemble\n"
    "4. Tuning: Use Optuna for Bayesian hyperparameter optimization\n"
    "5. Validation: Implement stratified k-fold cross-validation\n"
    "6. Threshold Optimization: Find optimal threshold based on cost-benefit analysis\n"
    "7. Calibration: Use Platt scaling or isotonic regression for calibrated probabilities\n"
    "8. MLOps: Set up MLflow for experiment tracking and model registry\n"
    "9. API: Build REST API with FastAPI for integration with loan processing systems\n"
    "10. Testing: Add unit tests, integration tests, and data validation tests",
)

pdf.add_question(
    "How would you handle a situation where the model's performance degrades over time?",
    "1. Diagnose the cause: Check for data drift, concept drift, or data quality issues\n"
    "2. Monitor feature distributions: Use PSI (Population Stability Index) to detect drift\n"
    "3. Check recent performance: Evaluate on recent data with known outcomes\n"
    "4. Retrain: Update model with recent data\n"
    "5. Feature engineering: Add new features if the data landscape has changed\n"
    "6. Model selection: Try different algorithms if the current one is no longer suitable\n"
    "7. Ensemble: Combine old and new models for smoother transitions\n"
    "8. Rollback: If new model performs worse, revert to previous version",
)

pdf.add_question(
    "Explain the bias-variance trade-off.",
    "Bias: Error from overly simplistic assumptions (underfitting).\n"
    "- High bias: Model is too simple, misses patterns\n"
    "- Example: Linear model on non-linear data\n"
    "\nVariance: Error from sensitivity to training data (overfitting).\n"
    "- High variance: Model memorizes training data, doesn't generalize\n"
    "- Example: Deep decision tree with no pruning\n"
    "\nTrade-off: As model complexity increases, bias decreases but variance increases.\n"
    "Optimal model: Balances both for minimum total error.\n"
    "\nIn this project:\n"
    "- Logistic Regression: Higher bias, lower variance\n"
    "- Random Forest: Lower bias, moderate variance (bagging reduces variance)\n"
    "- XGBoost: Low bias, controlled variance (regularization + early stopping)",
)

pdf.add_question(
    "What is regularization and how does it prevent overfitting?",
    "Regularization adds a penalty to the loss function for complex models:\n"
    "- L1 (Lasso): Adds |weights| penalty, can zero out features (feature selection)\n"
    "- L2 (Ridge): Adds weights^2 penalty, shrinks weights toward zero\n"
    "- Elastic Net: Combination of L1 and L2\n"
    "\nIn XGBoost:\n"
    "- reg_alpha: L1 regularization on leaf weights\n"
    "- reg_lambda: L2 regularization on leaf weights\n"
    "- max_depth: Limits tree complexity\n"
    "- min_child_weight: Minimum samples required for a split\n"
    "\nRegularization prevents the model from fitting noise in the training data.",
)

pdf.add_question(
    "What is the difference between predict() and predict_proba()?",
    "predict(): Returns the predicted class label (0 or 1).\n"
    "- Uses default threshold of 0.5\n"
    "- Returns: [0, 1, 0, 1, ...]\n"
    "\npredict_proba(): Returns the probability of each class.\n"
    "- Returns: [[0.8, 0.2], [0.3, 0.7], ...] where columns are [P(class=0), P(class=1)]\n"
    "- Allows custom threshold selection\n"
    "- Required for ROC-AUC calculation\n"
    "- Used in this project for probability-based risk classification\n"
    "\nIn the dashboard, we use predict_proba()[0][1] to get the default probability.",
)

pdf.add_question(
    "How do you choose the optimal classification threshold?",
    "The default threshold of 0.5 may not be optimal for imbalanced data:\n"
    "1. ROC Curve: Find threshold that maximizes (TPR - FPR)\n"
    "2. Precision-Recall Curve: Find threshold that maximizes F1 score\n"
    "3. Cost-Benefit Analysis:\n"
    "   - Cost_FN = cost of missing a default (loan amount)\n"
    "   - Cost_FP = cost of rejecting a good applicant (lost profit)\n"
    "   - Choose threshold that minimizes total expected cost\n"
    "4. Business Requirements:\n"
    "   - Conservative lender: Lower threshold (catch more defaults, accept more FPs)\n"
    "   - Aggressive lender: Higher threshold (approve more loans, accept more FNs)\n"
    "\nIn this project, we use probability bands (30%, 60%) instead of hard classification.",
)

pdf.add_question(
    "What is the difference between parametric and non-parametric models?",
    "Parametric models assume a specific functional form with fixed number of parameters:\n"
    "- Logistic Regression: Assumes linear decision boundary\n"
    "- Pros: Fast, interpretable, less data needed\n"
    "- Cons: May underfit if assumption is wrong\n"
    "\nNon-parametric models make no assumptions about functional form:\n"
    "- Random Forest, XGBoost: Learn complex patterns from data\n"
    "- Pros: Flexible, can capture any pattern\n"
    "- Cons: More data needed, slower, risk of overfitting\n"
    "\nIn this project, we use both: Logistic Regression (parametric baseline) and tree ensembles (non-parametric).",
)

pdf.add_question(
    "Walk me through how you would explain this project in an interview.",
    '"I built a Credit Risk Prediction System that helps lenders assess loan default risk.\n'
    "\nI started with Lending Club data containing ~9,600 loans with features like FICO score, "
    "DTI ratio, interest rate, and credit history. The target is whether the loan was fully paid or not.\n"
    "\nMy preprocessing pipeline handles missing values with median imputation, one-hot encodes "
    "categorical variables, and engineers 5 new features like income-to-installment ratio and "
    "delinquency risk score. I split the data into train/validation/test sets using stratified sampling.\n"
    "\nI trained three models: Logistic Regression as a baseline, Random Forest for comparison, "
    "and XGBoost as the primary model. XGBoost performed best with ~0.72 ROC-AUC. I handled class "
    "imbalance using scale_pos_weight.\n"
    "\nFor explainability, I implemented SHAP to provide both global feature importance and "
    "individual prediction explanations, which is critical for regulatory compliance.\n"
    "\nFinally, I built an interactive Streamlit dashboard where users can input applicant details "
    'and get real-time predictions with SHAP-based explanations."',
)

pdf.add_question(
    "What challenges did you face and how did you overcome them?",
    "1. Class Imbalance: The dataset had far more paid loans than defaults. Solved using scale_pos_weight "
    'in XGBoost and class_weight="balanced" in scikit-learn models.\n'
    "\n2. Feature Engineering: Needed domain knowledge to create meaningful features. Researched credit "
    "risk industry practices to create ratio-based and categorical features.\n"
    "\n3. Input Transformation: The dashboard receives raw input but the model expects transformed features. "
    "Implemented prepare_input_features() to replicate the preprocessing pipeline.\n"
    "\n4. Model Explainability: Raw predictions aren't enough for production. Integrated SHAP for "
    "transparent, regulatory-compliant explanations.\n"
    "\n5. Modular Design: Organized code into preprocessing, model, and explain modules for "
    "maintainability and reusability.",
)

pdf.add_question(
    "How does this project demonstrate your ML skills?",
    "1. End-to-end pipeline: From raw data to deployed dashboard\n"
    "2. Data preprocessing: Missing values, encoding, feature engineering\n"
    "3. Model selection: Comparing multiple algorithms with proper evaluation\n"
    "4. Class imbalance: Practical handling of real-world data challenges\n"
    "5. Explainability: SHAP integration for production-ready models\n"
    "6. Software engineering: Modular code, proper structure, documentation\n"
    "7. Deployment: Interactive dashboard with real-time predictions\n"
    "8. Domain knowledge: Credit risk understanding and feature engineering\n"
    "9. Best practices: Stratified splitting, early stopping, model serialization\n"
    "10. Communication: Clear explanations and visualizations",
)

# Save PDF
output_path = r"C:\Users\AKSHAYKUMAR\OneDrive\Desktop\MachineLearning\credit_risk\Credit_Risk_Interview_QA.pdf"
pdf.output(output_path)
print(f"PDF generated successfully: {output_path}")
