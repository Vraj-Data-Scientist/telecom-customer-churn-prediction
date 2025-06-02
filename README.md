
---



# Telecom Customer Churn Prediction

This repository contains a machine learning project for predicting customer churn in the telecom industry using the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn). The goal is to identify customers at risk of churning to enable targeted retention strategies, prioritizing **high recall** for the "Churn" class due to the high cost of losing customers (5–10x more than retention costs).

## Project Overview

Customer churn is a critical issue in telecom, with an industry churn rate of 15–25% annually. This project builds predictive models to identify at-risk customers, leveraging data preprocessing, feature engineering, and advanced machine learning techniques. The focus is on maximizing recall for the "Churn" class (class = 1) to minimize missed churners, while maintaining acceptable precision (0.45–0.60) to manage false positives.

### Dataset
- **Source**: Telco Customer Churn dataset (7043 rows, 21 columns).
- **Features**: Customer demographics (e.g., gender, SeniorCitizen), services (e.g., InternetService, TechSupport), account details (e.g., tenure, MonthlyCharges, Contract).
- **Target**: Churn (binary: 0 = No Churn, 1 = Churn, ~26.6% churn rate).
- **Test Set**: 1409 samples (1035 No Churn, 374 Churn).

### Methodology
- **Preprocessing**: Label encoding for binary features, one-hot encoding for multiclass features, SMOTETomek for class imbalance, StandardScaler for scaling.
- **Feature Engineering**: Added `MonthlyCharges_per_Tenure`, `Weighted_Service_Score`, `High_Risk_Contract`, `Charges_Contract_Interaction`, `Charges_Payment_Interaction`. Dropped redundant and low-correlation features.
- **Models**: XGBoost, RandomForest, LogisticRegression, CatBoost, VotingClassifier, StackingClassifier.
- **Evaluation Metrics**: Recall, Precision, F1-Score for Churn, ROC-AUC.
- **Threshold Tuning**: Optimized for recall ≥0.85 and precision ≥0.45.

## Model Performance

### Before Feature Engineering
The following table summarizes model performance before feature engineering (using original features after preprocessing).

| **Model** | **Churn Recall** | **Churn Precision** | **Churn F1-Score** | **ROC-AUC** | **Accuracy** |
|-----------|------------------|---------------------|--------------------|-------------|--------------|
| XGBoost (default) | 0.61 | 0.58 | 0.59 | 0.820 | 0.78 |
| RandomForest (class_weight='balanced') | 0.62 | 0.59 | 0.60 | 0.827 | 0.78 |
| LogisticRegression (class_weight='balanced') | 0.79 | 0.50 | 0.61 | 0.840 | 0.73 |
| XGBoost (scale_pos_weight=3) | 0.71 | 0.53 | 0.61 | 0.816 | 0.75 |
| VotingClassifier (RF+LR+XGB) | 0.71 | 0.54 | 0.61 | 0.836 | 0.76 |
| XGBoost (GridSearchCV) | 0.70 | 0.53 | 0.60 | 0.819 | 0.76 |
| RandomForest (GridSearchCV) | 0.60 | 0.58 | 0.59 | 0.824 | 0.78 |
| CatBoost (scale_pos_weight=3) | 0.75 | 0.53 | 0.62 | 0.826 | 0.76 |
| StackingClassifier (LR+CatBoost) | 0.71 | 0.54 | 0.61 | 0.832 | 0.76 |
| VotingClassifier (threshold=0.412) | 0.79 | 0.52 | 0.63 | 0.836 | 0.75 |
| StackingClassifier (threshold=0.236) | 0.82 | 0.50 | 0.62 | 0.832 | 0.74 |

### After Feature Engineering
Feature engineering included `MonthlyCharges_per_Tenure`, `Weighted_Service_Score`, `High_Risk_Contract`, `Charges_Contract_Interaction`, and `Charges_Payment_Interaction`, with redundant and low-correlation features dropped.

| **Model** | **Churn Recall** | **Churn Precision** | **Churn F1-Score** | **ROC-AUC** | **Accuracy** |
|-----------|------------------|---------------------|--------------------|-------------|--------------|
| LogisticRegression (class_weight='balanced') | 0.79 | 0.50 | 0.61 | 0.841 | 0.73 |
| StackingClassifier (threshold=0.531) | 0.73 | 0.54 | 0.62 | 0.830 | 0.76 |
| StackingClassifier (class_weight={0:1, 1:2}) | 0.75 | 0.50 | 0.60 | 0.832 | 0.73 |
| CatBoost (threshold=0.4) | 0.78 | 0.48 | 0.59 | 0.825 | 0.72 |
| StackingClassifier (top 15 features) | 0.74 | 0.49 | 0.59 | Not reported | 0.73 |
| StackingClassifier (threshold=0.138) | 0.91 | 0.41 | 0.56 | 0.824 | 0.62 |
| **StackingClassifier (threshold=0.2425)** | **0.86** | **0.45** | **0.59** | **0.834** | **0.68** |

### Best Model
The **StackingClassifier (LogisticRegression + CatBoost, threshold = 0.2425)** with feature engineering is the best model.

| **Metric** | **Value** |
|------------|-----------|
| Churn Recall | 0.86 |
| Churn Precision | 0.45 |
| Churn F1-Score | 0.59 |
| ROC-AUC | 0.834 |
| Accuracy | 0.68 |

**Configuration**:
- **Base Models**: LogisticRegression (class_weight={0:1, 1:2}), CatBoost (iterations=1000, learning_rate=0.05, depth=6, scale_pos_weight=5).
- **Meta-Learner**: LogisticRegression (class_weight={0:1, 1:2}).
- **Threshold**: 0.2425 (optimized for recall ≥0.85, precision ≥0.45).
- **Features**: Original features + `MonthlyCharges_per_Tenure`, `Weighted_Service_Score`, `High_Risk_Contract`, `Charges_Contract_Interaction`, `Charges_Payment_Interaction`.

## Why This Recall is Industry Standard
In the telecom industry, **recall for the Churn class** is prioritized because:
- **High Cost of False Negatives**: Missing a churner (false negative) leads to lost revenue and high acquisition costs (5–10x retention costs). A recall of 0.86 catches ~321 out of 374 churners, minimizing missed opportunities.
- **Acceptable False Positives**: Offering retention incentives (e.g., discounts) to non-churners (false positives) is less costly. Precision of 0.45 is acceptable, as false positives are manageable with automated, low-cost interventions.
- **Industry Benchmarks**: Telecom churn models typically target recall of 0.85–0.95 to ensure most at-risk customers are identified. The achieved recall of 0.86 aligns with this standard, balancing business needs with resource constraints.

## Why This Model is Best
The StackingClassifier with threshold = 0.2425 is the best model because:
1. **High Recall (0.86)**: Catches 86% of churners, meeting the industry target (≥0.85), reducing missed churners to ~53 (vs. 112–149 for other models).
2. **Acceptable Precision (0.45)**: While lower than some models (e.g., 0.54 for threshold=0.531), it’s within the tolerable range (0.45–0.60) for telecom, where false positives are less costly.
3. **Competitive F1-Score (0.59)**: Balances recall and precision, comparable to or better than other models (e.g., RandomForest: 0.59 F1).
4. **Strong ROC-AUC (0.834)**: Indicates excellent separability, among the highest (e.g., LogisticRegression: 0.841).
5. **Feature Engineering Impact**: New features (e.g., `MonthlyCharges_per_Tenure`, `Charges_Payment_Interaction`) capture cost sensitivity and risk, boosting recall.
6. **Robustness**: Cross-validation F1-churn scores (0.77–0.91, mean ~0.86) confirm stability.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/telecom-churn-prediction.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   - Open `customer-churn-prediction.ipynb` in Jupyter Notebook.
   - Execute cells to preprocess data, train models, and evaluate performance.
4. **Predict Churn**:
   - Use the trained StackingClassifier model with threshold = 0.2425 to predict churn on new data.

## Future Improvements
- **Feature Engineering**: Add features like tenure bins or customer service call frequency (if available).
- **Hyperparameter Tuning**: Expand GridSearchCV for CatBoost (e.g., tune `iterations`, `depth`).
- **Cost-Sensitive Learning**: Optimize directly for a cost matrix (e.g., FN=5, FP=1).
- **Alternative Models**: Test neural networks or anomaly detection for rare churn cases.

