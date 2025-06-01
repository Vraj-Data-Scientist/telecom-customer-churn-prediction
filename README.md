
---

# Telecom Customer Churn Prediction

This project focuses on predicting customer churn in the telecom industry using machine learning. By analyzing customer data, the project aims to identify patterns and predict which customers are likely to leave, enabling targeted retention strategies. The dataset used is the **Telco Customer Churn** dataset, containing 7,043 customer records with 21 attributes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Modeling](#modeling)
- [Results](#results)
- [Strategies to Reduce Churn](#strategies-to-reduce-churn)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Customer churn, when customers stop doing business with a company, is a critical issue in the telecom industry, with an annual churn rate of **15-25%**. Retaining customers is more cost-effective than acquiring new ones, making churn prediction essential for profitability and growth. This project:

- Explores customer churn patterns using data analysis and visualization.
- Builds and evaluates multiple machine learning models to predict churn.
- Provides actionable strategies to reduce churn based on model insights.

Key objectives include:
- Determining the percentage of churned vs. active customers.
- Identifying churn patterns based on gender, service type, and other features.
- Finding the most profitable services and features.
- Predicting high-risk customers for targeted retention.

## Dataset

The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) includes:
- **Rows**: 7,043 customers.
- **Columns**: 21 attributes, including:
  - **Churn**: Target variable (Yes/No).
  - **Services**: Phone, Internet, Online Security, Tech Support, Streaming, etc.
  - **Account Info**: Tenure, Contract, Payment Method, Monthly/Total Charges.
  - **Demographics**: Gender, Senior Citizen status, Partner, Dependents.

| Attribute | Description | Type |
|-----------|-------------|------|
| customerID | Unique customer identifier | Object |
| Churn | Whether customer churned (Yes/No) | Object |
| tenure | Months as a customer | Int64 |
| MonthlyCharges | Monthly billing amount | Float64 |
| TotalCharges | Total billed amount | Object (converted to Float64) |
| Contract | Contract type (Month-to-month, One year, Two year) | Object |

## Installation

To run this project, install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly missingno scikit-learn xgboost catboost
```

Clone the repository:

```bash
git clone https://github.com/your-username/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

## Usage

1. **Load the Dataset**:
   - Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project directory.
   - Run the Jupyter notebook (`customer-churn-prediction.ipynb`) or HTML file (`customer-churn-prediction.html`) to explore the analysis.

2. **Run the Analysis**:
   - Execute the notebook cells to preprocess data, visualize patterns, and train models.
   - The notebook includes code for data cleaning, visualization, and model evaluation.

3. **View Results**:
   - Check the **Results** section for model performance metrics.
   - Review visualizations (e.g., confusion matrices, ROC curves) for insights.

## Data Preprocessing

The dataset was preprocessed to ensure quality and compatibility with machine learning models:

- **Missing Values**:
  - Used `missingno` to visualize missing data; no major patterns found.
  - Converted `TotalCharges` to numeric (`pd.to_numeric(df.TotalCharges, errors='coerce')`), revealing 11 missing values.
  - Dropped 11 rows with `tenure == 0` (no impact on data).
  - Filled remaining `TotalCharges` missing values with the mean.

- **Encoding**:
  - Converted categorical columns (e.g., `gender`, `Contract`) to numeric using `LabelEncoder` for most features.
  - Applied one-hot encoding to `PaymentMethod`, `Contract`, and `InternetService`.
  - Mapped `SeniorCitizen` (0/1) to "No"/"Yes" for consistency, then encoded.

- **Scaling**:
  - Standardized numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to normalize ranges.

- **Data Split**:
  - Split into 70% training (`X_train`, `y_train`) and 30% testing (`X_test`, `y_test`) with `stratify=y` to maintain churn distribution.

| Step | Action | Tool/Method |
|------|--------|-------------|
| Missing Values | Drop rows with `tenure == 0`, fill `TotalCharges` with mean | `missingno`, `pd.to_numeric` |
| Encoding | Label encoding, one-hot encoding | `LabelEncoder`, `pd.get_dummies` |
| Scaling | Standardize numeric features | `StandardScaler` |
| Data Split | 70/30 train-test split | `train_test_split` |

## Data Visualization

Visualizations revealed key churn patterns:
- **Churn Rate**: 26.6% of customers churned (1,869/7,032).
- **Gender**: No significant churn difference (49.5% female, 50.5% male).
- **Contract Type**:
  - 75% of month-to-month contract customers churned vs. 13% (one-year) and 3% (two-year).
- **Payment Method**:
  - Electronic check users had higher churn; credit card/bank transfer users churned less.
- **Internet Service**:
  - Fiber optic users had a higher churn rate than DSL or no-internet customers.
- **Demographics**:
  - Customers without dependents or partners were more likely to churn.
  - Senior citizens (a small fraction) had a high churn rate.
- **Services**:
  - Lack of online security, tech support, or phone service increased churn.
  - Paperless billing users were more likely to churn.
- **Charges**:
  - Higher `MonthlyCharges` and lower `tenure` correlated with churn.
  - `TotalCharges` showed less clear separation.

| Feature | Insight | Visualization |
|---------|---------|---------------|
| Churn | 26.6% churn rate | Pie chart |
| Contract | Month-to-month: 75% churn | Histogram |
| Payment Method | Electronic check: High churn | Pie chart, Histogram |
| Internet Service | Fiber optic: High churn | Bar chart |
| Tenure | New customers churn more | Box plot |

## Modeling

Multiple machine learning models were trained to predict churn:
- **K-Nearest Neighbors (KNN)**: `n_neighbors=11`.
- **Support Vector Classifier (SVC)**: `random_state=1`.
- **RandomForestClassifier**: `n_estimators=500`, `max_features="sqrt"`, `max_leaf_nodes=30`.
- **Logistic Regression**.
- **Decision Tree Classifier**.
- **XGBoost Classifier**.
- **AdaBoost Classifier**.
- **Gradient Boosting Classifier**.
- **Voting Classifier**: Soft voting with Gradient Boosting, Logistic Regression, and AdaBoost.

**Training**:
- Features: All columns except `Churn` and `customerID`.
- Target: `Churn` (0: No, 1: Yes).
- Evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.

## Results

The table below compares the performance of all models on the test set (2,110 samples, 561 churn, 1,549 non-churn):

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-------|----------|-------------------|----------------|------------------|
| Voting Classifier | **0.816** | 0.68 | 0.57 | 0.62 |
| Random Forest | 0.814 | 0.71 | 0.51 | 0.59 |
| AdaBoost | 0.813 | 0.68 | 0.56 | 0.62 |
| Logistic Regression | 0.809 | 0.66 | 0.58 | 0.62 |
| Gradient Boosting | 0.808 | 0.67 | 0.55 | 0.60 |
| SVC | 0.808 | 0.69 | 0.50 | 0.58 |
| XGBoost | 0.780 | 0.60 | 0.55 | 0.57 |
| KNN | 0.776 | 0.59 | 0.52 | 0.55 |
| Decision Tree | 0.730 | 0.49 | 0.53 | 0.51 |

**Key Observations**:
- **Voting Classifier** achieved the highest accuracy (81.6%) and balanced performance.
- **Random Forest** had the highest precision (0.71) for churn but lower recall (0.51), missing many churn cases.
- **Logistic Regression** and **AdaBoost** performed well, with F1-scores of 0.62 for churn.
- **Decision Tree** had the lowest accuracy (73.0%) and poor precision (0.49).
- **Confusion Matrix (Random Forest)**:
  - True Negatives (TN): 1,400 (90.4% correct non-churn).
  - True Positives (TP): 324 (57.8% correct churn).
  - False Negatives (FN): 237 (missed churn cases, critical for retention).
  - False Positives (FP): 149 (less costly errors).

**Implications**:
- Models excel at predicting non-churn but struggle with churn (low recall).
- False negatives (e.g., 237 for Random Forest) indicate missed opportunities to retain at-risk customers.
- Future improvements could include handling class imbalance (e.g., SMOTE) or hyperparameter tuning.

## Strategies to Reduce Churn

Based on model predictions and data insights, the following strategies can reduce churn:
- **Know Your Customers**:
  - Use models to identify at-risk customers (e.g., 324 TP in Random Forest).
  - Analyze features like `TotalCharges`, `Contract`, and `InternetService`.
- **Improve Customer Service**:
  - Address dissatisfaction (e.g., fiber optic users, 237 FN cases).
  - Enhance support for customers with high `MonthlyCharges`.
- **Build Loyalty**:
  - Offer promotions for month-to-month contract holders (75% churn).
  - Provide discounts for long-term customers (low `tenure` churners).
- **Survey Churned Customers**:
  - Collect feedback to address issues (e.g., billing errors in `TotalCharges`).
  - Improve services like online security and tech support.

| Strategy | Action | Example |
|----------|--------|---------|
| Know Customers | Target at-risk customers | Retention offers for 324 TP |
| Improve Service | Enhance support | Resolve fiber optic complaints |
| Build Loyalty | Personalized promotions | Discounts for month-to-month users |
| Survey Churners | Collect feedback | Fix billing issues |

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure code follows PEP 8 standards and includes comments for clarity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



---

  - The HTML file is assumed to be a converted Jupyter notebook, so the README references both formats.
- **Actionable**: Includes installation, usage, and contribution guidelines for practical use.

If you need adjustments (e.g., specific repository details, additional sections, or a different format), let me know!
