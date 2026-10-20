# # RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

## 📌 Project Overview

RetailPulse is an AI-powered retail analytics platform designed to help businesses understand customer purchasing behaviour, analyse sales performance, predict customer churn, forecast demand, and optimize inventory decisions.

The platform combines data analytics, machine learning, and interactive visualizations to transform historical retail transaction data into meaningful business insights.

Using Python, Scikit-learn, Pandas, Plotly, and Streamlit, RetailPulse provides an interactive dashboard where users can explore sales trends, customer segments, churn-risk predictions, demand forecasts, and inventory recommendations.

## 🎯 Project Objectives

- Analyse historical retail sales and transaction data.
- Understand customer purchasing patterns and behaviour.
- Segment customers using machine learning.
- Identify customers who may be at risk of churning.
- Forecast future product demand using historical data.
- Support inventory planning and stock optimization.
- Present analytical results through interactive charts and tables.
- Help retailers make informed, data-driven business decisions.

## ✨ Key Features

### 1. Sales Analysis and Exploratory Data Analysis

- Analyse retail transactions and sales performance.
- Examine data quality and missing values.
- Visualize revenue and sales trends over time.
- Identify top-performing products and countries.
- Present key findings using charts and summary tables.

### 2. Customer Segmentation

- Apply the K-Means clustering algorithm.
- Group customers according to purchasing behaviour.
- Analyse customer purchasing frequency and monetary value.
- Visualize customer segments.
- Support targeted marketing and customer engagement strategies.

### 3. Customer Churn Prediction

- Use machine learning to estimate customer churn risk.
- Apply the Gradient Boosting algorithm in the churn prediction experiment.
- Evaluate model performance using ROC-AUC and precision-based metrics.
- Identify customers who may require additional attention.
- Support customer retention strategies and personalized marketing.

### 4. Demand Forecasting

- Analyse historical sales and demand patterns.
- Generate and examine demand forecasts.
- Compare actual and predicted values.
- Visualize forecasting results.
- Evaluate prediction performance using forecasting metrics.

### 5. Inventory Optimization

- Analyse product-level demand patterns.
- Examine inventory requirements and SKU-level recommendations.
- Support stock replenishment planning.
- Visualize inventory-related metrics and recommendations.
- Help retailers make informed inventory management decisions.

### 6. Interactive Analytics Dashboard

- Display analytical results through Streamlit.
- Present charts, tables, and key performance indicators.
- Organize analytics into separate dashboard sections.
- Support interactive exploration of retail business insights.
- Provide downloadable analytical tables where available.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Plotly | Interactive charts and visualizations |
| Streamlit | Interactive web dashboard |
| Joblib | Model and object serialization |
| Jupyter Notebook | Data analysis and experimentation |
| Visual Studio Code | Development environment |
| Git | Version control |
| GitHub | Source code management |
| GitHub Actions | Automated code checks and CI workflow |

## 🏗️ Project Architecture

RetailPulse follows a data analytics workflow that transforms retail transaction data into actionable insights.

1. Data Collection
   - Load historical retail transaction data.

2. Data Preprocessing
   - Examine data quality.
   - Handle missing and invalid values where required.
   - Prepare the data for analysis and modeling.

3. Exploratory Data Analysis
   - Analyse revenue, products, transactions, and customer behaviour.

4. Machine Learning
   - Perform customer segmentation.
   - Develop churn-risk predictions.
   - Evaluate model performance.

5. Demand Forecasting
   - Analyse historical demand.
   - Generate forecasts and compare predictions with actual values.

6. Inventory Optimization
   - Analyse product demand and inventory requirements.
   - Examine SKU-level recommendations.

7. Dashboard Development
   - Display results through interactive charts, tables, and metrics.

## 📂 Project Structure

```text
Retail/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── customer_segmentation.ipynb
│   ├── churn_prediction.ipynb
│   ├── demand_forecasting.ipynb
│   └── inventory_optimization.ipynb
│
├── models/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── k8s/
    ├── deployment.yaml
    └── service.yaml
