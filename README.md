# 

# Online Retail Customer Churn Analysis

## Project Overview

This project is an **end-to-end customer churn analysis and prediction system** built using the Online Retail dataset. The goal is to move beyond basic retention metrics and deliver an **analyst-level, business-ready churn modeling pipeline** with interpretable insights and actionable decisions.

The project progresses from **data cleaning and cohort analysis**, through **feature engineering and model building**, and culminates in **advanced model evaluation, threshold optimization, cost-based decisioning, and explainability (SHAP)**.

----------

## 1. Data Understanding

**Dataset:** Online Retail transactional data

**Key fields:**

-   InvoiceNo
    
-   InvoiceDate
    
-   CustomerID
    
-   Quantity
    
-   UnitPrice
    
-   Country
    

Each row represents a transaction line item. Customers can appear multiple times across invoices and dates.

----------

## 2. Data Cleaning & Preprocessing

To ensure analytical and modeling correctness, the following cleaning steps were applied:

### Removed / Filtered Data

-   **Missing CustomerID** → cannot attribute behavior
    
-   **Negative Quantity** → product returns/refunds (not purchase intent)
    
-   **Zero or Negative UnitPrice** → data noise
    
-   **Duplicate invoice lines** → prevent double counting
    

### Why this matters

Churn models depend on **true customer purchasing behavior**. Returns, missing IDs, or invalid prices introduce bias and distort features such as spend, frequency, and recency.

----------

## 3. Cohort & Retention Analysis

### Objective

Understand customer retention behavior over time before modeling churn.

### Method

-   Customers grouped by **first purchase month (cohort)**
    
-   Monthly active status tracked
    
-   Retention percentage calculated
    

### Output

-   **Cohort retention table**
-   **Heatmap of retention decay**
    ![Retention Heatmap](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/retention_heatmap.png)

### Insight

Retention drops sharply after the first month, validating churn as a meaningful and measurable outcome.

----------

## 4. Churn Definition

A customer is labeled as **churned (1)** if:

-   No purchases are made after a defined observation window
    

Otherwise labeled as **active (0)**.

This binary definition enables supervised classification while aligning with business intuition.

----------

## 5. Feature Engineering

### Observation Windows Tested

-   14 days
    
-   30 days
    
-   60 days
    
-   90 days
    ![Performance by Window](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/model_performance_by_window.png)

### Engineered Features (per customer)

-   `orders_w` – number of orders
    
-   `active_days_w` – unique purchase days
    
-   `total_quantity_w` – total items purchased
    
-   `total_spend_w` – monetary value
    
-   `avg_order_value_w`
    
-   `days_to_second_purchase`
    

### Why multiple windows?

Short windows capture **early intent**, longer windows capture **habit formation**. Comparing windows lets us balance **early prediction vs accuracy**.

----------

## 6. Baseline Modeling

### Models Used

-   Logistic Regression (baseline)
    
-   Random Forest
    
-   XGBoost
    

### Evaluation Metric

-   **ROC-AUC** (threshold independent, imbalance-aware)
    

### Key Findings

-   Performance improves consistently as the window increases
    
-   **90-day window performs best**
    
-   Random Forest outperforms Logistic Regression
    

----------

## 7. Model Performance Comparison

### Visuals

-   ROC-AUC vs Observation Window
    
-   ROC Curves (90-day)
    ![ROC curves](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/roc_curves_90d.png)

### Insight

Tree-based models capture **non-linear customer behavior patterns** better than linear models.

----------

## 8. Feature Importance Analysis

### Methods

-   Random Forest feature importance
    
-   XGBoost feature importance
    

### Top Drivers

1.  Active days
    
2.  Order frequency
    
3.  Quantity purchased
    

### Interpretation

Consistency of engagement matters more than raw spend.
![Feature Importance](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/feature_importance_comparison.png)

----------

## 9. Confusion Matrix (Default Threshold)

Evaluated classification errors at default threshold (0.5):

-   High false negatives (missed churners)
    ![Confusion Matrix 90d](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/confusion_matrix_rf_90d.png)

This motivated **threshold optimization**.

----------

## 10. Lift Curve Analysis

### Purpose

Measure marketing effectiveness when targeting top-risk customers.

### Result

Lift curve closely follows diagonal → model is **ranking customers meaningfully**, suitable for targeted interventions.
![Lift Curve](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/lift_curve_rf_90d.png)

----------

## 11. Threshold Optimization (F1-based)

### Approach

-   Sweep probability thresholds
    
-   Compute Precision, Recall, F1-score
    

### Result

-   **Optimal threshold ≈ 0.38**
    
-   Recall increases substantially
    ![Optimized Threshold](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/threshold_optimization_rf_90d.png)

### Business Meaning

Better at **catching churners early**, acceptable increase in false positives.

----------

## 12. Confusion Matrix @ Optimized Threshold

### Outcome

-   Much higher true churn capture
    
-   Explicit trade-off between precision and recall
    
![Confusion Matrix Optimized](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/confusion_matrix_rf_90d_optimized.png)
This is a **deployment-ready decision threshold**.

----------

## 13. Cost-Based Threshold Optimization (Advanced)

### Motivation

Not all errors cost the same:

-   False Negative (missed churn) is more expensive
    
-   False Positive (unnecessary offer) is cheaper
    

### Cost Assumptions (example)

-   FN cost = high
    
-   FP cost = low
    

### Result

-   **Cost-optimal threshold ≈ 0.13**
    
-   Minimizes total business cost
    ![Cost based Threshold](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/cost_based_threshold_rf_90d.png)

### Insight

Best statistical threshold ≠ best business threshold.

----------

## 14. Model Explainability with SHAP

### Why SHAP?

-   Explain individual predictions
    
-   Validate model logic
    
-   Build stakeholder trust
    

### Outputs

-   SHAP feature importance
    ![SHAP feature impoortance](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/shap_feature_importance_rf_90d.png)
-   SHAP summary (beeswarm)
    ![SHAP Summary](https://github.com/ArpitaRandive/Online-Retail-Customer-Churn-Analysis/blob/main/Images/shap_summary_rf_90d.png)

### Key Insights

-   Low activity days strongly increase churn risk
    
-   Early repeat purchase reduces churn probability
    
-   Spend alone is less predictive than engagement
    

SHAP confirms the model is **behaviorally sound**.

----------

## 15. Final Model Choice

**Selected Model:** Random Forest (90-day window)

**Why?**

-   Best ROC-AUC
    
-   Stable feature importance
    
-   Interpretable with SHAP
    
-   Business-aligned after threshold tuning
    

----------

## 16. What Makes This Analyst-Level

-   Multiple observation windows
    
-   Business-driven thresholding
    
-   Cost-sensitive decision making
    
-   Explainability (SHAP)
    
-   Clear linkage between metrics and actions
    

This goes beyond "train a model" and demonstrates **decision intelligence**.

----------

## 17. Future Enhancements

-   Regional / country-level churn models
    
-   Survival analysis (time-to-churn)
    
-   Campaign uplift modeling
    
-   Customer lifetime value integration
    

----------

## Conclusion

This project delivers a **complete churn analytics pipeline**—from raw data to actionable, explainable decisions. It balances statistical rigor with business relevance, making it suitable for real-world deployment and strong portfolio presentation.
