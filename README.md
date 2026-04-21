# 🏦 Bank Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Model-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

This project analyzes **bank customer churn** using a dataset of 10,000 customers. The goal is to identify key drivers of churn, segment at-risk customers, and provide actionable business recommendations — all skills directly relevant to financial services data analyst roles.

---

## 🎯 Business Problem

Customer churn is one of the most costly challenges for retail banks. Acquiring a new customer costs **5–7x more** than retaining an existing one. This analysis answers:

- **Who** is most likely to churn?
- **Why** are they leaving?
- **What** can the bank do to retain them?

---

## 🗂️ Dataset

| Attribute | Detail |
|---|---|
| Source | Synthetic dataset mirroring Kaggle Bank Churn structure |
| Rows | 10,000 customers |
| Columns | 12 features + 1 target (churn) |
| Target | `churn` (1 = Churned, 0 = Retained) |

**Features include:** Credit Score, Country, Gender, Age, Tenure, Account Balance, Number of Products, Credit Card Status, Active Member Flag, Estimated Salary

---

## 🔧 Tools & Libraries

| Tool | Purpose |
|---|---|
| `Python 3.9+` | Core language |
| `Pandas` | Data wrangling & SQL-style queries |
| `NumPy` | Numerical operations |
| `Matplotlib / Seaborn` | Static visualizations |
| `Plotly` | Interactive HTML charts |
| `Scikit-learn` | Random Forest classifier & model evaluation |

---

## 📊 Key Findings

### 1. Overall Churn Rate
- **~20% churn rate** across the customer base
- High-value customers (balance > $100K) represent a disproportionate churn risk

### 2. Age is the #1 Churn Driver
- Customers aged **41–60** churn at nearly **double the rate** of younger customers
- Suggests mid-career customers may be seeking better financial products elsewhere

### 3. Germany Outpaces Other Regions
- German customers churn at a **significantly higher rate** than France or Spain
- Potential root cause: product mismatch, regional competition, or service gaps

### 4. Inactive Members Are High Risk
- **Inactive members churn at ~2x the rate** of active members
- Engagement campaigns could substantially reduce overall churn

### 5. Product Over-Selling Backfires
- Customers with **3–4 products** show much higher churn than 1–2 product holders
- Suggests forced bundling may create dissatisfaction

---

## 🤖 Predictive Model

| Metric | Score |
|---|---|
| Model | Random Forest Classifier |
| ROC-AUC | ~0.86 |
| Top Features | Age, Balance, Active Member Status, Num Products, Credit Score |

---

## 💡 Business Recommendations

1. **Retention Program for 41–60 Age Group** — Loyalty rewards, dedicated relationship managers
2. **Germany Market Review** — Audit fees, product offerings, and competitor landscape
3. **Activation Campaigns** — Monthly financial health nudges for inactive members
4. **High-Value Customer Watch List** — Private banking support for balance > $100K churners
5. **Smart Cross-Sell** — Quality over quantity; avoid over-bundling products

---

## 📁 Project Structure
