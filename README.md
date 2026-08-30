# SBA Loan Approval & Default Analysis

A data mining and predictive analytics project using historical U.S. Small Business Administration (SBA) loan records from 1987 to 2014. This repository contains end-to-end data processing pipelines, exploratory visual analysis, and machine learning models (Logistic Regression, Random Forest, Gradient Boosting, Decision Trees) designed to evaluate credit risk and predict loan defaults.

## Executive Summary

The project analyzes 899,164 historical loan records containing 27 feature variables. The primary objective is to uncover patterns that differentiate loans paid in full (`P I F`) from those charged off (`CHGOFF`).

### Key Findings
* **Loan Distribution:** California (130,619 loans), Texas (70,458 loans), and New York (57,693 loans) represent the highest loan volumes. Retail Trade, Manufacturing, and Accommodation/Food Services account for ~43% of total disbursements.
* **Risk Drivers:** Higher default likelihood is associated with larger loan amounts, LowDoc program participation (+23% default risk), new businesses (+18% default likelihood), and zero-employee firms (21% default rate vs. 14% for firms with 10+ employees).
* **High-Risk Sectors:** Real Estate & Rental/Leasing (~30% default rate) and Finance & Insurance show elevated risk compared to Manufacturing and Healthcare.
* **Mitigating Factors:** Longer loan terms correlate negatively with default rates ($r = -0.32$). Businesses with revolving lines of credit exhibited a 34% lower default rate.

---

## Dataset Overview

The dataset consists of 27 features spanning borrower information, lender metrics, loan terms, and outcomes:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `LoanNr_ChkDgt` | Nominal | Unique identifier for each loan |
| `Name`, `City`, `State`, `Zip` | Nominal | Borrower geographic profile |
| `Bank`, `BankState` | Nominal | Lender information |
| `NAICS` | Nominal | Industry classification code |
| `NewExist` | Nominal | Business status (1 = Existing, 2 = New) |
| `FranchiseCode` | Nominal | Franchise indicator |
| `UrbanRural` | Nominal | 1 = Urban, 2 = Rural, 0 = Undefined |
| `RevLineCr`, `LowDoc` | Nominal | Revolving line of credit and Low Documentation flags (Y/N) |
| `Term` | Ratio | Loan term in months |
| `NoEmp`, `CreateJob`, `RetainedJob` | Ratio | Employee headcount and job impact metrics |
| `DisbursementGross`, `GrAppv`, `SBA_Appv` | Ratio | Financial values ($) for total disbursed, approved, and guaranteed amounts |
| `MIS_Status` | Nominal | Target variable (`P I F` vs. `CHGOFF`) |

---

## Technical Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Pipeline:** `scikit-learn`, `imbalanced-learn` (SMOTE)

To set up the development environment, install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
