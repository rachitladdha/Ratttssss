
# Employee Attrition Streamlit Dashboard

This repository contains a Streamlit dashboard (`app.py`) for exploring employee attrition and training/predicting with three classifiers:
Decision Tree, Random Forest, and Gradient Boosting.

Features:
- Interactive dashboard with 5 charts and filters (job role multi-select and satisfaction slider).
- Modeling tab to train models with a single click; shows accuracy table, confusion matrices, and feature importances.
- Predict tab to upload a new dataset, generate predictions and download CSV with predicted labels.
- Preprocessing: numeric imputation with mean, categorical imputation with mode, label encoding for categorical columns.

Usage:
1. Add your dataset file named `EA.csv` to the repo root or upload a CSV using the app sidebar.
2. Deploy on Streamlit Cloud by connecting your GitHub repo and pointing to this file.
3. Install dependencies (listed in `requirements.txt`) - Streamlit Cloud will handle this automatically.

Note: This package intentionally does not include pinned package versions to reduce Streamlit Cloud compatibility issues.
