
# Employee Attrition Intelligence — Streamlit App

Upload your HR dataset (`EA.csv`-style with an `Attrition` column) to:
- Explore **five** actionable charts with **Job Role filters** and a **satisfaction** slider.
- Train/evaluate **Decision Tree, Random Forest, Gradient Boosted Trees** with **5-fold stratified CV**.
- Generate **confusion matrices**, **ROC overlay**, **metrics table**, and **feature importance** plots.
- Predict on **new datasets** and **download** the results.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push `app.py` and `requirements.txt` (and optionally this `README.md`) to a GitHub repo (top-level, no folders).
2. Create a new Streamlit Cloud app pointing to `app.py`.
