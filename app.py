
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import json

st.set_page_config(layout="wide", page_title="Attrition Dashboard (Streamlit)", initial_sidebar_state="expanded")

# ---------------------- Helper functions ----------------------
@st.cache_data(show_spinner=False)
def detect_label_column(df):
    # Look for likely label columns
    candidates = [c for c in df.columns if c.lower() in ("attrition", "left", "resigned", "is_attrition", "is_left")]
    return candidates[0] if candidates else None

def detect_jobrole_column(df):
    candidates = [c for c in df.columns if "job" in c.lower() and "role" in c.lower() or c.lower()=="jobrole" or c.lower()=="role"]
    # fallback common names
    if not candidates:
        for name in ["JobRole", "Role", "Position"]:
            if name in df.columns:
                candidates.append(name)
                break
    return candidates[0] if candidates else None

def detect_satisfaction_column(df):
    # find a column with 'satisf' substring
    for c in df.columns:
        if "satisf" in c.lower():
            return c
    # fallback to likely names
    for name in ["JobSatisfaction", "SatisfactionLevel", "Satisfaction"]:
        if name in df.columns:
            return name
    return None

def preprocess(df, label_col=None, for_training=True):
    """
    - Impute numerical cols with mean.
    - Impute categorical cols with mode.
    - Label-encode categorical columns and the label (if present / non-numeric).
    Returns encoded df, and encoding maps.
    """
    df = df.copy()
    # Detect label if not provided
    if label_col is None and detect_label_column(df):
        label_col = detect_label_column(df)
    # We'll impute and encode on feature columns only (label preserved if present)
    feature_cols = [c for c in df.columns if c != label_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Impute numeric with mean
    for c in num_cols:
        df[c] = df[c].astype(float).fillna(df[c].astype(float).mean())
    # Impute categorical with mode
    for c in cat_cols:
        if df[c].isna().any():
            modes = df[c].mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else "Unknown"
            df[c] = df[c].fillna(fill).astype(str)
        else:
            df[c] = df[c].astype(str)
    # Drop rows with null label (can't train on missing labels)
    if label_col and df[label_col].isna().any():
        df = df[~df[label_col].isna()].copy()
    # Label encode categorical features and label if needed
    encoding_maps = {}
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoding_maps[c] = {str(k): int(v) for v, k in enumerate(le.classes_)}
        encoders[c] = le
    if label_col:
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            le_y = LabelEncoder()
            df[label_col] = le_y.fit_transform(df[label_col].astype(str))
            encoding_maps[label_col] = {str(k): int(v) for v, k in enumerate(le_y.classes_)}
            encoders[label_col] = le_y
        else:
            encoders[label_col] = None
    return df, encoding_maps, encoders, label_col

def train_models(X_train, y_train):
    RANDOM_STATE = 42
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
    return trained

def compute_metrics(trained_models, X_train, y_train, X_test, y_test):
    rows = []
    cms = {}
    reports = {}
    for name, model in trained_models.items():
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        rows.append({"Model": name, "Train Accuracy": acc_train, "Test Accuracy": acc_test})
        cm = confusion_matrix(y_test, y_pred_test, labels=[0,1])
        cms[name] = cm
        reports[name] = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    return pd.DataFrame(rows), cms, reports

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names, rotation=0)
    return fig

def plot_feature_importances(model, feature_names, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(6, max(3, top_n*0.25)))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], orient='h', ax=ax)
    ax.set_title("Feature importances (top {})".format(top_n))
    return fig

# ---------------------- Sidebar: Data load ----------------------
st.sidebar.title("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV). If none, use the example upload.", type=["csv"])
use_example = False
if uploaded is None:
    st.sidebar.info("No dataset uploaded. You can run with a sample dataset by uploading your EA.csv in the app or use your own.")
else:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded dataset: {}".format(uploaded.name))
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        df = None

# If there is a file named 'EA.csv' in the working dir (e.g., if you add it to the repo), prefer it
import os
if os.path.exists("EA.csv") and uploaded is None:
    try:
        df = pd.read_csv("EA.csv")
        st.sidebar.success("Loaded EA.csv from repo root.")
    except Exception:
        pass

if 'df' not in locals() or df is None:
    df = pd.DataFrame()  # empty placeholder

# ---------------------- Main layout ----------------------
st.title("Employee Attrition Dashboard")
st.markdown("Interactive Streamlit dashboard for exploring attrition and running ML models. Filters apply across charts.")

# Tabs: Dashboard, Modeling, Predict
tabs = st.tabs(["Dashboard", "Modeling & Train", "Predict New Data", "About & Instructions"])

# ---------------------- DASHBOARD TAB ----------------------
with tabs[0]:
    if df.empty:
        st.warning("No data loaded. Please upload a CSV file using the sidebar or add 'EA.csv' to the repo root.")
    else:
        st.subheader("Data snapshot")
        st.dataframe(df.head())

        # Detect key columns
        label_col = detect_label_column(df) or st.sidebar.text_input("Specify label column name (if not detected)", value="Attrition")
        job_col = detect_jobrole_column(df) or st.sidebar.text_input("Specify job role column name (if not detected)", value="JobRole")
        satis_col = detect_satisfaction_column(df) or st.sidebar.text_input("Specify satisfaction column name (if not detected)", value="JobSatisfaction")

        st.sidebar.markdown("### Dashboard Filters")
        # Job role multiselect
        job_options = ["(all)"]
        if job_col in df.columns:
            job_options = ["(all)"] + sorted(df[job_col].dropna().astype(str).unique().tolist())
        selected_jobs = st.sidebar.multiselect("Job role filter (multi-select)", job_options, default=["(all)"])
        # satisfaction slider (if column exists and numeric)
        satis_min, satis_max = 0.0, 1.0
        satis_default = None
        if satis_col in df.columns and pd.api.types.is_numeric_dtype(df[satis_col]):
            satis_min = float(df[satis_col].min())
            satis_max = float(df[satis_col].max())
            satis_default = (satis_min, satis_max)
            selected_satis = st.sidebar.slider("Satisfaction range", min_value=float(satis_min), max_value=float(satis_max), value=satis_default)
        else:
            selected_satis = None

        # Apply filters to a working df
        viz_df = df.copy()
        if "(all)" not in selected_jobs and selected_jobs:
            viz_df = viz_df[viz_df[job_col].astype(str).isin(selected_jobs)]
        if selected_satis is not None:
            viz_df = viz_df[(viz_df[satis_col] >= selected_satis[0]) & (viz_df[satis_col] <= selected_satis[1])]

        st.markdown("### Charts (filters applied)")
        # Chart 1: Attrition rate by Job Role (bar chart with percentage)
        st.markdown("#### 1) Attrition rate by Job Role")
        if job_col in viz_df.columns and label_col in viz_df.columns:
            grp = viz_df.groupby(job_col).agg(total=(label_col, "size"), attrited=(label_col, lambda x: (x.astype(str).str.lower()=="yes").sum() if x.dtype==object else (x==1).sum()))
            grp["attrition_rate"] = grp["attrited"] / grp["total"]
            grp_sorted = grp.sort_values("attrition_rate", ascending=False).reset_index()
            fig1, ax1 = plt.subplots(figsize=(8,4))
            sns.barplot(data=grp_sorted, x="attrition_rate", y=job_col, orient="h", ax=ax1)
            ax1.set_xlabel("Attrition rate (proportion)")
            st.pyplot(fig1)
        else:
            st.info("Job role or label column not detected for Chart 1. Use the sidebar to set names.")

        # Chart 2: Satisfaction vs Attrition (stacked counts)
        st.markdown("#### 2) Satisfaction vs Attrition (counts by satisfaction level)")
        if satis_col in viz_df.columns and label_col in viz_df.columns:
            # Create pivot
            df_piv = viz_df.copy()
            # normalize label to 'Yes'/'No' if possible
            df_piv["_label_str"] = df_piv[label_col].astype(str)
            piv = pd.crosstab(df_piv[satis_col], df_piv["_label_str"])
            fig2, ax2 = plt.subplots(figsize=(8,4))
            piv.plot(kind="bar", stacked=True, ax=ax2)
            ax2.set_xlabel(satis_col)
            ax2.set_ylabel("Counts")
            st.pyplot(fig2)
        else:
            st.info("Satisfaction or label column not detected for Chart 2.")

        # Chart 3: Attrition rate by Years at Company (line)
        st.markdown("#### 3) Attrition rate by Years at Company")
        years_col = None
        for c in ["YearsAtCompany", "Years At Company", "Years_in_Company", "YearsWithCompany"]:
            if c in viz_df.columns:
                years_col = c; break
        if years_col is None:
            # try to find anything with 'year' and 'company'
            for c in viz_df.columns:
                if "year" in c.lower() and "company" in c.lower():
                    years_col = c; break
        if years_col in viz_df.columns and label_col in viz_df.columns:
            grp2 = viz_df.groupby(years_col).agg(total=(label_col,"size"), attrited=(label_col, lambda x: (x.astype(str).str.lower()=="yes").sum() if x.dtype==object else (x==1).sum()))
            grp2["attrition_rate"] = grp2["attrited"] / grp2["total"]
            grp2 = grp2.reset_index().sort_values(years_col)
            fig3, ax3 = plt.subplots(figsize=(8,3))
            ax3.plot(grp2[years_col], grp2["attrition_rate"], marker='o')
            ax3.set_xlabel(years_col); ax3.set_ylabel("Attrition rate")
            st.pyplot(fig3)
        else:
            st.info("Years at company or label column not detected for Chart 3.")

        # Chart 4: Monthly Income distribution by Attrition (boxplot)
        st.markdown("#### 4) Monthly Income distribution by Attrition (boxplot)")
        income_candidates = [c for c in viz_df.columns if "income" in c.lower() or "salary" in c.lower() or "monthly" in c.lower()]
        income_col = income_candidates[0] if income_candidates else None
        if income_col and label_col in viz_df.columns:
            fig4, ax4 = plt.subplots(figsize=(8,4))
            sns.boxplot(x=viz_df[label_col].astype(str), y=viz_df[income_col], ax=ax4)
            ax4.set_xlabel("Attrition"); ax4.set_ylabel(income_col)
            st.pyplot(fig4)
        else:
            st.info("Income column or label not detected for Chart 4.")

        # Chart 5: Feature importance (Quick RandomForest on filtered data)
        st.markdown("#### 5) Quick feature importances (RandomForest trained on filtered data)")
        # Preprocess and train a RF quickly
        try:
            df_enc, encoding_maps, encoders, label_col_enc = preprocess(viz_df, label_col=label_col)
            X = df_enc.drop(columns=[label_col_enc])
            y = df_enc[label_col_enc].astype(int)
            if len(y.unique()) > 1 and len(X.columns)>0:
                # small train/test split
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
                rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
                rf.fit(Xtr, ytr)
                fig5 = plot_feature_importances(rf, X.columns.tolist(), top_n=min(20, X.shape[1]))
                if fig5:
                    st.pyplot(fig5)
                else:
                    st.info("Model does not expose feature importances.")
            else:
                st.info("Not enough data or features for feature importance chart.")
        except Exception as e:
            st.error(f"Error creating feature importance: {e}")

# ---------------------- MODELING & TRAIN TAB ----------------------
with tabs[1]:
    st.header("Train & Evaluate Models")
    st.markdown("This tab runs Decision Tree, Random Forest and Gradient Boosting on the (filtered) dataset. Click 'Train Models' to run.")
    if df.empty:
        st.warning("No dataset loaded. Upload CSV in the sidebar or add 'EA.csv' to repo root.")
    else:
        st.markdown("### Choose columns to include as features")
        # default: all except detected label
        label_col = detect_label_column(df) or st.text_input("Label column", value="Attrition")
        all_features = [c for c in df.columns if c != label_col]
        selected_features = st.multiselect("Select feature columns", all_features, default=all_features)
        if not selected_features:
            st.warning("Select at least one feature.")
        else:
            run = st.button("Train Models")
            if run:
                with st.spinner("Preprocessing and training..."):
                    df_enc, encoding_maps, encoders, label_col_enc = preprocess(df[[*selected_features, label_col]], label_col=label_col)
                    X = df_enc.drop(columns=[label_col_enc])
                    y = df_enc[label_col_enc].astype(int)
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
                    trained = train_models(X_train, y_train)
                    # Save to session_state so Predict tab can use
                    st.session_state['trained_models'] = trained
                    st.session_state['encoders'] = encoders
                    st.session_state['encoding_maps'] = encoding_maps
                    st.session_state['label_col'] = label_col_enc
                    st.session_state['feature_names'] = X.columns.tolist()
                    # Metrics
                    metrics_df, cms, reports = compute_metrics(trained, X_train, y_train, X_test, y_test)
                    st.success("Training complete.")
                    st.markdown("### Accuracy table")
                    st.dataframe(metrics_df.style.format({"Train Accuracy":"{:.3f}", "Test Accuracy":"{:.3f}"}))
                    # Confusion matrices
                    st.markdown("### Confusion Matrices (Test set)")
                    class_names = None
                    if label_col_enc in encoding_maps:
                        inv = {v:k for k,v in encoding_maps[label_col_enc].items()}
                        class_names = [inv.get(0, "0"), inv.get(1, "1")]
                    else:
                        class_names = ["0","1"]
                    for name, cm in cms.items():
                        st.markdown(f"**{name}**")
                        st.pyplot(plot_confusion_matrix(cm, class_names))
                    # Feature importances for each model
                    st.markdown("### Feature importances per model (if available)")
                    for name, model in trained.items():
                        st.markdown(f"**{name}**")
                        fig_imp = plot_feature_importances(model, st.session_state['feature_names'], top_n=min(20, len(st.session_state['feature_names'])))
                        if fig_imp:
                            st.pyplot(fig_imp)
                        else:
                            st.info(f"{name} has no feature_importances_.")
                    # Save models and artifacts into session state
                    st.success("Models saved into session (will persist while the app runs).")
            else:
                st.info("Click 'Train Models' to preprocess and train the three algorithms on the dataset.")

# ---------------------- PREDICT TAB ----------------------
with tabs[2]:
    st.header("Upload new data and predict Attrition")
    st.markdown("Upload a new CSV (same format as training data without the label or with label). Choose a trained model and predict. You can download the results.")
    uploaded_new = st.file_uploader("Upload new dataset for prediction (CSV)", type=["csv"], key="predict")
    model_choice = st.selectbox("Choose model (trained models)", options=["Auto-train and use Random Forest", "Decision Tree", "Random Forest", "Gradient Boosting"])
    if uploaded_new is not None:
        try:
            new_df = pd.read_csv(uploaded_new)
            st.write("Preview of uploaded data")
            st.dataframe(new_df.head())
            # Check session for trained models; if not, auto-train RF on full df
            if 'trained_models' not in st.session_state:
                st.warning("No trained models in session. Auto-training a Random Forest on the provided main dataset (if present).")
                if df.empty:
                    st.error("Cannot auto-train because no base dataset is loaded. Upload training data in the sidebar or train in Modeling tab.")
                else:
                    # train RF on df using all columns except detected label
                    label_col_main = detect_label_column(df) or st.text_input("Label column (for training)", value="Attrition")
                    df_enc_main, encoding_maps_main, encoders_main, label_col_enc_main = preprocess(df[[c for c in df.columns if c!=label_col_main] + [label_col_main]], label_col=label_col_main)
                    X = df_enc_main.drop(columns=[label_col_enc_main])
                    y = df_enc_main[label_col_enc_main].astype(int)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
                    rf.fit(Xtr, ytr)
                    st.session_state['trained_models'] = {"Random Forest": rf}
                    st.session_state['encoding_maps'] = encoding_maps_main
                    st.session_state['encoders'] = encoders_main
                    st.session_state['label_col'] = label_col_enc_main
                    st.session_state['feature_names'] = X.columns.tolist()
            # Prepare new data for prediction: impute & encode using training encoders if available
            # For simplicity apply preprocess() on the new df (this will fit new label encoders on the uploaded file)
            new_enc, new_maps, new_encoders, new_labelcol = preprocess(new_df, label_col=None, for_training=False)
            # Choose model
            chosen_model = None
            if model_choice == "Auto-train and use Random Forest":
                chosen_model = st.session_state['trained_models'].get("Random Forest")
            else:
                # pick model from session if available, else warn
                if 'trained_models' in st.session_state and model_choice in st.session_state['trained_models']:
                    chosen_model = st.session_state['trained_models'][model_choice]
                else:
                    st.warning(f"Model {model_choice} not found in session. Using Random Forest if available.")
                    chosen_model = st.session_state['trained_models'].get("Random Forest")
            if chosen_model is None:
                st.error("No model available for prediction. Train models first in Modeling tab or upload a training dataset.")
            else:
                # Align features: use session feature names if available
                feat_names = st.session_state.get('feature_names', new_enc.drop(columns=[new_labelcol]) .columns.tolist())
                # If columns mismatch try to intersect
                X_pred = new_enc.copy()
                if st.session_state.get('feature_names'):
                    # Add missing columns with zeros, drop extras
                    for c in st.session_state['feature_names']:
                        if c not in X_pred.columns:
                            X_pred[c] = 0
                    X_pred = X_pred[st.session_state['feature_names']]
                else:
                    X_pred = X_pred[feat_names]
                preds = chosen_model.predict(X_pred)
                # map label back to original if mapping exists
                label_map = st.session_state.get('encoding_maps', {}).get(st.session_state.get('label_col', ''), None)
                if label_map:
                    inv = {v:k for k,v in label_map.items()}
                    preds_readable = [inv.get(int(p), str(p)) for p in preds]
                else:
                    preds_readable = [str(int(p)) for p in preds]
                out_df = new_df.copy().reset_index(drop=True)
                out_df["Predicted_Attrition"] = preds_readable
                st.markdown("### Predictions preview")
                st.dataframe(out_df.head())
                # Download button
                csv = out_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv, file_name="predictions_with_attrition.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    else:
        st.info("Upload a CSV to run predictions (you can include or exclude the label column).")

# ---------------------- ABOUT & INSTRUCTIONS TAB ----------------------
with tabs[3]:
    st.header("About / Instructions")
    st.markdown("""
    **How to use this Streamlit app**
    1. Upload your EA.csv (or any employee dataset) using the sidebar. The app will try to auto-detect the label column 'Attrition', job role column and a satisfaction column.
    2. Use the Dashboard tab to explore 5 charts with filters (job role multi-select and satisfaction slider).
    3. Use the Modeling & Train tab to choose features and train Decision Tree, Random Forest and Gradient Boosting classifiers. Click 'Train Models' to run. Models persist in session memory while the app runs.
    4. Use the Predict tab to upload a new dataset and predict Attrition. You can download the data with predicted labels.
    \n\n**Notes & engineer choices**\n- Missing numeric values are imputed with column mean; categorical missing values are replaced with the mode.\n- Label encoding is applied per categorical column. The app tries to be robust to common column names but you can override names via the sidebar.\n- For production use, you should: (a) freeze encodings from training and apply identical transforms to new data, (b) add model versioning and persistence, (c) add more metrics such as AUC, precision/recall and class-weighting if classes are imbalanced.
    """)

