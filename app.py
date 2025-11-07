
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Employee Attrition Intelligence", layout="wide")
st.title("👔 Employee Attrition Intelligence — Streamlit Dashboard")
st.caption("Upload your HR dataset (must include an **Attrition** column with values 'Yes'/'No'). "
           "Explore insights, train models (DT, RF, GBRT, 5-fold stratified CV), and predict on new data.")

# ---------------------------
# Utilities
# ---------------------------
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")

def encode_features(df: pd.DataFrame, target_col: str):
    y_text = df[target_col].astype(str)
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=False)
    y_bin = (y_text.str.lower() == "yes").astype(int)
    return X, y_text, y_bin

def align_columns(X_new: pd.DataFrame, feature_list):
    X_new = pd.get_dummies(X_new, drop_first=False)
    missing = [c for c in feature_list if c not in X_new.columns]
    for c in missing:
        X_new[c] = 0
    extra = [c for c in X_new.columns if c not in feature_list]
    X_new = X_new.drop(columns=extra)
    return X_new[feature_list]

def model_pack(random_state=42):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=random_state),
    }

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No","Yes"]); ax.set_yticklabels(["No","Yes"])
    ax.set_xlabel("Predicted Attrition"); ax.set_ylabel("Actual Attrition")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center")
    fig.tight_layout()
    return fig

def make_download(dataframe: pd.DataFrame, filename: str = "predictions.csv"):
    buff = BytesIO()
    dataframe.to_csv(buff, index=False)
    buff.seek(0)
    st.download_button("⬇️ Download predictions CSV", buff, file_name=filename, mime="text/csv")

# ---------------------------
# Data Upload (base)
# ---------------------------
with st.sidebar:
    st.header("1) Base Dataset")
    base_file = st.file_uploader("Upload HR CSV (with 'Attrition')", type=["csv"], key="base")
    st.header("2) Settings")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

if base_file is None:
    st.info("Please upload your base HR dataset (e.g., **EA.csv**) from the sidebar.")
    st.stop()

df = load_data(base_file)
if "Attrition" not in df.columns:
    st.error("The uploaded file must contain an 'Attrition' column with values like 'Yes'/'No'.")
    st.stop()

# Basic type handling
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str)

# ---------------------------
# Filters (apply to all charts)
# ---------------------------
st.subheader("Filters")
jobrole_vals = sorted(df["JobRole"].dropna().unique()) if "JobRole" in df.columns else []
jobrole_sel = st.multiselect("Filter by Job Role", jobrole_vals, default=jobrole_vals if jobrole_vals else None)

sat_col = None
for c in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]:
    if c in df.columns:
        sat_col = c
        break
if sat_col:
    min_sat, max_sat = int(df[sat_col].min()), int(df[sat_col].max())
    sat_thr = st.slider(f"Minimum {sat_col}", min_sat, max_sat, min_sat, 1)
else:
    sat_thr = None
    st.warning("No satisfaction column (Job/Environment/Relationship) found — satisfaction filter disabled.")

def apply_filters(data):
    d = data.copy()
    if jobrole_sel and "JobRole" in d.columns:
        d = d[d["JobRole"].isin(jobrole_sel)]
    if sat_thr is not None and sat_col in d.columns:
        d = d[d[sat_col] >= sat_thr]
    return d

df_f = apply_filters(df)

# ===========================
# TABS
# ===========================
tab_dash, tab_model, tab_predict = st.tabs(["📊 Dashboard", "🤖 Modeling (DT/RF/GBRT)", "🔮 Predict on New Data"])

# ===========================
# 📊 Dashboard
# ===========================
with tab_dash:
    st.markdown("### Five Insightful Charts for HR Action")
    # Precompute helpful columns
    df_f["AttritionFlag"] = (df_f["Attrition"].str.lower() == "yes").astype(int)

    # 1) Attrition rate by Job Role (Bar)
    if "JobRole" in df_f.columns:
        role_rate = df_f.groupby("JobRole")["AttritionFlag"].mean().reset_index()
        c1 = alt.Chart(role_rate).mark_bar().encode(
            x=alt.X("AttritionFlag:Q", title="Attrition Rate"),
            y=alt.Y("JobRole:N", sort="-x", title="Job Role"),
            tooltip=["JobRole","AttritionFlag"]
        ).properties(height=350, title="Attrition Rate by Job Role")
        st.altair_chart(c1, use_container_width=True)
    else:
        st.info("JobRole column not found for Chart 1.")

    # 2) Attrition by OverTime & Gender (Stacked 100% bar)
    if set(["OverTime","Gender"]).issubset(df_f.columns):
        grp = df_f.groupby(["OverTime","Gender"])["AttritionFlag"].mean().reset_index()
        c2 = alt.Chart(grp).mark_bar().encode(
            x=alt.X("OverTime:N", title="OverTime"),
            y=alt.Y("AttritionFlag:Q", title="Attrition Rate"),
            color=alt.Color("Gender:N"),
            tooltip=["OverTime","Gender","AttritionFlag"]
        ).properties(height=320, title="Attrition Rate by OverTime & Gender")
        st.altair_chart(c2, use_container_width=True)
    else:
        st.info("OverTime/Gender columns not found for Chart 2.")

    # 3) Tenure curve — Attrition vs YearsAtCompany (Line)
    if "YearsAtCompany" in df_f.columns:
        yrs = df_f.groupby("YearsAtCompany")["AttritionFlag"].mean().reset_index()
        c3 = alt.Chart(yrs).mark_line(point=True).encode(
            x=alt.X("YearsAtCompany:Q", title="Years At Company"),
            y=alt.Y("AttritionFlag:Q", title="Attrition Rate"),
            tooltip=["YearsAtCompany","AttritionFlag"]
        ).properties(height=320, title="Attrition Rate vs Tenure")
        st.altair_chart(c3, use_container_width=True)
    else:
        st.info("YearsAtCompany not found for Chart 3.")

    # 4) Heatmap — Avg Monthly Income by JobLevel & Attrition
    if set(["MonthlyIncome","JobLevel","Attrition"]).issubset(df_f.columns):
        heat = df_f.groupby(["JobLevel","Attrition"])["MonthlyIncome"].mean().reset_index()
        c4 = alt.Chart(heat).mark_rect().encode(
            x=alt.X("JobLevel:O", title="Job Level"),
            y=alt.Y("Attrition:N", title="Attrition"),
            color=alt.Color("MonthlyIncome:Q", title="Avg Monthly Income"),
            tooltip=["JobLevel","Attrition","MonthlyIncome"]
        ).properties(height=300, title="Avg Monthly Income by Job Level & Attrition")
        st.altair_chart(c4, use_container_width=True)
    else:
        st.info("MonthlyIncome/JobLevel/Attrition not found for Chart 4.")

    # 5) Distribution — Distance from Home by Attrition (Boxplot)
    if set(["DistanceFromHome","Attrition"]).issubset(df_f.columns):
        box = alt.Chart(df_f).mark_boxplot().encode(
            x=alt.X("Attrition:N"),
            y=alt.Y("DistanceFromHome:Q"),
            color=alt.Color("Attrition:N"),
            tooltip=["Attrition","DistanceFromHome"]
        ).properties(height=320, title="Distance From Home distribution by Attrition")
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("DistanceFromHome/Attrition not found for Chart 5.")

# ===========================
# 🤖 Modeling
# ===========================
with tab_model:
    st.markdown("### Train & Evaluate Models (DT / RF / GBRT)")
    X, y_text, y_bin = encode_features(df, "Attrition")

    run = st.button("Run 5-fold Stratified CV & Evaluate", key="run_models")
    if run:
        X_train, X_test, y_train_text, y_test_text, y_train_bin, y_test_bin = train_test_split(
            X, y_text, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
        )

        models = model_pack(random_state=random_state)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        metrics_rows = []
        cms = {}
        rocs = {}
        roc_colors = {"Decision Tree":"tab:blue","Random Forest":"tab:orange","Gradient Boosted Tree":"tab:green"}

        fig_roc, ax_roc = plt.subplots(figsize=(6.6,6))
        ax_roc.plot([0,1],[0,1],"--",color="gray",label="Chance")

        for name, model in models.items():
            cv_acc = cross_val_score(model, X, y_bin, cv=cv, scoring="accuracy").mean()
            model.fit(X_train, y_train_text)

            ypred_tr = model.predict(X_train); ypred_te = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                proba_te = model.predict_proba(X_test)[:,1]
            else:
                proba_te = model.decision_function(X_test)

            # Metrics
            acc_tr = accuracy_score(y_train_text, ypred_tr)
            acc_te = accuracy_score(y_test_text, ypred_te)
            prec = precision_score(y_test_text, ypred_te, pos_label="Yes")
            rec = recall_score(y_test_text, ypred_te, pos_label="Yes")
            f1 = f1_score(y_test_text, ypred_te, pos_label="Yes")
            auc = roc_auc_score(y_test_bin, proba_te)

            metrics_rows.append([name, acc_tr, acc_te, prec, rec, f1, auc, cv_acc])

            # Confusion matrices
            cm_tr = confusion_matrix(y_train_text, ypred_tr, labels=["No","Yes"])
            cm_te = confusion_matrix(y_test_text, ypred_te, labels=["No","Yes"])
            cms[name] = (cm_tr, cm_te)

            # ROC
            fpr, tpr, _ = roc_curve(y_test_bin, proba_te)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=roc_colors[name])

        ax_roc.set_title("ROC Curves (Test Set)"); ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate"); ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        # Metrics table
        metrics_df = pd.DataFrame(metrics_rows, columns=["Algorithm","Training Accuracy","Testing Accuracy","Precision (Test, Yes)","Recall (Test, Yes)","F1 Score (Test, Yes)","AUC (Test)","CV (5-fold) Accuracy"]).set_index("Algorithm")
        st.dataframe(metrics_df.round(4), use_container_width=True)

        # Confusion matrices (train & test)
        st.markdown("#### Confusion Matrices")
        for name, (cm_tr, cm_te) in cms.items():
            c1, c2 = st.columns(2)
            with c1: st.pyplot(plot_cm(cm_tr, f"{name} — Training"))
            with c2: st.pyplot(plot_cm(cm_te, f"{name} — Testing"))

        # Feature importances
        st.markdown("#### Feature Importances")
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
                top = fi.head(20).iloc[::-1]
                fig, ax = plt.subplots(figsize=(8,6))
                ax.barh(top["feature"], top["importance"], color="goldenrod")
                ax.set_title(f"Top 20 Feature Importances — {name}"); ax.set_xlabel("Importance"); ax.set_ylabel("Feature")
                st.pyplot(fig)
                with st.expander(f"Show full table — {name}"):
                    st.dataframe(fi.reset_index(drop=True))
            else:
                st.info(f"{name} does not expose feature_importances_.")

        # Save best model in session for prediction tab (by AUC)
        best_name = metrics_df["AUC (Test)"].astype(float).idxmax()
        st.success(f"Recommended model: **{best_name}** (highest Test AUC)")
        st.session_state["best_model_name"] = best_name
        st.session_state["feature_cols"] = list(X_train.columns)
        st.session_state["trained_models"] = {n:m for n,m in models.items()}

# ===========================
# 🔮 Predict on New Data
# ===========================
with tab_predict:
    st.markdown("### Upload a New Dataset to Predict Attrition")
    st.caption("If you ran modeling above, the app will reuse the **best model and feature set**. Otherwise it will train a quick Random Forest on the base dataset silently.")

    new_file = st.file_uploader("Upload new CSV (with same HR structure; 'Attrition' optional)", type=["csv"], key="predict")
    if new_file is not None:
        new_df = load_data(new_file)
        st.dataframe(new_df.head())

        # Ensure we have a model & feature columns
        if "trained_models" not in st.session_state or "feature_cols" not in st.session_state:
            # quick train on base dataset (RF)
            X0, y0_text, y0_bin = encode_features(df, "Attrition")
            rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
            rf.fit(X0, y0_text)
            st.session_state["trained_models"] = {"Random Forest": rf}
            st.session_state["best_model_name"] = "Random Forest"
            st.session_state["feature_cols"] = list(pd.get_dummies(df.drop(columns=["Attrition"]), drop_first=False).columns)

        best_name = st.session_state["best_model_name"]
        model = st.session_state["trained_models"][best_name]
        feature_list = st.session_state["feature_cols"]

        # Prepare new data
        X_new = new_df.drop(columns=[c for c in ["Attrition"] if c in new_df.columns])
        X_new_aligned = align_columns(X_new, feature_list)

        # Predict
        preds = model.predict(X_new_aligned)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new_aligned)[:,1]
        else:
            # scale decision_function to [0,1] roughly for display if needed
            df_raw = model.decision_function(X_new_aligned)
            proba = (df_raw - df_raw.min())/(df_raw.max()-df_raw.min()+1e-9)

        out = new_df.copy()
        out["Predicted_Attrition"] = preds
        out["Predicted_Prob_Yes"] = proba.round(4)

        st.success(f"Predictions generated with **{best_name}**.")
        st.dataframe(out.head(50), use_container_width=True)
        make_download(out, filename="attrition_predictions.csv")
    else:
        st.info("Upload a new CSV above to generate predictions.")
