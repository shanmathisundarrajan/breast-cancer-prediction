import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

TOP_N = 10  # sliders in the quick tab; other features use dataset averages


@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, data


model, accuracy, data = train_model()
names = list(data.feature_names)
df = pd.DataFrame(data.data, columns=names)
importances = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
top_features = list(importances.index[:TOP_N])

mean_vals = df.mean()
benign_vals = df[data.target == 1].median()
malignant_vals = df[data.target == 0].median()


def show_result(p_malignant, p_benign):
    if p_benign >= 0.5:
        st.success(f"Benign ({p_benign:.0%} confidence)")
    else:
        st.error(f"Malignant ({p_malignant:.0%} confidence)")
    c1, c2 = st.columns(2)
    c1.metric("Benign probability", f"{p_benign:.0%}")
    c2.metric("Malignant probability", f"{p_malignant:.0%}")
    st.progress(p_benign, text="Benign probability")


def predict_row(values):
    proba = model.predict_proba(np.array(values, dtype=float).reshape(1, -1))[0]
    return float(proba[0]), float(proba[1])


def normalize(col):
    return str(col).strip().lower().replace("_", " ")


def predict_frame(frame):
    """Returns (result_df, error_message). Accepts column names with spaces or underscores."""
    frame = frame.copy()
    frame.columns = [normalize(c) for c in frame.columns]
    missing = [n for n in names if n not in frame.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)
    X = frame[names].apply(pd.to_numeric, errors="coerce")
    bad_rows = X.index[X.isnull().any(axis=1)].tolist()
    if bad_rows:
        shown = ", ".join(str(r + 2) for r in bad_rows[:10])  # +2 = spreadsheet row number
        return None, f"Empty or non-numeric values found in file rows: {shown}"
    proba = model.predict_proba(X.values)
    out = pd.DataFrame({
        "Prediction": np.where(proba[:, 1] >= 0.5, "Benign", "Malignant"),
        "Benign probability": proba[:, 1].round(3),
        "Malignant probability": proba[:, 0].round(3),
    })
    return pd.concat([out, X.reset_index(drop=True)], axis=1), None


def slider_key(f):
    return "slider_" + f


def load_sample(values):
    for f in top_features:
        st.session_state[slider_key(f)] = float(values[f])


for f in top_features:
    st.session_state.setdefault(slider_key(f), float(mean_vals[f]))

# ---------------- Page ----------------
st.title("🩺 Breast Cancer Prediction")
st.write(
    "Enter cell measurements and a Random Forest model, trained on the Wisconsin Diagnostic "
    "Breast Cancer dataset, predicts whether the tumor is **benign** or **malignant**."
)
st.info(
    "The 30 inputs are measurements from a digitized image of a fine needle aspirate "
    "(radius, texture, area, and so on), normally produced by a lab. "
    "Educational project, not a medical diagnosis tool."
)

tab1, tab2, tab3 = st.tabs(["Quick predict", "Enter all 30 values", "Upload CSV"])

# ---- Tab 1: sliders ----
with tab1:
    st.write(f"Adjust the {TOP_N} most important measurements. The other {len(names) - TOP_N} use dataset averages.")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load typical benign sample", on_click=load_sample, args=(benign_vals,))
    b2.button("Load typical malignant sample", on_click=load_sample, args=(malignant_vals,))

    row = mean_vals.copy()
    left, right = st.columns(2)
    for i, f in enumerate(top_features):
        lo, hi = float(df[f].min()), float(df[f].max())
        with (left if i % 2 == 0 else right):
            row[f] = st.slider(f.capitalize(), lo, hi, key=slider_key(f), step=(hi - lo) / 200)

    st.subheader("Prediction")
    show_result(*predict_row(row.values))

# ---- Tab 2: all 30 values ----
with tab2:
    st.write("Type in all 30 measurements, then click Predict. Defaults are dataset averages.")
    with st.form("full_form"):
        cols = st.columns(3)
        entered = []
        for i, n in enumerate(names):
            with cols[i % 3]:
                entered.append(st.number_input(n.capitalize(), value=float(round(mean_vals[n], 4)), format="%.4f"))
        submitted = st.form_submit_button("Predict")
    if submitted:
        st.subheader("Prediction")
        show_result(*predict_row(entered))

# ---- Tab 3: CSV upload ----
with tab3:
    st.write("Upload a CSV with one patient per row and these 30 columns. Download the template to see the format.")
    template = pd.concat([df[data.target == 1].head(1), df[data.target == 0].head(1)])
    st.download_button("Download CSV template", template.to_csv(index=False), "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type="csv")
    if uploaded is not None:
        try:
            result, error = predict_frame(pd.read_csv(uploaded))
        except Exception:
            result, error = None, "Couldn't read that file. Check that it's a valid CSV."
        if error:
            st.error(error)
        else:
            st.success(f"Predicted {len(result)} patient(s)")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download results", result.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------- Model info ----------------
st.divider()
st.subheader("About the model")
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{accuracy:.1%}")
m2.metric("Patients in dataset", len(df))
m3.metric("Features", len(names))
st.write("Most important features for the prediction:")
st.bar_chart(importances.head(10)[::-1], horizontal=True)
