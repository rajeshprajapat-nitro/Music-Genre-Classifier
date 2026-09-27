"""
app.py
------
Streamlit web app for the Music Genre Classifier.

Run with:
    streamlit run app.py

Features:
  - Live prediction: move sliders for each audio feature -> get predicted genre
    + class probabilities, updated instantly.
  - Batch prediction: upload a CSV of songs -> get predictions for all rows.
  - Model insights tab: accuracy, confusion matrices, correlation heatmap,
    tempo-vs-acousticness scatter -- all the plots from training.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")

# ---------------------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reference():
    return pd.read_csv("feature_reference.csv")


try:
    bundle = load_model()
    ref_df = load_reference()
except FileNotFoundError:
    st.error(
        "model.pkl / feature_reference.csv nahi mile. Pehle terminal me chalao:\n\n"
        "```\npython generate_dataset.py\npython music_genre_classifier.py\n```\n"
        "Uske baad `streamlit run app.py` dobara chalao."
    )
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
le = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]
model_name = bundle["model_name"]
model_acc = bundle["accuracy"]
genres = list(le.classes_)

FEATURE_RANGES = {
    "tempo": (40.0, 220.0, 120.0, "BPM"),
    "acousticness": (0.0, 1.0, 0.3, ""),
    "energy": (0.0, 1.0, 0.6, ""),
    "danceability": (0.0, 1.0, 0.5, ""),
    "instrumentalness": (0.0, 1.0, 0.1, ""),
    "valence": (0.0, 1.0, 0.5, ""),
    "loudness": (-30.0, 0.0, -8.0, "dB"),
    "speechiness": (0.0, 1.0, 0.08, ""),
    "liveness": (0.0, 1.0, 0.2, ""),
    "duration_ms": (60000.0, 400000.0, 220000.0, "ms"),
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎵 Music Genre Classifier")
st.caption(
    f"Powered by **{model_name}** · trained on numerical audio features only "
    f"(tempo, acousticness, energy, ...) · test accuracy **{model_acc:.1%}**"
)

tab1, tab2, tab3 = st.tabs(["🎚️ Live Predictor", "📂 Batch Predict (CSV)", "📊 Model Insights"])

# ---------------------------------------------------------------------------
# TAB 1: Live predictor with sliders
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Adjust audio features and get an instant genre prediction")

    preset_col, _ = st.columns([2, 3])
    with preset_col:
        preset = st.selectbox(
            "Quick preset (loads a typical profile, then tweak sliders below)",
            ["-- custom --"] + genres,
        )

    if preset != "-- custom --":
        defaults = ref_df[ref_df["genre"] == preset][feature_cols].mean()
    else:
        defaults = pd.Series({k: v[2] for k, v in FEATURE_RANGES.items()})

    col1, col2 = st.columns(2)
    values = {}
    for i, feat in enumerate(feature_cols):
        lo, hi, _, unit = FEATURE_RANGES[feat]
        target_col = col1 if i % 2 == 0 else col2
        label = f"{feat} ({unit})" if unit else feat
        values[feat] = target_col.slider(
            label, float(lo), float(hi), float(defaults[feat]), key=f"slider_{feat}"
        )

    X_input = pd.DataFrame([values])[feature_cols].values
    X_scaled = scaler.transform(X_input)

    pred_idx = model.predict(X_scaled)[0]
    pred_genre = le.inverse_transform([pred_idx])[0]

    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted Genre", pred_genre.upper())

    with right:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            proba_df = pd.DataFrame({"genre": genres, "probability": proba}).sort_values(
                "probability", ascending=True
            )
            fig = px.bar(
                proba_df, x="probability", y="genre", orientation="h",
                title="Class probabilities", range_x=[0, 1],
            )
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Is model (KNN) ke liye probability estimate available nahi hai — sirf predicted class dikh rahi hai.")

# ---------------------------------------------------------------------------
# TAB 2: Batch predict from uploaded CSV
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV with the same feature columns to predict genres in bulk")
    st.write(f"Required columns: `{', '.join(feature_cols)}`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns in uploaded CSV: {missing}")
        else:
            X_batch = batch_df[feature_cols].values
            X_batch_scaled = scaler.transform(X_batch)
            preds = model.predict(X_batch_scaled)
            batch_df["predicted_genre"] = le.inverse_transform(preds)
            st.success(f"Predicted genres for {len(batch_df)} songs.")
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Sample rows aap `music_genre_dataset.csv` se copy karke test kar sakte ho.")

# ---------------------------------------------------------------------------
# TAB 3: Model insights (plots generated during training)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Training results & diagnostics")

    st.markdown("#### Accuracy comparison")
    acc_rows = [{"model": k, "accuracy": v["accuracy"]} for k, v in bundle["all_results"].items()]
    acc_df = pd.DataFrame(acc_rows)
    fig_acc = px.bar(acc_df, x="model", y="accuracy", range_y=[0, 1], text_auto=".2%",
                      color="model", title="Test Accuracy by Model")
    fig_acc.update_layout(height=350)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("#### Best hyperparameters (from GridSearchCV)")
    for k, v in bundle["all_results"].items():
        st.write(f"**{k}**: {v['best_params']}")

    st.markdown("#### Confusion matrices, correlation heatmap & feature scatter (from training)")
    img_cols = st.columns(2)
    img_files = [
        ("plot_confusion_matrix_knn.png", "KNN Confusion Matrix"),
        ("plot_confusion_matrix_svm.png", "SVM Confusion Matrix"),
        ("plot_correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("plot_tempo_vs_acousticness.png", "Tempo vs Acousticness by Genre"),
    ]
    for i, (fname, caption) in enumerate(img_files):
        try:
            img_cols[i % 2].image(fname, caption=caption, use_container_width=True)
        except Exception:
            img_cols[i % 2].warning(f"{fname} not found — re-run music_genre_classifier.py first.")

    st.markdown("#### Genre feature profile (dataset averages)")
    profile = ref_df.groupby("genre")[feature_cols].mean().round(3)
    st.dataframe(profile, use_container_width=True)
