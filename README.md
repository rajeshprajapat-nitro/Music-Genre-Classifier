# 🎵 Music Genre Classifier (KNN + SVM)

Classifies songs into 10 genres — `classical, jazz, blues, country, reggae,
pop, rock, disco, hiphop, metal` — using **only numerical audio features**
(tempo, acousticness, energy, danceability, instrumentalness, valence,
loudness, speechiness, liveness, duration_ms).

##   ⚠️ About the dataset
This project was built to run against the real **GTZAN** audio-features
dataset, but this sandboxed environment couldn't reliably download it from
GitHub (404s / API rate limits). So `generate_dataset.py` creates a
**domain-informed synthetic dataset** — same structure, and per-genre
statistics grounded in real audio characteristics (e.g. classical = slow +
highly acoustic, metal = fast + loud + non-acoustic, hiphop = high
speechiness, etc.). This keeps the full ML pipeline realistic and working
end-to-end.

**To use the real GTZAN dataset instead:**
1. Download `features_30_sec.csv` from Kaggle:
   https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Replace `music_genre_dataset.csv` with it (make sure there's a `genre`
   column and numerical feature columns).
3. Update the `feature_cols` list in `music_genre_classifier.py` to match
   the real column names (e.g. `tempo`, `chroma_stft_mean`,
   `spectral_centroid_mean`, `mfcc1_mean` ... `mfcc20_mean`, etc.)
4. Re-run — nothing else changes.

## 📁 Files (all in this one folder)
| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds `music_genre_dataset.csv` (1500 songs × 10 features × 10 genres) |
| `music_genre_classifier.py` | Full pipeline: EDA → scaling → KNN + SVM → tuning → evaluation → saves `model.pkl` |
| `app.py` | **Streamlit web app** — the interactive UI (see below) |
| `music_genre_dataset.csv` | The dataset |
| `model.pkl` | Trained model + scaler + label encoder (used by `app.py`) |
| `feature_reference.csv` | Per-genre feature values (used for slider presets in the app) |
| `plot_correlation_heatmap.png` | Feature correlation matrix |
| `plot_tempo_vs_acousticness.png` | Scatter of the two headline features, colored by genre |
| `plot_confusion_matrix_knn.png` | KNN multi-class confusion matrix |
| `plot_confusion_matrix_svm.png` | SVM multi-class confusion matrix |
| `results_summary.txt` | Best hyperparameters + accuracy for each model |
| `requirements.txt` | All Python dependencies |

## ▶️ How to run

**Step 1 — install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — generate data & train the models (only needed once, or whenever you change the data):**
```bash
python generate_dataset.py          # creates music_genre_dataset.csv
python music_genre_classifier.py    # trains, tunes, evaluates, saves model.pkl
```

**Step 3 — launch the web app:**
```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your browser automatically.

## 🖥️ Web App Features
- **🎚️ Live Predictor** — sliders for every audio feature (tempo, acousticness,
  energy, ...); pick a genre preset to auto-fill typical values, then tweak
  them and watch the predicted genre + class-probability bar chart update
  instantly.
- **📂 Batch Predict (CSV)** — upload a CSV with the same feature columns and
  get genre predictions for every row, downloadable as a new CSV.
- **📊 Model Insights** — accuracy comparison, best hyperparameters,
  confusion matrices, correlation heatmap, and per-genre average feature
  profile table, all in one dashboard.

## 🧠 What this demonstrates (for your resume)
- **Multi-class classification** (10 classes) with two different algorithm
  families — instance-based (KNN) and margin-based (SVM with RBF kernel).
- **Feature scaling / standardization**: KNN and SVM are distance/margin
  based, so `StandardScaler` is fit only on training data (avoiding data
  leakage) before transforming both train and test sets. Without this,
  `duration_ms` (~10^5 scale) would completely dominate `acousticness`
  (0–1 scale) in distance calculations.
- **Hyperparameter tuning** via `GridSearchCV` (k and weighting for KNN;
  C and gamma for SVM) with 5-fold cross-validation.
- **Evaluation**: accuracy + full multi-class confusion matrix +
  per-class precision/recall/F1 (via `classification_report`), so you can
  see *which* genres get confused with each other (e.g. rock/country/blues
  often overlap acoustically — a genuinely interesting decision-boundary
  problem, not a trivial one).

## 📊 Sample results (synthetic data)
| Model | Accuracy | Best Params |
|---|---|---|
| KNN | ~0.73 | see `results_summary.txt` |
| SVM (RBF) | ~0.77 | see `results_summary.txt` |

Classical is near-perfectly separated (very distinct acoustic profile);
genres with overlapping tempo/energy ranges (pop/disco, blues/country/rock)
show the expected confusion — this is exactly the "complex decision
boundary" learning outcome the project is meant to demonstrate.
