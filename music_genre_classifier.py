"""
music_genre_classifier.py
--------------------------
Music Genre Classification using purely numerical audio features.

Pipeline:
  1. Load data (Pandas)
  2. EDA: feature correlation heatmap, per-genre distributions
  3. Train/test split (stratified)
  4. Feature scaling (StandardScaler) -- critical for KNN & SVM since they
     are distance/margin based and features like 'duration_ms' (~10^5)
     would otherwise dominate 'acousticness' (0-1).
  5. Train two models: K-Nearest Neighbors and SVM (RBF kernel)
  6. Hyperparameter tuning via GridSearchCV (k for KNN, C/gamma for SVM)
  7. Evaluate: Accuracy, multi-class Confusion Matrix, classification report
  8. Save all plots as PNGs + a results summary text file

Required tech stack: Pandas, Scikit-Learn (KNeighborsClassifier, SVC)
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

sns.set_style("whitegrid")

DATA_PATH = "music_genre_dataset.csv"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("\nGenre distribution:\n", df["genre"].value_counts())
    return df


def run_eda(df, feature_cols):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plot_correlation_heatmap.png", dpi=150)
    plt.close()

    # Tempo vs Acousticness scatter, colored by genre (the two "headline" features)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x="tempo", y="acousticness", hue="genre",
                     palette="tab10", alpha=0.7, s=40)
    plt.title("Tempo vs Acousticness by Genre")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_tempo_vs_acousticness.png", dpi=150)
    plt.close()

    print("Saved EDA plots: plot_correlation_heatmap.png, plot_tempo_vs_acousticness.png")


def plot_confusion(cm, labels, title, filename):
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_and_evaluate(X_train, X_test, y_train, y_test, label_names):
    results = {}

    # ---------------- KNN ----------------
    knn_param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    best_knn = knn_grid.best_estimator_
    knn_pred = best_knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_cm = confusion_matrix(y_test, knn_pred)

    print(f"\n=== K-Nearest Neighbors ===")
    print(f"Best params: {knn_grid.best_params_}")
    print(f"Test Accuracy: {knn_acc:.4f}")
    print(classification_report(y_test, knn_pred, target_names=label_names))
    plot_confusion(knn_cm, label_names, f"KNN Confusion Matrix (acc={knn_acc:.3f})",
                    "plot_confusion_matrix_knn.png")
    results["KNN"] = dict(model=best_knn, accuracy=knn_acc,
                           best_params=knn_grid.best_params_, cm=knn_cm)

    # ---------------- SVM ----------------
    svm_param_grid = {"C": [0.1, 1, 10, 50], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf"]}
    svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    svm_pred = best_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)

    print(f"\n=== SVM (RBF kernel) ===")
    print(f"Best params: {svm_grid.best_params_}")
    print(f"Test Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm_pred, target_names=label_names))
    plot_confusion(svm_cm, label_names, f"SVM Confusion Matrix (acc={svm_acc:.3f})",
                    "plot_confusion_matrix_svm.png")
    results["SVM"] = dict(model=best_svm, accuracy=svm_acc,
                           best_params=svm_grid.best_params_, cm=svm_cm)

    return results


def main():
    df = load_data()

    feature_cols = ["tempo", "acousticness", "energy", "danceability",
                     "instrumentalness", "valence", "loudness",
                     "speechiness", "liveness", "duration_ms"]

    run_eda(df, feature_cols)

    X = df[feature_cols].values
    y_raw = df["genre"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling -- fit ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, label_names)

    # Pick the better model to ship in the web app (by test accuracy)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\nBest model overall: {best_name} (acc={results[best_name]['accuracy']:.4f}) -> saved for the web app")

    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler,
            "label_encoder": le,
            "feature_cols": feature_cols,
            "accuracy": results[best_name]["accuracy"],
            "all_results": {k: {"accuracy": v["accuracy"], "best_params": v["best_params"]}
                             for k, v in results.items()},
        }, f)
    print("Saved: model.pkl (used by app.py)")

    # Also save the feature stats (min/max/mean) per genre -- handy for the app's
    # slider defaults and for a "typical genre profile" reference table.
    df[feature_cols + ["genre"]].to_csv("feature_reference.csv", index=False)

    # Summary
    with open("results_summary.txt", "w") as f:
        f.write("Music Genre Classification - Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {df.shape[0]} songs, {len(feature_cols)} numerical features, "
                f"{len(label_names)} genres\n")
        f.write(f"Features used: {', '.join(feature_cols)}\n\n")
        for name, res in results.items():
            f.write(f"{name}\n")
            f.write(f"  Best hyperparameters: {res['best_params']}\n")
            f.write(f"  Test Accuracy: {res['accuracy']:.4f}\n\n")

    print("\nSaved: results_summary.txt")
    print("Saved: plot_confusion_matrix_knn.png, plot_confusion_matrix_svm.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
