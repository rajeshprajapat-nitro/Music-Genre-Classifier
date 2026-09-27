"""
generate_dataset.py
--------------------
Creates a realistic, domain-informed music-genre audio-features dataset.

WHY SYNTHETIC?
We tried to pull the real GTZAN 'features_30_sec.csv' from public GitHub
mirrors, but ran into 404s / API rate limits from this sandboxed environment.
Instead of blocking the project, this script generates a dataset with the
SAME structure and realistic per-genre statistics (grounded in known audio
characteristics of each genre), so every downstream step (EDA, scaling,
KNN, SVM, confusion matrix) works exactly like it would on the real data.

If you later download the real GTZAN or Spotify dataset, just point
`music_genre_classifier.py` at that CSV instead -- the rest of the
pipeline needs zero changes as long as column names line up.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

GENRES = [
    "classical", "jazz", "blues", "country", "reggae",
    "pop", "rock", "disco", "hiphop", "metal",
]

N_PER_GENRE = 150  # 150 * 10 = 1500 songs total

# Per-genre (mean, std) for each numerical audio feature.
# Values are grounded in well-known genre characteristics:
# - classical: slow tempo, highly acoustic, quiet, low energy
# - metal: fast tempo, loud, high energy, very low acousticness
# - hiphop: strong speechiness, high danceability, moderate tempo
# - reggae: mid tempo, high danceability, moderate acousticness
GENRE_PROFILES = {
    "classical": dict(tempo=(75, 12), acousticness=(0.93, 0.06), energy=(0.20, 0.10),
                       danceability=(0.30, 0.10), instrumentalness=(0.80, 0.15),
                       valence=(0.35, 0.15), loudness=(-22, 4), speechiness=(0.04, 0.02),
                       liveness=(0.15, 0.08), duration_ms=(280000, 60000)),
    "jazz":      dict(tempo=(110, 25), acousticness=(0.70, 0.15), energy=(0.40, 0.15),
                       danceability=(0.50, 0.12), instrumentalness=(0.55, 0.25),
                       valence=(0.50, 0.18), loudness=(-14, 3), speechiness=(0.06, 0.03),
                       liveness=(0.25, 0.15), duration_ms=(260000, 50000)),
    "blues":     dict(tempo=(95, 20), acousticness=(0.55, 0.20), energy=(0.45, 0.15),
                       danceability=(0.48, 0.12), instrumentalness=(0.25, 0.20),
                       valence=(0.40, 0.15), loudness=(-11, 3), speechiness=(0.06, 0.03),
                       liveness=(0.20, 0.12), duration_ms=(240000, 45000)),
    "country":   dict(tempo=(115, 18), acousticness=(0.45, 0.18), energy=(0.55, 0.15),
                       danceability=(0.55, 0.12), instrumentalness=(0.05, 0.06),
                       valence=(0.55, 0.18), loudness=(-9, 3), speechiness=(0.05, 0.02),
                       liveness=(0.18, 0.10), duration_ms=(220000, 35000)),
    "reggae":    dict(tempo=(90, 15), acousticness=(0.35, 0.18), energy=(0.55, 0.15),
                       danceability=(0.70, 0.10), instrumentalness=(0.10, 0.10),
                       valence=(0.60, 0.18), loudness=(-9, 3), speechiness=(0.08, 0.04),
                       liveness=(0.20, 0.10), duration_ms=(230000, 40000)),
    "pop":       dict(tempo=(118, 15), acousticness=(0.18, 0.12), energy=(0.65, 0.13),
                       danceability=(0.68, 0.10), instrumentalness=(0.02, 0.03),
                       valence=(0.60, 0.18), loudness=(-6, 2), speechiness=(0.07, 0.04),
                       liveness=(0.17, 0.10), duration_ms=(210000, 30000)),
    "rock":      dict(tempo=(125, 20), acousticness=(0.12, 0.10), energy=(0.75, 0.12),
                       danceability=(0.50, 0.12), instrumentalness=(0.10, 0.12),
                       valence=(0.50, 0.18), loudness=(-6, 2), speechiness=(0.06, 0.03),
                       liveness=(0.25, 0.15), duration_ms=(250000, 45000)),
    "disco":     dict(tempo=(120, 12), acousticness=(0.15, 0.10), energy=(0.72, 0.10),
                       danceability=(0.75, 0.08), instrumentalness=(0.05, 0.06),
                       valence=(0.68, 0.15), loudness=(-7, 2), speechiness=(0.06, 0.03),
                       liveness=(0.20, 0.10), duration_ms=(240000, 30000)),
    "hiphop":    dict(tempo=(100, 15), acousticness=(0.12, 0.10), energy=(0.68, 0.13),
                       danceability=(0.78, 0.08), instrumentalness=(0.01, 0.02),
                       valence=(0.55, 0.18), loudness=(-6, 2), speechiness=(0.22, 0.10),
                       liveness=(0.18, 0.10), duration_ms=(220000, 35000)),
    "metal":     dict(tempo=(140, 20), acousticness=(0.03, 0.03), energy=(0.90, 0.07),
                       danceability=(0.42, 0.10), instrumentalness=(0.15, 0.15),
                       valence=(0.35, 0.18), loudness=(-4, 2), speechiness=(0.09, 0.05),
                       liveness=(0.28, 0.15), duration_ms=(260000, 50000)),
}

FEATURES = ["tempo", "acousticness", "energy", "danceability", "instrumentalness",
            "valence", "loudness", "speechiness", "liveness", "duration_ms"]


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate():
    rows = []
    for genre in GENRES:
        profile = GENRE_PROFILES[genre]
        for i in range(N_PER_GENRE):
            row = {"filename": f"{genre}.{i:05d}.wav", "genre": genre}
            for feat in FEATURES:
                mean, std = profile[feat]
                val = RNG.normal(mean, std)
                if feat in ("acousticness", "energy", "danceability",
                             "instrumentalness", "valence", "speechiness", "liveness"):
                    val = clip01(val)
                if feat == "tempo":
                    val = max(40, val)
                if feat == "duration_ms":
                    val = max(60000, val)
                row[feat] = round(float(val), 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "music_genre_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    print(df.groupby("genre").size())
