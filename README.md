# 🎵 Spotify Track Popularity Prediction

End-to-end predictive analytics project (KDD process) that predicts a track's Spotify **popularity score (0–100)** from its audio features, release information, title features and the reach/genres of its artists.

**Author:** Abhinava Akash Kotha (Roll No. 23070521075) – B.Tech CSE (AI/ML), Symbiosis Institute of Technology, Nagpur

## Problem
Supervised **regression**: given a track's characteristics, estimate its `popularity`. Primary metric: RMSE (also MAE and R²), compared against a predict-the-mean baseline.

## Datasets (2)
Both come from the Kaggle dataset **Spotify Dataset 1921–2020, 600k+ Tracks** (`yamaerenay/spotify-dataset-19212020-600k-tracks`). Full copies are included under `data/` and stored with Git LFS:

| File | Content | Size |
|------|---------|------|
| `data/tracks.csv` | ~600k tracks: popularity, audio features, release date, artist IDs | ~600k rows |
| `data/artists.csv` | Artist followers, genres, artist popularity | ~1.1M rows |

The notebook **downloads them automatically** with `kagglehub` – no manual download. Git LFS is required to check out the included full CSVs. The notebook also creates small samples (`tracks_sample.csv`, `artists_sample.csv`) in `data/`; if the Kaggle download fails, it falls back to these samples.

## Methodology (KDD stages)
1. Problem identification
2. Data selection & loading (two datasets)
3. Data understanding
4. Data integration – tracks joined to artists via `id_artists` (multi-artist aggregation)
5. Pre-processing – duplicates, invalid values, outliers, imputation (inside the pipeline)
6. Transformation – feature engineering (log followers, genre flags, title & date features)
7. Exploratory data analysis
8. Modelling – Baseline, Linear, Ridge, Random Forest, Histogram Gradient Boosting (+ RandomizedSearchCV tuning)
9. Evaluation – hold-out test set, residual analysis, error by popularity band, **leakage ablation study**
10. Interpretation – permutation importance, partial dependence
11. Conclusions, limitations, future work

> **Data-leakage note:** artist popularity is derived from track popularity, so it is excluded from the main model and evaluated separately in the ablation study.

## Repository structure
```
├── Spotify_Track_Popularity_Prediction.ipynb   # full implementation
├── README.md
├── requirements.txt
├── data/
│   ├── tracks.csv                              # full dataset (Git LFS)
│   ├── artists.csv                             # full dataset (Git LFS)
│   ├── tracks_sample.csv                       # created by the notebook (Section 12)
│   └── artists_sample.csv
└── models/                                     # saved model + result tables
```

## How to run
**Google Colab (recommended):** upload the notebook → *Runtime → Run all*. The first cell installs `kagglehub`.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook Spotify_Track_Popularity_Prediction.ipynb
```

## Results
*(Add your final table here after running the notebook – copy from `models/test_results.csv` and `models/ablation_results.csv`.)*

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Baseline (mean) | | | |
| Linear Regression | | | |
| Ridge Regression | | | |
| Random Forest | | | |
| Hist Gradient Boosting (tuned) | | | |

## Limitations
Popularity is a time-dependent snapshot; random (not time-based) split; only primary-artist genres used.
