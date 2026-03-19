# Exoplanet Detection via Machine Learning on Kepler Light Curves

A machine learning pipeline for detecting and classifying exoplanet transits from Kepler space telescope photometry data. Built as a collaborative project between a CS Lead [Tioluwani Enoch (@tioluwani-enoch)] (pipeline, ML) and a Physics Lead [Jacob Calan-Tolle (@jactlle)] (astrophysical validation, feature design).

---

## What This Does

This project takes raw Kepler light curve data from confirmed exoplanet host stars and eclipsing binaries and runs them through a full ML-ready preprocessing pipeline. The goal is to detect transit signals — the tiny dips in starlight caused by a planet passing in front of its host star — using a combination of signal processing and supervised machine learning.

The pipeline goes from raw FITS files all the way to a trained, physics-validated classifier that distinguishes real planet transits from eclipsing binary false positives.

---

## Project Structure

```
exoplanent-detection-blueprint/
│
├── data/
│   ├── raw/                              ← Kepler FITS files from MAST
│   └── processed/
│       ├── KIC_<id>/
│       │   ├── windows_raw.npy           ← fractional flux windows (feature extraction)
│       │   ├── windows_ml.npy            ← per-window normalized windows (ML input)
│       │   ├── centers.npy               ← center timestamps per window
│       │   └── meta.csv                  ← per-window metadata + physics columns
│       ├── KIC_<id>_lightcurve.csv       ← full stitched light curve per target
│       ├── combined_meta.csv             ← merged metadata across all targets
│       └── combined_features.csv         ← physics-filtered feature dataset
│
├── src/
│   ├── ingest.py                         ← download + parse Kepler data from MAST
│   ├── preprocess.py                     ← detrend, normalize, window light curves
│   ├── run_pipeline.py                   ← batch script — loops all targets
│   ├── features.py                       ← feature engineering + dataset builder
│   ├── model.py                          ← ML classifier + evaluation
│   ├── visualize.py                      ← plotting + result export
│   ├── check_gaps.py                     ← diagnostic — gap analysis tool
│   └── __init__.py
│
├── outputs/                              ← plots, model, result exports
│   ├── random_forest.joblib              ← saved trained model
│   ├── feature_importance.png            ← feature importance plot
│   └── roc_curve.png                     ← ROC curve
│
├── notebooks/                            ← Jupyter notebooks for exploration
├── .gitignore
└── README.md
```

---

## Training Targets

| KIC ID | Name | Type | Period (days) | Depth | Label |
|---|---|---|---|---|---|
| 11446443 | TrES-2b | Hot Jupiter | 2.471 | 1.44% | Planet ✓ |
| 5780885 | Kepler-7b | Hot Jupiter | 4.886 | ~1.0% | Planet ✓ |
| 11853905 | Kepler-4b | Neptune-sized | 3.213 | ~0.087% | Planet ✓ |
| 10619192 | Kepler-17b | Hot Jupiter | 1.486 | ~1.5% | Planet ✓ |
| 10874614 | Kepler-6b | Hot Jupiter | 3.235 | ~1.1% | Planet ✓ |
| 6922244 | Kepler-8b | Hot Jupiter | 3.523 | ~0.9% | Planet ✓ |
| 6541920 | Kepler-11b | Super-Earth | 10.304 | ~0.03% | Planet ✓ |
| 3544694 | KIC 3544694 | Eclipsing Binary | 3.846 | ~10% | False positive ✗ |

**Dataset stats:** 1,182 total samples — 447 planet transit windows (label=1), 200 EB eclipse windows (label=0), 535 non-transit negatives (label=0)

---

## Pipeline

### Phase 1 — Data Ingestion (`ingest.py`)

Downloads Kepler long-cadence (30-min) light curves from MAST via `lightkurve`. Extracts stellar parameters from FITS headers: Teff, log g, stellar radius, metallicity, Kepler magnitude.

### Phase 2 — Preprocessing (`preprocess.py` + `run_pipeline.py`)

Physics-approved pipeline order:

```
stitch → flatten → normalize → clip upward outliers
→ extract numpy → interpolate small gaps → window → save
```

Key decisions and the physics behind them:

**Stitching**: Each quarter is normalized individually before stitching via `corrector_func=lambda x: x.normalize()` to remove quarter-to-quarter flux offsets.

**Flattening**: `lightkurve`'s `flatten(window_length=1001)`. Window of 1001 cadences (~20 days) is long enough that the Savitzky-Golay filter cannot fit through a short transit. Rule: window must be ≥ 10× transit duration in cadences. `flatten()` must run on the native lightkurve object before any numpy extraction — extracting flux first causes the trend fit to lose metadata and over-flatten.

**Normalization**: `lc_flat.normalize()` via lightkurve only. No manual median divide afterward — double normalization washes out the ~0.016 transit signal entirely.

**Outlier clipping**: Upward outliers only (cosmic rays, flares, momentum dumps). Downward dips are never clipped — a planet can only block light, never add it.

**Gap handling**: Linear interpolation for gaps ≤ 10 consecutive cadences. Gaps > 10 cadences masked as NaN. All 64 large gaps in TrES-2b verified against known spacecraft events (Q8 absence, January 2013 safe mode, reaction wheel degradation) — no astrophysical signal lost.

**Windowing**: 201-cadence windows (~100 hours) with 50-cadence stride. Two output arrays:
- `windows_raw.npy` — fractional flux values preserved for feature extraction
- `windows_ml.npy` — per-window normalized for ML model input

Per-window `flux_in` and `flux_out` computed using known transit ephemeris (period, duration, t0) — not from BLS output.

### Phase 3 — Feature Engineering + ML Classification (`features.py`, `model.py`)

**Feature extraction:**

Three physics features extracted per window:

| Feature | Formula | Physics meaning |
|---|---|---|
| `norm_depth` | (flux_out − flux_in) / flux_out | Fractional transit depth |
| `dur_period_ratio` | duration / period | Transit duration as fraction of orbit |
| `radius_ratio` | √norm_depth | Planet-to-star radius ratio |

Physics filter applied to positive candidates only: `0.0005 < norm_depth < 0.1`, `0.5 < period < 500`, `0.001 < dur_period_ratio < 0.1`. Negative samples (non-transit windows) bypass the physics filter — applying a transit depth check to windows that are not supposed to contain transits is physically nonsensical.

**Eclipsing binary labeling (Option B, physics-approved):**

EB eclipse windows labeled 0 (false positives). EB non-eclipse windows excluded entirely — out-of-transit baseline is already well-represented by planet target negative samples. This keeps the physics filter semantically clean: it only validates things being claimed as planets.

**Model:**

Random Forest with `class_weight='balanced'` on 3 physics features across 8 targets.

**Operating parameters (physics-approved threshold=0.10):**

| Metric | Value |
|---|---|
| Precision | 0.855 |
| Recall | 0.589 |
| CV ROC-AUC | 0.974 ± 0.012 |
| True positives | 53 per batch |
| False positives | 9 per batch |

**Feature importances (balanced — all three contributing):**

| Feature | Importance |
|---|---|
| norm_depth | 0.372 |
| radius_ratio | 0.336 |
| dur_period_ratio | 0.293 |

**Honest limitations:**

The 0.974 CV ROC-AUC confirms the model has genuinely learned the physics of transit vs EB distinction. The recall ceiling (0.589) is a feature space limitation — 3 features derived from the same two flux measurements (flux_in, flux_out) hit a generalization ceiling on a diverse multi-target dataset.

Identified improvement path: ingress/egress slope and secondary eclipse depth — deferred to Phase 5 to avoid scope creep.

### Phase 4 — Visualization (`visualize.py`)

*In progress.*

### Phase 5 — Portfolio Packaging

*In progress.*

---

## Setup

**Requirements**: Python 3.8+, Windows/Mac/Linux

```bash
# Clone the repo
git clone https://github.com/your-username/exoplanent-detection-blueprint
cd exoplanent-detection-blueprint

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install numpy pandas matplotlib scipy scikit-learn astropy lightkurve jupyter notebook joblib
```

---

## Running the Pipeline

```bash
# Step 1 — run all targets through preprocessing
python src/run_pipeline.py

# Step 2 — extract features + build training dataset
python src/features.py

# Step 3 — train and evaluate the classifier
python src/model.py
```

Data downloads automatically from MAST on first run and caches locally in `data/raw/`. Full pipeline run takes ~15-20 minutes for all 8 targets.

---

## ML + Physics Collaboration

| Task | Owner |
|---|---|
| Pipeline architecture + ML implementation | CS Lead |
| Astrophysical parameter selection | Physics Lead |
| Preprocessing code | CS Lead |
| Preprocessing physics review + approval | Physics Lead |
| Feature transformation design | Physics Lead |
| Feature engineering code | CS Lead |
| Negative sampling strategy | Joint |
| EB labeling approach | Physics Lead |
| ML threshold tuning | Physics Lead |
| Visualization + result export | CS Lead |
| Physical interpretation of ML outputs | Physics Lead |

Every preprocessing decision in this repo has been reviewed and approved by the Physics Lead before being committed.

---

## Key Physics Decisions Log

| Decision | Rationale | Approved by |
|---|---|---|
| flatten() before numpy extraction | lightkurve loses trend metadata after .value extraction | Physics Lead |
| Clip upward outliers only | Transits are always negative dips — planets block light, never add it | Physics Lead |
| window_length=1001 for flatten | Must be ≥10× transit duration to avoid fitting through transit bottoms | Physics Lead |
| Median normalization via lightkurve | Manual divide after flatten = double normalization, kills transit signal | Physics Lead |
| 2× duration buffer for negative sampling | Prevents real transits being mislabeled as negatives | Physics Lead |
| EB eclipse windows labeled 0 only | Non-eclipse EB windows excluded — baseline already covered by planet negatives | Physics Lead |
| threshold=0.10 operating point | recall-heavy by design — Physics Lead validates false positives manually | Physics Lead |

---

## Resume Bullets

- Built end-to-end Kepler exoplanet detection pipeline in Python using `lightkurve`, `astropy`, and `scikit-learn` across 8 confirmed targets
- Preprocessed 500,000+ photometric cadences across 8 Kepler targets with physics-informed detrending, gap analysis, and windowing
- Trained Random Forest classifier achieving CV ROC-AUC of 0.974 on planet transit vs eclipsing binary classification
- Designed dual-output windowing system separating raw fractional flux (feature extraction) from ML-normalized windows (model input)
- Diagnosed and resolved triple-normalization bug that was washing out transit signals — recovered 1.44% depth from TrES-2b
- Collaborated across CS/Physics domains with documented physics approval checkpoints at every pipeline stage

---

## Data Sources

- Light curves: [MAST Archive](https://mast.stsci.edu) via `lightkurve`
- Planet parameters: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)
- Eclipsing binary catalog: [Kepler EB Catalog — Villanova/MAST](https://archive.stsci.edu/kepler/eclipsing_binaries.html)