# Exoplanet Detection via Machine Learning on Kepler Light Curves

A machine learning pipeline for detecting and classifying exoplanet transits from Kepler space telescope photometry data. Built as a collaborative project between a CS Lead (pipeline, ML) and a Physics Lead (astrophysical validation, feature design).

---

## What This Does

This project takes raw Kepler light curve data from a confirmed exoplanet host star (KIC 11446443 / TrES-2b) and runs it through a full ML-ready preprocessing pipeline. The goal is to detect transit signals — the tiny dips in starlight caused by a planet passing in front of its host star — using a combination of signal processing and supervised machine learning.

The pipeline goes from raw FITS files all the way to a windowed, physics-validated dataset ready for model training.

---

## Project Structure

```
exoplanent-detection-blueprint/
│
├── data/
│   ├── raw/                        ← Kepler FITS files from MAST
│   └── processed/
│       └── KIC_11446443/
│           ├── windows_raw.npy     ← fractional flux windows (feature extraction)
│           ├── windows_ml.npy      ← per-window normalized windows (ML input)
│           ├── centers.npy         ← center timestamps per window
│           └── meta.csv            ← per-window metadata + physics columns
│           KIC_11446443_lightcurve.csv  ← full stitched light curve
│
├── src/
│   ├── ingest.py                   ← download + parse Kepler data from MAST
│   ├── preprocess.py               ← detrend, normalize, window light curves
│   ├── features.py                 ← feature engineering (transit depth, period, etc.)
│   ├── model.py                    ← ML classifier
│   ├── visualize.py                ← plotting + result export
│   └── __init__.py
│
├── notebooks/                      ← Jupyter notebooks for exploration
├── outputs/                        ← plots, result exports
├── .gitignore
└── README.md
```

---

## Target Star

| Parameter | Value |
|---|---|
| KIC ID | 11446443 |
| Common name | TrES-2 / Kepler-1 |
| Planet | TrES-2b |
| Transit depth | ~1.6% (0.016 fractional flux) |
| Orbital period | 2.47 days |
| Transit duration | ~1.7 hours |
| Stellar Teff | 5850 K |
| Stellar log g | 4.455 |
| Stellar radius | 0.95 R☉ |
| Kepler magnitude | 11.338 |
| Quarters available | 15 (Q0–Q17, missing Q8, Q12, Q16) |

---

## Pipeline

### Phase 1 — Data Ingestion (`ingest.py`)

Downloads Kepler long-cadence (30-min) light curves from MAST via `lightkurve`. Extracts stellar parameters from FITS headers: Teff, log g, stellar radius, metallicity, Kepler magnitude.

### Phase 2 — Preprocessing (`preprocess.py`)

Physics-approved pipeline order:

```
stitch → flatten → normalize → clip upward outliers
→ extract numpy → interpolate small gaps → window → save
```

Key decisions and the physics behind them:

**Stitching**: Each quarter is normalized individually before stitching via `corrector_func=lambda x: x.normalize()` to remove quarter-to-quarter flux offsets.

**Flattening**: Uses `lightkurve`'s built-in `flatten(window_length=1001)`. Window length of 1001 cadences (~20 days) is long enough that the Savitzky-Golay filter cannot fit through TrES-2b's 1.7-hour transit. Rule of thumb: window must be ≥ 10× transit duration in cadences.

**Normalization**: `lc_flat.normalize()` via lightkurve only. No manual median divide afterward — double normalization washes out the ~0.016 transit signal entirely.

**Outlier clipping**: Upward outliers only (cosmic rays, flares, momentum dumps). Downward dips are never clipped — a planet can only block light, never add it. Transit dips are always negative excursions.

**Gap handling**: Linear interpolation for gaps ≤ 10 consecutive cadences. Gaps > 10 cadences are masked as NaN. Known large gaps (BKJD 808–902, 1280–1322, 1413) correspond to Q8 absence, January 2013 safe mode event, and reaction wheel degradation near end of mission — all verified and safe to mask.

**Windowing**: 201-cadence windows (~100 hours) with 50-cadence stride. Two output arrays:
- `windows_raw.npy` — fractional flux values preserved for feature extraction
- `windows_ml.npy` — per-window normalized for ML model input

Remaining NaNs inside valid windows are filled with the window median, not zero. Zeroing creates fake flat-bottomed dips that look like shallow transits to the ML model.

**Final output stats:**
- 1,031 windows of shape (1031, 201)
- Flux range: 0.9856 to 1.0026
- Transit depth recovered: 0.0144 (1.44%)

### Phase 3 — Feature Engineering (`features.py`)

*In progress.*

Planned features: transit depth (normalized), duration-to-period ratio, BLS power, ingress/egress slope, secondary eclipse depth, odd/even transit depth difference, centroid shift, CDPP noise floor.

### Phase 4 — ML Classification (`model.py`)

*In progress.*

### Phase 5 — Visualization (`visualize.py`)

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
pip install numpy pandas matplotlib scipy scikit-learn astropy lightkurve jupyter notebook
```

---

## Running the Pipeline

```bash
# Step 1 — ingest data
python src/ingest.py

# Step 2 — preprocess
python src/preprocess.py
```

Data downloads automatically from MAST on first run and caches locally in `data/raw/`.

---

## ML + Physics Collaboration

This project is split between a CS Lead and a Physics Lead:

| Task | Owner |
|---|---|
| Pipeline architecture + ML implementation | CS Lead |
| Astrophysical parameter selection | Physics Lead |
| Preprocessing code | CS Lead |
| Preprocessing physics review + approval | Physics Lead |
| Feature transformation design | Physics Lead |
| Feature engineering code | CS Lead |
| ML threshold tuning | Physics Lead |
| Visualization + result export | CS Lead |
| Physical interpretation of ML outputs | Physics Lead |

Every preprocessing decision in this repo has been reviewed and approved by the Physics Lead before being committed.

---

## Resume Bullets

- Built end-to-end Kepler exoplanet detection pipeline in Python using `lightkurve`, `astropy`, and `scikit-learn`
- Preprocessed 52,000+ photometric cadences across 15 Kepler quarters: detrending, sigma clipping, gap analysis, and windowing
- Recovered 1.44% transit depth signal (TrES-2b) through physics-informed normalization pipeline
- Designed dual-output windowing system separating raw fractional flux (feature extraction) from ML-normalized windows (model input)
- Collaborated across CS/Physics domains with documented physics approval checkpoints at each pipeline stage

---

## Data Sources

- Light curves: [MAST Archive](https://mast.stsci.edu) via `lightkurve`
- Planet parameters: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)