import numpy as np
import pandas as pd
import os
import lightkurve as lk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 201
STEP_SIZE   = 50


# ── 1. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slice into overlapping fixed-size windows.
    Returns TWO arrays:
      windows_raw — fractional flux preserved (for feature extraction)
      windows_ml  — per-window normalized (for ML model input)
    Windows with >10% NaN skipped.
    """
    windows_raw, windows_ml, centers = [], [], []

    for start in range(0, len(flux) - window_size + 1, step_size):
        end    = start + window_size
        window = np.array(flux[start:end], dtype=np.float32)

        if np.isnan(window).sum() > window_size * 0.1:
            continue
        if np.isnan(window).any():
            window_median = np.nanmedian(window)
            window = np.where(np.isnan(window), window_median, window)

        # Raw — preserve fractional flux (transit depth intact)
        windows_raw.append(window.copy())

        # ML — per-window normalize for model input
        median = np.median(window)
        std    = np.std(window)
        if std > 0:
            window_ml = (window - median) / std
        else:
            window_ml = window - median
        windows_ml.append(window_ml)

        centers.append(time[start + window_size // 2])

    return (
        np.array(windows_raw, dtype=np.float32),
        np.array(windows_ml,  dtype=np.float32),
        np.array(centers)
    )


# ── 2. SAVE ───────────────────────────────────────────────────────────────────

def save_windows(windows_raw, windows_ml, centers, kic_id, label,
                 period_days=None, duration_hours=None,
                 depth_raw=None, flux_out=None, flux_in=None):
    """
    Save both window arrays to disk.
      windows_raw.npy — fractional flux, transit depth preserved
      windows_ml.npy  — per-window normalized, for ML model input
      centers.npy     — center timestamp per window
      meta.csv        — per-window metadata with physics columns
    """
    out_dir = os.path.join(PROCESSED_DIR, f"KIC_{kic_id}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "windows_raw.npy"), windows_raw)
    np.save(os.path.join(out_dir, "windows_ml.npy"),  windows_ml)
    np.save(os.path.join(out_dir, "centers.npy"),     centers)

    meta = pd.DataFrame({
        "window_index":   range(len(windows_raw)),
        "center_time":    centers,
        "kic_id":         kic_id,
        "label":          label,
        "period_days":    period_days,
        "duration_hours": duration_hours,
        "depth_raw":      depth_raw,
        "flux_out":       flux_out,
        "flux_in":        flux_in
    })
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)

    print(f"  Saved {len(windows_raw)} windows → {out_dir}")
    print(f"  windows_raw.npy — fractional flux (feature extraction)")
    print(f"  windows_ml.npy  — normalized (ML model input)")
    return meta


# ── 3. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, label=1):
    """
    Physics-approved pipeline order:
      stitch → flatten → normalize → clip upward only
      → extract numpy → interpolate gaps
      → save CSV → window (raw + ml) → save

    Core rules:
      - flatten() on native lightkurve object before any numpy extraction
      - normalize() via lightkurve only — no manual median divide after
      - clip upward outliers ONLY — transits are negative dips, never clip down
      - windows_raw preserves fractional flux for feature extraction
      - windows_ml applies per-window normalization for ML model input
    """
    print(f"\nPreprocessing KIC {kic_id}...")

    # Step 1 — stitch with per-quarter normalization
    lc_stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    print(f"  Stitched: {len(lc_stitched.flux)} cadences | "
          f"baseline {lc_stitched.time.value[0]:.1f} – "
          f"{lc_stitched.time.value[-1]:.1f} BKJD")

    # Step 2 — flatten on native lightkurve object
    # window_length=1001 (~500 hrs, ~20 days) physics-approved
    # long enough that SavGol cannot fit through a 1.7 hr transit
    lc_flat, trend = lc_stitched.flatten(window_length=1001, return_trend=True)
    print(f"  Flattened: trend removed")

    # Step 3 — normalize using lightkurve built-in ONLY
    # do NOT manually divide by median after — double normalization
    # washes out the ~0.016 transit signal entirely
    lc_normalized = lc_flat.normalize()
    print(f"  Normalized: flux centered at 1.0")

    # Step 4 — clip upward outliers ONLY (physics-approved)
    # Transits are negative dips — a planet can only block light, never add it
    # Upward spikes = cosmic rays, flares, artifacts → clip
    # Downward dips = transits, stellar eclipses → never touch
    median_flux  = np.nanmedian(lc_normalized.flux.value)
    std_flux     = np.nanstd(lc_normalized.flux.value)
    outlier_mask = lc_normalized.flux.value > median_flux + 2.5 * std_flux
    lc_clipped   = lc_normalized[~outlier_mask]
    n_removed    = int(outlier_mask.sum())
    print(f"  Clipped {n_removed} upward outliers (transits preserved)")

    # Step 5 — extract numpy arrays (all lightkurve ops done)
    time     = np.array(lc_clipped.time.value,     dtype=np.float64)
    flux     = np.array(lc_clipped.flux.value,     dtype=np.float64)
    flux_err = np.array(lc_clipped.flux_err.value, dtype=np.float64)

    print(f"  Flux range after clip: "
          f"{np.nanmin(flux):.6f} to {np.nanmax(flux):.6f}")

    # Step 6 — interpolate small gaps only (<=10 cadences)
    # large gaps stay NaN and get skipped during windowing
    flux_series = pd.Series(flux)
    flux_interp = flux_series.interpolate(
        method='linear',
        limit=10,
        limit_direction='both'
    ).values

    # flux_err normalized by median
    median        = np.nanmedian(flux_interp)
    flux_err_norm = flux_err / median

    print(f"  Flux range after interpolate: "
          f"{np.nanmin(flux_interp):.6f} to {np.nanmax(flux_interp):.6f}")

    # Step 7 — save full light curve CSV
    lc_df = pd.DataFrame({
        "time_BKJD": time,
        "flux_norm":  flux_interp,
        "flux_err":   flux_err_norm
    })
    lc_df.to_csv(
        os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv"),
        index=False
    )
    print(f"  Saved full light curve CSV")

    # Physics columns — computed from full light curve
    flux_out  = float(np.nanmedian(flux_interp))
    flux_in   = float(np.nanmin(flux_interp))
    depth_raw = float(flux_out - flux_in)

    period_days    = 289.86
    duration_hours = 7.4

    # Step 8 — window into raw + ml arrays
    windows_raw, windows_ml, centers = window_lightcurve(time, flux_interp)
    print(f"  Windowed: {len(windows_raw)} windows of size {WINDOW_SIZE}")
    print(f"  Raw flux range in windows: "
          f"{windows_raw.min():.6f} to {windows_raw.max():.6f}")

    # Step 9 — save
    meta = save_windows(
        windows_raw, windows_ml, centers, kic_id, label,
        period_days=period_days,
        duration_hours=duration_hours,
        depth_raw=depth_raw,
        flux_out=flux_out,
        flux_in=flux_in
    )
    return windows_raw, windows_ml, centers, meta


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    kic_id = 11446443
    label  = 1

    print("Searching for local or cached light curves...")
    search_result = lk.search_lightcurve(
        "KIC 11446443",
        mission="Kepler",
        author="Kepler",
        exptime=1800
    )

    lc_collection = search_result.download_all(
        download_dir=os.path.join(PROJECT_ROOT, "data", "raw")
    )

    if lc_collection is not None:
        windows_raw, windows_ml, centers, meta = preprocess_target(
            kic_id, lc_collection, label
        )

        print("\nSample meta:")
        print(meta.head())
        print(f"\nwindows_raw shape: {windows_raw.shape}")
        print(f"windows_ml shape:  {windows_ml.shape}")
        print(f"Raw min/max:       {windows_raw.min():.6f} / {windows_raw.max():.6f}")
        print(f"ML  min/max:       {windows_ml.min():.4f} / {windows_ml.max():.4f}")
    else:
        print("No data found. Check your data/raw directory.")
