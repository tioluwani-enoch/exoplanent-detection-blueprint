import numpy as np
import pandas as pd
import os
import lightkurve as lk
from astropy.stats import sigma_clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 201
STEP_SIZE   = 50


# ── 1. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slice into overlapping fixed-size windows.
    Each window independently normalized.
    Windows with >10% NaN skipped (large spacecraft gaps).
    """
    windows, centers = [], []

    for start in range(0, len(flux) - window_size + 1, step_size):
        end    = start + window_size
        window = flux[start:end]

        if np.isnan(window).sum() > window_size * 0.1:
            continue

        window = np.nan_to_num(window, nan=0.0)

        median = np.median(window)
        std    = np.std(window)
        if std > 0:
            window = (window - median) / std

        windows.append(window)
        centers.append(time[start + window_size // 2])

    return np.array(windows, dtype=np.float32), np.array(centers)


# ── 2. SAVE ───────────────────────────────────────────────────────────────────

def save_windows(windows, centers, kic_id, label,
                 period_days=None, duration_hours=None,
                 depth_raw=None, flux_out=None, flux_in=None):
    out_dir = os.path.join(PROCESSED_DIR, f"KIC_{kic_id}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "windows.npy"), windows)
    np.save(os.path.join(out_dir, "centers.npy"), centers)

    meta = pd.DataFrame({
        "window_index":   range(len(windows)),
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
    print(f"  Saved {len(windows)} windows → {out_dir}")
    return meta


# ── 3. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, label=1):
    """
    Physics-approved pipeline order:
      stitch → flatten → sigma clip → interpolate → normalize → window → save
    
    Core rule: never extract .flux.value into numpy until all
    lightkurve operations are done. flatten() needs the lightkurve
    object intact to fit the Savitzky-Golay trend correctly.
    """
    print(f"\nPreprocessing KIC {kic_id}...")

    # Step 1 — stitch with per-quarter normalization
    lc_stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    print(f"  Stitched: {len(lc_stitched.flux)} cadences | "
          f"baseline {lc_stitched.time.value[0]:.1f} – "
          f"{lc_stitched.time.value[-1]:.1f} BKJD")

    # Step 2 — flatten FIRST on native lightkurve object
    # window_length=101 (~50 hrs) — physics-approved
    # Must be shorter than TrES-2b period (~119 cadences)
    lc_flat, trend = lc_stitched.flatten(window_length=101, return_trend=True)
    print(f"  Flattened: trend removed")

    # Step 3 — sigma clip using lightkurve's built-in method
    lc_clipped = lc_flat.remove_outliers(sigma=2.5)
    n_removed  = len(lc_flat.flux) - len(lc_clipped.flux)
    print(f"  Sigma clipped: {n_removed} outliers removed")

    # Step 4 — extract numpy arrays (lightkurve operations are done)
    time     = np.array(lc_clipped.time.value,     dtype=np.float64)
    flux     = np.array(lc_clipped.flux.value,     dtype=np.float64)
    flux_err = np.array(lc_clipped.flux_err.value, dtype=np.float64)

    # Step 5 — interpolate small gaps (≤10 cadences), mask large ones
    flux_series = pd.Series(flux)
    flux_interp = flux_series.interpolate(
        method='linear',
        limit=10,
        limit_direction='both'
    ).values
    print(f"  Interpolated small gaps")

    # Step 6 — normalize flux_err by median before final normalization
    median       = np.nanmedian(flux_interp)
    std          = np.nanstd(flux_interp)
    flux_norm    = (flux_interp - median) / std if std > 0 else flux_interp - median
    flux_err_norm = flux_err / median

    print(f"  Flux range: {np.nanmin(flux_norm):.4f} to {np.nanmax(flux_norm):.4f}")

    # Step 7 — save full light curve CSV with flux_err
    lc_df = pd.DataFrame({
        "time_BKJD": time,
        "flux_norm":  flux_norm,
        "flux_err":   flux_err_norm
    })
    lc_df.to_csv(
        os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv"),
        index=False
    )
    print(f"  Saved full light curve CSV")

    # Physics columns
    flux_out  = float(np.nanmedian(flux_norm))
    flux_in   = float(np.nanmin(flux_norm))
    depth_raw = float(flux_out - flux_in)

    period_days    = 289.86
    duration_hours = 7.4

    # Step 8 — window
    windows, centers = window_lightcurve(time, flux_norm)
    print(f"  Windowed: {len(windows)} windows of size {WINDOW_SIZE}")

    # Step 9 — save windows
    meta = save_windows(
        windows, centers, kic_id, label,
        period_days=period_days,
        duration_hours=duration_hours,
        depth_raw=depth_raw,
        flux_out=flux_out,
        flux_in=flux_in
    )
    return windows, centers, meta


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
        windows, centers, meta = preprocess_target(kic_id, lc_collection, label)

        print("\nSample meta:")
        print(meta.head())
        print(f"\nWindows array shape: {windows.shape}")
        print(f"Min flux value:      {windows.min():.4f}")
        print(f"Max flux value:      {windows.max():.4f}")
    else:
        print("No data found. Check your data/raw directory.")
