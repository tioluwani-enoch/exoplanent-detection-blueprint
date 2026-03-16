import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 201   # cadences per window (~100 hrs at 30-min cadence)
STEP_SIZE   = 50    # stride between windows


# ── 1. SIGMA CLIPPING ─────────────────────────────────────────────────────────

def clean_flux(flux):
    """
    Sigma clip at 3-sigma to remove cosmic rays and spacecraft artifacts.
    Physics note: clips outliers before detrending so the Savitzky-Golay
    baseline isn't pulled by spikes.
    """
    flux = np.array(flux, dtype=np.float64)
    clipped = sigma_clip(flux, sigma=3, maxiters=5, masked=True)
    flux[clipped.mask] = np.nan
    return flux


# ── 2. DETRENDING ─────────────────────────────────────────────────────────────

def detrend_flux(flux, window_length=101, polyorder=2):
    """
    Remove long-term stellar variability using a Savitzky-Golay filter.
    Physics note: dividing out the trend preserves fractional transit depth.
    window_length=101 cadences ~ 50 hrs, safely longer than Kepler-22b's
    7.4 hr transit so the transit shape is not smoothed away.
    """
    flux = np.array(flux, dtype=np.float64)
    mask = np.isfinite(flux)

    if mask.sum() < window_length:
        return flux

    # Interpolate over NaNs before filtering
    flux_filled = flux.copy()
    flux_filled[~mask] = np.interp(
        np.where(~mask)[0],
        np.where(mask)[0],
        flux[mask]
    )

    trend = savgol_filter(flux_filled, window_length=window_length, polyorder=polyorder)
    detrended = flux / trend
    detrended[~mask] = np.nan
    return detrended


# ── 3. NORMALIZATION ──────────────────────────────────────────────────────────

def normalize_flux(flux):
    """
    Normalize to zero median, unit std deviation.
    Physics note: median used instead of mean so rare transit dips
    don't shift the out-of-transit baseline.
    """
    flux = np.array(flux, dtype=np.float32)
    median = np.nanmedian(flux)
    std    = np.nanstd(flux)
    if std == 0:
        return flux - median
    return (flux - median) / std


# ── 4. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slice the light curve into overlapping fixed-size windows.
    Each window is independently normalized so the model sees
    local flux shape, not absolute quarter-to-quarter offsets.
    """
    windows, centers = [], []

    for start in range(0, len(flux) - window_size + 1, step_size):
        end    = start + window_size
        window = flux[start:end]

        # Skip windows with more than 10% missing data
        if np.isnan(window).sum() > window_size * 0.1:
            continue

        window = np.nan_to_num(window, nan=0.0)

        # Normalize each window independently
        median = np.median(window)
        std    = np.std(window)
        if std > 0:
            window = (window - median) / std

        windows.append(window)
        centers.append(time[start + window_size // 2])

    return np.array(windows, dtype=np.float32), np.array(centers)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────

def save_windows(windows, centers, kic_id, label,
                 period_days=None, duration_hours=None,
                 depth_raw=None, flux_out=None, flux_in=None):
    """
    Save windowed dataset to disk.
      windows.npy — shape (n_windows, window_size)
      centers.npy — center timestamp per window
      meta.csv    — per-window metadata with physics columns
    """
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

    print(f"Saved {len(windows)} windows for KIC {kic_id} → {out_dir}")
    return meta


# ── 6. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, label=1):
    """
    Full preprocessing pipeline for one target:
      stitch (per-quarter normalize) → sigma clip → detrend → normalize → window → save
    """
    print(f"\nPreprocessing KIC {kic_id}...")

    # Normalize each quarter to the same baseline before stitching
    # This fixes quarter-to-quarter flux offsets before anything else runs
    stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    time = stitched.time.value
    flux = stitched.flux.value

    print(f"  Stitched: {len(flux)} cadences | "
          f"baseline {time[0]:.1f} – {time[-1]:.1f} BKJD")

    # Sigma clip at 3-sigma to remove cosmic rays and artifacts
    flux_cleaned = clean_flux(flux)
    n_clipped = int(np.isnan(flux_cleaned).sum() - np.isnan(flux).sum())
    print(f"  Sigma clipped: {n_clipped} outliers removed")

    # Detrend stellar variability
    flux_detrended = detrend_flux(flux_cleaned)
    print(f"  Detrended: {np.isnan(flux_detrended).sum()} NaNs remaining")

    # Global normalize
    flux_normalized = normalize_flux(flux_detrended)
    print(f"  Flux range after normalize: "
          f"{np.nanmin(flux_normalized):.4f} to {np.nanmax(flux_normalized):.4f}")

    # Compute physics columns for CSV
    flux_out  = float(np.nanmedian(flux_normalized))
    flux_in   = float(np.nanmin(flux_normalized))
    depth_raw = float(flux_out - flux_in)

    # Kepler-22b known values (features.py will compute these dynamically)
    period_days    = 289.86
    duration_hours = 7.4

    # Window
    windows, centers = window_lightcurve(time, flux_normalized)
    print(f"  Windowed: {len(windows)} windows of size {WINDOW_SIZE}")

    # Save
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
    import lightkurve as lk
    sys.path.insert(0, os.path.dirname(__file__))

    kic_id = 11446443
    label  = 1  # confirmed planet host

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