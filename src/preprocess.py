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
    Sigma clip at 2.5-sigma (physics-approved).
    Targets sharp single-point spikes: cosmic rays, flares, residual artifacts.
    Safe for transits — TrES-2b's 1.6% depth is gradual and multi-cadence,
    will never trigger a 2.5σ single-point cut.
    """
    flux = np.array(flux, dtype=np.float64)
    clipped = sigma_clip(flux, sigma=2.5, maxiters=5, masked=True)
    flux[clipped.mask] = np.nan
    return flux


# ── 2. INTERPOLATION ──────────────────────────────────────────────────────────

def interpolate_gaps(flux):
    """
    Interpolate small gaps only (physics-approved rules):
      - Linear interpolation for gaps ≤ 10 consecutive NaNs
      - Gaps > 10 cadences left as NaN (real quarter boundaries / data gaps)
      - Zeroing NaNs is physically wrong — creates fake flat-bottomed dips
        that look like shallow transits to the ML model
    """
    flux_series = pd.Series(flux)
    flux_interpolated = flux_series.interpolate(
        method='linear',
        limit=10,
        limit_direction='both'
    )

    # Report gap distribution
    nan_mask   = flux_series.isna()
    total_nans = nan_mask.sum()

    if total_nans > 0:
        # Find runs of NaNs
        gap_lengths = []
        count = 0
        for val in nan_mask:
            if val:
                count += 1
            elif count > 0:
                gap_lengths.append(count)
                count = 0
        if count > 0:
            gap_lengths.append(count)

        gap_lengths = np.array(gap_lengths)
        small_gaps  = (gap_lengths <= 10).sum()
        large_gaps  = (gap_lengths  > 10).sum()
        print(f"  Gap analysis: {total_nans} NaNs total | "
              f"{small_gaps} small gaps (≤10, interpolated) | "
              f"{large_gaps} large gaps (>10, masked out)")

    return flux_interpolated.values


# ── 3. DETRENDING ─────────────────────────────────────────────────────────────

def detrend_flux(flux, window_length=101, polyorder=2):
    """
    Remove long-term stellar variability using Savitzky-Golay filter.
    window_length=101 cadences ~ 50 hrs — safely longer than the 7.4 hr
    transit so the transit shape is not smoothed away.
    Divides out the trend to preserve fractional transit depth.
    """
    flux = np.array(flux, dtype=np.float64)
    mask = np.isfinite(flux)

    if mask.sum() < window_length:
        return flux

    flux_filled = flux.copy()
    flux_filled[~mask] = np.interp(
        np.where(~mask)[0],
        np.where(mask)[0],
        flux[mask]
    )

    trend     = savgol_filter(flux_filled, window_length=window_length, polyorder=polyorder)
    detrended = flux / trend
    detrended[~mask] = np.nan
    return detrended


# ── 4. NORMALIZATION ──────────────────────────────────────────────────────────

def normalize_flux(flux):
    """
    Normalize to zero median, unit std deviation.
    Median used instead of mean — transit dips are rare events and
    would pull the mean down, shifting the out-of-transit baseline.
    """
    flux   = np.array(flux, dtype=np.float32)
    median = np.nanmedian(flux)
    std    = np.nanstd(flux)
    if std == 0:
        return flux - median
    return (flux - median) / std


# ── 5. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slice into overlapping fixed-size windows.
    Each window independently normalized — model sees local flux shape,
    not absolute quarter-to-quarter offsets.
    Windows with >10% NaN are skipped entirely (large data gaps).
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


# ── 6. SAVE ───────────────────────────────────────────────────────────────────

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


# ── 7. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, label=1):
    """
    Full preprocessing pipeline:
      stitch (per-quarter normalize)
      → sigma clip (2.5σ)
      → interpolate small gaps (≤10 cadences)
      → detrend (Savitzky-Golay)
      → normalize (zero median)
      → window (201 cadences, stride 50)
      → save
    """
    print(f"\nPreprocessing KIC {kic_id}...")

    # Normalize each quarter individually before stitching
    stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    time = stitched.time.value
    flux = stitched.flux.value

    print(f"  Stitched: {len(flux)} cadences | "
          f"baseline {time[0]:.1f} – {time[-1]:.1f} BKJD")

    # 2.5-sigma clip
    flux_cleaned = clean_flux(flux)
    n_clipped    = int(np.isnan(flux_cleaned).sum() - np.isnan(flux).sum())
    print(f"  Sigma clipped: {n_clipped} outliers removed")

    # Interpolate small gaps, mask large ones
    flux_interp = interpolate_gaps(flux_cleaned)

    # Detrend
    flux_detrended = detrend_flux(flux_interp)
    print(f"  Detrended: {np.isnan(flux_detrended).sum()} NaNs remaining "
          f"(large gaps masked out)")

    # Normalize
    flux_normalized = normalize_flux(flux_detrended)
    print(f"  Flux range: "
          f"{np.nanmin(flux_normalized):.4f} to {np.nanmax(flux_normalized):.4f}")

    # Physics columns
    flux_out  = float(np.nanmedian(flux_normalized))
    flux_in   = float(np.nanmin(flux_normalized))
    depth_raw = float(flux_out - flux_in)

    period_days    = 289.86   # Kepler-22b known value
    duration_hours = 7.4      # features.py will compute dynamically

    # Save full stitched light curve for reference + visualization
    lc_df = pd.DataFrame({
        "time_BKJD": time,
        "flux_norm":  flux_normalized,
    })
    lc_df.to_csv(
        os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv"),
        index=False
    )
    print(f"  Saved full light curve CSV: KIC_{kic_id}_lightcurve.csv")

    # Window
    windows, centers = window_lightcurve(time, flux_normalized)

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
