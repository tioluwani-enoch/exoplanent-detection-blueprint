import numpy as np
import pandas as pd
import os
import lightkurve as lk

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE    = 201
STEP_SIZE      = 50
PERIOD_DAYS    = 2.47063
DURATION_HOURS = 1.7
T0_BKJD        = 120.595


# ── 1. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slice into overlapping fixed-size windows.
    Returns:
      windows_raw — fractional flux preserved (feature extraction)
      windows_ml  — per-window normalized (ML model input)
      centers     — center timestamp per window
      flux_ins    — median flux inside transit for each window
      flux_outs   — median flux outside transit for each window
    Windows with >10% NaN are skipped.
    """
    windows_raw = []
    windows_ml  = []
    centers     = []
    flux_ins    = []
    flux_outs   = []

    half_dur = (DURATION_HOURS / 24.0) / 2.0   # transit half-duration in days

    for start in range(0, len(flux) - window_size + 1, step_size):
        end      = start + window_size
        window   = np.array(flux[start:end],    dtype=np.float32)
        time_win = np.array(time[start:end],    dtype=np.float64)
        center_t = time_win[window_size // 2]

        # Skip windows with >10% NaN
        if np.isnan(window).sum() > window_size * 0.1:
            continue

        # Fill remaining NaNs with window median
        if np.isnan(window).any():
            wmed   = np.nanmedian(window)
            window = np.where(np.isnan(window), wmed, window)

        # Per-window flux_in / flux_out using known transit ephemeris
        in_mask  = (time_win >= center_t - half_dur) & \
                   (time_win <= center_t + half_dur)
        out_mask = (time_win >= center_t - PERIOD_DAYS / 2.0) & \
                   (time_win <= center_t + PERIOD_DAYS / 2.0) & \
                   ~in_mask

        flux_in_val  = float(np.nanmedian(window[in_mask]))  \
                       if in_mask.sum()  > 0 else float(np.nanmin(window))
        flux_out_val = float(np.nanmedian(window[out_mask])) \
                       if out_mask.sum() > 0 else float(np.nanmedian(window))

        flux_ins.append(flux_in_val)
        flux_outs.append(flux_out_val)

        # Raw window — fractional flux intact
        windows_raw.append(window.copy())

        # ML window — per-window normalized
        med = np.median(window)
        std = np.std(window)
        windows_ml.append((window - med) / std if std > 0 else window - med)

        centers.append(center_t)

    return (
        np.array(windows_raw, dtype=np.float32),
        np.array(windows_ml,  dtype=np.float32),
        np.array(centers,     dtype=np.float64),
        np.array(flux_ins,    dtype=np.float32),
        np.array(flux_outs,   dtype=np.float32),
    )


# ── 2. SAVE ───────────────────────────────────────────────────────────────────

def save_windows(windows_raw, windows_ml, centers, kic_id, label,
                 flux_ins, flux_outs):
    """
    Save both window arrays + metadata to disk.
      windows_raw.npy — fractional flux (feature extraction)
      windows_ml.npy  — normalized (ML model input)
      centers.npy     — center timestamp per window
      meta.csv        — per-window metadata with physics columns
    """
    out_dir = os.path.join(PROCESSED_DIR, f"KIC_{kic_id}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "windows_raw.npy"), windows_raw)
    np.save(os.path.join(out_dir, "windows_ml.npy"),  windows_ml)
    np.save(os.path.join(out_dir, "centers.npy"),     centers)

    depth_per_window = flux_outs - flux_ins

    meta = pd.DataFrame({
        "window_index":   range(len(windows_raw)),
        "center_time":    centers,
        "kic_id":         kic_id,
        "label":          label,
        "period_days":    PERIOD_DAYS,
        "duration_hours": DURATION_HOURS,
        "depth_raw":      depth_per_window,
        "flux_out":       flux_outs,
        "flux_in":        flux_ins,
    })
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)

    print(f"  Saved {len(windows_raw)} windows → {out_dir}")
    print(f"  windows_raw.npy — fractional flux (feature extraction)")
    print(f"  windows_ml.npy  — normalized (ML model input)")
    return meta


# ── 3. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, label=1):
    """
    Physics-approved pipeline:
      stitch → flatten (w=1001) → normalize → clip upward only
      → extract numpy → interpolate small gaps (≤10)
      → save CSV → window (raw + ml) → save
    """
    print(f"\nPreprocessing KIC {kic_id}...")

    # Step 1 — stitch with per-quarter normalization
    lc_stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    print(f"  Stitched: {len(lc_stitched.flux)} cadences | "
          f"baseline {lc_stitched.time.value[0]:.1f} – "
          f"{lc_stitched.time.value[-1]:.1f} BKJD")

    # Step 2 — flatten on native lightkurve object
    # window_length=1001 (~20 days) — longer than any transit, safe for TrES-2b
    lc_flat, trend = lc_stitched.flatten(window_length=1001, return_trend=True)
    print(f"  Flattened: trend removed")

    # Step 3 — normalize via lightkurve only (no manual divide)
    lc_normalized = lc_flat.normalize()
    print(f"  Normalized: flux centered at 1.0")

    # Step 4 — clip upward outliers ONLY
    # Planets block light → transits are always negative dips → never clip down
    median_flux  = np.nanmedian(lc_normalized.flux.value)
    std_flux     = np.nanstd(lc_normalized.flux.value)
    outlier_mask = lc_normalized.flux.value > median_flux + 2.5 * std_flux
    lc_clipped   = lc_normalized[~outlier_mask]
    print(f"  Clipped {int(outlier_mask.sum())} upward outliers (transits preserved)")

    # Step 5 — extract numpy arrays (all lightkurve ops done)
    time     = np.array(lc_clipped.time.value,     dtype=np.float64)
    flux     = np.array(lc_clipped.flux.value,     dtype=np.float64)
    flux_err = np.array(lc_clipped.flux_err.value, dtype=np.float64)

    print(f"  Flux range after clip: "
          f"{np.nanmin(flux):.6f} to {np.nanmax(flux):.6f}")

    # Step 6 — interpolate small gaps (≤10 cadences), mask large ones
    flux_series = pd.Series(flux)
    flux_interp = flux_series.interpolate(
        method='linear',
        limit=10,
        limit_direction='both'
    ).values

    median        = np.nanmedian(flux_interp)
    flux_err_norm = flux_err / median

    print(f"  Flux range after interpolate: "
          f"{np.nanmin(flux_interp):.6f} to {np.nanmax(flux_interp):.6f}")

    # Step 7 — save full light curve CSV
    lc_df = pd.DataFrame({
        "time_BKJD": time,
        "flux_norm":  flux_interp,
        "flux_err":   flux_err_norm,
    })
    lc_df.to_csv(
        os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv"),
        index=False
    )
    print(f"  Saved full light curve CSV")

    # Step 8 — window
    windows_raw, windows_ml, centers, flux_ins, flux_outs = \
        window_lightcurve(time, flux_interp)

    print(f"  Windowed: {len(windows_raw)} windows of size {WINDOW_SIZE}")
    print(f"  Raw flux range:      {windows_raw.min():.6f} to {windows_raw.max():.6f}")

    n_transit = int((flux_outs - flux_ins > 0.005).sum())
    print(f"  Transit windows detected: {n_transit} (depth > 0.5%)")

    # Step 9 — save
    meta = save_windows(
        windows_raw, windows_ml, centers, kic_id, label,
        flux_ins, flux_outs
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
        print(meta.head(10))
        print(f"\nwindows_raw shape: {windows_raw.shape}")
        print(f"windows_ml shape:  {windows_ml.shape}")
        print(f"Raw min/max:       {windows_raw.min():.6f} / {windows_raw.max():.6f}")
        print(f"ML  min/max:       {windows_ml.min():.4f} / {windows_ml.max():.4f}")
    else:
        print("No data found. Check your data/raw directory.")
