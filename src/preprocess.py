import numpy as np
import pandas as pd
import os
import lightkurve as lk

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 201
STEP_SIZE   = 50

TARGETS = {
    11446443: {
        "name":           "TrES-2b",
        "period_days":    2.47063,
        "duration_hours": 1.7,
        "t0_bkjd":        120.595,
        "label":          1,
    },
    5780885: {
        "name":           "Kepler-7b",
        "period_days":    4.885525,
        "duration_hours": 4.5,
        "t0_bkjd":        134.179,
        "label":          1,
    },
    11853905: {
        "name":           "Kepler-4b",
        "period_days":    3.21346,
        "duration_hours": 3.95,
        "t0_bkjd":        123.613,
        "label":          1,
    },
    3544694: {
        "name":           "KIC 3544694 (EB)",
        "period_days":    3.8457,
        "duration_hours": 2.1,
        "t0_bkjd":        667.427,
        "label":          0,
    },
    10619192: {
        "name":           "Kepler-17b",
        "period_days":    1.4857,
        "duration_hours": 2.067,
        "t0_bkjd":        352.678,
        "label":          1,
    },
    10874614: {
        "name":           "Kepler-6b",
        "period_days":    3.2347,
        "duration_hours": 3.23,
        "t0_bkjd":        121.627,
        "label":          1,
    },
    6922244: {
        "name":           "Kepler-8b",
        "period_days":    3.5225,
        "duration_hours": 3.28,
        "t0_bkjd":        121.489,
        "label":          1,
    },
    6431596: {
        "name":           "KIC 6431596 (EB)",
        "period_days":    3.8457,
        "duration_hours": 2.1,
        "t0_bkjd":        334.427,
        "label":          0,
    },
}


# ── 1. WINDOWING ──────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, period_days, duration_hours, t0_bkjd,
                      window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    windows_raw = []
    windows_ml  = []
    centers     = []
    flux_ins    = []
    flux_outs   = []

    half_dur = (duration_hours / 24.0) / 2.0

    for start in range(0, len(flux) - window_size + 1, step_size):
        end      = start + window_size
        window   = np.array(flux[start:end],  dtype=np.float32)
        time_win = np.array(time[start:end],  dtype=np.float64)
        center_t = time_win[window_size // 2]

        if np.isnan(window).sum() > window_size * 0.1:
            continue

        if np.isnan(window).any():
            wmed   = np.nanmedian(window)
            window = np.where(np.isnan(window), wmed, window)

        in_mask  = (time_win >= center_t - half_dur) & \
                   (time_win <= center_t + half_dur)
        out_mask = (time_win >= center_t - period_days / 2.0) & \
                   (time_win <= center_t + period_days / 2.0) & \
                   ~in_mask

        flux_in_val  = float(np.nanmedian(window[in_mask]))  \
                       if in_mask.sum()  > 0 else float(np.nanmin(window))
        flux_out_val = float(np.nanmedian(window[out_mask])) \
                       if out_mask.sum() > 0 else float(np.nanmedian(window))

        flux_ins.append(flux_in_val)
        flux_outs.append(flux_out_val)
        windows_raw.append(window.copy())

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
                 flux_ins, flux_outs, period_days, duration_hours):
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
        "period_days":    period_days,
        "duration_hours": duration_hours,
        "depth_raw":      depth_per_window,
        "flux_out":       flux_outs,
        "flux_in":        flux_ins,
    })
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    print(f"  Saved {len(windows_raw)} windows → {out_dir}")
    return meta


# ── 3. FULL PIPELINE ──────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, target_config=None):
    if target_config is None:
        target_config = TARGETS[kic_id]

    label          = target_config["label"]
    period_days    = target_config["period_days"]
    duration_hours = target_config["duration_hours"]
    t0_bkjd        = target_config["t0_bkjd"]

    print(f"\nPreprocessing KIC {kic_id} — {target_config['name']}...")

    lc_stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    print(f"  Stitched: {len(lc_stitched.flux)} cadences | "
          f"baseline {lc_stitched.time.value[0]:.1f} – "
          f"{lc_stitched.time.value[-1]:.1f} BKJD")

    lc_flat, trend = lc_stitched.flatten(window_length=1001, return_trend=True)
    print(f"  Flattened: trend removed")

    lc_normalized = lc_flat.normalize()
    print(f"  Normalized: flux centered at 1.0")

    median_flux  = np.nanmedian(lc_normalized.flux.value)
    std_flux     = np.nanstd(lc_normalized.flux.value)
    outlier_mask = lc_normalized.flux.value > median_flux + 2.5 * std_flux
    lc_clipped   = lc_normalized[~outlier_mask]
    print(f"  Clipped {int(outlier_mask.sum())} upward outliers")

    time     = np.array(lc_clipped.time.value,     dtype=np.float64)
    flux     = np.array(lc_clipped.flux.value,     dtype=np.float64)
    flux_err = np.array(lc_clipped.flux_err.value, dtype=np.float64)

    print(f"  Flux range: {np.nanmin(flux):.6f} to {np.nanmax(flux):.6f}")

    flux_series = pd.Series(flux)
    flux_interp = flux_series.interpolate(
        method='linear', limit=10, limit_direction='both'
    ).values

    median        = np.nanmedian(flux_interp)
    flux_err_norm = flux_err / median

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

    windows_raw, windows_ml, centers, flux_ins, flux_outs = \
        window_lightcurve(time, flux_interp, period_days, duration_hours, t0_bkjd)

    n_transit = int((flux_outs - flux_ins > 0.005).sum())
    print(f"  Windowed: {len(windows_raw)} windows | "
          f"{n_transit} transit windows detected")

    meta = save_windows(
        windows_raw, windows_ml, centers, kic_id, label,
        flux_ins, flux_outs, period_days, duration_hours
    )
    return windows_raw, windows_ml, centers, meta


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    kic_id = 11446443
    config = TARGETS[kic_id]

    print("Searching for local or cached light curves...")
    search_result = lk.search_lightcurve(
        f"KIC {kic_id}", mission="Kepler", author="Kepler", exptime=1800
    )
    lc_collection = search_result.download_all(
        download_dir=os.path.join(PROJECT_ROOT, "data", "raw")
    )

    if lc_collection is not None:
        windows_raw, windows_ml, centers, meta = preprocess_target(
            kic_id, lc_collection, config
        )
        print(f"\nwindows_raw shape: {windows_raw.shape}")
        print(f"Raw min/max: {windows_raw.min():.6f} / {windows_raw.max():.6f}")