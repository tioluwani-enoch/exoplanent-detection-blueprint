"""
preprocess.py — Phase 2: Data Processing

KEY CHANGE FROM ORIGINAL:
    Instead of hardcoding period, duration, and t0 for each star,
    we now DERIVE them from the light curve using Box Least Squares (BLS).
    This means the pipeline can process a completely unknown star.

    The TARGETS dict now only holds:
        - name  (for labeling / display)
        - label (1 = confirmed planet, 0 = eclipsing binary — training only)

    Everything else (period, duration, t0) is computed from the data.

WHAT BLS DOES (simple explanation):
    Box Least Squares tries thousands of possible periods. For each one,
    it slides a "box" (a flat dip) across the phase-folded light curve
    and measures how well that box fits. The period that produces the
    deepest, best-fitting box wins. It also tells us:
        - The best-fit period (how often the dip repeats)
        - t0 (when the first dip happens)
        - Duration (how long each dip lasts)

PIPELINE FLOW:
    1. Download raw Kepler light curves
    2. Stitch quarters together, flatten trends, normalize, clip outliers
    3. Run BLS to find the best candidate period/t0/duration
    4. Window the light curve into overlapping chunks for ML
    5. Save everything to disk
"""

import numpy as np
import pandas as pd
import os
import lightkurve as lk
import warnings

warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Windowing parameters (unchanged from original)
# WINDOW_SIZE = 201 cadences ≈ 4.2 days at 30-min Kepler cadence
# STEP_SIZE = 50 cadences = overlap so we don't miss transits on window edges
WINDOW_SIZE = 201
STEP_SIZE   = 50

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS — now only name + label (for training). No hardcoded orbital params.
#
# label = 1 → confirmed exoplanet host (positive training example)
# label = 0 → eclipsing binary (negative training example — looks like a
#              planet transit but isn't)
#
# When you run this on a NEW unknown star, you don't need a TARGETS entry
# at all — just call preprocess_target(kic_id, lc_collection) and it will
# derive everything from the light curve.
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = {
    11446443: {
        "name":  "TrES-2b",
        "label": 1,
    },
    5780885: {
        "name":  "Kepler-7b",
        "label": 1,
    },
    11853905: {
        "name":  "Kepler-4b",
        "label": 1,
    },
    3544694: {
        "name":  "KIC 3544694 (EB)",
        "label": 0,
    },
    10619192: {
        "name":  "Kepler-17b",
        "label": 1,
    },
    10874614: {
        "name":  "Kepler-6b",
        "label": 1,
    },
    6922244: {
        "name":  "Kepler-8b",
        "label": 1,
    },
    6431596: {
        "name":  "KIC 6431596 (EB)",
        "label": 0,
    },
}


# ── 1. BLS PERIOD FINDING ────────────────────────────────────────────────────
#
# This is the new core function. It replaces the hardcoded period/duration/t0.
#
# How it works step by step:
#   1. We give BLS a range of trial periods to search (0.5 to 30 days covers
#      most hot Jupiters and short-period planets Kepler is best at finding).
#   2. For each trial period, BLS phase-folds the light curve and slides a
#      box-shaped model across it, measuring the signal strength.
#   3. The period with the strongest signal is our best candidate.
#   4. BLS also returns the transit epoch (t0) and duration at that period.
#   5. We return all three values so the rest of the pipeline can use them.
#
# The "power" value tells us how confident the detection is — higher = better.
# We also return it so features.py can use it as a feature if desired.
# ─────────────────────────────────────────────────────────────────────────────

def find_period_bls(lc_flat):
    """
    Run Box Least Squares on a flattened light curve to find the
    best-fit period, transit epoch (t0), and duration.

    WHY WE USE ASTROPY DIRECTLY INSTEAD OF LIGHTKURVE'S WRAPPER:
    lightkurve's to_periodogram(method='bls') calls astropy's
    BoxLeastSquares.autoperiod() under the hood, which generates a
    "conservative" period grid. For Kepler's ~1470-day baseline, this
    grid has tens of millions of trial periods — way too many to fit
    in memory. By using astropy's BLS directly, we can control exactly
    how many trial periods to use.

    We use 50,000 periods log-spaced from 0.5 to 30 days. Log-spacing
    means we sample short periods more finely (where precision matters
    most for transit detection) and long periods more coarsely. 50k
    trials is more than enough to find any real transit signal.

    Parameters
    ----------
    lc_flat : lightkurve.LightCurve
        A flattened, normalized light curve (trends already removed).

    Returns
    -------
    dict with keys:
        period_days    : float  — best-fit orbital period in days
        t0_bkjd        : float  — time of first transit (BKJD)
        duration_hours : float  — transit duration in hours
        bls_power      : float  — strength of the BLS detection (higher = better)
        bls_snr        : float  — signal-to-noise of the BLS peak
        bls_depth      : float  — depth of the best-fit box model

    Returns None if BLS fails or finds no significant signal.
    """
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    try:
        # Get time and flux as plain numpy arrays
        time = lc_flat.time.value
        flux = lc_flat.flux.value

        # Remove any NaNs — BLS can't handle them
        mask = np.isfinite(flux) & np.isfinite(time)
        time = time[mask]
        flux = flux[mask]

        if len(time) < 100:
            print(f"    Too few valid points ({len(time)}) for BLS")
            return None

        # Create the BLS model using astropy directly
        model = BoxLeastSquares(time * u.day, flux)

        # Build our own period grid: 50,000 periods from 0.5 to 30 days
        # Log-spaced so short periods (where most hot Jupiters live) get
        # finer sampling than long periods
        periods = np.logspace(np.log10(0.5), np.log10(30.0), 50000) * u.day

        # Transit durations to try: 0.05 to 0.3 days (1.2 to 7.2 hours)
        # This covers everything from short hot Jupiter transits to longer ones
        durations = np.linspace(0.05, 0.3, 10) * u.day

        print(f"    Searching {len(periods)} trial periods...")

        # Run BLS — this is the actual computation
        # For each trial period, BLS phase-folds the data and finds the
        # best-fitting box (transit) shape
        results = model.power(periods, durations)

        # Find the period with the highest BLS power
        best_idx      = np.argmax(results.power)
        best_period   = float(results.period[best_idx].value)     # days
        best_power    = float(results.power[best_idx])
        best_duration = float(results.duration[best_idx].value)   # days
        best_t0       = float(results.transit_time[best_idx].value)  # BKJD
        best_depth    = float(results.depth[best_idx])

        # Convert duration from days to hours
        duration_hours = best_duration * 24.0

        # Compute signal-to-noise: how many standard deviations the best
        # peak is above the mean of the power spectrum
        all_power  = results.power
        mean_power = float(np.nanmean(all_power))
        std_power  = float(np.nanstd(all_power))
        bls_snr    = (best_power - mean_power) / std_power if std_power > 0 else 0.0

        # Sanity checks: reject obviously bad results
        if best_period <= 0 or duration_hours <= 0 or duration_hours > best_period * 24:
            print(f"    BLS returned unphysical values — skipping")
            return None

        # If the BLS signal-to-noise is very low, there's probably no real
        # periodic signal in this light curve
        if bls_snr < 3.0:
            print(f"    BLS SNR too low ({bls_snr:.1f}) — weak/no periodic signal")
            # Still return results but flag it — the ML model can learn from this

        print(f"    BLS result: P={best_period:.5f}d, "
              f"t0={best_t0:.3f}, dur={duration_hours:.2f}h, "
              f"SNR={bls_snr:.1f}, depth={best_depth:.6f}")

        return {
            "period_days":    float(best_period),
            "t0_bkjd":        float(best_t0),
            "duration_hours": float(duration_hours),
            "bls_power":      float(best_power),
            "bls_snr":        float(bls_snr),
            "bls_depth":      float(best_depth),
        }

    except Exception as e:
        print(f"    BLS failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── 2. WINDOWING ──────────────────────────────────────────────────────────────
#
# This slides a window across the light curve and extracts chunks.
# Each chunk becomes one sample for the ML model.
#
# Two versions of each window are saved:
#   - windows_raw: the actual flux values (for visualization)
#   - windows_ml:  median-subtracted and divided by std (normalized for ML)
#
# We also compute flux_in (flux during transit) and flux_out (flux outside
# transit) for each window. The difference between these is the transit depth.
#
# This function is mostly unchanged from the original — the only difference
# is that period/duration/t0 now come from BLS instead of a lookup table.
# ─────────────────────────────────────────────────────────────────────────────

def window_lightcurve(time, flux, period_days, duration_hours, t0_bkjd,
                      window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Slide a window across the light curve and extract overlapping chunks.

    Parameters
    ----------
    time, flux     : arrays of time (BKJD) and normalized flux
    period_days    : orbital period (from BLS)
    duration_hours : transit duration (from BLS)
    t0_bkjd        : transit epoch (from BLS)
    window_size    : number of cadences per window (201 ≈ 4.2 days)
    step_size      : how far to slide between windows (50 = overlapping)

    Returns
    -------
    windows_raw  : (N, window_size) array of raw flux windows
    windows_ml   : (N, window_size) array of normalized windows for ML
    centers      : (N,) array of center times for each window
    flux_ins     : (N,) median flux inside transit for each window
    flux_outs    : (N,) median flux outside transit for each window
    """
    windows_raw = []
    windows_ml  = []
    centers     = []
    flux_ins    = []
    flux_outs   = []

    # Half the transit duration in days — defines the "in-transit" zone
    half_dur = (duration_hours / 24.0) / 2.0

    for start in range(0, len(flux) - window_size + 1, step_size):
        end      = start + window_size
        window   = np.array(flux[start:end],  dtype=np.float32)
        time_win = np.array(time[start:end],  dtype=np.float64)
        center_t = time_win[window_size // 2]

        # Skip windows that are >10% NaN (probably in a data gap)
        if np.isnan(window).sum() > window_size * 0.1:
            continue

        # Fill remaining NaNs with median (simple gap-filling)
        if np.isnan(window).any():
            wmed   = np.nanmedian(window)
            window = np.where(np.isnan(window), wmed, window)

        # Measure flux inside vs outside the expected transit window
        # "in_mask" = cadences within half a transit duration of center
        # "out_mask" = cadences within half a period but NOT in transit
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

        # Normalize for ML: subtract median, divide by std
        # This makes all windows comparable regardless of overall brightness
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


# ── 3. SAVE ───────────────────────────────────────────────────────────────────
#
# Saves windowed data + metadata to disk.
# Now includes BLS-derived parameters (period, duration, t0, SNR, power)
# so downstream code knows what BLS found.
# ─────────────────────────────────────────────────────────────────────────────

def save_windows(windows_raw, windows_ml, centers, kic_id, label,
                 flux_ins, flux_outs, bls_result):
    """
    Save windowed light curve data to disk.

    The metadata CSV now includes BLS-derived period, duration, t0, and
    signal quality metrics — no hardcoded values.
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
        # These now come from BLS — derived from the data, not looked up
        "period_days":    bls_result["period_days"],
        "duration_hours": bls_result["duration_hours"],
        "t0_bkjd":        bls_result["t0_bkjd"],
        "bls_power":      bls_result["bls_power"],
        "bls_snr":        bls_result["bls_snr"],
        "depth_raw":      depth_per_window,
        "flux_out":       flux_outs,
        "flux_in":        flux_ins,
    })
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    print(f"  Saved {len(windows_raw)} windows → {out_dir}")
    return meta


# ── 4. FULL PIPELINE FOR ONE TARGET ──────────────────────────────────────────
#
# This is the main function that processes a single star.
#
# WHAT CHANGED:
#   Old version: took period/duration/t0 from TARGETS dict (hardcoded)
#   New version: runs BLS on the light curve to derive them
#
# STEP BY STEP:
#   1. Stitch all quarters into one continuous light curve
#   2. Flatten (remove long-term trends from spacecraft drift)
#   3. Normalize (center flux at 1.0)
#   4. Clip upward outliers (cosmic rays, instrument glitches)
#   5. Run BLS to find the best period, t0, duration
#   6. Window the light curve into overlapping chunks
#   7. Save everything
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_target(kic_id, lc_collection, target_config=None):
    """
    Full preprocessing pipeline for one star.

    Parameters
    ----------
    kic_id         : int — Kepler Input Catalog ID
    lc_collection  : lightkurve.LightCurveCollection — downloaded quarter data
    target_config  : dict (optional) — only needs 'name' and 'label' for training.
                     If None, defaults to label=-1 (unknown) for inference on new stars.

    Returns
    -------
    windows_raw, windows_ml, centers, meta — or None if BLS finds nothing
    """
    # Default config for unknown stars (inference mode)
    if target_config is None:
        target_config = {"name": f"KIC {kic_id}", "label": -1}

    label = target_config.get("label", -1)  # -1 = unknown (not in training set)
    name  = target_config.get("name", f"KIC {kic_id}")

    print(f"\nPreprocessing KIC {kic_id} — {name}...")

    # ── Step 1: Stitch quarters ──
    # Kepler observed in ~90-day "quarters". Each quarter has slightly
    # different flux levels due to spacecraft rotation. Stitching
    # normalizes each quarter to 1.0 then joins them into one time series.
    lc_stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    print(f"  Stitched: {len(lc_stitched.flux)} cadences | "
          f"baseline {lc_stitched.time.value[0]:.1f} – "
          f"{lc_stitched.time.value[-1]:.1f} BKJD")

    # ── Step 2: Flatten ──
    # Removes long-period trends (stellar variability, instrument drift)
    # using a Savitzky-Golay-like filter. window_length=1001 means it
    # smooths over ~20 days — long enough to preserve transit dips but
    # short enough to remove slow trends.
    lc_flat, trend = lc_stitched.flatten(window_length=1001, return_trend=True)
    print(f"  Flattened: trend removed")

    # ── Step 3: Normalize ──
    # Divides all flux values by the median, centering everything at 1.0.
    # A transit dip will now show as flux < 1.0.
    lc_normalized = lc_flat.normalize()
    print(f"  Normalized: flux centered at 1.0")

    # ── Step 4: Clip upward outliers ──
    # Remove points that are too bright (cosmic ray hits, etc.)
    # We only clip UPWARD because downward dips might be real transits.
    median_flux  = np.nanmedian(lc_normalized.flux.value)
    std_flux     = np.nanstd(lc_normalized.flux.value)
    outlier_mask = lc_normalized.flux.value > median_flux + 2.5 * std_flux
    lc_clipped   = lc_normalized[~outlier_mask]
    print(f"  Clipped {int(outlier_mask.sum())} upward outliers")

    # ── Step 5: Run BLS to find period, t0, duration ──
    # THIS IS THE KEY NEW STEP. Instead of looking up known values,
    # we ask the data: "Is there a repeating dip? If so, what's its period?"
    print(f"  Running BLS period search...")
    bls_result = find_period_bls(lc_clipped)

    if bls_result is None:
        print(f"  BLS found no significant signal — skipping this target")
        return None

    period_days    = bls_result["period_days"]
    duration_hours = bls_result["duration_hours"]
    t0_bkjd        = bls_result["t0_bkjd"]

    # ── Prepare arrays for windowing ──
    time     = np.array(lc_clipped.time.value,     dtype=np.float64)
    flux     = np.array(lc_clipped.flux.value,     dtype=np.float64)
    flux_err = np.array(lc_clipped.flux_err.value, dtype=np.float64)

    print(f"  Flux range: {np.nanmin(flux):.6f} to {np.nanmax(flux):.6f}")

    # Interpolate small gaps (up to 10 consecutive NaNs)
    flux_series = pd.Series(flux)
    flux_interp = flux_series.interpolate(
        method='linear', limit=10, limit_direction='both'
    ).values

    median        = np.nanmedian(flux_interp)
    flux_err_norm = flux_err / median

    # Save the full processed light curve as CSV (for visualization later)
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

    # ── Step 6: Window the light curve ──
    # Slide a 201-cadence window across the time series, stepping by 50.
    # Each window becomes one sample for the ML model.
    windows_raw, windows_ml, centers, flux_ins, flux_outs = \
        window_lightcurve(time, flux_interp, period_days, duration_hours, t0_bkjd)

    if len(windows_raw) == 0:
        print(f"  No valid windows produced — skipping")
        return None

    n_transit = int((flux_outs - flux_ins > 0.005).sum())
    print(f"  Windowed: {len(windows_raw)} windows | "
          f"{n_transit} show possible dips (depth > 0.005)")

    # ── Step 7: Save ──
    meta = save_windows(
        windows_raw, windows_ml, centers, kic_id, label,
        flux_ins, flux_outs, bls_result
    )
    return windows_raw, windows_ml, centers, meta


# ── MAIN ──────────────────────────────────────────────────────────────────────
# Run on a single target for testing

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
        result = preprocess_target(kic_id, lc_collection, config)
        if result is not None:
            windows_raw, windows_ml, centers, meta = result
            print(f"\nwindows_raw shape: {windows_raw.shape}")
            print(f"Raw min/max: {windows_raw.min():.6f} / {windows_raw.max():.6f}")
        else:
            print("Preprocessing returned no results.")