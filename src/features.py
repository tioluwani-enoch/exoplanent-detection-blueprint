import numpy as np
import pandas as pd
import os

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHTCURVE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "KIC_11446443_lightcurve.csv")
PARAMS_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "meta.csv")
WINDOWS_ML     = os.path.join(PROJECT_ROOT, "data", "processed", "windows_ml.npy")
OUTPUT_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "KIC_11446443_features.csv")

PERIOD_DAYS    = 2.47063
DURATION_HOURS = 1.7
T0_BKJD        = 120.595
RANDOM_SEED    = 42


def compute_features(period, duration, depth_raw, flux_out, flux_in, stellar_radius):
    """
    Transform raw transit parameters into physically grounded ML features.
    Returns None if values fall outside physically valid ranges.
    Physics filter — only applied to POSITIVE candidates, never to negatives.
    """
    norm_depth       = (flux_out - flux_in) / flux_out
    dur_period_ratio = duration / period
    radius_ratio     = norm_depth ** 0.5 if norm_depth > 0 else 0.0

    if not (0.001 < norm_depth < 0.1):
        return None
    if not (0.5 < period < 500):
        return None
    if not (0.001 < dur_period_ratio < 0.1):
        return None

    return {
        "norm_depth":       norm_depth,
        "dur_period_ratio": dur_period_ratio,
        "radius_ratio":     radius_ratio,
    }


def compute_flux_in_out(lc_df, center_time, duration_hours, period_days):
    """
    Compute flux_in and flux_out for a given window from the light curve.
    """
    half_duration = (duration_hours / 24) / 2

    in_transit = lc_df[
        (lc_df["time_BKJD"] >= center_time - half_duration) &
        (lc_df["time_BKJD"] <= center_time + half_duration)
    ]

    half_window = period_days / 2
    out_transit = lc_df[
        (lc_df["time_BKJD"] >= center_time - half_window) &
        (lc_df["time_BKJD"] <= center_time + half_window) &
        ~(
            (lc_df["time_BKJD"] >= center_time - half_duration) &
            (lc_df["time_BKJD"] <= center_time + half_duration)
        )
    ]

    if len(in_transit) == 0 or len(out_transit) == 0:
        return None, None

    flux_in  = np.nanmedian(in_transit["flux_norm"].values)
    flux_out = np.nanmedian(out_transit["flux_norm"].values)
    return flux_in, flux_out


def sample_negatives(meta_df, n_samples, positive_centers):
    """
    Sample genuine non-transit windows from meta.csv.
    Physics-approved rules:
      - Center times must fall well outside ±(duration/2) of any known transit
      - Randomly sampled with fixed seed for reproducibility
      - Physics filter is NOT applied — negatives don't need transit validation
    """
    half_dur_days = (DURATION_HOURS / 24.0) / 2.0

    # Build all known transit centers from ephemeris
    t_start = meta_df["center_time"].min()
    t_end   = meta_df["center_time"].max()
    n_transits = int((t_end - t_start) / PERIOD_DAYS) + 2
    transit_centers = np.array([
        T0_BKJD + i * PERIOD_DAYS
        for i in range(-5, n_transits + 5)
    ])

    def is_near_transit(t):
        # Exclude windows within 2x the transit half-duration of any transit center
        # 2x gives a clean buffer without excluding the entire light curve
        return np.any(np.abs(t - transit_centers) < half_dur_days * 2)

    # Filter to genuine non-transit windows
    non_transit_mask = meta_df["center_time"].apply(
        lambda t: not is_near_transit(t)
    )
    non_transit_df = meta_df[non_transit_mask].reset_index(drop=True)

    print(f"  Non-transit windows available: {len(non_transit_df)}")

    # Sample reproducibly
    rng         = np.random.default_rng(RANDOM_SEED)
    sample_idx  = rng.choice(len(non_transit_df), size=min(n_samples, len(non_transit_df)), replace=False)
    sampled     = non_transit_df.iloc[sample_idx].reset_index(drop=True)

    return sampled


def build_feature_dataset(params_csv, lightcurve_csv, output_csv):
    """
    1. Extract positive samples using physics filter
    2. Sample equal number of negatives from non-transit windows
    3. Combine into balanced training dataset
    4. Save to CSV
    """
    df    = pd.read_csv(params_csv)
    lc_df = pd.read_csv(lightcurve_csv)
    print(f"Loaded {len(df)} windows from {params_csv}")

    # ── Positive samples ──────────────────────────────────────────
    records  = []
    rejected = 0

    for _, row in df.iterrows():
        flux_in, flux_out = compute_flux_in_out(
            lc_df,
            center_time    = row["center_time"],
            duration_hours = row["duration_hours"],
            period_days    = row["period_days"]
        )

        if flux_in is None or flux_out is None:
            rejected += 1
            continue

        features = compute_features(
            period         = row["period_days"],
            duration       = row["duration_hours"] / 24,
            depth_raw      = row["depth_raw"],
            flux_out       = flux_out,
            flux_in        = flux_in,
            stellar_radius = row.get("stellar_radius_rs", np.nan),
        )

        if features is None:
            rejected += 1
            continue

        records.append({
            "kic_id":            row["kic_id"],
            "window_index":      row["window_index"],
            "center_time":       row["center_time"],
            "label":             1,
            "stellar_radius_rs": row.get("stellar_radius_rs", np.nan),
            "period_days":       row["period_days"],
            "duration_hours":    row["duration_hours"],
            "flux_out":          flux_out,
            "flux_in":           flux_in,
            "norm_depth":        features["norm_depth"],
            "dur_period_ratio":  features["dur_period_ratio"],
            "radius_ratio":      features["radius_ratio"],
        })

    positives_df = pd.DataFrame(records)
    n_positives  = len(positives_df)
    print(f"\n  Positive samples (transit):     {n_positives}")
    print(f"  Rejected by physics filter:     {rejected}")

    # ── Negative samples ──────────────────────────────────────────
    # Sample equal number of non-transit windows
    # Physics filter NOT applied — negatives just need to be non-transit
    neg_windows = sample_negatives(
        df,
        n_samples        = n_positives,
        positive_centers = positives_df["center_time"].values
    )

    neg_records = []
    for _, row in neg_windows.iterrows():
        flux_in, flux_out = compute_flux_in_out(
            lc_df,
            center_time    = row["center_time"],
            duration_hours = DURATION_HOURS,
            period_days    = PERIOD_DAYS
        )

        if flux_in is None or flux_out is None:
            continue

        norm_depth       = (flux_out - flux_in) / flux_out if flux_out != 0 else 0.0
        dur_period_ratio = (DURATION_HOURS / 24.0) / PERIOD_DAYS
        radius_ratio     = norm_depth ** 0.5 if norm_depth > 0 else 0.0

        neg_records.append({
            "kic_id":            row["kic_id"],
            "window_index":      row["window_index"],
            "center_time":       row["center_time"],
            "label":             0,
            "stellar_radius_rs": row.get("stellar_radius_rs", np.nan),
            "period_days":       PERIOD_DAYS,
            "duration_hours":    DURATION_HOURS,
            "flux_out":          flux_out,
            "flux_in":           flux_in,
            "norm_depth":        norm_depth,
            "dur_period_ratio":  dur_period_ratio,
            "radius_ratio":      radius_ratio,
        })

    negatives_df = pd.DataFrame(neg_records)
    print(f"  Negative samples (no transit):  {len(negatives_df)}")

    # ── Combine + save ────────────────────────────────────────────
    combined_df = pd.concat([positives_df, negatives_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    print(f"\nFeature extraction complete:")
    print(f"  Total samples:   {len(combined_df)}")
    print(f"  Positives:       {int(combined_df['label'].sum())}")
    print(f"  Negatives:       {int((combined_df['label'] == 0).sum())}")
    print(f"  Output saved to: {output_csv}")
    print(f"\nSample output:")
    print(combined_df[["kic_id", "label", "norm_depth",
                        "dur_period_ratio", "radius_ratio"]].head(10).to_string())

    return combined_df


if __name__ == "__main__":
    build_feature_dataset(PARAMS_CSV, LIGHTCURVE_CSV, OUTPUT_CSV)
