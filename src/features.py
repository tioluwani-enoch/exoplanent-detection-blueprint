import numpy as np
import pandas as pd
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHTCURVE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "KIC_11446443_lightcurve.csv")
PARAMS_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "meta.csv")
OUTPUT_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "KIC_11446443_features.csv")


def compute_features(period, duration, depth_raw, flux_out, flux_in, stellar_radius):
    """
    Transform raw transit parameters into physically grounded ML features.
    Returns None if values fall outside physically valid ranges.
    """
    norm_depth       = (flux_out - flux_in) / flux_out
    dur_period_ratio = duration / period
    radius_ratio     = norm_depth ** 0.5

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
    Compute flux_in and flux_out for a given window directly from the light curve.
    """
    half_duration = (duration_hours / 24) / 2  # in days

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


def build_feature_dataset(params_csv: str, lightcurve_csv: str, output_csv: str):
    """
    Reads the windows metadata CSV, runs compute_features() on each row,
    filters out physically invalid entries, and saves the feature dataset.
    """
    df    = pd.read_csv(params_csv)
    lc_df = pd.read_csv(lightcurve_csv)
    print(f"Loaded {len(df)} windows from {params_csv}")

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
            "label":             row["label"],
            "stellar_radius_rs": row.get("stellar_radius_rs", np.nan),
            "period_days":       row["period_days"],
            "duration_hours":    row["duration_hours"],
            "flux_out":          flux_out,
            "flux_in":           flux_in,
            "norm_depth":        features["norm_depth"],
            "dur_period_ratio":  features["dur_period_ratio"],
            "radius_ratio":      features["radius_ratio"],
        })

    features_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    features_df.to_csv(output_csv, index=False)

    print(f"\nFeature extraction complete:")
    print(f"  Total windows:   {len(df)}")
    print(f"  Passed filters:  {len(features_df)}")
    print(f"  Rejected:        {rejected}")
    print(f"  Output saved to: {output_csv}")
    print(f"\nSample output:")
    print(features_df[["kic_id", "label", "norm_depth", "dur_period_ratio", "radius_ratio"]].head(10).to_string())

    return features_df


if __name__ == "__main__":
    build_feature_dataset(PARAMS_CSV, LIGHTCURVE_CSV, OUTPUT_CSV)