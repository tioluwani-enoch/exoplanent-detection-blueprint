import numpy as np
import pandas as pd
import os

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
LIGHTCURVE_DIR = PROCESSED_DIR
OUTPUT_CSV     = os.path.join(PROCESSED_DIR, "combined_features.csv")

RANDOM_SEED    = 42
NORM_DEPTH_MIN = 0.0005   # lowered from 0.001 — physics-approved


def compute_features(period, duration, flux_out, flux_in):
    norm_depth       = (flux_out - flux_in) / flux_out if flux_out != 0 else 0.0
    dur_period_ratio = duration / period
    radius_ratio     = norm_depth ** 0.5 if norm_depth > 0 else 0.0

    if not (NORM_DEPTH_MIN < norm_depth < 0.1):
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
    half_duration = (duration_hours / 24) / 2
    half_window   = period_days / 2

    in_transit = lc_df[
        (lc_df["time_BKJD"] >= center_time - half_duration) &
        (lc_df["time_BKJD"] <= center_time + half_duration)
    ]
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

    return (
        np.nanmedian(in_transit["flux_norm"].values),
        np.nanmedian(out_transit["flux_norm"].values)
    )


def sample_negatives(meta_df, n_samples, period_days, duration_hours, t0_bkjd):
    half_dur_days = (duration_hours / 24.0) / 2.0
    t_start       = meta_df["center_time"].min()
    t_end         = meta_df["center_time"].max()
    n_transits    = int((t_end - t_start) / period_days) + 2

    transit_centers = np.array([
        t0_bkjd + i * period_days
        for i in range(-5, n_transits + 5)
    ])

    def is_near_transit(t):
        return np.any(np.abs(t - transit_centers) < half_dur_days * 2)

    non_transit_mask = meta_df["center_time"].apply(lambda t: not is_near_transit(t))
    non_transit_df   = meta_df[non_transit_mask].reset_index(drop=True)

    rng        = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(
        len(non_transit_df),
        size=min(n_samples, len(non_transit_df)),
        replace=False
    )
    return non_transit_df.iloc[sample_idx].reset_index(drop=True)


def process_one_target(kic_id, meta_df, lc_df):
    period_days    = meta_df["period_days"].iloc[0]
    duration_hours = meta_df["duration_hours"].iloc[0]
    t0_bkjd        = meta_df["center_time"].min()
    label          = int(meta_df["label"].iloc[0])

    records  = []
    rejected = 0

    for _, row in meta_df.iterrows():
        flux_in, flux_out = compute_flux_in_out(
            lc_df, row["center_time"], duration_hours, period_days
        )
        if flux_in is None:
            rejected += 1
            continue

        features = compute_features(period_days, duration_hours / 24,
                                    flux_out, flux_in)
        if features is None:
            rejected += 1
            continue

        records.append({
            "kic_id":            kic_id,
            "window_index":      row["window_index"],
            "center_time":       row["center_time"],
            "label":             1,
            "period_days":       period_days,
            "duration_hours":    duration_hours,
            "flux_out":          flux_out,
            "flux_in":           flux_in,
            "norm_depth":        features["norm_depth"],
            "dur_period_ratio":  features["dur_period_ratio"],
            "radius_ratio":      features["radius_ratio"],
        })

    positives_df = pd.DataFrame(records)
    n_pos        = len(positives_df)

    # Sample negatives — physics filter NOT applied
    neg_windows = sample_negatives(
        meta_df, n_pos, period_days, duration_hours, t0_bkjd
    )

    neg_records = []
    for _, row in neg_windows.iterrows():
        flux_in, flux_out = compute_flux_in_out(
            lc_df, row["center_time"], duration_hours, period_days
        )
        if flux_in is None:
            continue

        nd  = (flux_out - flux_in) / flux_out if flux_out != 0 else 0.0
        dpr = (duration_hours / 24.0) / period_days
        rr  = nd ** 0.5 if nd > 0 else 0.0

        neg_records.append({
            "kic_id":            kic_id,
            "window_index":      row["window_index"],
            "center_time":       row["center_time"],
            "label":             0,
            "period_days":       period_days,
            "duration_hours":    duration_hours,
            "flux_out":          flux_out,
            "flux_in":           flux_in,
            "norm_depth":        nd,
            "dur_period_ratio":  dpr,
            "radius_ratio":      rr,
        })

    negatives_df = pd.DataFrame(neg_records)
    combined     = pd.concat([positives_df, negatives_df], ignore_index=True)

    print(f"  KIC {kic_id}: {n_pos} positives, "
          f"{len(negatives_df)} negatives, {rejected} rejected")
    return combined


def build_combined_feature_dataset():
    combined_meta = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_meta.csv"))
    all_records   = []

    for kic_id, group in combined_meta.groupby("kic_id"):
        kic_id   = int(kic_id)
        lc_path  = os.path.join(LIGHTCURVE_DIR, f"KIC_{kic_id}_lightcurve.csv")

        if not os.path.exists(lc_path):
            print(f"  KIC {kic_id}: light curve CSV not found — skipping")
            continue

        lc_df    = pd.read_csv(lc_path)
        result   = process_one_target(kic_id, group.reset_index(drop=True), lc_df)
        all_records.append(result)

    if not all_records:
        print("No features extracted.")
        return None

    final_df = pd.concat(all_records, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFeature extraction complete:")
    print(f"  Total samples:  {len(final_df)}")
    print(f"  Positives:      {int(final_df['label'].sum())}")
    print(f"  Negatives:      {int((final_df['label'] == 0).sum())}")
    print(f"  Output → {OUTPUT_CSV}")
    print(f"\nPer-target breakdown:")
    print(final_df.groupby(["kic_id", "label"]).size().unstack(fill_value=0).to_string())

    return final_df


if __name__ == "__main__":
    build_combined_feature_dataset()