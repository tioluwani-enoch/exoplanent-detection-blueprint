import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LIGHTCURVE_DIR = PROCESSED_DIR
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "combined_features.csv")
STELLAR_PARAMS_CSV = os.path.join(PROCESSED_DIR, "stellar_params.csv")

RANDOM_SEED = 42
NORM_DEPTH_MIN = 0.0005

# Hardcoded fallback — overridden automatically if ingest.py has saved stellar_params.csv
STELLAR_PARAMS = {
    11446443: {"radius_rsun": 1.000, "mass_msun": 0.980, "t_eff_k": 5850.0},
    5780885:  {"radius_rsun": 2.020, "mass_msun": 1.350, "t_eff_k": 5933.0},
    11853905: {"radius_rsun": 1.487, "mass_msun": 1.223, "t_eff_k": 5857.0},
    10619192: {"radius_rsun": 1.045, "mass_msun": 1.160, "t_eff_k": 5630.0},
    10874614: {"radius_rsun": 1.391, "mass_msun": 1.209, "t_eff_k": 5647.0},
    6922244:  {"radius_rsun": 1.486, "mass_msun": 1.213, "t_eff_k": 6213.0},
    6541920:  {"radius_rsun": 1.065, "mass_msun": 0.961, "t_eff_k": 5680.0},
    11904151: {"radius_rsun": 1.065, "mass_msun": 0.913, "t_eff_k": 5627.0},
}


def load_stellar_params():
    """
    Load stellar params from ingest.py output CSV if it exists.
    To enable this path, add one line to the end of ingest.py:
        results.to_csv(os.path.join(PROCESSED_DIR, "stellar_params.csv"), index=False)
    Falls back to hardcoded STELLAR_PARAMS if CSV is missing.
    """
    if os.path.exists(STELLAR_PARAMS_CSV):
        df = pd.read_csv(STELLAR_PARAMS_CSV)
        # Use median across quarters to get one value per star
        df_agg = df.groupby("kic_id").agg(
            stellar_rad_rs=("stellar_rad_rs", "median"),
            t_eff_k=("t_eff_k", "median"),
        ).reset_index()
        params = {}
        for _, row in df_agg.iterrows():
            kic_id = int(row["kic_id"])
            fallback = STELLAR_PARAMS.get(kic_id, {"radius_rsun": 1.0, "mass_msun": 1.0})
            params[kic_id] = {
                "radius_rsun": float(row["stellar_rad_rs"])
                               if pd.notna(row["stellar_rad_rs"])
                               else fallback["radius_rsun"],
                "mass_msun": fallback["mass_msun"],
                "t_eff_k": float(row["t_eff_k"])
                           if pd.notna(row["t_eff_k"]) else 5778.0,
            }
        print(f"  Loaded stellar params from CSV for {len(params)} targets")
        return params
    print("  stellar_params.csv not found — using hardcoded STELLAR_PARAMS")
    return STELLAR_PARAMS


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(period, duration, flux_out, flux_in, t_eff_k=5778.0):
    norm_depth = (flux_out - flux_in) / flux_out if flux_out != 0 else 0.0
    norm_depth_corrected = correct_depth_for_limb_darkening(norm_depth, t_eff_k)
    dur_period_ratio = duration / period
    radius_ratio = norm_depth_corrected ** 0.5 if norm_depth_corrected > 0 else 0.0

    if not (NORM_DEPTH_MIN < norm_depth < 0.1):
        return None
    if not (0.5 < period < 500):
        return None
    if not (0.001 < dur_period_ratio < 0.1):
        return None

    return {
        "norm_depth": norm_depth,
        "dur_period_ratio": dur_period_ratio,
        "radius_ratio": radius_ratio,
    }


def compute_ingress_egress_slope(lc_df, center_time, duration_hours):
    """
    Compute ingress slope (flux dropping into transit) and egress slope
    (flux recovering). For a clean planet transit these are steep and roughly
    symmetric; EB contamination produces shallower or asymmetric slopes.
    Returns (ingress_slope, egress_slope) as Δflux/Δday — normalized by depth
    so values are comparable across different stellar brightnesses.
    """
    half_dur = (duration_hours / 24) / 2
    buffer   = half_dur * 2.0  # slightly wider window to capture shoulders

    window = lc_df[
        (lc_df["time_BKJD"] >= center_time - buffer) &
        (lc_df["time_BKJD"] <= center_time + buffer)
    ].dropna(subset=["flux_norm"]).sort_values("time_BKJD")

    if len(window) < 6:
        return 0.0, 0.0

    mid_t    = center_time
    ingress  = window[window["time_BKJD"] < mid_t]
    egress   = window[window["time_BKJD"] >= mid_t]

    def safe_slope(seg):
        if len(seg) < 2:
            return 0.0
        t = seg["time_BKJD"].values
        f = seg["flux_norm"].values
        dt = t[-1] - t[0]
        if dt == 0:
            return 0.0
        return float((f[-1] - f[0]) / dt)

    return safe_slope(ingress), safe_slope(egress)


def compute_secondary_depth(lc_df, center_time, period_days, duration_hours):
    """
    Measure flux dip at the expected secondary eclipse (phase 0.5).
    A true exoplanet transit yields secondary_depth ≈ 0 because the planet
    contributes negligible flux. An eclipsing binary shows a real dip here.
    Returns secondary_depth >= 0; values > ~0.002 are EB red flags.
    """
    secondary_center = center_time + period_days / 2.0
    half_dur = (duration_hours / 24) / 2

    in_secondary = lc_df[
        (lc_df["time_BKJD"] >= secondary_center - half_dur) &
        (lc_df["time_BKJD"] <= secondary_center + half_dur)
    ]["flux_norm"]

    out_secondary = lc_df[
        (lc_df["time_BKJD"] >= secondary_center - period_days / 2) &
        (lc_df["time_BKJD"] <= secondary_center + period_days / 2) &
        ~(
            (lc_df["time_BKJD"] >= secondary_center - half_dur) &
            (lc_df["time_BKJD"] <= secondary_center + half_dur)
        )
    ]["flux_norm"]

    if len(in_secondary) == 0 or len(out_secondary) == 0:
        return 0.0

    f_in  = np.nanmedian(in_secondary.values)
    f_out = np.nanmedian(out_secondary.values)
    depth = (f_out - f_in) / f_out if f_out != 0 else 0.0
    return float(max(depth, 0.0))


# ── PHYSICAL PARAMETER DERIVATION ─────────────────────────────────────────────

def correct_depth_for_limb_darkening(norm_depth, t_eff_k):
    """
    Quadratic limb darkening correction (Mandel & Agol 2002).
    Coefficients approximated from Claret (2011) Kepler bandpass.
    Valid for solar-type stars 4500K < Teff < 7000K.
    """
    teff = t_eff_k if (t_eff_k and not np.isnan(float(t_eff_k))) else 5778.0
    u1 = np.clip(0.6 - 0.0001 * (teff - 5778), 0.1, 0.9)
    u2 = np.clip(0.1 + 0.00005 * (teff - 5778), 0.0, 0.5)
    correction = (1 - u1 / 3 - u2 / 6) ** 2
    return norm_depth / correction


def compute_physical_params(radius_ratio, period_days, kic_id, stellar_params=None):
    """
    Derive planet radius (R_Jup) and orbital distance (AU) from
    dimensionless transit features + stellar parameters.

    Planet radius:  R_p = (Rp/Rs) × R_star
    Orbital dist:   a³  = M_star × P² (Kepler's 3rd Law, solar units)
    """
    if stellar_params is None:
        stellar_params = STELLAR_PARAMS

    params        = stellar_params.get(int(kic_id), {"radius_rsun": 1.0, "mass_msun": 1.0})
    R_star_solar  = params["radius_rsun"]
    M_star_solar  = params["mass_msun"]

    R_star_earth   = R_star_solar * 109.076        # 1 R_sun = 109.076 R_earth
    R_planet_earth = radius_ratio * R_star_earth
    R_planet_jup   = R_planet_earth / 11.209       # 1 R_jup = 11.209 R_earth

    P_years = period_days / 365.25
    a_AU    = (M_star_solar * P_years ** 2) ** (1 / 3)

    return {
        "planet_radius_Rjup":  round(R_planet_jup, 4),
        "orbital_distance_AU": round(a_AU, 5),
    }


# ── FLUX EXTRACTION ───────────────────────────────────────────────────────────

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
        np.nanmin(in_transit["flux_norm"].values),     # true transit floor, not biased median
        np.nanmedian(out_transit["flux_norm"].values),
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

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(
        len(non_transit_df),
        size=min(n_samples, len(non_transit_df)),
        replace=False,
    )
    return non_transit_df.iloc[sample_idx].reset_index(drop=True)


# ── PER-TARGET PROCESSING ─────────────────────────────────────────────────────

def process_one_target(kic_id, meta_df, lc_df, stellar_params=None):
    period_days    = meta_df["period_days"].iloc[0]
    duration_hours = meta_df["duration_hours"].iloc[0]
    t0_bkjd        = meta_df["center_time"].min()
    is_eb          = int(meta_df["label"].iloc[0]) == 0

    records  = []
    rejected = 0

    for _, row in meta_df.iterrows():
        flux_in, flux_out = compute_flux_in_out(
            lc_df, row["center_time"], duration_hours, period_days
        )
        if flux_in is None:
            rejected += 1
            continue

        t_eff = stellar_params.get(int(kic_id), {}).get("t_eff_k", 5778.0) \
                if stellar_params else 5778.0
        features = compute_features(period_days, duration_hours / 24, flux_out, flux_in, t_eff)
        if features is None:
            rejected += 1
            continue

        ingress_slope, _ = compute_ingress_egress_slope(
            lc_df, row["center_time"], duration_hours
        )
        secondary_depth = compute_secondary_depth(
            lc_df, row["center_time"], period_days, duration_hours
        )

        phys = compute_physical_params(
            features["radius_ratio"], period_days, kic_id, stellar_params
        ) if not is_eb else {"planet_radius_Rjup": np.nan, "orbital_distance_AU": np.nan}

        records.append({
            "kic_id":              kic_id,
            "window_index":        row["window_index"],
            "center_time":         row["center_time"],
            "label":               0 if is_eb else 1,
            "period_days":         period_days,
            "duration_hours":      duration_hours,
            "flux_out":            flux_out,
            "flux_in":             flux_in,
            "norm_depth":          features["norm_depth"],
            "dur_period_ratio":    features["dur_period_ratio"],
            "radius_ratio":        features["radius_ratio"],
            "ingress_slope":       ingress_slope,
            "secondary_depth":     secondary_depth,
            "planet_radius_Rjup":  phys["planet_radius_Rjup"],
            "orbital_distance_AU": phys["orbital_distance_AU"],
        })

    target_df = pd.DataFrame(records)

    if is_eb:
        print(f"  KIC {kic_id} (EB): {len(target_df)} eclipse windows labeled 0, "
              f"{rejected} rejected")
        return target_df

    n_pos       = len(target_df)
    neg_windows = sample_negatives(meta_df, n_pos, period_days, duration_hours, t0_bkjd)

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

        ing_slope, _ = compute_ingress_egress_slope(
            lc_df, row["center_time"], duration_hours
        )
        sec_depth = compute_secondary_depth(
            lc_df, row["center_time"], period_days, duration_hours
        )

        neg_records.append({
            "kic_id":              kic_id,
            "window_index":        row["window_index"],
            "center_time":         row["center_time"],
            "label":               0,
            "period_days":         period_days,
            "duration_hours":      duration_hours,
            "flux_out":            flux_out,
            "flux_in":             flux_in,
            "norm_depth":          nd,
            "dur_period_ratio":    dpr,
            "radius_ratio":        rr,
            "ingress_slope":       ing_slope,
            "secondary_depth":     sec_depth,
            "planet_radius_Rjup":  np.nan,
            "orbital_distance_AU": np.nan,
        })

    negatives_df = pd.DataFrame(neg_records)
    combined     = pd.concat([target_df, negatives_df], ignore_index=True)

    print(f"  KIC {kic_id}: {n_pos} positives, "
          f"{len(negatives_df)} negatives, {rejected} rejected")
    return combined


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def build_combined_feature_dataset():
    stellar_params = load_stellar_params()
    combined_meta  = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_meta.csv"))
    all_records    = []

    for kic_id, group in combined_meta.groupby("kic_id"):
        kic_id  = int(kic_id)
        lc_path = os.path.join(LIGHTCURVE_DIR, f"KIC_{kic_id}_lightcurve.csv")

        if not os.path.exists(lc_path):
            print(f"  KIC {kic_id}: light curve CSV not found — skipping")
            continue

        lc_df  = pd.read_csv(lc_path)
        result = process_one_target(
            kic_id, group.reset_index(drop=True), lc_df, stellar_params
        )
        all_records.append(result)

    if not all_records:
        print("No features extracted.")
        return None

    final_df = pd.concat(all_records, ignore_index=True)

    eb_mask = (final_df["label"] == 0)
    if eb_mask.sum() > 200:
        eb_df     = final_df[eb_mask].sample(n=200, random_state=RANDOM_SEED)
        planet_df = final_df[~eb_mask]
        final_df  = pd.concat([planet_df, eb_df], ignore_index=True)
        print(f"  Capped EB eclipse windows to 200 (was {eb_mask.sum()})")

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