import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Target config — extend as more targets are added
TARGETS = {
    11446443: {"name": "TrES-2b",   "period": 2.47063,  "t0": 120.595},
    5780885:  {"name": "Kepler-7b", "period": 4.885525, "t0": 134.179},
    11853905: {"name": "Kepler-4b", "period": 3.21346,  "t0": 123.613},
    10619192: {"name": "Kepler-17b","period": 1.4857,   "t0": 352.678},
    10874614: {"name": "Kepler-6b", "period": 3.2347,   "t0": 121.627},
    6922244:  {"name": "Kepler-8b", "period": 3.5225,   "t0": 121.489},
    6541920:  {"name": "Kepler-11b","period": 10.3039,  "t0": 856.738},
    11904151: {"name": "Kepler-10b","period": 0.8374907,"t0": 201.087},
}


def load_lightcurve(kic_id):
    path = os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv")
    if not os.path.exists(path):
        print(f"Light curve not found for KIC {kic_id} — skipping")
        return None
    return pd.read_csv(path)


def load_features(kic_id=None):
    """Load combined features, optionally filtered by KIC ID."""
    path = os.path.join(PROCESSED_DIR, "combined_features.csv")
    if not os.path.exists(path):
        print("combined_features.csv not found")
        return None
    df = pd.read_csv(path)
    if kic_id is not None:
        df = df[df["kic_id"] == kic_id]
    return df


# ── Plot 1: Full light curve with labeled transit detections ──────────────────
def plot_full_lightcurve(kic_id):
    df     = load_lightcurve(kic_id)
    feats  = load_features(kic_id)
    config = TARGETS.get(kic_id, {})
    if df is None or feats is None:
        return

    detections     = feats[feats["label"] == 1]
    duration_hours = feats["duration_hours"].iloc[0] if len(feats) > 0 else 1.7
    half_dur = max(duration_hours / 24) / 2

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # Before — raw light curve, no annotations
    axes[0].plot(df["time_BKJD"], df["flux_norm"],
                 lw=0.4, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Normalized Flux")
    axes[0].set_title(f"KIC {kic_id} ({config.get('name','')}) — Raw Light Curve (Before Detection)")
    axes[0].set_ylim(df["flux_norm"].quantile(0.001), df["flux_norm"].quantile(0.999))

    # After — light curve with detected transit windows highlighted
    axes[1].plot(df["time_BKJD"], df["flux_norm"],
                 lw=0.4, color="steelblue", alpha=0.8)
    for _, row in detections.iterrows():
        axes[1].axvspan(row["center_time"] - half_dur,
                        row["center_time"] + half_dur,
                        color="red", alpha=0.35, linewidth=0)
    axes[1].set_ylabel("Normalized Flux")
    axes[1].set_xlabel("Time (BKJD)")
    axes[1].set_title(f"KIC {kic_id} ({config.get('name','')}) — With {len(detections)} Detected Transits (After Detection)")
    axes[1].set_ylim(df["flux_norm"].quantile(0.001), df["flux_norm"].quantile(0.999))

    transit_patch = mpatches.Patch(color="red", alpha=0.35, label="Detected Transit")
    axes[1].legend(handles=[transit_patch], loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, f"KIC_{kic_id}_before_after.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Phase-folded light curve ─────────────────────────────────────────
def plot_phase_folded(kic_id):
    df     = load_lightcurve(kic_id)
    config = TARGETS.get(kic_id, {})
    if df is None or not config:
        return

    period = config["period"]
    t0     = config["t0"]
    name   = config.get("name", f"KIC {kic_id}")

    df_clean = df.dropna(subset=["flux_norm"]).copy()
    phase    = ((df_clean["time_BKJD"] - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(phase, df_clean["flux_norm"],
               s=0.4, color="steelblue", alpha=0.4, rasterized=True)
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"{name} (KIC {kic_id}) — Phase-Folded Light Curve  |  P = {period:.5f} days")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(df_clean["flux_norm"].quantile(0.001),
                df_clean["flux_norm"].quantile(0.999))

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, f"KIC_{kic_id}_phasefolded.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Physical parameters summary chart ─────────────────────────────────
def plot_physical_params():
    feats = load_features()
    if feats is None:
        return

    planets = feats[(feats["label"] == 1) &
                    feats["planet_radius_Rjup"].notna() &
                    feats["orbital_distance_AU"].notna()]

    # One row per KIC ID — median values
    summary = planets.loc[planets.groupby("kic_id")["norm_depth"].idxmax()][
    ["kic_id", "planet_radius_Rjup", "orbital_distance_AU", "norm_depth"]
].reset_index(drop=True)

    # Add planet names
    summary["name"] = summary["kic_id"].map(
        {k: v["name"] for k, v in TARGETS.items()}
    ).fillna(summary["kic_id"].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: planet radius
    axes[0].barh(summary["name"], summary["planet_radius_Rjup"],
                 color="steelblue", edgecolor="white")
    axes[0].axvline(1.0, color="orange", linestyle="--",
                    linewidth=1.2, label="1 Jupiter radius")
    axes[0].set_xlabel("Planet Radius (R_Jup)")
    axes[0].set_title("Estimated Planet Radii")
    axes[0].legend()

    # Right: orbital distance
    axes[1].barh(summary["name"], summary["orbital_distance_AU"],
                 color="coral", edgecolor="white")
    axes[1].axvline(1.0, color="green", linestyle="--",
                    linewidth=1.2, label="1 AU (Earth's orbit)")
    axes[1].set_xlabel("Orbital Distance (AU)")
    axes[1].set_title("Estimated Orbital Distances")
    axes[1].legend()

    plt.suptitle("Physical Parameter Estimates — Detected Planets", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "physical_params_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Export: clean results CSV ─────────────────────────────────────────────────
def export_results_csv():
    feats = load_features()
    if feats is None:
        return

    planets = feats[feats["label"] == 1].copy()
    planets["planet_name"] = planets["kic_id"].map(
        {k: v["name"] for k, v in TARGETS.items()}
    )

    summary = planets.groupby(["kic_id", "planet_name"]).agg(
        period_days         = ("period_days",          "first"),
        duration_hours      = ("duration_hours",        "first"),
        median_norm_depth   = ("norm_depth",            "median"),
        median_radius_ratio = ("radius_ratio",          "median"),
        planet_radius_Rjup  = ("planet_radius_Rjup",   "median"),
        orbital_distance_AU = ("orbital_distance_AU",  "median"),
        transits_detected   = ("window_index",          "count"),
    ).reset_index()

    out = os.path.join(OUTPUTS_DIR, "detection_results.csv")
    summary.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(summary.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for kic_id in TARGETS:
        plot_full_lightcurve(kic_id)
        plot_phase_folded(kic_id)

    plot_physical_params()
    export_results_csv()