import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGETS = {
    11446443: {"name": "TrES-2b",    "period": 2.47063,   "t0": 120.595},
    5780885:  {"name": "Kepler-7b",  "period": 4.885525,  "t0": 134.179},
    11853905: {"name": "Kepler-4b",  "period": 3.21346,   "t0": 123.613},
    10619192: {"name": "Kepler-17b", "period": 1.4857,    "t0": 352.678},
    10874614: {"name": "Kepler-6b",  "period": 3.2347,    "t0": 121.627},
    6922244:  {"name": "Kepler-8b",  "period": 3.5225,    "t0": 121.489},
    11904151: {"name": "Kepler-10b", "period": 0.8374907, "t0": 201.087},
}


def load_lightcurve(kic_id):
    path = os.path.join(PROCESSED_DIR, f"KIC_{kic_id}_lightcurve.csv")
    if not os.path.exists(path):
        print(f"  Light curve not found for KIC {kic_id} — skipping")
        return None
    return pd.read_csv(path)


def load_features(kic_id=None):
    path = os.path.join(PROCESSED_DIR, "combined_features.csv")
    if not os.path.exists(path):
        print("  combined_features.csv not found — run features.py first")
        return None
    df = pd.read_csv(path)
    if kic_id is not None:
        df = df[df["kic_id"] == kic_id]
    return df


# ── Plot 1: Before / After detection ─────────────────────────────────────────

def plot_full_lightcurve(kic_id):
    df     = load_lightcurve(kic_id)
    feats  = load_features(kic_id)
    config = TARGETS.get(kic_id, {})

    if df is None or feats is None or len(feats) == 0:
        print(f"  Skipping KIC {kic_id} — no data")
        return

    # Use model predictions for the after panel, not training labels
    # predicted_transit column written by model.py at threshold=0.10
    if "predicted_transit" in feats.columns:
        detections = feats[feats["predicted_transit"] == 1]
    else:
        print(f"  Warning: predicted_transit column not found — run model.py first")
        detections = feats[feats["label"] == 1]

    duration_hours  = feats["duration_hours"].iloc[0] if len(feats) > 0 else 1.7

    # Actual half-duration in days (~0.07 days for TrES-2b)
    # Too narrow to see at full 1500-day scale — use visual width instead
    half_dur_actual = (duration_hours / 24) / 2
    half_dur_visual = max(half_dur_actual, 3.0)

    name    = config.get("name", f"KIC {kic_id}")
    y_low   = df["flux_norm"].quantile(0.001)
    y_high  = df["flux_norm"].quantile(0.999)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # Panel 1 — Before: raw light curve, no annotations
    axes[0].plot(df["time_BKJD"], df["flux_norm"],
                 lw=0.4, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Normalized Flux")
    axes[0].set_title(f"KIC {kic_id} ({name}) — Before Detection")
    axes[0].set_ylim(y_low, y_high)

    # Panel 2 — After: light curve with model-predicted transit windows
    axes[1].plot(df["time_BKJD"], df["flux_norm"],
                 lw=0.4, color="steelblue", alpha=0.8)
    for _, row in detections.iterrows():
        axes[1].axvspan(
            row["center_time"] - half_dur_visual,
            row["center_time"] + half_dur_visual,
            color="red", alpha=0.25, linewidth=0
        )
    axes[1].set_ylabel("Normalized Flux")
    axes[1].set_xlabel("Time (BKJD)")
    axes[1].set_title(
        f"KIC {kic_id} ({name}) — After Detection "
        f"({len(detections)} transit windows flagged)"
    )
    axes[1].set_ylim(y_low, y_high)

    transit_patch = mpatches.Patch(color="red", alpha=0.35,
                                   label="Model-predicted transit")
    axes[1].legend(handles=[transit_patch], loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, f"KIC_{kic_id}_before_after.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Plot 2: Phase-folded light curve ─────────────────────────────────────────

def plot_phase_folded(kic_id):
    df     = load_lightcurve(kic_id)
    config = TARGETS.get(kic_id, {})

    if df is None or not config:
        return

    period   = config["period"]
    t0       = config["t0"]
    name     = config.get("name", f"KIC {kic_id}")
    df_clean = df.dropna(subset=["flux_norm"]).copy()

    phase          = ((df_clean["time_BKJD"] - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(phase, df_clean["flux_norm"],
               s=0.4, color="steelblue", alpha=0.4, rasterized=True)
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(
        f"{name} (KIC {kic_id}) — Phase-Folded Light Curve  "
        f"|  P = {period:.5f} days"
    )
    # Narrow window to show transit clearly, bin points for visibility
    phase_window = min(0.05, (config["period"] * 3) / (config["period"] * 24))
    ax.set_xlim(-0.05, 0.05)
    # Bin the phase curve so transit isn't buried in scatter
    bin_edges = np.linspace(-0.05, 0.05, 100)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = [
        df_clean["flux_norm"][(phase >= bin_edges[i]) & (phase < bin_edges[i+1])].mean()
        for i in range(len(bin_centers))
    ]
    ax.plot(bin_centers, bin_means, color="steelblue", lw=1.5, zorder=3)
    ax.set_ylim(
    df_clean["flux_norm"].quantile(0.001),
    df_clean["flux_norm"].quantile(0.999)
)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, f"KIC_{kic_id}_phasefolded.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Plot 3: Physical parameters summary ──────────────────────────────────────

def plot_physical_params():
    # Read directly from detection_results.csv so plot always matches the CSV
    results_path = os.path.join(OUTPUTS_DIR, "detection_results.csv")
    if not os.path.exists(results_path):
        print("  detection_results.csv not found — run export_results_csv() first")
        return

    summary = pd.read_csv(results_path)

    if "planet_radius_Rjup" not in summary.columns or \
       "orbital_distance_AU" not in summary.columns:
        print("  planet_radius_Rjup / orbital_distance_AU columns not found in results CSV")
        return

    summary = summary.dropna(subset=["planet_radius_Rjup", "orbital_distance_AU"])

    if len(summary) == 0:
        print("  No physical parameter data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(summary["planet_name"], summary["planet_radius_Rjup"],
                 color="steelblue", edgecolor="white")
    axes[0].axvline(1.0, color="orange", linestyle="--",
                    linewidth=1.2, label="1 Jupiter radius")
    axes[0].set_xlabel("Planet Radius (R_Jup)")
    axes[0].set_title("Estimated Planet Radii")
    axes[0].legend()

    axes[1].barh(summary["planet_name"], summary["orbital_distance_AU"],
                 color="coral", edgecolor="white")
    axes[1].axvline(1.0, color="green", linestyle="--",
                    linewidth=1.2, label="1 AU (Earth orbit)")
    axes[1].set_xlabel("Orbital Distance (AU)")
    axes[1].set_title("Estimated Orbital Distances")
    axes[1].legend()

    plt.suptitle("Physical Parameter Estimates — Detected Planets", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "physical_params_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Export: results CSV ───────────────────────────────────────────────────────

def export_results_csv():
    feats = load_features()
    if feats is None:
        return

    # Use model predictions if available, otherwise fall back to labels
    label_col = "predicted_transit" if "predicted_transit" in feats.columns \
                else "label"

    planets = feats[feats[label_col] == 1].copy()
    planets["planet_name"] = planets["kic_id"].map(
        {k: v["name"] for k, v in TARGETS.items()}
    )

    agg_dict = {
        "period_days":         ("period_days",      "first"),
        "duration_hours":      ("duration_hours",   "first"),
        "median_norm_depth":   ("norm_depth",        "median"),
        "median_radius_ratio": ("radius_ratio",      "median"),
        "transits_detected":   ("window_index",      "count"),
    }

    if "planet_radius_Rjup" in feats.columns:
        agg_dict["planet_radius_Rjup"] = ("planet_radius_Rjup", "median")
    if "orbital_distance_AU" in feats.columns:
        agg_dict["orbital_distance_AU"] = ("orbital_distance_AU", "median")
    if "prediction_proba" in feats.columns:
        agg_dict["mean_confidence"] = ("prediction_proba", "mean")

    summary = planets.groupby(["kic_id", "planet_name"]).agg(
        **agg_dict
    ).reset_index()

    out = os.path.join(OUTPUTS_DIR, "detection_results.csv")
    summary.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(summary.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 4 — Visualization")
    print("=" * 55)

    for kic_id in TARGETS:
        print(f"\nPlotting KIC {kic_id} — {TARGETS[kic_id]['name']}...")
        plot_full_lightcurve(kic_id)
        plot_phase_folded(kic_id)

    print("\nGenerating summary plots...")
    export_results_csv()
    plot_physical_params()

    print("\nDone. All plots saved to /outputs/")