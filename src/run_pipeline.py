"""
run_pipeline.py — Single entry point for the full training pipeline

This script replaces both the old ingest.py AND run_pipeline.py.
Instead of downloading data twice (once in ingest, once here), it does
everything in one pass per target:

    1. Search the Kepler archive for the star
    2. Download all available quarters (once)
    3. Extract stellar parameters from the FITS headers (what ingest.py did)
    4. Run preprocessing + BLS period-finding (what preprocess.py does)
    5. Collect all results

OUTPUTS:
    data/processed/stellar_params.csv   — stellar properties for all targets
    data/processed/combined_meta.csv    — windowed light curve metadata
    data/processed/KIC_*/               — per-target window files
    data/processed/KIC_*_lightcurve.csv — per-target processed light curves

After this finishes, run:
    features.py → model.py → visualize.py
"""

import os
import sys
import numpy as np
import pandas as pd
import lightkurve as lk

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_target, TARGETS

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── STELLAR PARAMETER EXTRACTION ─────────────────────────────────────────────
#
# This is what ingest.py used to do as a separate step.
# Now it happens inside the same loop that downloads data, so we
# don't download anything twice.
#
# These parameters describe the STAR, not the planet:
#   - t_eff_k:        Surface temperature (Kelvin). Hotter stars are bigger/bluer.
#   - log_g:          Surface gravity. Tells you if it's a main-sequence star or giant.
#   - fe_h:           Metallicity. Metal-rich stars are more likely to host planets.
#   - stellar_rad_rs: Stellar radius in solar radii. Needed to convert
#                     radius_ratio (Rp/Rs) into actual planet size.
#   - kepmag:         Kepler magnitude (brightness). Dimmer stars = noisier data.
#
# We extract these from the FITS file headers that come with each quarter's
# light curve. Since the values should be the same across quarters (it's the
# same star), we take the median later in features.py.
# ─────────────────────────────────────────────────────────────────────────────

def extract_stellar_params(lc_collection, kic_id):
    """
    Pull stellar properties from the FITS headers of downloaded light curves.

    Parameters
    ----------
    lc_collection : lightkurve.LightCurveCollection — the downloaded quarters
    kic_id        : int — Kepler Input Catalog ID

    Returns
    -------
    List of dicts, one per quarter, with stellar + light curve summary stats.
    """
    records = []
    for lc in lc_collection:
        hdr = lc.meta  # FITS header metadata
        records.append({
            # --- Star identifiers ---
            "kic_id":          hdr.get("KEPLERID",  kic_id),
            "quarter":         hdr.get("QUARTER",   np.nan),
            "ra":              hdr.get("RA_OBJ",    np.nan),
            "dec":             hdr.get("DEC_OBJ",   np.nan),

            # --- Stellar physical parameters (from Kepler pipeline) ---
            "t_eff_k":         hdr.get("TEFF",      np.nan),
            "log_g":           hdr.get("LOGG",      np.nan),
            "fe_h":            hdr.get("FEH",       np.nan),
            "stellar_rad_rs":  hdr.get("RADIUS",    np.nan),
            "kepmag":          hdr.get("KEPMAG",    np.nan),

            # --- Light curve summary stats (useful for quick QA) ---
            "flux_mean":       np.nanmean(lc.flux.value),
            "flux_std":        np.nanstd(lc.flux.value),
            "flux_min":        np.nanmin(lc.flux.value),
            "n_cadences":      len(lc.flux),
            "time_baseline_d": float(lc.time[-1].value - lc.time[0].value),
        })
    return records


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_all_targets():
    """
    One-pass pipeline: download → extract stellar params → preprocess with BLS.

    For each target in TARGETS:
      1. Search Kepler archive
      2. Download all quarters (saved to data/raw/ for caching)
      3. Extract stellar parameters from FITS headers
      4. Run preprocess_target() which does flatten → normalize → BLS → window
      5. Collect everything
    """
    all_meta           = []
    all_stellar_params = []
    succeeded          = []
    failed             = []

    for kic_id, config in TARGETS.items():
        print(f"\n{'='*55}")
        print(f"  KIC {kic_id} — {config['name']}")
        print(f"{'='*55}")

        try:
            # ── Step 1: Search the archive ──
            search_result = lk.search_lightcurve(
                f"KIC {kic_id}",
                mission="Kepler",
                author="Kepler",
                exptime=1800     # 30-min cadence (standard Kepler long cadence)
            )

            if len(search_result) == 0:
                print(f"  No data found — skipping")
                failed.append(kic_id)
                continue

            print(f"  Found {len(search_result)} quarters")

            # ── Step 2: Download (cached — won't re-download if already on disk) ──
            lc_collection = search_result.download_all(
                download_dir=RAW_DIR
            )

            if lc_collection is None:
                print(f"  Download failed — skipping")
                failed.append(kic_id)
                continue

            # ── Step 3: Extract stellar parameters (replaces ingest.py) ──
            stellar_records = extract_stellar_params(lc_collection, kic_id)
            all_stellar_params.extend(stellar_records)
            print(f"  Extracted stellar params from {len(stellar_records)} quarters")

            # Quick summary of what we got from the headers
            teff = stellar_records[0].get("t_eff_k", "N/A")
            rad  = stellar_records[0].get("stellar_rad_rs", "N/A")
            mag  = stellar_records[0].get("kepmag", "N/A")
            print(f"    Teff={teff}K, R*={rad}Rsun, Kepmag={mag}")

            # ── Step 4: Preprocess (flatten, BLS, window) ──
            result = preprocess_target(
                kic_id, lc_collection, target_config=config
            )

            if result is None:
                print(f"  BLS found no signal or preprocessing failed — skipping")
                failed.append(kic_id)
                continue

            windows_raw, windows_ml, centers, meta = result
            all_meta.append(meta)
            succeeded.append(kic_id)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(kic_id)
            continue

    # ── Save stellar parameters CSV (what ingest.py used to produce) ──
    if all_stellar_params:
        stellar_df = pd.DataFrame(all_stellar_params)
        stellar_path = os.path.join(PROCESSED_DIR, "stellar_params.csv")
        stellar_df.to_csv(stellar_path, index=False)
        print(f"\n  Stellar params saved → {stellar_path}")
        print(f"  Covers {stellar_df['kic_id'].nunique()} unique stars, "
              f"{len(stellar_df)} total quarter records")

    # ── Save combined meta CSV ──
    if not all_meta:
        print("\nNo targets processed successfully.")
        return None

    combined = pd.concat(all_meta, ignore_index=True)
    meta_path = os.path.join(PROCESSED_DIR, "combined_meta.csv")
    combined.to_csv(meta_path, index=False)

    # ── Print summary ──
    print(f"\n{'='*55}")
    print(f"  Pipeline complete")
    print(f"{'='*55}")
    print(f"  Succeeded: {len(succeeded)} targets {succeeded}")
    print(f"  Failed:    {len(failed)} targets {failed}")
    print(f"  Total windows: {len(combined)}")
    print(f"  Combined meta → {meta_path}")

    # Show BLS-derived parameters so you can sanity-check them
    # Compare these against known values to verify BLS is working:
    #   TrES-2b should be ~2.47 days
    #   Kepler-7b should be ~4.89 days
    #   etc.
    print(f"\n  BLS-derived parameters per target:")
    print(f"  {'KIC':>10}  {'Name':<20}  {'Period(d)':>10}  {'Dur(h)':>7}  "
          f"{'t0':>10}  {'BLS_SNR':>8}  {'Label':>6}")
    print(f"  {'-'*80}")
    for kic_id in succeeded:
        target_meta = combined[combined["kic_id"] == kic_id]
        period = target_meta["period_days"].iloc[0]
        dur    = target_meta["duration_hours"].iloc[0]
        t0     = target_meta["t0_bkjd"].iloc[0]
        snr    = target_meta["bls_snr"].iloc[0]
        label  = target_meta["label"].iloc[0]
        name   = TARGETS.get(kic_id, {}).get("name", "unknown")
        print(f"  {kic_id:>10}  {name:<20}  {period:>10.4f}  {dur:>7.2f}  "
              f"{t0:>10.2f}  {snr:>8.1f}  {label:>6}")

    print(f"\n  Per-target window summary:")
    summary = combined.groupby("kic_id").agg(
        windows=("window_index", "count"),
        label=("label", "first"),
        transit_windows=("depth_raw", lambda x: (x > 0.005).sum())
    )
    print(summary.to_string())

    print(f"\n  Next steps:")
    print(f"    python src/features.py")
    print(f"    python src/model.py")
    print(f"    python src/visualize.py")

    return combined


if __name__ == "__main__":
    run_all_targets()