import os
import sys
import pandas as pd
import lightkurve as lk

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_target, TARGETS

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def run_all_targets():
    all_meta     = []
    succeeded    = []
    failed       = []

    for kic_id, config in TARGETS.items():
        print(f"\n{'='*55}")
        print(f"  KIC {kic_id} — {config['name']}")
        print(f"{'='*55}")

        try:
            search_result = lk.search_lightcurve(
                f"KIC {kic_id}",
                mission="Kepler",
                author="Kepler",
                exptime=1800
            )

            if len(search_result) == 0:
                print(f"  No data found — skipping")
                failed.append(kic_id)
                continue

            print(f"  Found {len(search_result)} quarters")
            lc_collection = search_result.download_all(
                download_dir=os.path.join(PROJECT_ROOT, "data", "raw")
            )

            if lc_collection is None:
                print(f"  Download failed — skipping")
                failed.append(kic_id)
                continue

            windows_raw, windows_ml, centers, meta = preprocess_target(
                kic_id, lc_collection, target_config=config
            )
            all_meta.append(meta)
            succeeded.append(kic_id)

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(kic_id)
            continue

    if not all_meta:
        print("\nNo targets processed successfully.")
        return None

    # Merge all meta CSVs into combined training set
    combined = pd.concat(all_meta, ignore_index=True)
    out_path = os.path.join(PROCESSED_DIR, "combined_meta.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Batch complete")
    print(f"{'='*55}")
    print(f"  Succeeded: {len(succeeded)} targets {succeeded}")
    print(f"  Failed:    {len(failed)} targets {failed}")
    print(f"  Total windows: {len(combined)}")
    print(f"  Combined meta → {out_path}")
    print(f"\n  Per-target summary:")
    summary = combined.groupby("kic_id").agg(
        windows=("window_index", "count"),
        label=("label", "first"),
        transit_windows=("depth_raw", lambda x: (x > 0.005).sum())
    )
    print(summary.to_string())

    return combined


if __name__ == "__main__":
    run_all_targets()