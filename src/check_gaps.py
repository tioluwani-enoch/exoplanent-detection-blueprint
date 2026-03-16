import numpy as np
import pandas as pd
import lightkurve as lk
import os
from astropy.stats import sigma_clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_large_gaps():
    print("Loading light curves...")
    search_result = lk.search_lightcurve(
        "KIC 11446443",
        mission="Kepler",
        author="Kepler",
        exptime=1800
    )
    lc_collection = search_result.download_all(
        download_dir=os.path.join(PROJECT_ROOT, "data", "raw")
    )

    # Stitch with per-quarter normalization
    stitched = lc_collection.stitch(corrector_func=lambda x: x.normalize())
    time = stitched.time.value
    flux = stitched.flux.value

    # Convert to plain numpy array first (lightkurve returns masked arrays)
    flux = np.array(flux, dtype=np.float64)
    flux[~np.isfinite(flux)] = np.nan

    # 2.5-sigma clip
    clipped = sigma_clip(flux, sigma=2.5, maxiters=5, masked=True)
    flux[clipped.mask] = np.nan

    # Find all runs of NaNs
    nan_mask = np.isnan(flux)
    gaps = []
    i = 0
    while i < len(nan_mask):
        if nan_mask[i]:
            start = i
            while i < len(nan_mask) and nan_mask[i]:
                i += 1
            end = i
            length = end - start
            center_time = time[start + length // 2] if (start + length // 2) < len(time) else time[-1]
            gaps.append({
                "gap_index":   len(gaps),
                "start_idx":   start,
                "end_idx":     end,
                "length":      length,
                "center_bkjd": round(center_time, 1),
                "type":        "LARGE" if length > 10 else "small"
            })
        else:
            i += 1

    all_gaps = pd.DataFrame(gaps)
    large_gaps = all_gaps[all_gaps["type"] == "LARGE"].reset_index(drop=True)

    print(f"\nTotal gaps found: {len(all_gaps)}")
    print(f"Small gaps (≤10): {(all_gaps['type'] == 'small').sum()}")
    print(f"Large gaps (>10): {len(large_gaps)}")

    print("\n── Large gaps detail ──────────────────────────────────────")
    print(f"{'Gap':>4}  {'Center BKJD':>12}  {'Length':>8}  {'Note'}")
    print("-" * 55)

    # Known Kepler quarter boundary times (approximate BKJD)
    quarter_boundaries = [
        131, 170, 260, 350, 442, 532, 591, 683, 773,
        838, 930, 1021, 1098, 1182, 1273, 1372, 1471
    ]

    for _, gap in large_gaps.iterrows():
        # Check if gap is near a quarter boundary
        nearest = min(quarter_boundaries, key=lambda q: abs(q - gap.center_bkjd))
        dist = abs(nearest - gap.center_bkjd)
        if dist < 10:
            note = f"quarter boundary (~Q{quarter_boundaries.index(nearest)})"
        else:
            note = "*** INVESTIGATE — mid-quarter gap ***"

        print(f"{int(gap.gap_index):>4}  {gap.center_bkjd:>12.1f}  "
              f"{int(gap.length):>8}  {note}")

    print("\n── Missing quarters check ─────────────────────────────────")
    print("Expected missing: Q8, Q12, Q16 (not in your 15 files)")
    print("Q8  ≈ BKJD 773–838")
    print("Q12 ≈ BKJD 1021–1098")
    print("Q16 ≈ BKJD 1372–1471")

    return large_gaps

if __name__ == "__main__":
    large_gaps = check_large_gaps()