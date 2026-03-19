import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Observations
import os


# CHECKPOINT: Need to have data query and basic filtering in place
# TARGET: Kepler exoplanet host star light curves


source = 'KIC 11446443'  # Defining target (KIC 11446443 = TrES-2b, confirmed exoplanet host)


# Defining output path — relative to project root (one level up from /src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)



# Gathering Coordinates
coord = SkyCoord.from_name(source)
target_ra  = coord.ra.deg
target_dec = coord.dec.deg


# Printing Coordinates
print(f"Target: {source}")
print(f"RA: {target_ra:.2f} deg, Dec: {target_dec:.2f} deg")


# Building the query
search_result = lk.search_lightcurve(
    f"KIC 11446443",
    mission="Kepler",
    author="Kepler",   # Official Kepler pipeline (PDCSAP flux — systematics corrected)
    exptime=1800       # Long cadence: 30-min intervals (standard for transit detection)
)

# Launch query
print(f"Downloaded {len(search_result)} light curve file(s)")
print(search_result)


# Filtering: only grab quarters with confirmed data (mirrors your phot_g_mean_mag < 17 filter)
lc_collection = search_result.download_all(download_dir=RAW_DIR)


# Extracting astrophysically relevant parameters from FITS headers
records = []
for lc in lc_collection:
    hdr = lc.meta
    records.append({
        # --- Stellar identifiers ---
        "kic_id":          hdr.get("KEPLERID",  np.nan),
        "quarter":         hdr.get("QUARTER",   np.nan),
        "ra":              hdr.get("RA_OBJ",    np.nan),
        "dec":             hdr.get("DEC_OBJ",   np.nan),

        # --- Stellar physical parameters (from FITS header) ---
        # These are the astrophysical params that matter for transit detection
        "t_eff_k":         hdr.get("TEFF",      np.nan),   # Effective temperature
        "log_g":           hdr.get("LOGG",      np.nan),   # Surface gravity
        "fe_h":            hdr.get("FEH",       np.nan),   # Metallicity
        "stellar_rad_rs":  hdr.get("RADIUS",    np.nan),   # Stellar radius (solar radii)
        "kepmag":          hdr.get("KEPMAG",    np.nan),   # Kepler magnitude

        # --- Light curve summary stats (ML feature candidates) ---
        "flux_mean":       np.nanmean(lc.flux.value),
        "flux_std":        np.nanstd(lc.flux.value),       # Proxy for photometric noise
        "flux_min":        np.nanmin(lc.flux.value),       # Candidate transit depth floor
        "n_cadences":      len(lc.flux),
        "time_baseline_d": float(lc.time[-1].value - lc.time[0].value),
    })

results = pd.DataFrame(records)
print(f"\nExtracted parameters for {len(results)} quarters")
print(results[["kic_id", "quarter", "t_eff_k", "log_g", "stellar_rad_rs", "kepmag", "flux_std"]].to_string())

results.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "stellar_params.csv"), index=False)