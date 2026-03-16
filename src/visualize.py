import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "KIC_11446443_lightcurve.csv")

# %%
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} cadences")
print(df.head())

# %%
output_path = os.path.join(PROJECT_ROOT, "outputs", "KIC_11446443_lightcurve.png")
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

plt.figure(figsize=(14, 4))
plt.plot(df["time_BKJD"], df["flux_norm"], lw=0.5, color='steelblue')
plt.xlabel("Time (BKJD)")
plt.ylabel("Normalized Flux")
plt.title("KIC 11446443 — Full Stitched Light Curve (15 Quarters)")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Plot saved to: {output_path}")

# %%
# Zoom into one known transit window for TrES-2b
# Known period ~2.47 days, zoom into a small window
mask = (df["time_BKJD"] > 120) & (df["time_BKJD"] < 135)
df_zoom = df[mask]

plt.figure(figsize=(14, 4))
plt.plot(df_zoom["time_BKJD"], df_zoom["flux_norm"], lw=1.0, color='steelblue', marker='o', markersize=1.5)
plt.xlabel("Time (BKJD)")
plt.ylabel("Normalized Flux")
plt.title("KIC 11446443 — Zoomed Window (BKJD 120–135)")
plt.tight_layout()

zoom_path = os.path.join(PROJECT_ROOT, "outputs", "KIC_11446443_zoomed.png")
plt.savefig(zoom_path, dpi=150)
print(f"Zoomed plot saved to: {zoom_path}") 

# %%
# Phase fold on TrES-2b's known period to reveal transit shape
period = 2.47063   # days
t0 = 120.595       # approximate first transit center (BKJD)

df_clean = df.dropna(subset=["flux_norm"])
phase = ((df_clean["time_BKJD"] - t0) % period) / period
phase[phase > 0.5] -= 1  # center transit at phase 0

plt.figure(figsize=(10, 4))
plt.scatter(phase, df_clean["flux_norm"], s=0.3, color='steelblue', alpha=0.4)
plt.xlabel("Orbital Phase")
plt.ylabel("Normalized Flux")
plt.title("KIC 11446443 — Phase Folded Light Curve (TrES-2b, P=2.47 days)")
plt.xlim(-0.1, 0.1)
plt.tight_layout()

phase_path = os.path.join(PROJECT_ROOT, "outputs", "KIC_11446443_phasefolded.png")
plt.savefig(phase_path, dpi=150)
print(f"Phase folded plot saved to: {phase_path}")