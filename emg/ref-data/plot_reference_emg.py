"""
plot_reference_emg.py
=====================
Plots the reference EMG profiles from 3-gold/reference_emg.csv.

Each muscle is shown with:
  - A solid line  → grand mean across 97 subjects
  - A shaded band → mean ± 1 SD  (fluctuation / normal corridor)

Two figures are produced:
  Figure 1 – All 4 muscles on one shared axes (overlay view)
  Figure 2 – 2×2 subplot grid, one panel per muscle (detailed view)

Outputs saved to 3-gold/plots/:
  emg_reference_overlay.png
  emg_reference_subplots.png

Usage:
  python plot_reference_emg.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "3-gold", "reference_emg.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "3-gold", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 2. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

x = df["gait_cycle_pct"].values          # 0 to 100 (integer, 101 points)

muscles = {
    #  key      display name              mean column    sd column      color
    "VL": ("Vastus Lateralis (VL)",    "VL_mean",  "VL_sd",  "#1565C0"),  # blue
    "TA": ("Tibialis Anterior (TA)",   "TA_mean",  "TA_sd",  "#E65100"),  # orange
    "ST": ("Semitendinosus (ST)",      "ST_mean",  "ST_sd",  "#2E7D32"),  # green
    "GM": ("Gastrocnemius Med. (GM)",  "GM_mean",  "GM_sd",  "#6A1B9A"),  # purple
}

# ── 3. Figure 1 – Overlay (all muscles on one axes) ───────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 5))

for key, (label, mean_col, sd_col, color) in muscles.items():
    mean = df[mean_col].values
    sd   = df[sd_col].values

    # solid line = grand mean
    ax1.plot(x, mean,
             color=color, linewidth=2.2, label=label, zorder=3)

    # shaded band = mean ± 1 SD
    ax1.fill_between(x, mean - sd, mean + sd,
                     color=color, alpha=0.15, zorder=2)

# gait-phase reference bands
ax1.axvspan(0,  60, alpha=0.04, color="gray",  label="Stance phase (0–60%)")
ax1.axvspan(60, 100, alpha=0.07, color="navy", label="Swing phase (60–100%)")
ax1.axvline(60, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

ax1.set_xlabel("Gait Cycle (%)", fontsize=12)
ax1.set_ylabel("Normalized EMG  (0 – 1)", fontsize=12)
ax1.set_title("Reference EMG Profiles — All Muscles\n"
              "Solid line = grand mean  |  Shaded band = mean ± 1 SD  "
              "(97 subjects, peak-per-step normalization)",
              fontsize=11, pad=10)
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.05, 1.10)
ax1.set_xticks(range(0, 101, 10))
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

fig1.tight_layout()
out1 = os.path.join(PLOTS_DIR, "emg_reference_overlay.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")

# ── 4. Figure 2 – 2×2 Subplots (one panel per muscle) ────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=False)
axes = axes.flatten()

for idx, (key, (label, mean_col, sd_col, color)) in enumerate(muscles.items()):
    ax  = axes[idx]
    mean = df[mean_col].values
    sd   = df[sd_col].values
    upper = mean + sd
    lower = mean - sd

    # gait-phase background
    ax.axvspan(0,  60,  alpha=0.05, color="gray")
    ax.axvspan(60, 100, alpha=0.10, color="navy")
    ax.axvline(60, color="gray", linewidth=0.9, linestyle="--", alpha=0.5)

    # fluctuation band (mean ± 1 SD)
    ax.fill_between(x, lower, upper,
                    color=color, alpha=0.20,
                    label="Mean ± 1 SD")

    # upper and lower SD boundary lines (dashed, thin)
    ax.plot(x, upper, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.plot(x, lower, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    # grand mean solid line
    ax.plot(x, mean, color=color, linewidth=2.5, label="Grand mean", zorder=4)

    # mark peak
    peak_idx = int(np.argmax(mean))
    ax.scatter(peak_idx, mean[peak_idx],
               color=color, s=60, zorder=5, edgecolors="white", linewidths=1.2)
    ax.annotate(f"Peak: {peak_idx}%",
                xy=(peak_idx, mean[peak_idx]),
                xytext=(peak_idx + 5, mean[peak_idx] + 0.04),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_title(label, fontsize=11, color=color, fontweight="bold")
    ax.set_ylabel("Normalized EMG", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(-0.05, lower.min() - 0.05),
                min(1.15,  upper.max() + 0.08))
    ax.set_xticks(range(0, 101, 10))
    ax.grid(True, alpha=0.3, linestyle="--")

    # legend (only first panel, to avoid repetition)
    if idx == 0:
        line_patch  = mpatches.Patch(color=color, alpha=0.9,
                                      label="Grand mean (solid line)")
        band_patch  = mpatches.Patch(color=color, alpha=0.25,
                                      label="Mean ± 1 SD (shaded band)")
        stance_patch = mpatches.Patch(color="gray", alpha=0.15,
                                       label="Stance phase (0–60%)")
        swing_patch  = mpatches.Patch(color="navy", alpha=0.25,
                                       label="Swing phase (60–100%)")
        ax.legend(handles=[line_patch, band_patch, stance_patch, swing_patch],
                  fontsize=7.5, loc="lower right", framealpha=0.85)

# x-axis label on bottom row only
for ax in axes[2:]:
    ax.set_xlabel("Gait Cycle (%)", fontsize=10)

fig2.suptitle(
    "Reference EMG Profiles — Individual Muscles\n"
    "Grand mean ± 1 SD  |  97 subjects  |  Peak-per-step normalization",
    fontsize=12, y=1.01
)
fig2.tight_layout()
out2 = os.path.join(PLOTS_DIR, "emg_reference_subplots.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

# ── 5. Quick data summary ──────────────────────────────────────────────────────
print("\n== Reference EMG Summary ==================================")
print(f"  Data source  : {CSV_PATH}")
print(f"  Gait points  : 101  (0% to 100%)")
print(f"  Subjects     : 97   (outlier-pruned from 120)")
print(f"  Normalization: Peak-per-step  -> scale 0 to 1")
print()
print(f"  {'Muscle':<6}  {'Peak at':<10}  {'Mean range':<22}  {'Avg SD'}")
print(f"  {'------':<6}  {'-------':<10}  {'----------':<22}  {'------'}")
for key, (label, mean_col, sd_col, color) in muscles.items():
    mean = df[mean_col].values
    sd   = df[sd_col].values
    peak = int(np.argmax(mean))
    print(f"  {key:<6}  {peak:>3}% gc      "
          f"[{mean.min():.3f} - {mean.max():.3f}]           "
          f"{sd.mean():.3f}")
print("===========================================================")
print(f"\nPlots saved to: {PLOTS_DIR}")
