"""
Phase 4 – Validation and Statistical Evaluation
================================================
Input : 3-gold/reference_emg.csv        – final mean ± SD profiles
        3-gold/subject_profiles.npy     – per-subject arrays (K, 101, 4)

Steps:
  A. Physiological plausibility check
       Verify that each muscle's peak activation falls within the
       expected gait-cycle window from the biomechanics literature.
  B. Intraclass Correlation Coefficient (ICC)
       Compute ICC(3,k) across subject-level profiles for each muscle.
       Good reliability: ICC > 0.75; Excellent: ICC > 0.90.
  C. Visualizations
       1. Spaghetti plot – all subject profiles overlaid per muscle
       2. Reference band plot – grand mean ± 1 SD per muscle
  D. Ridge Regression functional validation
       Feature : step-wise normalized EMG at each gait-cycle point
       Target  : gait_cycle_pct (proxy for phase progression)
       Metric  : R², RMSE on held-out 20% (stratified by subject)

Outputs:
  3-gold/plots/spaghetti_{muscle}.png
  3-gold/plots/reference_band.png
  3-gold/validation_report.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR   = os.path.join(BASE_DIR, "3-gold")
PLOTS_DIR  = os.path.join(GOLD_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MUSCLES = ["VL", "TA", "ST", "GM"]

# ── expected activation windows (% gait cycle) ─────────────────────────────────
# Source: Winter (2009); Perry & Burnfield (2010); Konrad (2005)
# Each muscle may have multiple valid activation windows (primary + secondary).
EXPECTED_PEAKS = {
    "VL": [(0,  25)],               # early stance – weight acceptance
    "TA": [(0,   5), (60, 100)],    # initial contact (0-5%) AND swing (60-100%)
    "ST": [(0,  15), (80, 100)],    # early stance (0-15%) AND late swing (80-100%)
    "GM": [(30,  60)],              # late stance – push-off propulsion
}

REPORT_LINES = []


def log(msg: str):
    print(msg)
    REPORT_LINES.append(msg)


# ── A. Physiological plausibility ─────────────────────────────────────────────
def check_physiological_plausibility(ref_df: pd.DataFrame) -> bool:
    log("\n── A. Physiological Plausibility Check ──")
    all_ok = True
    for muscle, windows in EXPECTED_PEAKS.items():
        col = f"{muscle}_mean"
        if col not in ref_df.columns:
            log(f"  {muscle}: MISSING column '{col}'")
            all_ok = False
            continue
        series   = ref_df[col].values
        peak_pct = int(np.argmax(series))
        ok = any(lo <= peak_pct <= hi for lo, hi in windows)
        window_str = " or ".join(f"{lo}–{hi}%" for lo, hi in windows)
        status = "OK" if ok else "WARNING"
        log(
            f"  {muscle}: peak at {peak_pct}% gait cycle "
            f"(expected {window_str})  [{status}]"
        )
        if not ok:
            all_ok = False
    return all_ok


# ── B. ICC computation (ICC 3,k – two-way mixed, consistency, average) ─────────
def compute_icc(profiles: np.ndarray) -> dict[str, float]:
    """
    profiles: (K, 101, 4)  where K = number of subjects
    Returns dict {muscle: ICC value}
    """
    K = profiles.shape[0]   # subjects (raters)
    n = profiles.shape[1]   # time points (targets / items)
    icc_values = {}

    for i, muscle in enumerate(MUSCLES):
        X = profiles[:, :, i].T   # (n=101, K subjects)

        grand_mean = X.mean()
        subj_means = X.mean(axis=0)  # (K,)
        item_means = X.mean(axis=1)  # (n,)

        SSt  = ((X - grand_mean) ** 2).sum()
        SSs  = K  * ((item_means - grand_mean) ** 2).sum()   # between items (rows)
        SSr  = n  * ((subj_means - grand_mean) ** 2).sum()   # between subjects (cols)
        SSe  = SSt - SSs - SSr

        MSs  = SSs / (n - 1)
        MSe  = SSe / ((n - 1) * (K - 1))

        if (MSs + (K - 1) * MSe) > 0:
            icc = (MSs - MSe) / (MSs + (K - 1) * MSe)
        else:
            icc = float("nan")

        icc_values[muscle] = float(np.clip(icc, -1.0, 1.0))

    return icc_values


# ── C. Visualizations ─────────────────────────────────────────────────────────
def plot_spaghetti(profiles: np.ndarray, ref_df: pd.DataFrame):
    gait_pct = np.arange(101)
    for i, muscle in enumerate(MUSCLES):
        fig, ax = plt.subplots(figsize=(9, 4))
        # individual subject profiles
        for k in range(profiles.shape[0]):
            ax.plot(gait_pct, profiles[k, :, i], color="steelblue",
                    alpha=0.25, linewidth=0.8)
        # grand mean
        ax.plot(gait_pct, ref_df[f"{muscle}_mean"].values,
                color="crimson", linewidth=2.5, label="Grand mean")
        ax.set_xlabel("Gait cycle (%)", fontsize=11)
        ax.set_ylabel("Normalized EMG", fontsize=11)
        ax.set_title(f"Spaghetti Plot – {muscle}", fontsize=13)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"spaghetti_{muscle}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log(f"  Saved: {out}")


def plot_reference_band(ref_df: pd.DataFrame):
    gait_pct = np.arange(101)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    colors = {"VL": "#1f77b4", "TA": "#ff7f0e", "ST": "#2ca02c", "GM": "#d62728"}

    for i, muscle in enumerate(MUSCLES):
        ax   = axes[i]
        mean = ref_df[f"{muscle}_mean"].values
        sd   = ref_df[f"{muscle}_sd"].values
        c    = colors[muscle]
        ax.fill_between(gait_pct, mean - sd, mean + sd, alpha=0.25, color=c,
                        label="Mean ± 1 SD")
        ax.plot(gait_pct, mean, color=c, linewidth=2.0, label="Grand mean")
        ax.set_title(muscle, fontsize=12)
        ax.set_ylabel("Normalized EMG", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in axes[2:]:
        ax.set_xlabel("Gait cycle (%)", fontsize=10)

    fig.suptitle("Reference EMG Profiles – Grand Mean ± 1 SD", fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "reference_band.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"  Saved: {out}")


# ── D. Ridge Regression functional validation ──────────────────────────────────
def ridge_validation(profiles: np.ndarray):
    """
    Feature : (K*101, 4) – each row = 4-muscle EMG at one gait-cycle point
    Target  : gait_cycle_pct (0–100), repeated for each subject
    Groups  : subject index (for stratified split)
    """
    log("\n── D. Ridge Regression Functional Validation ──")
    K, T, C = profiles.shape   # subjects, time points, channels
    X = profiles.reshape(-1, C)                       # (K*101, 4)
    y = np.tile(np.arange(T), K).astype(float)       # (K*101,)
    groups = np.repeat(np.arange(K), T)               # (K*101,)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])

    r2   = r2_score(y[test_idx], y_pred)
    rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
    nrmse = rmse / (y.max() - y.min()) * 100.0

    log(f"  Train size: {len(train_idx)}  |  Test size: {len(test_idx)}")
    log(f"  Ridge (α=1) – R² = {r2:.4f}  |  RMSE = {rmse:.2f}%  |  NRMSE = {nrmse:.2f}%")

    if r2 >= 0.70:
        log("  → Model quality: ACCEPTABLE (R² ≥ 0.70)")
    elif r2 >= 0.50:
        log("  → Model quality: MODERATE (R² ≥ 0.50)")
    else:
        log("  → Model quality: LOW – review reference data or model features")

    return r2, rmse, nrmse


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("PHASE 4 – VALIDATION")
    log("=" * 60)

    ref_path   = os.path.join(GOLD_DIR, "reference_emg.csv")
    subj_path  = os.path.join(GOLD_DIR, "subject_profiles.npy")
    if not os.path.exists(ref_path) or not os.path.exists(subj_path):
        raise FileNotFoundError(
            "Gold outputs not found. Run phase3_gold.py first."
        )

    ref_df   = pd.read_csv(ref_path)
    profiles = np.load(subj_path)   # (K, 101, 4)
    K = profiles.shape[0]
    log(f"Reference profiles loaded: {K} subjects  |  shape: {profiles.shape}")

    # A. Physiological plausibility
    plausible = check_physiological_plausibility(ref_df)
    log(f"  Plausibility result: {'PASS' if plausible else 'REVIEW REQUIRED'}")

    # B. ICC
    log("\n── B. Intraclass Correlation Coefficient (ICC 3,k) ──")
    icc_vals = compute_icc(profiles)
    for muscle, icc in icc_vals.items():
        rating = "Excellent" if icc >= 0.90 else ("Good" if icc >= 0.75 else "Poor")
        log(f"  {muscle}: ICC = {icc:.4f}  [{rating}]")

    # C. Plots
    log("\n── C. Visualizations ──")
    plot_spaghetti(profiles, ref_df)
    plot_reference_band(ref_df)

    # D. Ridge Regression
    r2, rmse, nrmse = ridge_validation(profiles)

    # Write report
    report_path = os.path.join(GOLD_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\n  validation_report.txt    → {report_path}")

    log("\n── VALIDATION SUMMARY ───────────────────────────────────")
    log(f"  Physiological plausibility : {'PASS' if plausible else 'REVIEW'}")
    for muscle, icc in icc_vals.items():
        log(f"  ICC({muscle})               : {icc:.4f}")
    log(f"  Ridge R²                   : {r2:.4f}")
    log(f"  Ridge RMSE                 : {rmse:.2f}% gait cycle")
    log("────────────────────────────────────────────────────────\n")

    return ref_df, profiles


if __name__ == "__main__":
    main()
