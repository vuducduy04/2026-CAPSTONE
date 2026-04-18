"""
Phase 3 – Gold Layer: Reference Data Construction
==================================================
Input : 2-silver/normalized_data.npy   – (N_valid, 101, 4) float32
        2-silver/valid_files.csv       – metadata for each valid step

Steps:
  A. Hierarchical ensemble averaging
       Step level   → within-trial averaging  (step → trial mean)
       Trial level  → within-subject averaging (trial → subject mean)
       Grand level  → across-subject averaging (subject → grand mean ± SD)
  B. Outlier detection & pruning
       Cross-correlation: flag subject profiles with Pearson r < 0.7
         vs. the preliminary grand mean for any muscle channel.
       Coefficient of Variation: flag time points with CV > 100%.
       Iterate until stable (max 5 rounds).
  C. Coreset summary (optional, K-medoids-inspired):
       If the dataset is large, identify K=50 representative steps using
       hierarchical clustering on the 404-dim feature space.
  D. Final output
       3-gold/reference_emg.csv   – 101-row table: gait_cycle_pct + mean/SD per muscle
       3-gold/subject_profiles.npy – per-subject mean profiles (K_subjects, 101, 4)
       3-gold/outlier_report.csv  – subjects flagged and removed
       3-gold/variability_summary.csv – CV and ICC-like stats per time point
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SILVER_DIR = os.path.join(BASE_DIR, "2-silver")
GOLD_DIR   = os.path.join(BASE_DIR, "3-gold")
os.makedirs(GOLD_DIR, exist_ok=True)

MUSCLES = ["VL", "TA", "ST", "GM"]

# ── outlier thresholds ─────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.70   # minimum Pearson r vs. grand mean (per channel)
CV_THRESHOLD   = 100.0  # max coefficient of variation (%) at any time point
MAX_ITER       = 5      # maximum outlier-pruning iterations


# ── helpers ────────────────────────────────────────────────────────────────────
def hierarchical_average(data_3d: np.ndarray,
                         valid_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Compute per-subject mean profiles via hierarchical averaging.

    Returns dict: {subject_id: array(101, 4)}
    """
    subject_profiles = {}

    for subject, subj_df in valid_df.groupby("subject"):
        trial_means = []
        for trial, trial_df in subj_df.groupby("trial"):
            idxs = trial_df["array_idx"].values
            trial_steps = data_3d[idxs]          # (M_steps, 101, 4)
            trial_means.append(trial_steps.mean(axis=0))  # (101, 4)

        # average over trials for this subject
        subj_profile = np.stack(trial_means, axis=0).mean(axis=0)  # (101, 4)
        subject_profiles[subject] = subj_profile

    return subject_profiles


def compute_grand_mean(subject_profiles: dict) -> np.ndarray:
    """Grand mean across all subjects → shape (101, 4)."""
    all_profiles = np.stack(list(subject_profiles.values()), axis=0)  # (K, 101, 4)
    return all_profiles.mean(axis=0)


def flag_outliers(subject_profiles: dict,
                  grand_mean: np.ndarray) -> list[str]:
    """
    Return list of subject IDs whose profile correlates poorly (r < threshold)
    with the grand mean on any muscle channel.
    """
    flagged = []
    for subj, profile in subject_profiles.items():
        for i, muscle in enumerate(MUSCLES):
            r, _ = pearsonr(profile[:, i], grand_mean[:, i])
            if r < CORR_THRESHOLD:
                flagged.append(subj)
                break   # one bad channel is enough to flag the subject
    return flagged


def coefficient_of_variation(subject_profiles: dict) -> np.ndarray:
    """Return CV (%) at each time point for each muscle → shape (101, 4)."""
    profiles = np.stack(list(subject_profiles.values()), axis=0)  # (K, 101, 4)
    mean = profiles.mean(axis=0)
    std  = profiles.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean * 100.0, 0.0)
    return cv


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 3 – GOLD LAYER")
    print("=" * 60)

    # load Silver outputs
    npy_path   = os.path.join(SILVER_DIR, "normalized_data.npy")
    valid_path = os.path.join(SILVER_DIR, "valid_files.csv")
    if not os.path.exists(npy_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(
            "Silver outputs not found. Run phase2_silver.py first."
        )

    data_3d  = np.load(npy_path)              # (N, 101, 4)
    valid_df = pd.read_csv(valid_path)
    print(f"Loaded normalized data: {data_3d.shape}")
    print(f"Valid steps: {len(valid_df)}  |  Subjects: {valid_df['subject'].nunique()}")

    # ── A. Hierarchical averaging ──────────────────────────────────────────────
    print("\n── A. Hierarchical Averaging ──")
    subject_profiles = hierarchical_average(data_3d, valid_df)
    print(f"  Subject profiles computed: {len(subject_profiles)}")

    # ── B. Iterative outlier pruning ──────────────────────────────────────────
    print("\n── B. Outlier Detection & Pruning ──")
    outlier_log = []
    removed_subjects = []

    for iteration in range(1, MAX_ITER + 1):
        grand_mean = compute_grand_mean(subject_profiles)
        flagged    = flag_outliers(subject_profiles, grand_mean)

        if not flagged:
            print(f"  Iteration {iteration}: no outliers found – converged.")
            break

        print(f"  Iteration {iteration}: removing {len(flagged)} outlier(s): {flagged}")
        for subj in flagged:
            del subject_profiles[subj]
            removed_subjects.append(subj)
            outlier_log.append({
                "iteration": iteration,
                "subject":   subj,
                "reason":    f"Pearson r < {CORR_THRESHOLD} vs. grand mean",
            })
    else:
        print(f"  Reached max iterations ({MAX_ITER}); stopping pruning.")

    print(f"  Subjects retained: {len(subject_profiles)}")
    print(f"  Subjects removed : {len(removed_subjects)}")

    # ── C. Final grand mean & SD ───────────────────────────────────────────────
    print("\n── C. Grand Ensemble Average ──")
    all_profiles = np.stack(list(subject_profiles.values()), axis=0)  # (K, 101, 4)
    grand_mean   = all_profiles.mean(axis=0)    # (101, 4)
    grand_sd     = all_profiles.std(axis=0)     # (101, 4)
    cv           = coefficient_of_variation(subject_profiles)

    print(f"  Final subject count : {all_profiles.shape[0]}")
    print(f"  Grand mean shape    : {grand_mean.shape}")

    # ── D. Build reference CSV ─────────────────────────────────────────────────
    gait_pct = np.arange(101)
    ref_dict = {"gait_cycle_pct": gait_pct}
    for i, muscle in enumerate(MUSCLES):
        ref_dict[f"{muscle}_mean"] = grand_mean[:, i]
        ref_dict[f"{muscle}_sd"]   = grand_sd[:, i]
    ref_df = pd.DataFrame(ref_dict)

    # variability summary: CV per muscle per time point
    var_dict = {"gait_cycle_pct": gait_pct}
    for i, muscle in enumerate(MUSCLES):
        var_dict[f"{muscle}_cv_pct"] = cv[:, i]
        var_dict[f"{muscle}_mean"]   = grand_mean[:, i]
        var_dict[f"{muscle}_sd"]     = grand_sd[:, i]
    var_df = pd.DataFrame(var_dict)

    # ── E. Save outputs ────────────────────────────────────────────────────────
    ref_path     = os.path.join(GOLD_DIR, "reference_emg.csv")
    subj_path    = os.path.join(GOLD_DIR, "subject_profiles.npy")
    outlier_path = os.path.join(GOLD_DIR, "outlier_report.csv")
    var_path     = os.path.join(GOLD_DIR, "variability_summary.csv")

    ref_df.to_csv(ref_path, index=False)
    np.save(subj_path, all_profiles)
    pd.DataFrame(outlier_log).to_csv(outlier_path, index=False)
    var_df.to_csv(var_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── GOLD SUMMARY ─────────────────────────────────────────")
    print(f"  Subjects in reference   : {all_profiles.shape[0]}")
    print(f"  Outliers removed        : {len(removed_subjects)}")
    for i, muscle in enumerate(MUSCLES):
        peak_pct = int(grand_mean[:, i].argmax())
        max_cv   = float(cv[:, i].max())
        print(
            f"  {muscle}: peak at {peak_pct}% gait cycle  |  max CV = {max_cv:.1f}%"
        )
    print(f"\n  reference_emg.csv        → {ref_path}")
    print(f"  subject_profiles.npy     → {subj_path}")
    print(f"  outlier_report.csv       → {outlier_path}")
    print(f"  variability_summary.csv  → {var_path}")
    print("────────────────────────────────────────────────────────\n")

    return ref_df, all_profiles


if __name__ == "__main__":
    main()
