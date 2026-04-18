"""
Phase 2 – Silver Layer: Quality Filtering & Amplitude Normalization
===================================================================
Input : 1-bronze/metadata_master.csv  (list of valid files from Phase 1)
        output/*.csv                (raw step files)

Steps:
  A. Load only files that passed Bronze integrity checks.
  B. Quality filtering – reject steps where:
       • Any channel is flat (std < flat_threshold)
       • Any channel shows clipping (>= 10 consecutive identical samples)
       • Peak-to-mean ratio < 1.5 for any channel  (electrode dropout)
       • Channel maximum is zero (dead channel)
  C. Amplitude normalization – Method B (Peak-Per-Step):
       Each channel divided by its own maximum within the step.
       Produces values in [0, 1] for every retained step.
  D. Save:
       2-silver/normalized_data.npy  – float32 array (N_valid, 101, 4)
       2-silver/valid_files.csv      – list of retained files + metadata
       2-silver/quality_report.csv  – per-file pass/fail with reasons
"""

import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BRONZE_DIR = os.path.join(BASE_DIR, "1-bronze")
SILVER_DIR = os.path.join(BASE_DIR, "2-silver")
os.makedirs(SILVER_DIR, exist_ok=True)

MUSCLES = ["VL", "TA", "ST", "GM"]

# ── quality thresholds ─────────────────────────────────────────────────────────
FLAT_STD_THRESHOLD  = 1e-6   # channel is flat if std < this
CLIP_RUN_LENGTH     = 10     # consecutive identical samples → clipping
PEAK_MEAN_MIN_RATIO = 1.5    # peak/mean must exceed this for valid activation


# ── helpers ────────────────────────────────────────────────────────────────────
def has_clipping(channel: np.ndarray, run_length: int = CLIP_RUN_LENGTH) -> bool:
    """Return True if any run of identical consecutive values >= run_length."""
    if len(channel) < run_length:
        return False
    diffs = np.diff(channel)
    count = 1
    for d in diffs:
        if abs(d) < 1e-10:
            count += 1
            if count >= run_length:
                return True
        else:
            count = 1
    return False


def quality_filter(arr: np.ndarray) -> tuple[bool, list[str]]:
    """
    Evaluate a (101, 4) array.
    Returns (passes: bool, reasons: list[str]).
    """
    reasons = []
    for i, muscle in enumerate(MUSCLES):
        ch = arr[:, i]
        ch_max  = ch.max()
        ch_std  = ch.std()
        ch_mean = ch.mean()

        # dead channel
        if ch_max == 0.0:
            reasons.append(f"{muscle}:DEAD_CHANNEL")
            continue

        # flat signal
        if ch_std < FLAT_STD_THRESHOLD:
            reasons.append(f"{muscle}:FLAT(std={ch_std:.2e})")

        # clipping
        if has_clipping(ch):
            reasons.append(f"{muscle}:CLIPPING")

        # peak-to-mean ratio
        if ch_mean > 0:
            ratio = ch_max / ch_mean
            if ratio < PEAK_MEAN_MIN_RATIO:
                reasons.append(f"{muscle}:LOW_PEAK_MEAN(ratio={ratio:.2f})")

    return (len(reasons) == 0), reasons


def normalize_step(arr: np.ndarray) -> np.ndarray:
    """
    Peak-per-step normalization (Method B).
    Each channel divided by its maximum; result in [0, 1].
    Channels with max == 0 are left as-is (already zero).
    """
    normed = arr.copy().astype(np.float32)
    for i in range(arr.shape[1]):
        ch_max = arr[:, i].max()
        if ch_max > 0:
            normed[:, i] = arr[:, i] / ch_max
    return normed


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 – SILVER LAYER")
    print("=" * 60)

    # load bronze metadata
    meta_path = os.path.join(BRONZE_DIR, "metadata_master.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "metadata_master.csv not found. Run phase1_bronze.py first."
        )
    meta_df = pd.read_csv(meta_path)

    # load integrity report to get passing files
    integ_path = os.path.join(BRONZE_DIR, "integrity_report.csv")
    integ_df = pd.read_csv(integ_path)
    bronze_pass = set(integ_df.loc[integ_df["pass"] == True, "filename"].tolist())
    print(f"Bronze-passing files : {len(bronze_pass)}")

    # filter metadata to bronze-passing only
    meta_df = meta_df[meta_df["filename"].isin(bronze_pass)].reset_index(drop=True)

    quality_rows  = []
    valid_rows    = []
    normalized_arrays = []

    for i, row in meta_df.iterrows():
        fname = row["filename"]
        fpath = row["file_path"]

        # load
        try:
            arr = pd.read_csv(fpath, header=None).values.astype(np.float64)
        except Exception as e:
            quality_rows.append({
                "filename": fname, "subject": row["subject"],
                "trial": row["trial"], "step": row["step"],
                "pass": False, "reasons": f"READ_ERROR:{e}",
            })
            continue

        # quality filter
        passes, reasons = quality_filter(arr)
        quality_rows.append({
            "filename": fname,
            "subject":  row["subject"],
            "trial":    row["trial"],
            "step":     row["step"],
            "pass":     passes,
            "reasons":  "; ".join(reasons) if reasons else "PASS",
        })

        if not passes:
            continue

        # normalize
        normed = normalize_step(arr)
        normalized_arrays.append(normed)
        valid_rows.append({
            "filename": fname,
            "subject":  row["subject"],
            "trial":    row["trial"],
            "step":     row["step"],
            "array_idx": len(normalized_arrays) - 1,
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(meta_df):
            print(f"  Processed {i+1}/{len(meta_df)} | Valid so far: {len(valid_rows)}")

    # stack into 3-D array
    if not normalized_arrays:
        raise RuntimeError("No valid files survived quality filtering.")

    data_3d = np.stack(normalized_arrays, axis=0)   # (N, 101, 4)
    print(f"\nNormalized array shape: {data_3d.shape}")

    # save
    npy_path    = os.path.join(SILVER_DIR, "normalized_data.npy")
    valid_path  = os.path.join(SILVER_DIR, "valid_files.csv")
    quality_path = os.path.join(SILVER_DIR, "quality_report.csv")

    np.save(npy_path, data_3d)
    pd.DataFrame(valid_rows).to_csv(valid_path, index=False)
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)

    # summary
    n_total = len(quality_rows)
    n_pass  = len(valid_rows)
    n_fail  = n_total - n_pass

    print("\n── SILVER SUMMARY ──────────────────────────────────────")
    print(f"  Bronze-passed files   : {n_total}")
    print(f"  Quality-passed steps  : {n_pass}")
    print(f"  Quality-rejected      : {n_fail}  ({n_fail/n_total*100:.1f}%)")
    print(f"  normalized_data.npy   → {npy_path}  shape={data_3d.shape}")
    print(f"  valid_files.csv       → {valid_path}")
    print(f"  quality_report.csv    → {quality_path}")
    print("────────────────────────────────────────────────────────\n")

    return data_3d, pd.DataFrame(valid_rows)


if __name__ == "__main__":
    main()
