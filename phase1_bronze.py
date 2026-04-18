"""
Phase 1 – Bronze Layer: Raw Data Ingestion and Cataloging
=========================================================
Tasks:
  1. Parse filenames → subject, trial, step identifiers
  2. Compute SHA-256 hash for every file (cryptographic provenance)
  3. Verify shape: each file must be exactly 101 rows × 4 columns
  4. Check for NaN, Inf, and missing values
  5. Check amplitude plausibility (values should be finite and ≥ 0)
  6. Compute per-channel descriptive statistics
  7. Write outputs:
       1-bronze/metadata_master.csv  – one row per file, full provenance
       1-bronze/integrity_report.csv – pass/fail per file with reasons
"""

import os, hashlib, re
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
BRONZE_DIR  = os.path.join(BASE_DIR, "1-bronze")
os.makedirs(BRONZE_DIR, exist_ok=True)

# column mapping (from existing average_emg.py)
MUSCLES = ["VL", "TA", "ST", "GM"]

EXPECTED_ROWS = 101
EXPECTED_COLS = 4

# amplitude plausibility: processed EMG envelopes are non-negative
AMP_MIN = 0.0
AMP_MAX = 10.0   # generous upper bound; flag if any value exceeds this


# ── helpers ────────────────────────────────────────────────────────────────────
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(name: str) -> dict:
    """Extract subject, trial, step from 'S001_Trial01_Step02.csv'."""
    m = re.match(r"(S\d+)_Trial(\d+)_Step(\d+)", name, re.IGNORECASE)
    if m:
        return {
            "subject": m.group(1).upper(),
            "trial":   int(m.group(2)),
            "step":    int(m.group(3)),
        }
    return {"subject": "UNKNOWN", "trial": -1, "step": -1}


def check_file(path: str) -> dict:
    """
    Load one CSV and run all integrity checks.
    Returns a dict with pass/fail flags and statistics.
    """
    result = {
        "file_path": path,
        "sha256": sha256(path),
        "ok": True,
        "fail_reasons": [],
    }

    # load
    try:
        df = pd.read_csv(path, header=None)
    except Exception as e:
        result["ok"] = False
        result["fail_reasons"].append(f"READ_ERROR:{e}")
        return result

    # shape
    rows, cols = df.shape
    result["n_rows"] = rows
    result["n_cols"] = cols
    if rows != EXPECTED_ROWS or cols != EXPECTED_COLS:
        result["ok"] = False
        result["fail_reasons"].append(
            f"SHAPE_MISMATCH:expected {EXPECTED_ROWS}×{EXPECTED_COLS}, got {rows}×{cols}"
        )

    arr = df.values.astype(float)

    # NaN / Inf
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    result["nan_count"] = nan_count
    result["inf_count"] = inf_count
    if nan_count > 0:
        result["ok"] = False
        result["fail_reasons"].append(f"NAN:{nan_count}")
    if inf_count > 0:
        result["ok"] = False
        result["fail_reasons"].append(f"INF:{inf_count}")

    # amplitude plausibility
    arr_clean = arr[np.isfinite(arr)]
    global_min = float(arr_clean.min()) if arr_clean.size else float("nan")
    global_max = float(arr_clean.max()) if arr_clean.size else float("nan")
    result["global_min"] = global_min
    result["global_max"] = global_max
    if np.isfinite(global_min) and global_min < AMP_MIN:
        result["ok"] = False
        result["fail_reasons"].append(f"NEGATIVE_VALUES:min={global_min:.6f}")
    if np.isfinite(global_max) and global_max > AMP_MAX:
        result["ok"] = False
        result["fail_reasons"].append(f"AMPLITUDE_SATURATION:max={global_max:.6f}")

    # missing data fraction (NaN ratio)
    total = rows * cols
    result["missing_pct"] = round(nan_count / total * 100, 3) if total > 0 else 0.0
    if result["missing_pct"] > 5.0:
        result["ok"] = False
        result["fail_reasons"].append(f"EXCESSIVE_MISSING:{result['missing_pct']:.1f}%")

    # per-channel statistics (only if shape is correct)
    if rows == EXPECTED_ROWS and cols == EXPECTED_COLS:
        for i, muscle in enumerate(MUSCLES):
            ch = arr[:, i]
            result[f"{muscle}_mean"] = round(float(np.nanmean(ch)), 6)
            result[f"{muscle}_std"]  = round(float(np.nanstd(ch)),  6)
            result[f"{muscle}_min"]  = round(float(np.nanmin(ch)),  6)
            result[f"{muscle}_max"]  = round(float(np.nanmax(ch)),  6)

    result["fail_reasons"] = "; ".join(result["fail_reasons"]) if result["fail_reasons"] else "PASS"
    return result


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 1 – BRONZE LAYER")
    print("=" * 60)

    csv_files = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".csv") and re.match(r"S\d+_Trial\d+_Step\d+", f, re.IGNORECASE)
    )
    print(f"Found {len(csv_files)} CSV files in {OUTPUT_DIR}")

    metadata_rows = []
    integrity_rows = []

    for i, fname in enumerate(csv_files, 1):
        path = os.path.join(OUTPUT_DIR, fname)
        parsed = parse_filename(fname)
        check  = check_file(path)

        meta = {
            "filename":     fname,
            "file_path":    path,
            "subject":      parsed["subject"],
            "trial":        parsed["trial"],
            "step":         parsed["step"],
            "sha256":       check["sha256"],
            "n_rows":       check.get("n_rows", -1),
            "n_cols":       check.get("n_cols", -1),
            "nan_count":    check.get("nan_count", -1),
            "inf_count":    check.get("inf_count", -1),
            "missing_pct":  check.get("missing_pct", -1),
            "global_min":   check.get("global_min", float("nan")),
            "global_max":   check.get("global_max", float("nan")),
        }
        # append per-muscle stats
        for muscle in MUSCLES:
            for stat in ["mean", "std", "min", "max"]:
                key = f"{muscle}_{stat}"
                meta[key] = check.get(key, float("nan"))

        metadata_rows.append(meta)

        integ = {
            "filename": fname,
            "subject":  parsed["subject"],
            "trial":    parsed["trial"],
            "step":     parsed["step"],
            "pass":     check["ok"],
            "reasons":  check["fail_reasons"],
        }
        integrity_rows.append(integ)

        if i % 100 == 0 or i == len(csv_files):
            n_fail = sum(1 for r in integrity_rows if not r["pass"])
            print(f"  Processed {i}/{len(csv_files)} | Failures so far: {n_fail}")

    # save outputs
    meta_df   = pd.DataFrame(metadata_rows)
    integ_df  = pd.DataFrame(integrity_rows)

    meta_path  = os.path.join(BRONZE_DIR, "metadata_master.csv")
    integ_path = os.path.join(BRONZE_DIR, "integrity_report.csv")

    meta_df.to_csv(meta_path,  index=False)
    integ_df.to_csv(integ_path, index=False)

    # summary
    n_total = len(integrity_rows)
    n_pass  = sum(1 for r in integrity_rows if r["pass"])
    n_fail  = n_total - n_pass

    print("\n── BRONZE SUMMARY ──────────────────────────────────────")
    print(f"  Total files   : {n_total}")
    print(f"  Passed        : {n_pass}")
    print(f"  Failed        : {n_fail}")
    print(f"  metadata_master.csv  → {meta_path}")
    print(f"  integrity_report.csv → {integ_path}")
    print("────────────────────────────────────────────────────────\n")

    return n_pass, n_fail


if __name__ == "__main__":
    main()
