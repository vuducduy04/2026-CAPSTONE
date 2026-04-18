"""
run_pipeline.py – Master script: run all four pipeline phases in sequence
=========================================================================
Usage:
    python run_pipeline.py           # run all phases
    python run_pipeline.py --phase 2 # run a single phase (1-4)

Phases:
    1  Bronze  – metadata cataloging & integrity checks
    2  Silver  – amplitude normalization & quality filtering
    3  Gold    – ensemble averaging & outlier pruning
    4  Validate– ICC, physiological checks, plots, Ridge regression
"""

import sys
import time
import argparse
import traceback


def banner(title: str):
    line = "═" * 60
    print(f"\n{line}\n  {title}\n{line}")


def run_phase(num: int):
    if num == 1:
        banner("PHASE 1 – BRONZE")
        import phase1_bronze as m
        m.main()
    elif num == 2:
        banner("PHASE 2 – SILVER")
        import phase2_silver as m
        m.main()
    elif num == 3:
        banner("PHASE 3 – GOLD")
        import phase3_gold as m
        m.main()
    elif num == 4:
        banner("PHASE 4 – VALIDATION")
        import phase4_validation as m
        m.main()
    else:
        raise ValueError(f"Unknown phase: {num}")


def main():
    parser = argparse.ArgumentParser(
        description="EMG Reference Data Pipeline – run all phases or a specific one"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only this phase (1–4). Omit to run all phases."
    )
    args = parser.parse_args()

    phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4]
    phase_names = {
        1: "Bronze – Metadata & Integrity",
        2: "Silver – Normalization & Quality",
        3: "Gold   – Ensemble & Outlier Pruning",
        4: "Validation – ICC, Plots, Ridge",
    }

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   EMG Reference Data Pipeline                        ║")
    print("║   Medallion Architecture: Bronze → Silver → Gold     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"Phases scheduled: {phases_to_run}\n")

    overall_start = time.time()
    results = {}

    for phase_num in phases_to_run:
        t0 = time.time()
        try:
            run_phase(phase_num)
            elapsed = time.time() - t0
            results[phase_num] = ("OK", elapsed)
            print(f"\n[Phase {phase_num}] Completed in {elapsed:.1f}s\n")
        except Exception:
            elapsed = time.time() - t0
            results[phase_num] = ("FAILED", elapsed)
            print(f"\n[Phase {phase_num}] FAILED after {elapsed:.1f}s")
            traceback.print_exc()
            print(
                "\nPipeline halted. Fix the error above before proceeding to the next phase."
            )
            sys.exit(1)

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Pipeline Complete – Summary                        ║")
    print("╠══════════════════════════════════════════════════════╣")
    for phase_num, (status, elapsed) in results.items():
        name = phase_names[phase_num]
        print(f"║  Phase {phase_num}  {name:<38} {status:>6}  {elapsed:>5.1f}s  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total elapsed: {total_elapsed:.1f}s" + " " * (38 - len(f"{total_elapsed:.1f}")) + "              ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print("Key output files:")
    print("  1-bronze/metadata_master.csv   – full file provenance")
    print("  1-bronze/integrity_report.csv  – pass/fail per raw file")
    print("  2-silver/normalized_data.npy   – quality-filtered, normalized steps")
    print("  2-silver/quality_report.csv    – per-step quality decisions")
    print("  3-gold/reference_emg.csv       – FINAL reference: mean ± SD per muscle")
    print("  3-gold/subject_profiles.npy    – per-subject mean profiles")
    print("  3-gold/outlier_report.csv      – subjects removed as outliers")
    print("  3-gold/variability_summary.csv – CV statistics per gait-cycle %")
    print("  3-gold/validation_report.txt   – full validation text report")
    print("  3-gold/plots/                  – spaghetti + reference band figures\n")


if __name__ == "__main__":
    main()
